# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-strict

"""C++ and Python code generation for AOT-T compiled kernels.

Generates:
  - kernel.h      (header with gridDims, tuner meta, selector proto)
  - kernel.cpp    (cubin externs, loaders, launchers, selector)
  - _torch_op.cpp (torch op registration)
  - _meta.py      (Python autotuner meta function)
"""

import textwrap
from collections import Counter
from typing import Any

# @manual=//triton:triton
import triton
from generative_recommenders.ops.triton_aot.compile.arg_descriptor import (
    ArgDescriptor,
    CONSTANT_CPP_OP_CTYPE,
    CONSTANT_SELECTOR_CTYPE,
    CONSTANT_TORCH_SCHEMA,
    ConstantArg,
    PointerArg,
    scalar_cpp_op_ctype,
    scalar_torch_schema,
    ScalarArg,
)
from generative_recommenders.ops.triton_aot.compile.spec_processing import (
    KernelSpec,
    OpsUnit,
)
from generative_recommenders.ops.triton_aot.compile.stable_types import (
    PY_TYPES_TO_CPP_TYPES,
    SCALAR_TYPES,
)
from generative_recommenders.ops.triton_aot.compile.utils import (
    hash_kernel_name,
    unwrap_heuristic,
)
from generative_recommenders.ops.triton_aot.shared.compat import get_scratch_parameters
from generative_recommenders.ops.triton_aot.shared.types import AUTOTUNE_ATTRs, CTYPES
from generative_recommenders.ops.triton_aot.templates.template_utils import (
    load_template,
    render_template,
)
from triton.runtime.jit import JITFunction


# ---------------------------------------------------------------------------
# Kernel naming and binary generation
# ---------------------------------------------------------------------------


def gen_kernel_name(
    fn: Any,
    spec: KernelSpec,
    cc: int | str,
) -> str:
    name = fn.__name__
    sig = "_".join([p.replace("*", "p") for p in spec.signature.values()])
    const = "_".join(map(str, spec.constants.values()))
    cc_str = f"sm{cc}"
    autotune_configs = []
    autotune_configs.append(f"w{spec.num_warps}")
    autotune_configs.append(f"s{spec.num_stages}")
    # AMD only
    autotune_configs.append(f"matrix{spec.matrix_instr_nonkdim}")
    autotune_configs.append(f"wave{spec.waves_per_eu}")
    autotune_configs.append(f"kpack{spec.kpack}")
    # See kernel_suffix in triton/compiler/code_generator.py
    suffix = ""
    for i, _ in enumerate(spec.signature):
        suffix += str(i)
        if i in spec.divisible_by_16:
            suffix += "d"
        if i in spec.divisible_by_8:
            suffix += "e"
    return "_".join([name, cc_str, sig, const] + autotune_configs + [suffix])


def gen_cubin(kernel_name: str, kernel: Any, install_dir: str, backend: str) -> str:
    """Generate kernel binary file (.cubin or .hsaco) and return extern declaration.

    Args:
        kernel_name: Full kernel name including specialization suffix.
        kernel: Compiled Triton kernel object containing binary in kernel.asm.
        install_dir: Directory to write binary file.
        backend: GPU backend ("cuda" or "hip").

    Returns:
        C++ extern declaration for the kernel binary array.
    """
    hashed = hash_kernel_name(kernel_name)
    if backend == "hip":
        binary_file = f"{install_dir}/{hashed}.hsaco"
        with open(binary_file, "wb") as hsaco:
            hsaco.write(kernel.asm["hsaco"])
        target_symbol_name = f"{kernel_name}_cubin"
    else:
        binary_file = f"{install_dir}/{hashed}.cubin"
        with open(binary_file, "wb") as cubin:
            cubin.write(kernel.asm["cubin"])
        target_symbol_name = f"{kernel_name}_cubin"

    # We return extern declarations for both the array and its pointer.
    # The pointer is used by gen_loader() to generate R_X86_64_64 relocations
    # instead of R_X86_64_32, which allows the .triton section to be placed
    # beyond the 4GB address limit in large binaries.
    # Note: The pointer is volatile to prevent optimizer constant-propagation.
    return f'extern "C" {{ extern unsigned char {target_symbol_name}[]; extern const void* volatile {target_symbol_name}_ptr; }}'


def gen_loader(kernel_name: str, cubin_name: str, shared: int) -> str:
    # TODO(changpan): Extract inline cuModuleLoadData/cuModuleGetFunction error
    # handling into a shared helper to reduce generated code size.
    return textwrap.dedent(
        f"""
        CUfunction load_{kernel_name}(void)
        {{
            thread_local std::unordered_map<int32_t, CUfunction> cache;
            auto idx = torch::stable::accelerator::getCurrentDeviceIndex();
            auto res = cache.find(idx);
            if (res != cache.end()) {{
                return res->second;
            }}
            CUfunction func;
            CUmodule mod_ptr;
            CUresult err;
            // Use pointer to cubin data to generate R_X86_64_64 relocation
            // instead of R_X86_64_32, allowing cubin data to be placed beyond 4GB
            const void *image = {kernel_name}_cubin_ptr;

            err = cuModuleLoadData(&mod_ptr, image);
            if (err != 0) {{
                const char* errStr;
                cuGetErrorString(err, &errStr);
                throw std::runtime_error("cuModuleLoadData failed for {kernel_name}: error " + std::to_string(err) + " (" + (errStr ? errStr : "unknown") + ")");
            }}

            err = cuModuleGetFunction(&func, mod_ptr, "{cubin_name}");
            if (err != 0) {{
                const char* errStr;
                cuGetErrorString(err, &errStr);
                throw std::runtime_error("cuModuleGetFunction failed for {kernel_name}: error " + std::to_string(err) + " (" + (errStr ? errStr : "unknown") + ")");
            }}

            check_errors({shared}, func);
            cache.emplace(idx, func);
            return func;
        }}
    """
    )


# ---------------------------------------------------------------------------
# Launcher codegen (per-spec)
# ---------------------------------------------------------------------------


def gen_launcher_params(
    descriptors: list[ArgDescriptor],
    signature: dict[int, str],
) -> str:
    args = ["gridDims grid"]
    for d in descriptors:
        if d.index in signature:
            if isinstance(d, PointerArg):
                ctype = "void*"
            else:
                ctype = CTYPES[signature[d.index]]
            args.append(f"{ctype} {d.name}")
    return ", ".join(args)


def gen_launch_args(
    func: JITFunction[list[Any]],
    spec: KernelSpec,
) -> list[str]:
    """Generate kernel launch argument list (pointers to non-constant arguments)."""
    args = []
    for i, arg in enumerate(func.arg_names):
        if i in spec.constants:
            continue
        assert i in spec.signature, f"Argument {i} ({arg}) does not appear in signature"
        args.append(f"&{arg}")
    return args


def gen_launcher(
    kernel_name: str,
    func: JITFunction[list[Any]],
    kernel: Any,
    shared: int,
    warp_size: int,
    spec: KernelSpec,
    descriptors: list[ArgDescriptor],
) -> str:
    params = gen_launcher_params(descriptors, spec.signature)
    args = gen_launch_args(func, spec)

    scratch_declarations, scratch_args = get_scratch_parameters(kernel)
    args.extend(scratch_args)

    args_str = ", ".join(args)

    return textwrap.dedent(
        f"""
        void {kernel_name}({params}) {{
            CUfunction func = load_{kernel_name}();
            cudaStream_t stream = grid.stream ? grid.stream : triton_aot_get_current_stream();
            {scratch_declarations}
            void *args[] = {{ {args_str} }};
            auto res = cuLaunchKernel(func, grid.x, grid.y, grid.z, {warp_size} * {spec.num_warps}, 1, 1, {shared}, stream, args, NULL);
            TRITON_AOT_CU_CHECK(res);
        }}
    """
    )


# ---------------------------------------------------------------------------
# Selector codegen (invariant)
# ---------------------------------------------------------------------------


def gen_selector_params(
    descriptors: list[ArgDescriptor],
) -> str:
    """Generate C++ selector function parameter list."""
    args = ["gridDims grid"]
    for d in descriptors:
        if isinstance(d, PointerArg):
            args.append(f"const std::optional<torch::stable::Tensor>& {d.name}")
        elif isinstance(d, ScalarArg):
            args.append(f"{CTYPES[d.triton_dtype]} {d.name}")
        elif isinstance(d, ConstantArg):
            args.append(f"{CONSTANT_SELECTOR_CTYPE[d.python_type]} {d.name}")

    for name, value in AUTOTUNE_ATTRs.items():
        args.append(f"{type(value).__name__} {name}")
    return ", ".join(args)


def gen_launcher_call_args(
    descriptors: list[ArgDescriptor],
    signature: dict[int, str],
) -> str:
    args = ["grid"]
    for d in descriptors:
        if d.index in signature:
            if isinstance(d, PointerArg):
                args.append(f"{d.name}.value().data_ptr()")
            elif isinstance(d, ScalarArg) and signature[d.index] != d.triton_dtype:
                args.append(f"static_cast<{CTYPES[signature[d.index]]}>({d.name})")
            else:
                args.append(d.name)
    return ", ".join(args)


def gen_guarded_calls(  # noqa: C901
    func: JITFunction[list[Any]],
    unit: OpsUnit,
    descriptors: list[ArgDescriptor],
) -> str:
    desc_by_idx: dict[int, ArgDescriptor] = {d.index: d for d in descriptors}
    calls = []
    for spec in unit.specs:
        kernel_name = gen_kernel_name(func, spec, unit.cc)
        args = gen_launcher_call_args(descriptors, spec.signature)
        guards = ""

        # Guard on tensor dtypes (per-spec: different specs may have different dtypes)
        for i, ttype in spec.signature.items():
            d = desc_by_idx[i]
            if not isinstance(d, PointerArg):
                continue
            arg = d.name
            atype = SCALAR_TYPES[ttype]
            guards += f"if ({arg}.has_value()) "
            guards += f"if ({arg}.value().scalar_type() == {atype}) "

        # Guard on int range (spec uses narrower type than selector)
        for i, dtype in spec.signature.items():
            d = desc_by_idx[i]
            if isinstance(d, ScalarArg) and dtype != d.triton_dtype:
                if dtype == "i32":
                    guards += f"if (fits_i32({d.name})) "

        # Guard on constant values.
        for i, val in spec.constants.items():
            arg = desc_by_idx[i].name
            if isinstance(val, bool):
                guards += f"if ({arg}) " if val else f"if (!({arg})) "
            elif isinstance(val, str):
                guards += f'if ({arg} == "{val}") '
            elif val is None:
                guards += f"if (!{arg}.has_value()) "
            else:
                guards += f"if ({arg} == {val}) "

        # Guard on special constants
        for name in AUTOTUNE_ATTRs.keys():
            guards += f"if ({name} == {getattr(spec, name)}) "

        # Guard on divisible_by_16
        for i in spec.divisible_by_16:
            arg = desc_by_idx[i].name
            if i in spec.signature:
                ttype = spec.signature[i]
                if ttype.startswith("*"):
                    guards += f"if ((((uintptr_t){arg}.value().data_ptr()) % 16) == 0) "
                else:
                    guards += f"if (({arg} % 16) == 0) "
            elif i in spec.constants:
                assert (spec.constants[i] % 16) == 0

        # Guard on divisible_by_8
        for i in spec.divisible_by_8:
            arg = desc_by_idx[i].name
            if i in spec.signature:
                ttype = spec.signature[i]
                # divisible_by_8 is only applied to int
                if not ttype.startswith("*"):
                    guards += f"if (({arg} % 8) == 0) "
            elif i in spec.constants:
                assert (spec.constants[i] % 8) == 0

        # Call the specialization.
        calls.append(f"{guards}return {kernel_name}({args});\n")
    return "".join(calls)


def gen_selector_proto(
    descriptors: list[ArgDescriptor],
    func_name: str,
) -> str:
    params = gen_selector_params(descriptors)
    # Add Triton's default values for num warps/stages, etc
    for name, value in AUTOTUNE_ATTRs.items():
        params = params.replace(name, f"{name}={value}")
    return f"void {func_name}({params});"


def gen_failure_msg(
    descriptors: list[ArgDescriptor],
) -> str:
    """Generate C++ ``<<``-chain for the dispatch-failure error message.

    Groups parameters by category (Tensors / Scalars / Constants /
    Autotune / Device).  Tensor entries include aligned16 status.
    """
    tensors: list[str] = []
    scalars: list[str] = []
    constants: list[str] = []

    for d in descriptors:
        if isinstance(d, PointerArg):
            dtype_expr = (
                f"({d.name}.has_value()"
                f" ? c10::toString({d.name}.value().scalar_type())"
                f' : "nullptr")'
            )
            align_expr = (
                f"(({d.name}.has_value()"
                f" && (((uintptr_t){d.name}.value().data_ptr()) % 16) == 0)"
                f' ? "true" : "false")'
            )
            tensors.append(
                f'" {d.name}=" << {dtype_expr} << "(aligned16=" << {align_expr} << ")"'
            )
        elif isinstance(d, ScalarArg):
            scalars.append(f'" {d.name}=" << {d.name}')
        elif isinstance(d, ConstantArg):
            constants.append(f'" {d.name}=" << {d.name}')

    autotune: list[str] = [f'" {n}=" << {n}' for n in AUTOTUNE_ATTRs]

    sections: list[str] = []
    if tensors:
        sections.append('"\\n  Tensors:" << ' + " << ".join(tensors))
    if scalars:
        sections.append('"\\n  Scalars:" << ' + " << ".join(scalars))
    if constants:
        sections.append('"\\n  Constants:" << ' + " << ".join(constants))
    sections.append('"\\n  Autotune:" << ' + " << ".join(autotune))
    sections.append('"\\n  Device: cc=" << cc')

    return " << ".join(sections)


def gen_selector(
    func: JITFunction[list[Any]],
    unit: OpsUnit,
    descriptors: list[ArgDescriptor],
) -> str:
    params = gen_selector_params(descriptors)
    guarded_calls = gen_guarded_calls(func, unit, descriptors)
    failure_msg = gen_failure_msg(descriptors)
    return f"""
        void {func.__name__}({params}) {{
            auto cc = compute_capability();
            if (grid.x * grid.y * grid.z > 0) {{
                {guarded_calls}
                std::stringstream ss;
                ss << "[TritonAOT] No implementation found for {func.__name__}" << {failure_msg};
                throw std::runtime_error(ss.str());
            }}
        }}
    """


# ---------------------------------------------------------------------------
# Torch op codegen (invariant)
# ---------------------------------------------------------------------------


def gen_cpp_op_params(
    descriptors: list[ArgDescriptor],
) -> str:
    args = []
    for d in descriptors:
        if isinstance(d, PointerArg):
            args.append(f"std::optional<torch::stable::Tensor> {d.name}")
        elif isinstance(d, ScalarArg):
            args.append(f"{scalar_cpp_op_ctype(d.triton_dtype)} {d.name}")
        elif isinstance(d, ConstantArg):
            args.append(f"{CONSTANT_CPP_OP_CTYPE[d.python_type]} {d.name}")
    for name, value in AUTOTUNE_ATTRs.items():
        args.append(f"{PY_TYPES_TO_CPP_TYPES[type(value)]} {name}")
    return ", ".join(args)


def gen_torch_op_params(
    descriptors: list[ArgDescriptor],
    default_values: dict[str, Any],
) -> str:
    args = []

    def gen_str_wrap(value: Any) -> Any:
        return f'\\"{value}\\"' if isinstance(value, str) else value

    def gen_default_str(arg: str) -> str:
        return (
            f" = {gen_str_wrap(default_values[arg])}" if arg in default_values else ""
        )

    for d in descriptors:
        df_str = gen_default_str(d.name)
        if isinstance(d, PointerArg):
            t = chr(ord("a") + d.index)
            args.append(f"Tensor({t}!)? {d.name}")
        elif isinstance(d, ScalarArg):
            args.append(f"{scalar_torch_schema(d.triton_dtype)} {d.name}{df_str}")
        elif isinstance(d, ConstantArg):
            args.append(f"{CONSTANT_TORCH_SCHEMA[d.python_type]} {d.name}{df_str}")
    for name, value in AUTOTUNE_ATTRs.items():
        args.append(f"{type(value).__name__} {name}={value}")
    return ", ".join(args)


def gen_torch_op(
    func: JITFunction[list[Any]],
    descriptors: list[ArgDescriptor],
    default_values: dict[str, Any],
) -> str:
    cpp_params = gen_cpp_op_params(descriptors)
    torch_params = gen_torch_op_params(descriptors, default_values)
    arg_names = list(func.arg_names) + list(AUTOTUNE_ATTRs.keys())
    args = ", ".join(arg_names)

    # Generate a comment noting which tensor params are non-optional but
    # promoted to Tensor? for TorchScript compatibility.
    promoted = [
        d.name for d in descriptors if isinstance(d, PointerArg) and not d.is_optional
    ]
    type_comment = ""
    if promoted:
        type_comment = (
            f"// Note: {', '.join(promoted)} are non-optional but use Tensor? "
            "for TorchScript compatibility.\n"
            "// Dispatch uses HAS_XXX constexpr ints, not tensor presence.\n"
        )
    return textwrap.dedent(
        f"""
        namespace {{
        triton::aot::gridDims dims_from_vec(
            const std::vector<int64_t>& grid
        ) {{
          return triton::aot::gridDims(
              grid.size() > 0 ? grid[0] : 1,
              grid.size() > 1 ? grid[1] : 1,
              grid.size() > 2 ? grid[2] : 1
          );
        }}

        {type_comment}void {func.__name__}_op(
            std::vector<int64_t> grid,
            {cpp_params}
        ) {{
            triton::aot::{func.__name__}(
                dims_from_vec(grid),
                {args}
            );
        }}

        void {func.__name__}_dummy_op(
            std::vector<int64_t> grid,
            {cpp_params}
        ) {{
            // Do nothing.  The op is a dummy for model transform,
            // processing, and splitting services.
        }}
        }}

        STABLE_TORCH_LIBRARY_FRAGMENT(triton_aot, m) {{
          m.def("{func.__name__}(int[] grid, {torch_params}) -> ()");
        }}
        STABLE_TORCH_LIBRARY_IMPL(triton_aot, CUDA, m) {{
          m.impl("{func.__name__}", TORCH_BOX(&{func.__name__}_op));
        }}

        STABLE_TORCH_LIBRARY_IMPL(triton_aot, CPU, m) {{
          m.impl("{func.__name__}", TORCH_BOX(&{func.__name__}_dummy_op));
        }}

        STABLE_TORCH_LIBRARY_IMPL(triton_aot, Meta, m) {{
          m.impl("{func.__name__}", TORCH_BOX(&{func.__name__}_dummy_op));
        }}
        """
    )


# ---------------------------------------------------------------------------
# Tuner meta codegen
# ---------------------------------------------------------------------------


def key_names_and_idx(func: Any) -> tuple[list[str], list[int]]:
    if hasattr(func, "key_idx"):
        arg_names = [func.arg_names[idx] for idx in func.key_idx]
        key_idx = func.key_idx
    else:
        arg_names = func.keys
        key_idx = [func.arg_names.index(arg) for arg in arg_names]
    return arg_names, key_idx


def is_non_empty_mapping_of_type(obj: object, value_type: type[Any]) -> bool:
    """Check if object is a non-empty dict with all values of specific type"""
    if not obj or not isinstance(obj, dict):
        return False

    return all(isinstance(value, value_type) for value in obj.values())


_LAUNCH_PARAM_NAMES: list[str] = ["num_warps", "num_stages"]


def gen_tuner_meta_py(
    func: Any,
    tuner_fallback: bool,
    unit: OpsUnit,
) -> str:
    vals = []

    guard_list = []

    # Use custom meta generation function if available
    if hasattr(func, "gen_autotune_select_meta_src"):
        return func.gen_autotune_select_meta_src(unit.constant_types)

    if hasattr(func, "cache") and is_non_empty_mapping_of_type(
        func.cache, triton.runtime.autotuner.Config
    ):
        # auto tuned configs
        arg_names, key_idx = key_names_and_idx(func)

        in_args = ", ".join(
            [
                f"{name}: {unit.constant_types[idx].__name__ if idx in unit.constant_types else 'int'}"
                for idx, name in zip(key_idx, arg_names)
            ]
        )

        cfg_first = next(iter(func.cache.values()))
        return_names = list(cfg_first.kwargs.keys()) + _LAUNCH_PARAM_NAMES

        for key, cfg in func.cache.items():
            val = list(cfg.kwargs.values()) + [cfg.num_warps, cfg.num_stages]
            val = tuple(val)
            vals.append(val)
            equations = []
            for arg, value in zip(arg_names, key):
                if isinstance(value, str):
                    equations.append(f"{arg} == '{value}'")
                elif isinstance(value, bool):
                    equations.append(f"{arg} == {int(value)}")
                else:
                    equations.append(f"{arg} == {value}")
            guard_list.append(f"if {' and '.join(equations)}: return {val}")

    else:
        # default configs — single spec, use specs[0]
        in_args = ""
        arg_names = list(_LAUNCH_PARAM_NAMES)
        return_names = list(_LAUNCH_PARAM_NAMES)
        val = unit.specs[0].num_warps, unit.specs[0].num_stages
        vals.append(val)

    name = unwrap_heuristic(func, JITFunction).__name__
    meta = name + "_meta"

    guards = "\n        ".join(guard_list)

    fmt_args = ", ".join([f"{{{arg_name}}}" for arg_name in arg_names])

    raise_runtime_error_str = (
        f"""raise RuntimeError(f"No autotuning config found for {name}({fmt_args})")"""
    )
    fallback_str = f"""return {Counter(vals).most_common(1)[0][0]}"""

    returns_comment = f"# Returns: ({', '.join(return_names)})"

    return textwrap.dedent(
        f"""
    def {meta}({in_args}):
        {returns_comment}
        {guards}
        {fallback_str if tuner_fallback else raise_runtime_error_str}
    """
    )


def gen_tuner_meta_cpp(
    func: Any,
    tuner_fallback: bool,
    constant_types: dict[int, type[Any]],
) -> str:
    # TODO(changpan): This C++ inline _meta is currently dead code — no C++ caller
    # invokes it.  The Python _meta.py (gen_tuner_meta_py) is the only consumer.
    # Double check, try remove this and the TUNER_META_CPP template region.
    def infer_arg_type(idx: int) -> str:
        if idx in constant_types:
            return PY_TYPES_TO_CPP_TYPES[constant_types[idx]]
        else:
            return "int64_t"

    arg_names, key_idx = key_names_and_idx(func)

    in_args = ", ".join(
        [f"{infer_arg_type(idx)} {name}" for idx, name in zip(key_idx, arg_names)]
    )

    vals = []
    guard_list = []
    for key, cfg in func.cache.items():
        val = list(cfg.kwargs.values()) + [cfg.num_warps, cfg.num_stages]
        val = tuple(val)
        vals.append(val)
        equations = []
        for arg, value in zip(arg_names, key):
            if isinstance(value, str):
                equations.append(f'{arg} == "{value}"')
            elif isinstance(value, bool):
                equations.append(f"{arg} == {int(value)}")
            else:
                equations.append(f"{arg} == {value}")
        guard_list.append(f"if ({' && '.join(equations)}) return std::make_tuple{val};")
    guards = "\n        ".join(guard_list)
    name = unwrap_heuristic(func, JITFunction).__name__
    meta = name + "_meta"
    fmt_args = ", ".join([f"{arg_name}" for arg_name in arg_names])
    raise_runtime_error_str = f"""throw std::runtime_error("No autotuning config found for {name}({fmt_args})");"""
    fallback_str = f"""return std::make_tuple{Counter(vals).most_common(1)[0][0]};"""
    # Infer the return type from the actual values
    return_type = _infer_return_type(vals[0])
    return textwrap.dedent(
        f"""
    inline std::tuple<{return_type}> {meta}({in_args}) {{
        {guards}
        {fallback_str if tuner_fallback else raise_runtime_error_str}
    }}
    """
    )


def _infer_return_type(vals: tuple[Any, ...]) -> str:
    types = [PY_TYPES_TO_CPP_TYPES.get(type(val)) for val in vals]
    try:
        # pyre-fixme[6]: For 1st argument expected
        #  `Iterable[typing_extensions.LiteralString]` but got `List[Optional[str]]`.
        return ", ".join(types)
    except TypeError:  # one of the types cannot be inferred, e.g. `None`
        raise ValueError("Cannot infer return type from `vals`")


# ---------------------------------------------------------------------------
# Top-level codegen entry points
# ---------------------------------------------------------------------------


def generate_header_content(
    tuned_func: triton.runtime.autotuner.Autotuner | None,
    func: JITFunction[list[Any]],
    unit: OpsUnit,
    descriptors: list[ArgDescriptor],
    tuner_fallback: bool,
) -> str:
    """Generate the content of the .h header file."""
    h_template = load_template("kernel.h")
    tuner_meta_cpp = (
        gen_tuner_meta_cpp(tuned_func, tuner_fallback, unit.constant_types)
        if tuned_func
        else ""
    )
    selector_proto = gen_selector_proto(descriptors, func.__name__)
    return render_template(
        h_template,
        {
            "TUNER_META_CPP": tuner_meta_cpp,
            "SELECTOR_PROTO": selector_proto,
        },
    )


def generate_kernel_cpp_content(
    func: JITFunction[list[Any]],
    unit: OpsUnit,
    descriptors: list[ArgDescriptor],
    prefix: str,
    generated_specs: list[str],
    backend: str,
) -> str:
    """Generate the content of the kernel .cpp file.

    All tensor params use Tensor? for TorchScript compatibility.
    Dispatch relies on HAS_XXX constexpr ints, not tensor presence.
    """
    cpp_template = load_template("kernel.cpp")
    kernel_specs = "\n".join(generated_specs)
    selector = gen_selector(func, unit, descriptors)

    # On AMD, apply hipification to generated code (KERNEL_SPECS, SELECTOR)
    # Templates are already hipified at load time (from hip/ subdirectory)
    if backend == "hip":
        from torch._inductor.codegen.aoti_hipify_utils import maybe_hipify_code_wrapper

        kernel_specs = maybe_hipify_code_wrapper(kernel_specs, force_hipify=True)
        selector = maybe_hipify_code_wrapper(selector, force_hipify=True)

    cpp_content = render_template(
        cpp_template,
        {
            "HEADER_INCLUDE": f'#include "{prefix}.h"\n',
            "KERNEL_SPECS": kernel_specs,
            "SELECTOR": selector,
        },
    )
    return cpp_content


def generate_torch_op_content(
    func: JITFunction[list[Any]],
    descriptors: list[ArgDescriptor],
    prefix: str,
    default_values: dict[str, Any],
) -> str:
    """Generate the content of the torch_op .cpp file."""
    torch_template = load_template("torch_op.cpp")
    torch_op_content = gen_torch_op(func, descriptors, default_values)
    torch_content = render_template(
        torch_template,
        {
            "HEADER_INCLUDE": f'#include "{prefix}.h"\n',
            "TORCH_OP": torch_op_content,
        },
    )
    return torch_content
