# pyre-strict
import ast
import inspect
import os
from typing import Any, Callable, Dict, List, Optional

from generative_recommenders.ops.triton_aot.compile.compile_state import (
    get_aott_compile_path,
    get_triton_aot_kernel_specs,
)
from generative_recommenders.ops.triton_aot.compile.utils import unwrap_heuristic
from generative_recommenders.ops.triton_aot.shared.compat import get_kernel_name
from generative_recommenders.ops.triton_aot.transform.import_utils import (
    get_original_import_header,
    rewrite_package_imports,
)
from generative_recommenders.ops.triton_aot.types import TritonAOT
from pyre_extensions import none_throws
from torch import package
from torch.fx import GraphModule

# @manual=//triton:triton
from triton.runtime.autotuner import Autotuner
from triton.runtime.jit import JITFunction, KernelInterface


def _is_torch_package_module(module_name: str) -> bool:
    """Check if a module name is from torch.package namespace."""
    return module_name.startswith("<torch_package")


def _strip_torch_package_prefix(module_name: str) -> str:
    """Strip the torch.package namespace prefix from a module name.

    Example:
        '<torch_package_0>.generative_recommenders.ops.triton_aot.triton_layer_norm'
        -> 'generative_recommenders.ops.triton_aot.triton_layer_norm'
    """
    if _is_torch_package_module(module_name):
        # Remove '<torch_package_N>.' prefix
        return module_name.split(".", 1)[1]
    return module_name


def _get_clean_module_basename(module_name: str) -> str:
    """Get the basename of a module, stripping torch.package prefix if present.

    Example:
        '<torch_package_0>.generative_recommenders.ops.triton_aot.triton_layer_norm'
        -> 'triton_layer_norm'
        'generative_recommenders.ops.triton_aot.triton_layer_norm'
        -> 'triton_layer_norm'
    """
    clean_name = _strip_torch_package_prefix(module_name)
    return clean_name.rsplit(".", 1)[-1]


def _extract_function_source(module_source: str, fn_name: str) -> str:
    """Extract a function's source code from module source.

    Parses the module source and extracts just the function definition.
    """
    tree = ast.parse(module_source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == fn_name:
            return ast.unparse(node)
    raise ValueError(f"Function '{fn_name}' not found in module source")


def _get_module_and_source(
    target: Callable[..., Any],
    package_importer: Optional[package.PackageImporter],
) -> tuple[Any, str, str]:
    """Get module, module source, and function source for a callable.

    Handles both regular modules and torch.package loaded modules.

    Args:
        target: The callable (function) to get source for
        package_importer: Optional PackageImporter for torch.package modules

    Returns:
        Tuple of (module, module_source, function_source)
    """
    module_name = target.__module__
    fn_name = target.__name__

    if _is_torch_package_module(module_name) and package_importer is not None:
        # Handle torch.package namespace
        real_module_name = _strip_torch_package_prefix(module_name)
        assert real_module_name.startswith(
            "generative_recommenders.ops.triton_aot"
        ) or real_module_name.startswith("prime_perf_optimizer"), (
            f"Expected module under 'generative_recommenders.ops.triton_aot' or 'prime_perf_optimizer', got: {real_module_name}"
        )

        # Get module source from package
        module_source = package_importer.get_source(real_module_name)

        # Import the module through the package importer
        fn_module = package_importer.import_module(real_module_name)

        # Extract function source from module source
        fn_source = _extract_function_source(module_source, fn_name)

        return fn_module, module_source, fn_source
    else:
        # Standard module handling
        fn_module = inspect.getmodule(target)
        module_source = inspect.getsource(none_throws(fn_module))
        fn_source = inspect.getsource(target)

        return fn_module, module_source, fn_source


def _calls_triton_aot_kernel(node: ast.FunctionDef, kernel_name: str) -> bool:
    """
    kernel_name is the JIT function name (e.g. "_weighted_layer_norm_fwd"),
    which may differ from the wrapper function name (e.g.
    "_triton_aot_swish_layer_norm").  We match by looking for a
    Subscript-call ``kernel_name[grid](...)`` inside the function body.
    """
    for child in ast.walk(node):
        if (
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Subscript)
            and isinstance(child.func.value, ast.Name)
            and child.func.value.id == kernel_name
        ):
            return True
    return False


def _is_torch_jit_unused(d: ast.expr) -> bool:
    """Check if a decorator AST node represents @torch.jit.unused."""
    return (
        isinstance(d, ast.Attribute)
        and d.attr == "unused"
        and isinstance(d.value, ast.Attribute)
        and d.value.attr == "jit"
        and isinstance(d.value.value, ast.Name)
        and d.value.value.id == "torch"
    )


def strip_jit_unused_decorator(
    node: ast.FunctionDef, kernel_name: str
) -> ast.FunctionDef:
    """Strip @torch.jit.unused if the function body calls ``kernel_name[grid](...)``.

    kernel_name is the TritonAOT kernel's JIT function name (e.g.
    ``_weighted_layer_norm_fwd``), not the wrapper function name.  This avoids
    relying on a naming convention on the wrapper function itself.
    """
    if _calls_triton_aot_kernel(node, kernel_name):
        node.decorator_list = [
            d for d in node.decorator_list if not _is_torch_jit_unused(d)
        ]
    return node


class TritonAOTOperatorTransform(ast.NodeTransformer):
    def __init__(self, kernel: Any) -> None:
        super().__init__()
        self._kernel: Any = kernel
        self._kernel_jit_fn: JITFunction[List[Any]] = unwrap_heuristic(
            kernel, return_type=JITFunction
        )
        self._kernel_autotuner: Optional[Autotuner] = unwrap_heuristic(
            kernel, return_type=Autotuner
        )
        self._kernel_name: str = get_kernel_name(self._kernel_jit_fn)
        # Only transform the function body
        self._autotune_params: List[str] = (
            list(list(self._kernel_autotuner.cache.values())[0].kwargs.keys())
            if self._kernel_autotuner is not None
            else []
        )
        self._autotune_params += ["num_warps", "num_stages"]

        self._lambda_arg_name: Optional[str] = None
        self._grid_name: Optional[str] = None
        self._autotune_key_id: Optional[Dict[str, int]] = None
        self._autotune_key_map: Optional[Dict[str, ast.expr]] = None
        self._kernel_meta: Optional[ast.Assign] = None

        if self._kernel_autotuner is not None:
            autotune_key_id: Dict[str, int] = {}
            self._autotune_key_id = autotune_key_id
            # pyre-ignore[16]: JITFunction has arg_names at runtime
            for key in self._kernel_autotuner.keys:
                autotune_key_id[key] = self._kernel_jit_fn.arg_names.index(key)

    def generate_function_meta(self) -> None:
        targets = [
            ast.Name(id=param, ctx=ast.Store()) for param in self._autotune_params
        ]
        autotune_key_map = self._autotune_key_map
        kernel_autotuner = self._kernel_autotuner
        call = ast.Call(
            func=ast.Name(id=f"{self._kernel_name}_meta", ctx=ast.Load()),
            args=[
                none_throws(autotune_key_map)[key]
                for key in none_throws(kernel_autotuner).keys
            ]
            if kernel_autotuner is not None
            else [],
            keywords=[],
        )
        self._kernel_meta = ast.Assign(
            # pyre-ignore[6]: ast.Assign targets type
            targets=[ast.Tuple(elts=targets, ctx=ast.Store())],
            value=call,
        )

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        strip_jit_unused_decorator(node, self._kernel_name)

        new_body: List[ast.stmt] = []
        stmts = node.body
        for stmt in stmts:
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name) and target.id == self._grid_name:
                        assert self._kernel_meta is not None
                        self._kernel_meta.lineno = stmt.lineno
                        new_body.append(self._kernel_meta)
            new_body.append(self.visit(stmt))
        node.body = new_body
        return node

    def visit_Assign(self, node: ast.Assign) -> ast.Assign:
        for target in node.targets:
            if isinstance(target, ast.Name) and isinstance(node.value, ast.Lambda):
                lambda_node = node.value
                self._lambda_arg_name = lambda_node.args.args[0].arg
                lambda_body = lambda_node.body
                assert isinstance(lambda_body, ast.Tuple)
                new_elts: List[ast.expr] = []
                for elt in lambda_body.elts:
                    new_elts.append(self.visit(elt))
                node.value = ast.Tuple(elts=new_elts, ctx=ast.Load())
                self._lambda_arg_name = None
        return node

    def visit_Subscript(self, node: ast.Subscript) -> ast.expr:
        if isinstance(node.value, ast.Name) and node.value.id == self._lambda_arg_name:
            assert isinstance(node.slice, ast.Constant)
            assert isinstance(node.slice.value, str)
            var_name = node.slice.value
            # pyre-ignore
            node = ast.Name(id=var_name, ctx=ast.Load())
        return node

    def visit_Expr(self, node: ast.Expr) -> ast.Expr:
        if isinstance(node.value, ast.Call):
            call = node.value
            if (
                isinstance(call.func, ast.Subscript)
                and isinstance(call.func.value, ast.Name)
                and call.func.value.id == self._kernel_name
            ):
                grid_arg = call.func.slice
                new_func = ast.Attribute(
                    value=ast.Attribute(
                        value=ast.Attribute(
                            value=ast.Name(id="torch", ctx=ast.Load()),
                            attr="ops",
                            ctx=ast.Load(),
                        ),
                        attr="triton_aot",
                        ctx=ast.Load(),
                    ),
                    attr=self._kernel_name,
                    ctx=ast.Load(),
                )
                new_args = [grid_arg] + call.args
                new_keywords = call.keywords + [
                    ast.keyword(arg=param, value=ast.Name(id=param, ctx=ast.Load()))
                    for param in self._autotune_params
                ]
                node.value = ast.Call(
                    func=new_func,
                    args=new_args,
                    keywords=new_keywords,
                )
        return node

    def contains_triton_call(self, node: ast.AST) -> bool:
        for child in ast.walk(node):
            if (
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Subscript)
                # pyre-ignore[16]: ast.expr may have `id` attribute at runtime
                and child.func.value.id == self._kernel_name
            ):
                # pyrefly: ignore [missing-attribute]
                self._grid_name = child.func.slice.id

                if self._kernel_autotuner is not None:
                    autotune_key_map: Dict[str, ast.expr] = {}
                    self._autotune_key_map = autotune_key_map
                    # pyre-ignore[16]: Autotuner has keys at runtime
                    for key in self._kernel_autotuner.keys:
                        found_key = False
                        for keyword in child.keywords:
                            if keyword.arg == key:
                                autotune_key_map[key] = keyword.value
                                found_key = True
                                break

                        if not found_key:
                            autotune_key_id = self._autotune_key_id
                            assert autotune_key_id is not None
                            assert key in autotune_key_id
                            key_id = autotune_key_id[key]
                            autotune_key_map[key] = child.args[key_id]

                self.generate_function_meta()
                return True
        return False

    def contains_lambda(self, node: ast.AST) -> bool:
        for child in ast.walk(node):
            if isinstance(child, ast.Lambda):
                return True
        return False

    def _get_grid_name(self, node: ast.AST) -> Optional[str]:
        for child in ast.walk(node):
            if (
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Subscript)
                # pyre-ignore[16]: ast.expr may have `id` attribute at runtime
                and child.func.value.id == self._kernel_name
            ):
                # pyrefly: ignore [missing-attribute]
                return child.func.slice.id
        return None

    def generate_so_loading_code(
        self,
        node: ast.AST,
        abs_triton_aot_path: str,
    ) -> str:
        """Return auto-generated code to load the compiled kernel at runtime.

        If *node* contains a call to this transformer's kernel, returns
        ``import importlib.util`` + meta-module loading + ``torch.ops.load_library``
        code.  Otherwise returns an empty string.

        This method also sets up internal transformer state (grid name,
        autotune key map, etc.) via ``contains_triton_call`` as a side effect.

        Example for _addmm_fwd kernel:
            kernel_dir = "triton_addmm__addmm_fwd"
            meta_module_path = "/path/to/triton_aot_compile/triton_addmm__addmm_fwd/_addmm_fwd_meta.py"
            so_path = "/path/to/triton_aot_compile/triton_addmm__addmm_fwd/addmm_fwd.so"
        """
        if not self.contains_triton_call(node):
            return ""

        kernel_dir = f"{_get_clean_module_basename(self._kernel_jit_fn.__module__)}_{self._kernel_name}"

        meta_module_path = os.path.join(
            abs_triton_aot_path, kernel_dir, f"{self._kernel_name}_meta.py"
        )

        so_path = os.path.join(
            abs_triton_aot_path,
            kernel_dir,
            f"{self._kernel_name.lstrip('_')}.so",
        )

        return f"""
# Auto-generated by triton_aot.kernel_wrapper_codegen
import importlib.util
_meta_spec = importlib.util.spec_from_file_location("{self._kernel_name}_meta", "{meta_module_path}")
_meta_module = importlib.util.module_from_spec(_meta_spec)
_meta_spec.loader.exec_module(_meta_module)
{self._kernel_name}_meta = _meta_module.{self._kernel_name}_meta

torch.ops.load_library("{so_path}")
"""


def _find_triton_aot_kernel(
    node_target: Any,
    kernel_specs: Dict[KernelInterface[List[Any]], List[Dict[str, List[Any]]]],
) -> Optional[TritonAOT]:
    """Find the single TritonAOT kernel referenced in a node target's globals.

    Scans ``node_target.__globals__`` for ``TritonAOT`` instances, validates
    that every instance appears in *kernel_specs*, and asserts at most one
    kernel is present (per the one-kernel-per-wrapper invariant).

    Returns the kernel, or ``None`` if the function references no kernels.
    """
    kernels: set[TritonAOT] = set()
    for _, var in node_target.__globals__.items():
        if isinstance(var, TritonAOT):
            if var.fn in kernel_specs:
                kernels.add(var)
            else:
                raise RuntimeError(
                    f"Cannot find TritonAOT kernel {var.fn} in TRITON_AOT_KERNEL_SPECS"
                )

    if len(kernels) == 0:
        return None

    fn_name = node_target.__name__
    assert len(kernels) == 1, (
        f"Expected exactly 1 kernel per wrapper function '{fn_name}', "
        f"got {len(kernels)}"
    )
    (kernel_obj,) = kernels
    return kernel_obj


def _generate_wrapper_files(
    node_target: Any,
    kernel: TritonAOT,
    compile_path: str,
    package_importer: Optional[package.PackageImporter],
) -> None:
    """Generate ``_original.py`` and ``_wrapper.py`` for a single kernel.

    Creates a per-kernel subdirectory under *compile_path*, writes the
    original function source, then AST-transforms the wrapper to replace
    ``kernel[grid](...)`` with ``torch.ops.triton_aot.*`` calls.
    """
    fn_name = node_target.__name__

    jit_fn = none_throws(
        unwrap_heuristic(kernel, return_type=JITFunction),
        f"Failed to unwrap kernel to JITFunction: {kernel}",
    )
    kernel_dir = (
        f"{_get_clean_module_basename(jit_fn.__module__)}_{get_kernel_name(jit_fn)}"
    )
    output_dir = os.path.join(compile_path, kernel_dir)
    os.makedirs(output_dir, exist_ok=True)

    _, module_code, wrapper_code = _get_module_and_source(node_target, package_importer)
    import_header = get_original_import_header(module_code)

    with open(os.path.join(output_dir, f"{fn_name}_original.py"), "w") as f:
        f.write(import_header)
        f.write(wrapper_code)

    # When source comes from a torch package, rewrite interned
    # imports to use _package_importer, which
    # is injected by replace_kernels at load time.  Must happen
    # before auto-generated code is appended (so stdlib imports
    # like ``import importlib.util`` are not touched).
    if package_importer is not None:
        import_header = rewrite_package_imports(import_header, package_importer)

    tree = ast.parse(wrapper_code)
    transformer = TritonAOTOperatorTransform(kernel=kernel)
    import_header += transformer.generate_so_loading_code(tree, compile_path)
    tree = transformer.visit(tree)

    new_source_code = ast.unparse(tree)

    with open(os.path.join(output_dir, f"{fn_name}_wrapper.py"), "w") as f:
        f.write(import_header)
        f.write(new_source_code)


def kernel_wrapper_codegen(
    module: GraphModule, packageImporter: package.PackageImporter | None = None
) -> None:
    """
    Generate wrapper files for TritonAOT kernels.
    Requirement: under wrapper.py, @triton.jit kernel/func is imported without 'as' alias.

    For each function containing TritonAOT kernels, generates:
    - {fn_name}_original.py: Original source code with imports
    - {fn_name}_wrapper.py: Transformed wrapper that uses torch.ops.triton_aot
    """
    compile_path = get_aott_compile_path()
    if not os.path.exists(compile_path):
        os.makedirs(compile_path)

    transformed_ops: set[Callable[..., Any]] = set()
    kernel_specs = get_triton_aot_kernel_specs()
    for node in module.graph.nodes:
        if node.op == "call_function" and hasattr(node.target, "__globals__"):
            if node.target not in transformed_ops:
                transformed_ops.add(node.target)
            else:
                continue

            kernel = _find_triton_aot_kernel(node.target, kernel_specs)
            if kernel is not None:
                _generate_wrapper_files(
                    node.target, kernel, compile_path, packageImporter
                )
