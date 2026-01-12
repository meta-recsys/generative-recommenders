"""
Modified from https://github.com/Dao-AILab/flash-attention/blob/main/hopper/setup.py
"""

import itertools
import os
import platform
import subprocess
import sys
import sysconfig
import warnings
from pathlib import Path

import torch
from packaging.version import parse, Version
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDA_HOME, CUDAExtension

# ninja build does not work unless include_dirs are abs path
this_dir = os.path.dirname(os.path.abspath(__file__))
PACKAGE_NAME = "hstu"
# FORCE_BUILD: Force a fresh build locally, instead of attempting to find prebuilt wheels
# SKIP_CUDA_BUILD: Intended to allow CI to use a simple `python setup.py sdist` run to copy over raw files, without any cuda compilation
FORCE_BUILD = os.getenv("FLASH_ATTENTION_FORCE_BUILD", "FALSE") == "TRUE"
SKIP_CUDA_BUILD = os.getenv("FLASH_ATTENTION_SKIP_CUDA_BUILD", "FALSE") == "TRUE"
# For CI, we want the option to build with C++11 ABI since the nvcr images use C++11 ABI
FORCE_CXX11_ABI = os.getenv("FLASH_ATTENTION_FORCE_CXX11_ABI", "FALSE") == "TRUE"

# HACK: we monkey patch pytorch's _write_ninja_file to pass
# "-gencode arch=compute_sm90a,code=sm_90a" to files ending in '_sm90.cu',
# and pass "-gencode arch=compute_sm80,code=sm_80" to files ending in '_sm80.cu'
from torch.utils.cpp_extension import (
    _is_cuda_file,
    _join_cuda_home,
    _join_rocm_home,
    _maybe_write,
    COMMON_HIP_FLAGS,
    get_cxx_compiler,
    IS_HIP_EXTENSION,
    IS_WINDOWS,
    SUBPROCESS_DECODE_ARGS,
)

DISABLE_BACKWARD = os.getenv("FLASH_ATTENTION_DISABLE_BACKWARD", "FALSE") == "TRUE"
DISABLE_FP16 = os.getenv("FLASH_ATTENTION_DISABLE_FP16", "TRUE") == "TRUE"
DISABLE_HDIM64 = os.getenv("FLASH_ATTENTION_DISABLE_HDIM64", "TRUE") == "TRUE"
DISABLE_HDIM96 = os.getenv("FLASH_ATTENTION_DISABLE_HDIM96", "TRUE") == "TRUE"
DISABLE_HDIM128 = os.getenv("FLASH_ATTENTION_DISABLE_HDIM128", "FALSE") == "TRUE"
DISABLE_HDIM192 = os.getenv("FLASH_ATTENTION_DISABLE_HDIM192", "TRUE") == "TRUE"
DISABLE_HDIM256 = os.getenv("FLASH_ATTENTION_DISABLE_HDIM256", "TRUE") == "TRUE"
DISABLE_SM8x = os.getenv("FLASH_ATTENTION_DISABLE_SM80", "TRUE") == "TRUE"


def _write_ninja_file(
    path,
    cflags,
    post_cflags,
    cuda_cflags,
    cuda_post_cflags,
    cuda_dlink_post_cflags,
    sources,
    objects,
    ldflags,
    library_target,
    with_cuda,
    **kwargs,  # kwargs (ignored) to absorb new flags in torch.utils.cpp_extension
) -> None:
    r"""Write a ninja file that does the desired compiling and linking.

    `path`: Where to write this file
    `cflags`: list of flags to pass to $cxx. Can be None.
    `post_cflags`: list of flags to append to the $cxx invocation. Can be None.
    `cuda_cflags`: list of flags to pass to $nvcc. Can be None.
    `cuda_postflags`: list of flags to append to the $nvcc invocation. Can be None.
    `sources`: list of paths to source files
    `objects`: list of desired paths to objects, one per source.
    `ldflags`: list of flags to pass to linker. Can be None.
    `library_target`: Name of the output library. Can be None; in that case,
                      we do no linking.
    `with_cuda`: If we should be compiling with CUDA.
    """

    def sanitize_flags(flags):
        if flags is None:
            return []
        else:
            return [flag.strip() for flag in flags]

    cflags = sanitize_flags(cflags)
    post_cflags = sanitize_flags(post_cflags)
    cuda_cflags = sanitize_flags(cuda_cflags)
    cuda_post_cflags = sanitize_flags(cuda_post_cflags)
    cuda_dlink_post_cflags = sanitize_flags(cuda_dlink_post_cflags)
    ldflags = sanitize_flags(ldflags)

    # Sanity checks...
    assert len(sources) == len(objects)
    assert len(sources) > 0

    compiler = get_cxx_compiler()

    # Version 1.3 is required for the `deps` directive.
    config = ["ninja_required_version = 1.3"]
    config.append(f"cxx = {compiler}")
    if with_cuda or cuda_dlink_post_cflags:
        if IS_HIP_EXTENSION:
            nvcc = _join_rocm_home("bin", "hipcc")
        else:
            nvcc = _join_cuda_home("bin", "nvcc")
        if "PYTORCH_NVCC" in os.environ:
            nvcc_from_env = os.getenv(
                "PYTORCH_NVCC"
            )  # user can set nvcc compiler with ccache using the environment variable here
        else:
            nvcc_from_env = nvcc
        config.append(f"nvcc_from_env = {nvcc_from_env}")
        config.append(f"nvcc = {nvcc}")

    if IS_HIP_EXTENSION:
        post_cflags = COMMON_HIP_FLAGS + post_cflags
    flags = [f"cflags = {' '.join(cflags)}"]
    flags.append(f"post_cflags = {' '.join(post_cflags)}")
    if with_cuda:
        flags.append(f"cuda_cflags = {' '.join(cuda_cflags)}")
        flags.append(f"cuda_post_cflags = {' '.join(cuda_post_cflags)}")
        cuda_post_cflags_sm80 = [
            s if s != "arch=compute_90a,code=sm_90a" else "arch=compute_80,code=sm_80"
            for s in cuda_post_cflags
        ]
        flags.append(f"cuda_post_cflags_sm80 = {' '.join(cuda_post_cflags_sm80)}")
        cuda_post_cflags_sm80_sm90 = cuda_post_cflags + [
            "-gencode",
            "arch=compute_80,code=sm_80",
        ]
        flags.append(
            f"cuda_post_cflags_sm80_sm90 = {' '.join(cuda_post_cflags_sm80_sm90)}"
        )
        cuda_post_cflags_sm100 = [
            s
            if s != "arch=compute_90a,code=sm_90a"
            else "arch=compute_100a,code=sm_100a"
            for s in cuda_post_cflags
        ]
        flags.append(f"cuda_post_cflags_sm100 = {' '.join(cuda_post_cflags_sm100)}")
    flags.append(f"cuda_dlink_post_cflags = {' '.join(cuda_dlink_post_cflags)}")
    flags.append(f"ldflags = {' '.join(ldflags)}")

    # Turn into absolute paths so we can emit them into the ninja build
    # file wherever it is.
    sources = [os.path.abspath(file) for file in sources]

    # See https://ninja-build.org/build.ninja.html for reference.
    compile_rule = ["rule compile"]
    if IS_WINDOWS:
        compile_rule.append(
            "  command = cl /showIncludes $cflags -c $in /Fo$out $post_cflags"
        )
        compile_rule.append("  deps = msvc")
    else:
        compile_rule.append(
            "  command = $cxx -MMD -MF $out.d $cflags -c $in -o $out $post_cflags"
        )
        compile_rule.append("  depfile = $out.d")
        compile_rule.append("  deps = gcc")

    if with_cuda:
        cuda_compile_rule = ["rule cuda_compile"]
        nvcc_gendeps = ""
        # --generate-dependencies-with-compile is not supported by ROCm
        # Nvcc flag `--generate-dependencies-with-compile` is not supported by sccache, which may increase build time.
        if (
            torch.version.cuda is not None
            and os.getenv("TORCH_EXTENSION_SKIP_NVCC_GEN_DEPENDENCIES", "0") != "1"
        ):
            cuda_compile_rule.append("  depfile = $out.d")
            cuda_compile_rule.append("  deps = gcc")
            # Note: non-system deps with nvcc are only supported
            # on Linux so use --generate-dependencies-with-compile
            # to make this work on Windows too.
            nvcc_gendeps = (
                "--generate-dependencies-with-compile --dependency-output $out.d"
            )
        cuda_compile_rule_sm80 = (
            ["rule cuda_compile_sm80"]
            + cuda_compile_rule[1:]
            + [
                f"  command = $nvcc_from_env {nvcc_gendeps} $cuda_cflags -c $in -o $out $cuda_post_cflags_sm80"
            ]
        )
        cuda_compile_rule_sm80_sm90 = (
            ["rule cuda_compile_sm80_sm90"]
            + cuda_compile_rule[1:]
            + [
                f"  command = $nvcc_from_env {nvcc_gendeps} $cuda_cflags -c $in -o $out $cuda_post_cflags_sm80_sm90"
            ]
        )
        cuda_compile_rule_sm100 = (
            ["rule cuda_compile_sm100"]
            + cuda_compile_rule[1:]
            + [
                f"  command = $nvcc_from_env {nvcc_gendeps} $cuda_cflags -c $in -o $out $cuda_post_cflags_sm100"
            ]
        )
        cuda_compile_rule.append(
            f"  command = $nvcc_from_env {nvcc_gendeps} $cuda_cflags -c $in -o $out $cuda_post_cflags"
        )

    # Emit one build rule per source to enable incremental build.
    build = []
    for source_file, object_file in zip(sources, objects):
        is_cuda_source = _is_cuda_file(source_file) and with_cuda
        if is_cuda_source:
            if source_file.endswith("_sm90.cu"):
                rule = "cuda_compile"
            elif source_file.endswith("_sm80.cu"):
                rule = "cuda_compile_sm80"
            elif source_file.endswith("_sm100.cu"):
                rule = "cuda_compile_sm100"
            else:
                rule = "cuda_compile_sm80_sm90"
        else:
            rule = "compile"
        if IS_WINDOWS:
            source_file = source_file.replace(":", "$:")
            object_file = object_file.replace(":", "$:")
        source_file = source_file.replace(" ", "$ ")
        object_file = object_file.replace(" ", "$ ")
        build.append(f"build {object_file}: {rule} {source_file}")

    if cuda_dlink_post_cflags:
        devlink_out = os.path.join(os.path.dirname(objects[0]), "dlink.o")
        devlink_rule = ["rule cuda_devlink"]
        devlink_rule.append("  command = $nvcc $in -o $out $cuda_dlink_post_cflags")
        devlink = [f"build {devlink_out}: cuda_devlink {' '.join(objects)}"]
        objects += [devlink_out]
    else:
        devlink_rule, devlink = [], []

    if library_target is not None:
        link_rule = ["rule link"]
        if IS_WINDOWS:
            cl_paths = (
                subprocess.check_output(["where", "cl"])
                .decode(*SUBPROCESS_DECODE_ARGS)
                .split("\r\n")
            )
            if len(cl_paths) >= 1:
                cl_path = os.path.dirname(cl_paths[0]).replace(":", "$:")
            else:
                raise RuntimeError("MSVC is required to load C++ extensions")
            link_rule.append(
                f'  command = "{cl_path}/link.exe" $in /nologo $ldflags /out:$out'
            )
        else:
            link_rule.append("  command = $cxx $in $ldflags -o $out")

        link = [f"build {library_target}: link {' '.join(objects)}"]

        default = [f"default {library_target}"]
    else:
        link_rule, link, default = [], [], []

    # 'Blocks' should be separated by newlines, for visual benefit.
    blocks = [config, flags, compile_rule]
    if with_cuda:
        blocks.append(cuda_compile_rule)  # type: ignore[possibly-undefined]
        blocks.append(cuda_compile_rule_sm80)  # type: ignore[possibly-undefined]
        blocks.append(cuda_compile_rule_sm80_sm90)  # type: ignore[possibly-undefined]
        blocks.append(cuda_compile_rule_sm100)  # type: ignore[possibly-undefined]
    blocks += [devlink_rule, link_rule, build, devlink, link, default]
    content = "\n\n".join("\n".join(b) for b in blocks)
    # Ninja requires a new lines at the end of the .ninja file
    content += "\n"
    _maybe_write(path, content)


# Monkey patching
torch.utils.cpp_extension._write_ninja_file = _write_ninja_file


def get_platform():
    """
    Returns the platform name as used in wheel filenames.
    """
    if sys.platform.startswith("linux"):
        return "linux_x86_64"
    elif sys.platform == "darwin":
        mac_version = ".".join(platform.mac_ver()[0].split(".")[:2])
        return f"macosx_{mac_version}_x86_64"
    elif sys.platform == "win32":
        return "win_amd64"
    else:
        raise ValueError("Unsupported platform: {}".format(sys.platform))


def get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output(
        [cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True
    )
    output = raw_output.split()
    release_idx = output.index("release") + 1
    bare_metal_version = parse(output[release_idx].split(",")[0])

    return raw_output, bare_metal_version


def check_if_cuda_home_none(global_option: str) -> None:
    if CUDA_HOME is not None:
        return
    # warn instead of error because user could be downloading prebuilt wheels, so nvcc won't be necessary
    # in that case.
    warnings.warn(
        f"{global_option} was requested, but nvcc was not found.  Are you sure your environment has nvcc available?  "
        "If you're installing within a container from https://hub.docker.com/r/pytorch/pytorch, "
        "only images whose names contain 'devel' will provide nvcc."
    )


def nvcc_threads_args():
    nvcc_threads = os.getenv("NVCC_THREADS") or "4"
    return ["--threads", nvcc_threads]


exe_extension = sysconfig.get_config_var("EXE")


cmdclass = {}
ext_modules = []

# We want this even if SKIP_CUDA_BUILD because when we run python setup.py sdist we want the .hpp
# files included in the source distribution, in case the user compiles from source.
subprocess.run(["git", "submodule", "update", "--init", "cutlass"])

if not SKIP_CUDA_BUILD:
    print("\n\ntorch.__version__  = {}\n\n".format(torch.__version__))
    TORCH_MAJOR = int(torch.__version__.split(".")[0])
    TORCH_MINOR = int(torch.__version__.split(".")[1])

    check_if_cuda_home_none(PACKAGE_NAME)
    _, bare_metal_version = get_cuda_bare_metal_version(CUDA_HOME)
    if bare_metal_version < Version("12.3"):
        raise RuntimeError(
            f"FlashAttention-3 is only supported on CUDA 12.3 and above, get {bare_metal_version} from {CUDA_HOME}"
        )

    cc_flag = []
    cc_flag.append("-gencode")
    cc_flag.append("arch=compute_90a,code=sm_90a")

    # HACK: The compiler flag -D_GLIBCXX_USE_CXX11_ABI is set to be the same as
    # torch._C._GLIBCXX_USE_CXX11_ABI
    # https://github.com/pytorch/pytorch/blob/8472c24e3b5b60150096486616d98b7bea01500b/torch/utils/cpp_extension.py#L920
    if FORCE_CXX11_ABI:
        torch._C._GLIBCXX_USE_CXX11_ABI = True
    repo_dir = Path(this_dir).parent
    cutlass_dir = repo_dir / "cpp" / "cutlass"

    feature_args = (
        []
        + ["-DOSS_ENV"]
        + (["-DFLASHATTENTION_DISABLE_BACKWARD"] if DISABLE_BACKWARD else [])
        + (["-DFLASHATTENTION_DISABLE_FP16"] if DISABLE_FP16 else [])
        + ["-DFLASHATTENTION_DISABLE_FP8"]
        + (["-DFLASHATTENTION_DISABLE_HDIM64"] if DISABLE_HDIM64 else [])
        + (["-DFLASHATTENTION_DISABLE_HDIM96"] if DISABLE_HDIM96 else [])
        + (["-DFLASHATTENTION_DISABLE_HDIM128"] if DISABLE_HDIM128 else [])
        + (["-DFLASHATTENTION_DISABLE_HDIM192"] if DISABLE_HDIM192 else [])
        + (["-DFLASHATTENTION_DISABLE_HDIM256"] if DISABLE_HDIM256 else [])
        + (["-DFLASHATTENTION_DISABLE_SM8x"] if DISABLE_SM8x else [])
    )

    DTYPE = ["bf16"] + (["fp16"] if not DISABLE_FP16 else [])
    HEAD_DIMENSIONS = (
        []
        + ([64] if not DISABLE_HDIM64 else [])
        + ([96] if not DISABLE_HDIM96 else [])
        + ([128] if not DISABLE_HDIM128 else [])
        + ([192] if not DISABLE_HDIM192 else [])
        + ([256] if not DISABLE_HDIM256 else [])
    )
    sources_fwd_sm80 = [
        f"hstu_attention/instantiations/flash_fwd_hdim{hdim}_{dtype}_sm80.cu"
        for hdim, dtype in itertools.product(HEAD_DIMENSIONS, DTYPE)
    ]
    sources_bwd_sm80 = [
        f"hstu_attention/instantiations/flash_bwd_hdim{hdim}_{dtype}_sm80.cu"
        for hdim, dtype in itertools.product(HEAD_DIMENSIONS, DTYPE)
    ]
    sources_fwd_sm90 = [
        f"hstu_attention/instantiations/flash_fwd_hdim{hdim}_{dtype}_sm90.cu"
        for hdim, dtype in itertools.product(HEAD_DIMENSIONS, DTYPE)
    ]
    sources_bwd_sm90 = [
        f"hstu_attention/instantiations/flash_bwd_hdim{hdim}_{dtype}_sm90.cu"
        for hdim, dtype in itertools.product(HEAD_DIMENSIONS, DTYPE)
    ]
    if DISABLE_BACKWARD:
        sources_bwd_sm90 = []
        sources_bwd_sm80 = []
    sources = (
        [
            "hstu_attention/flash_api.cpp",
            "hstu_attention/flash_common.cpp",
            "hstu_attention/flash_cpu_dummy.cpp",
            "hstu_attention/flash_meta.cpp",
        ]
        + (sources_fwd_sm80 if not DISABLE_SM8x else [])
        + sources_fwd_sm90
        + (sources_bwd_sm80 if not DISABLE_SM8x else [])
        + sources_bwd_sm90
    )
    nvcc_flags = [
        "-O3",
        "-std=c++17",
        "--ftemplate-backtrace-limit=0",  # To debug template code
        "--use_fast_math",
        # "--keep",
        # "--ptxas-options=--verbose,--register-usage-level=5,--warn-on-local-memory-usage",  # printing out number of registers
        "--resource-usage",  # printing out number of registers
        # f"--split-compile={os.getenv('NVCC_THREADS', '4')}",  # split-compile is faster
        "-lineinfo",  # TODO: disable this for release to reduce binary size
        "-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED",  # Necessary for the WGMMA shapes that we use
        "-DCUTLASS_ENABLE_GDC_FOR_SM90",  # For PDL
        "-DCUTLASS_DEBUG_TRACE_LEVEL=0",  # Can toggle for debugging
        "-DNDEBUG",  # Important, otherwise performance is severely impacted
        "-Xfatbin",  # compress all binary sections
        "-compress-all",
    ]
    if get_platform() == "win_amd64":
        nvcc_flags.extend(
            [
                "-D_USE_MATH_DEFINES",  # for M_LN2
                "-Xcompiler=/Zc:__cplusplus",  # sets __cplusplus correctly, CUTLASS_CONSTEXPR_IF_CXX17 needed for cutlass::gcd
            ]
        )
    include_dirs = [
        Path(this_dir),
        cutlass_dir / "include",
    ]

    ext_modules.append(
        CUDAExtension(
            name=f"{PACKAGE_NAME}._C",
            sources=sources,
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17", "-DPy_LIMITED_API=0x03090000"]
                + feature_args,
                "nvcc": nvcc_threads_args() + nvcc_flags + cc_flag + feature_args,
            },
            include_dirs=include_dirs,
            py_limited_api=True,
        )
    )


setup(
    name=PACKAGE_NAME,
    version="0.1.0",
    packages=find_packages(
        exclude=(
            "build",
            "csrc",
            "include",
            "tests",
            "dist",
            "docs",
            "benchmarks",
        )
    ),
    py_modules=["cuda_hstu_attention"],
    description="FlashAttention HSTU",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: Unix",
    ],
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.8",
    install_requires=[
        "torch",
        "einops",
        "packaging",
        "ninja==1.11.1.1",
    ],
)
