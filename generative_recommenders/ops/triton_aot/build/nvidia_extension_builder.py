# pyre-strict
"""
NVIDIA CUDA extension builder for Triton AOT kernels.

This module contains the NVIDIAExtensionBuilder class and common utilities
shared between NVIDIA and AMD extension builders.
"""

import logging
import os
import re

# @manual=//generative_recommenders/ops/triton_aot/build:torch_cpp_headers
import pkg_resources
from generative_recommenders.ops.triton_aot.build import cubin_embedder
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from torch.utils import cpp_extension
from torch.utils.cpp_extension import CUDA_HOME

# TODO investigate if other lib path can be found through build_paths
from triton.fb.build import build_paths  # @manual=//triton/fb:build


logger: logging.Logger = logging.getLogger(__name__)

# Path components for torch API headers (torch/csrc/api/include/torch/types.h)
# Used to locate bundled headers regardless of header_namespace setting
TORCH_API_INCLUDE_PATH_PARTS: tuple[str, ...] = ("torch", "csrc", "api", "include")

# Package and resource names for bundled torch headers
BUNDLED_HEADERS_PACKAGE: str = "generative_recommenders.ops.triton_aot.build"
BUNDLED_HEADERS_RESOURCE: str = "torch_cpp_headers"

# Generated files, for cubin embedder
EMBEDDED_CUBIN_FILENAME: str = "embedded_kernels_autogen.cpp"

# Regex pattern for extracting kernel names from cubin variable declarations
CUBIN_VAR_PATTERN: re.Pattern[str] = re.compile(r"extern unsigned char (\w+)_cubin\[\]")

# for compile/setup
CPP_STANDARD: str = "c++20"


class SoBuildExtension(build_ext):
    """
    By default, setuptools generates .so files with platform-specific suffixes like:
        addmm_fwd.cpython-39-x86_64-linux-gnu.so
    This class overrides that behavior to produce simpler names:
        addmm_fwd.so

    We inherit from setuptools' build_ext directly (instead of PyTorch's BuildExtension),
    as we only compile C++ wrapper code, not CUDA kernels (cubins are pre-compiled by Triton)

    Uses triton.fb.build.build_paths for compiler configuration in fbcode.
    """

    def get_ext_filename(self, fullname: str) -> str:
        return f"{fullname}.so"

    def build_extensions(self) -> None:
        """Override to set compiler from build_paths before building extensions."""
        compiler_path = build_paths.cc
        # pyre-ignore[16]: compiler is dynamically set by parent class
        self.compiler.set_executables(
            compiler=compiler_path,
            compiler_so=compiler_path,
            compiler_cxx=compiler_path,
            linker_so=f"{compiler_path} -shared",
            linker_exe=compiler_path,
        )
        logger.info(f"Using compiler: {compiler_path}")
        # pyre-ignore[16]: build_extensions is inherited from distutils
        super().build_extensions()


def extract_kernel_variants_from_cpp_files(source_dir: str) -> list[str]:
    """
    Extract kernel variant names from .cpp files by finding cubin variable declarations.

    Matches pattern: extern unsigned char xxx_cubin[];
    Returns kernel variant names (e.g., '_addmm_fwd_sm80_pfp32_pfp32_pfp32_pfp32_i32_')
    with '_cubin' suffix removed.
    """
    kernel_variants = []
    cpp_files = [f for f in os.listdir(source_dir) if f.endswith(".cpp")]

    for cpp_file in cpp_files:
        cpp_path = os.path.join(source_dir, cpp_file)
        with open(cpp_path, "r") as f:
            content = f.read()
            matches = CUBIN_VAR_PATTERN.findall(content)
            kernel_variants.extend(matches)

    return kernel_variants


class ExtensionBuilder:
    """
    Extension builder for NVIDIA CUDA backend.

    Handles CUDA-specific include paths, library directories, and compilation flags.
    This is the parent class for GPU extension builders.
    """

    source_dir: str
    kernel_name: str
    output_dir: str
    ext_name: str
    gpu_toolkit_path: str

    def __init__(
        self,
        source_dir: str,
        kernel_name: str,
        output_dir: str = "/tmp",
        gpu_toolkit_path: str | None = None,
    ) -> None:
        """
        Initialize the extension builder.

        Args:
            source_dir: Directory containing the generated C++ sources and cubin/hsaco files.
            kernel_name: Name of the kernel (e.g., "_addmm_fwd").
            output_dir: Directory to place the built .so file.
            gpu_toolkit_path: Path to GPU toolkit. Defaults to CUDA_HOME if not provided.
        """
        self.source_dir = source_dir
        self.kernel_name = kernel_name
        self.output_dir = output_dir
        self.ext_name = kernel_name.lstrip("_")
        if gpu_toolkit_path is None:
            if CUDA_HOME is None:
                raise RuntimeError(
                    "CUDA_HOME is not set. Install CUDA toolkit or set CUDA_HOME environment variable."
                )
            self.gpu_toolkit_path = CUDA_HOME
        else:
            self.gpu_toolkit_path = gpu_toolkit_path

        if not os.path.exists(self.gpu_toolkit_path):
            raise RuntimeError(f"GPU toolkit not found at {self.gpu_toolkit_path}. ")

    def get_gpu_include_dirs(self) -> list[str]:
        """Return CUDA include directory path, validated."""
        include_dir = os.path.join(self.gpu_toolkit_path, "include")
        cuda_header = os.path.join(include_dir, "cuda.h")
        if not os.path.exists(cuda_header):
            raise RuntimeError(
                f"CUDA header not found at {cuda_header}. "
                f"CUDA Toolkit (not just Runtime) must be installed."
            )
        return [include_dir]

    def get_gpu_library_dirs(self) -> list[str]:
        """
        Return directories containing CUDA libraries (libcuda.so or stubs).

        Supports two directory layouts:
          - OSS:  $CUDA_HOME/lib64/stubs/, $CUDA_HOME/lib64/, etc.
          - Meta: $CUDA_HOME/lib/cuda-<version>/stubs/
        """
        # OSS: standard CUDA installation paths
        candidates = [
            os.path.join(self.gpu_toolkit_path, "lib64/stubs"),
            os.path.join(self.gpu_toolkit_path, "lib/stubs"),
            os.path.join(self.gpu_toolkit_path, "lib64"),
            os.path.join(self.gpu_toolkit_path, "lib"),
        ]

        # $CUDA_HOME/lib/cuda-<version>/stubs/ (prioritized over OSS paths)
        lib_dir = os.path.join(self.gpu_toolkit_path, "lib")
        if os.path.isdir(lib_dir):
            for name in os.listdir(lib_dir):
                if name.startswith("cuda-"):
                    stubs = os.path.join(lib_dir, name, "stubs")
                    if os.path.isdir(stubs):
                        candidates.insert(0, stubs)

        result = [d for d in candidates if os.path.isdir(d)]
        if not result:
            raise RuntimeError(
                f"No CUDA library directories found in {self.gpu_toolkit_path}. Searched: {candidates}"
            )
        return result

    def get_libraries(self) -> list[str]:
        """Return CUDA libraries to link against."""
        return ["cuda"]

    def get_extra_compile_args(self) -> list[str]:
        """Return compiler arguments, with verbose flag for distutils mode."""
        args = [
            f"-std={CPP_STANDARD}",
            "-fPIC",  # Position Independent Code, required for shared libraries
            "-DUSE_CUDA",  # Makes shim.h CUDA declarations visible
            "-v",  # Verbose compiler output to help debug compilation issues
        ]
        return args

    def generate_embedded_kernels(
        self, output_filename: str, kernel_variants: list[str]
    ) -> None:
        """Generate embedded cubin cpp file."""
        cubin_embedder.generate_cpp_for_kernel_binaries(
            output_filename=output_filename,
            kernel_variants=kernel_variants,
            binary_dir=self.source_dir,
        )

    def get_torch_device_type(self) -> str:
        """Return the torch device type for include path lookup."""
        return "cuda"

    def get_torch_include_dirs(self) -> list[str]:
        """
        Return torch include directories for C++ extension compilation.

        Includes both standard torch headers and Meta's bundled headers.
        """
        device_type = self.get_torch_device_type()
        include_dirs = list(cpp_extension.include_paths(device_type))

        # Meta: bundled torch headers via pkg_resources
        bundled_headers_path = pkg_resources.resource_filename(
            BUNDLED_HEADERS_PACKAGE, BUNDLED_HEADERS_RESOURCE
        )
        if not os.path.isdir(bundled_headers_path):
            raise RuntimeError(
                f"Bundled torch headers not found at {bundled_headers_path}. "
                f"Ensure {BUNDLED_HEADERS_PACKAGE}:{BUNDLED_HEADERS_RESOURCE} is in deps."
            )

        include_dirs.append(bundled_headers_path)

        # Find torch/csrc/api/include directory to determine the correct include paths.
        # this is to handle header_namespace setting in the Buck target.
        api_include_suffix = os.path.join(*TORCH_API_INCLUDE_PATH_PARTS)
        api_include = None
        for root, _, _ in os.walk(bundled_headers_path):
            if root.endswith(api_include_suffix):
                api_include = root
                break

        assert api_include, (
            f"{api_include_suffix} not found under {bundled_headers_path}"
        )
        include_dirs.append(api_include)

        # Add parent of torch/csrc/api/include for #include <torch/csrc/...>
        # Go up len(TORCH_API_INCLUDE_PATH_PARTS) levels to reach the prefix
        prefix = api_include
        for _ in range(len(TORCH_API_INCLUDE_PATH_PARTS)):
            prefix = os.path.dirname(prefix)
        if prefix != bundled_headers_path:
            include_dirs.append(prefix)

        return include_dirs

    def validate_source_files(self) -> tuple[str, str]:
        """Validate that required source files exist and return their paths."""
        cpp_file = os.path.join(self.source_dir, f"{self.kernel_name}.cpp")
        torch_op_file = os.path.join(
            self.source_dir, f"{self.kernel_name}_torch_op.cpp"
        )
        assert os.path.exists(cpp_file), f"Kernel source not found: {cpp_file}"
        assert os.path.exists(torch_op_file), (
            f"Torch op source not found: {torch_op_file}"
        )
        return cpp_file, torch_op_file

    def build(self) -> str:
        """
        Build a Triton AOT kernel as a PyTorch C++ extension.

        Returns:
            Path to the built .so file.

        Raises:
            RuntimeError: If CUDA/HIP is not properly configured or build fails.
            AssertionError: If required source files are missing.
        """
        # Validate source files
        cpp_file, torch_op_file = self.validate_source_files()

        kernel_variants = extract_kernel_variants_from_cpp_files(self.source_dir)
        assert kernel_variants, f"No cubin references found in {self.source_dir}/*.cpp"

        # Generate embedded cubin/hsaco cpp file
        embedded_cubin_filename = f"{self.kernel_name}_embedded_kernels_autogen.cpp"
        embedded_cubin_cpp = os.path.join(self.output_dir, embedded_cubin_filename)
        self.generate_embedded_kernels(embedded_cubin_cpp, kernel_variants)

        # Get all include and library directories
        gpu_include_dirs = self.get_gpu_include_dirs()
        gpu_lib_dirs = self.get_gpu_library_dirs()
        torch_include_dirs = self.get_torch_include_dirs()
        libraries = self.get_libraries()
        extra_compile_args = self.get_extra_compile_args()

        ext_module = Extension(
            name=self.ext_name,
            sources=[cpp_file, torch_op_file, embedded_cubin_cpp],
            include_dirs=gpu_include_dirs + torch_include_dirs,
            library_dirs=gpu_lib_dirs,
            libraries=libraries,
            extra_compile_args=extra_compile_args,
            language="c++",
        )

        setup(
            name=self.ext_name,
            ext_modules=[ext_module],
            script_args=[
                "build_ext",
                f"--build-lib={self.output_dir}",
                f"--build-temp={self.output_dir}/build_temp",
            ],
            cmdclass={"build_ext": SoBuildExtension},
        )

        so_path = os.path.join(self.output_dir, f"{self.ext_name}.so")
        assert os.path.exists(so_path), f"Build failed: .so file not found at {so_path}"

        return so_path
