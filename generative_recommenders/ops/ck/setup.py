# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Builds the Composable Kernel (CK) HSTU attention PyTorch extension for ROCm.

    cd generative_recommenders/ops/ck
    GPU_ARCHS="gfx942;gfx950" MAX_JOBS=64 python setup.py install

Importing ``generative_recommenders.ops.ck.ck_hstu_attention`` then registers
``torch.ops.ck_hstu_experimental.{ck_hstu_mha,ck_hstu_mha_fwd,ck_hstu_mha_bwd}``.

Environment variables:
    GPU_ARCHS: ';'-separated AMD GPU archs to build for (gfx942, gfx950,
        gfx1250). Default: "gfx942;gfx950". Backward is not built when gfx1250
        is in the list.
    CK_DIR: an existing composable_kernel checkout to use instead of the
        composable_kernel git submodule next to this file.
    CK_HSTU_BUILD_DIR: where the .cpp -> .hip source copies are written.
        Default: ./build/hip_srcs.
"""

import glob
import os
import shutil
import subprocess
from pathlib import Path

import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

this_dir = Path(__file__).resolve().parent
kernel_dir = this_dir / "hstu_attention"

PACKAGE_NAME = "ck_hstu_experimental"
SUPPORTED_ARCHS = ("gfx942", "gfx950", "gfx1250")


def get_gpu_archs() -> list[str]:
    archs = [
        a.strip()
        for a in os.getenv("GPU_ARCHS", "gfx942;gfx950").split(";")
        if a.strip()
    ]
    unsupported = [a for a in archs if a not in SUPPORTED_ARCHS]
    if unsupported:
        raise ValueError(
            f"Unsupported GPU_ARCHS {unsupported}; supported: {SUPPORTED_ARCHS}"
        )
    return archs


def get_ck_include_dir() -> Path:
    ck_dir = os.getenv("CK_DIR")
    if ck_dir:
        ck_root = Path(ck_dir)
    else:
        subprocess.run(
            ["git", "submodule", "update", "--init", "composable_kernel"],
            cwd=this_dir,
            check=False,
        )
        ck_root = this_dir / "composable_kernel"
    include_dir = ck_root / "include"
    if not (include_dir / "ck_tile").is_dir():
        raise RuntimeError(
            f"composable_kernel headers not found under {include_dir}; run "
            "`git submodule update --init` or set CK_DIR."
        )
    return include_dir


def get_arch_flags(archs: list[str]) -> list[str]:
    flags = [f"--offload-arch={a}" for a in archs]
    # HSTU tile/LDS configs are chosen per build, not per device pass, so a
    # fat binary that includes gfx942 must use the gfx94 config for every arch.
    if "gfx942" in archs:
        flags.append("-DBUILD_HSTU_FOR_GFX94")
    # CK_TILE_TARGET_* lets host code pick the same tile configs as device
    # code (CK forbids __gfx*__ macros in host compilation).
    if "gfx950" in archs:
        flags.append("-DCK_TILE_TARGET_GFX950")
    if "gfx1250" in archs:
        flags += ["-DCK_TILE_TARGET_GFX1250", "-DHSTU_DISABLE_BWD"]
        # Upstream's CMakeLists.txt adds this for gfx125 targets.
        flags.append("-fno-slp-vectorize")
    else:
        flags += ["-mllvm", "-amdgpu-coerce-illegal-types=1"]
    return flags


def get_kernel_sources(archs: list[str]) -> list[str]:
    """
    Copies the kernel .cpp sources to .hip so they are compiled as device code
    (torch.utils.cpp_extension compiles .cpp files with the host compiler).
    Dropout instances are not built: the ck_hstu_mha ops never request dropout.
    """
    srcs = sorted(glob.glob(str(kernel_dir / "hstu_attention_*.cpp")))
    srcs += sorted(glob.glob(str(kernel_dir / "instances" / "*.cpp")))
    srcs = [
        s
        for s in srcs
        if "hstu_attention_torch_ops_" not in s
        and "_has_dropout_" not in s
        and not ("gfx1250" in archs and "backward" in os.path.basename(s))
    ]
    out_dir = Path(os.getenv("CK_HSTU_BUILD_DIR", str(this_dir / "build" / "hip_srcs")))
    out_dir.mkdir(parents=True, exist_ok=True)
    hip_srcs = []
    for s in srcs:
        dst = out_dir / (Path(s).stem + ".hip")
        if not dst.exists() or dst.read_bytes() != Path(s).read_bytes():
            shutil.copyfile(s, dst)
        hip_srcs.append(str(dst))
    return hip_srcs


def get_ext_modules() -> list[CUDAExtension]:
    if torch.version.hip is None:
        raise RuntimeError("The CK HSTU attention extension requires ROCm PyTorch.")
    archs = get_gpu_archs()
    ck_include_dir = get_ck_include_dir()
    sources = [
        str(kernel_dir / "hstu_attention_torch_ops.hip"),
        str(kernel_dir / "hstu_attention_torch_ops_common.cpp"),
        str(kernel_dir / "hstu_attention_torch_ops_cpu.cpp"),
    ] + get_kernel_sources(archs)

    common_flags = [
        "-O3",
        "-std=c++20",
        "-DCK_ENABLE_INT8",
        "-DCK_ENABLE_FP8",
        "-DCK_ENABLE_BF8",
        "-DCK_ENABLE_FP16",
        "-DCK_ENABLE_FP32",
        "-DCK_ENABLE_FP64",
        "-DCK_ENABLE_BF16",
        # Round-to-nearest bf16 conversion.
        "-DCK_TILE_FLOAT_TO_BFLOAT16_DEFAULT=3",
    ]
    hip_flags = (
        common_flags
        + get_arch_flags(archs)
        + [
            "--offload-compress",
            # PyTorch's default HIP flags disable these; CK needs them.
            "-U__HIP_NO_HALF_OPERATORS__",
            "-U__HIP_NO_HALF_CONVERSIONS__",
        ]
    )
    cxx_flags = common_flags + (["-DHSTU_DISABLE_BWD"] if "gfx1250" in archs else [])
    return [
        CUDAExtension(
            name=f"{PACKAGE_NAME}._ck_hstu_experimental_C",
            sources=sources,
            include_dirs=[
                str(kernel_dir),
                str(kernel_dir / "instances"),
                str(ck_include_dir),
            ],
            extra_compile_args={"cxx": cxx_flags, "nvcc": hip_flags},
        )
    ]


setup(
    name=PACKAGE_NAME,
    version="0.1.0",
    description="Composable Kernel HSTU attention for AMD GPUs",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Unix",
    ],
    license_files=["LICENSE", "hstu_attention/LICENSE"],
    ext_modules=get_ext_modules(),
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.10",
    install_requires=["torch"],
)
