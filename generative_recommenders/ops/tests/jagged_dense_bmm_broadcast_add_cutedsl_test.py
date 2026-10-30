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

#!/usr/bin/env python3

# Tests for the CuTeDSL (Blackwell tcgen05) backend of jagged_dense_bmm_broadcast_add. A target
# of their own: they need B200 and nvidia-cutlass-dsl >= 4.5.2, while the Triton tests in
# jagged_tensors_test run on any GPU with the default toolchain.

import contextlib
import random
import unittest
from typing import Any, Iterator, List, Tuple
from unittest import mock

import torch
from generative_recommenders.common import gpu_unavailable, HammerKernel
from generative_recommenders.ops import jagged_tensors
from generative_recommenders.ops.tests.jagged_tensors_test import (
    run_jagged_dense_bmm_broadcast_add_test,
)
from generative_recommenders.ops.utils import is_sm100_plus
from hypothesis import given, settings, strategies as st, Verbosity


def _seq_offsets(lengths: List[int], dtype: torch.dtype = torch.int64) -> torch.Tensor:
    offsets = torch.zeros(len(lengths) + 1, dtype=dtype, device=torch.device("cuda"))
    offsets[1:] = torch.cumsum(torch.tensor(lengths, device=offsets.device), 0)
    return offsets


def _ref_ddense_dbias(
    seq_offsets: torch.Tensor, jagged: torch.Tensor, d_out: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """fp32 per group: d_dense[g] = jagged[g].T @ d_out[g], d_bias[g] = d_out[g].sum(0)."""
    o = seq_offsets.tolist()
    d_dense = torch.zeros(
        len(o) - 1, jagged.shape[1], d_out.shape[1], device=jagged.device
    )
    d_bias = torch.zeros(len(o) - 1, d_out.shape[1], device=jagged.device)
    for g in range(len(o) - 1):
        if o[g + 1] > o[g]:
            d_dense[g] = (
                jagged[o[g] : o[g + 1]].float().T @ d_out[o[g] : o[g + 1]].float()
            )
            d_bias[g] = d_out[o[g] : o[g + 1]].float().sum(0)
    return d_dense, d_bias


def _ref_djagged(
    seq_offsets: torch.Tensor, d_out: torch.Tensor, dense: torch.Tensor
) -> torch.Tensor:
    """fp32 per group: d_jagged[g] = d_out[g] @ dense[g].T."""
    o = seq_offsets.tolist()
    return torch.cat(
        [d_out[o[g] : o[g + 1]].float() @ dense[g].float().T for g in range(len(o) - 1)]
    )


def _assert_close_fp32(actual: torch.Tensor, ref: torch.Tensor, msg: str = "") -> None:
    """bf16 kernel output against an fp32 reference: bf16 rounding plus summation order."""
    torch.testing.assert_close(
        actual.float(), ref, atol=5e-2, rtol=2e-2, msg=lambda m: f"{msg}: {m}"
    )


class _CudaArray:
    def __init__(self, ptr: int, shape: Tuple[int, ...], typestr: str) -> None:
        self.__cuda_array_interface__ = {
            "shape": shape,
            "typestr": typestr,
            "data": (ptr, False),
            "version": 3,
        }


class _GuardPage:
    """Device memory followed by reserved but unmapped VA. A tensor placed so it ends at the
    last mapped byte faults on any read past its end, deterministically -- the caching
    allocator usually has more memory mapped after a tensor, which hides such reads."""

    def __init__(self) -> None:
        # pyrefly: ignore [missing-import]  # cuda-bindings ships no type stubs
        import cuda.bindings.driver as cu

        def ok(ret: object) -> Any:
            ret = ret if isinstance(ret, tuple) else (ret,)
            assert ret[0] == cu.CUresult.CUDA_SUCCESS, ret[0]
            return ret[1] if len(ret) > 1 else None

        torch.empty(1, device="cuda")  # initialise the primary context
        prop = cu.CUmemAllocationProp()
        prop.type = cu.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
        prop.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        prop.location.id = torch.cuda.current_device()
        gran = ok(
            cu.cuMemGetAllocationGranularity(
                prop,
                cu.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM,
            )
        )
        self.mapped: int = 8 * gran
        va = int(ok(cu.cuMemAddressReserve(self.mapped + 2 * gran, 0, 0, 0)))
        ok(cu.cuMemMap(va, self.mapped, 0, ok(cu.cuMemCreate(self.mapped, prop, 0)), 0))
        access = cu.CUmemAccessDesc()
        access.location = prop.location
        access.flags = cu.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        ok(cu.cuMemSetAccess(va, self.mapped, [access], 1))
        self.end: int = va + self.mapped  # first unmapped byte

    def place(self, src: torch.Tensor) -> torch.Tensor:
        """A copy of `src` (bf16 or int32) whose last byte is the last mapped byte."""
        nbytes = src.numel() * src.element_size()
        ptr = self.end - nbytes
        assert ptr % 16 == 0 and nbytes <= self.mapped, (ptr, nbytes)
        typestr = {torch.bfloat16: "<f2", torch.int32: "<i4"}[src.dtype]
        raw = torch.as_tensor(_CudaArray(ptr, tuple(src.shape), typestr), device="cuda")
        out = raw.view(torch.bfloat16) if src.dtype == torch.bfloat16 else raw
        out.copy_(src)
        return out


class JaggedDenseBmmBroadcastAddCuTeDSLTest(unittest.TestCase):
    # tcgen05 MMA with cta_group=2 is Blackwell-only.
    @unittest.skipIf(not is_sm100_plus(), "CuTeDSL kernel requires sm_100+ (Blackwell)")
    @unittest.skipIf(*gpu_unavailable)
    # pyrefly: ignore [bad-argument-type]
    @given(
        batch_size=st.integers(4, 8),
        max_seq_len=st.integers(50, 500),
        D=st.sampled_from([64, 128, 256, 512]),
        K=st.sampled_from([64, 128, 256, 512]),
        # bf16 only by scope. fp16 is close to free later (ab_dtype is a parameter);
        # fp32 needs a tf32 MMA path.
        dtype=st.just(torch.bfloat16),
        contiguous=st.just(True),
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=20,
        deadline=None,
    )
    # pyrefly: ignore [bad-argument-type]
    def test_jagged_dense_bmm_broadcast_add_cutedsl(
        self, *args: Any, **kwargs: Any
    ) -> None:
        """Blackwell tcgen05 CuTeDSL kernel, forward only."""
        run_jagged_dense_bmm_broadcast_add_test(
            *args,
            **kwargs,
            test_backward=False,
            atol=None,
            rtol=None,
            ref_kernel=HammerKernel.PYTORCH,
            real_kernel=HammerKernel.CUTEDSL,
        )

    # tcgen05 MMA with cta_group=2 is Blackwell-only.
    @unittest.skipIf(not is_sm100_plus(), "CuTeDSL kernel requires sm_100+ (Blackwell)")
    @unittest.skipIf(*gpu_unavailable)
    # pyrefly: ignore [bad-argument-type]
    @given(
        batch_size=st.integers(4, 8),
        max_seq_len=st.integers(50, 500),
        # Backward: D is the N dim of the d_dense GEMM and K its M dim. Include sizes
        # that are not multiples of the 256-wide tiles (multiples of 8 only: TMA needs
        # 16-byte-aligned row strides). K is the d_jagged contraction: 64 is a
        # single k-block, 832 more than the production 768.
        D=st.sampled_from([64, 136, 256, 520]),
        K=st.sampled_from([64, 200, 256, 832]),
        dtype=st.just(torch.bfloat16),
        contiguous=st.just(True),
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=20,
        deadline=None,
    )
    # pyrefly: ignore [bad-argument-type]
    def test_jagged_dense_bmm_broadcast_add_cutedsl_backward(
        self, *args: Any, **kwargs: Any
    ) -> None:
        """Blackwell tcgen05 CuTeDSL kernels, forward and backward (d_jagged, d_dense, d_bias)."""
        run_jagged_dense_bmm_broadcast_add_test(
            *args,
            **kwargs,
            test_backward=True,
            atol=None,
            rtol=None,
            ref_kernel=HammerKernel.PYTORCH,
            real_kernel=HammerKernel.CUTEDSL,
        )

    _CUTEDSL_DEEP_K_ATOL = 1e-3
    # (name, batch_size, lengths, max_seq_len, D_contraction, K_output, atol)
    _CUTEDSL_SHAPE_CASES = [
        # No skipped tiles at a length spanning many M tiles: the deadlock regime.
        ("no_skipped_tiles", 8, [2048] * 8, 2048, 2176, 768, _CUTEDSL_DEEP_K_ATOL),
        # Same, deeper persistent loop (20 M tiles per group).
        ("long_groups_no_skip", 8, [5120] * 8, 5120, 256, 256, None),
        # Production dims, plus an empty group, a single-row group and a partial tile.
        (
            "production_dims",
            4,
            [3000, 5000, 1, 0],
            8192,
            2176,
            768,
            _CUTEDSL_DEEP_K_ATOL,
        ),
        # Empty-group position matters: leading, interior, and all-but-one.
        ("empty_first", 4, [0, 501, 501, 501], 512, 256, 256, None),
        ("empty_middle", 4, [501, 0, 501, 501], 512, 256, 256, None),
        ("only_one_nonempty", 4, [0, 0, 0, 1007], 1024, 256, 256, None),
        # One group far longer than the rest.
        ("skewed", 4, [8009, 149, 149, 149], 8192, 256, 256, None),
        ("single_group", 1, [2023], 2048, 256, 256, None),
        # Group shorter than one 128-row M tile.
        ("short_group", 4, [17, 501, 501, 501], 512, 256, 256, None),
        # N tiling: 320 is one full 256 tile plus a partial, 832 is three plus a
        # partial. Both exercise the per-element epilogue fallback, which the
        # vectorised store path never takes.
        ("n_one_full_one_partial", 4, [501] * 4, 512, 256, 320, None),
        ("n_multi_tile_partial_tail", 4, [501] * 4, 512, 256, 832, None),
    ]

    def _run_cutedsl_shape_cases(self, test_backward: bool) -> None:
        for (
            name,
            batch_size,
            lengths,
            max_seq_len,
            d,
            k,
            atol,
        ) in self._CUTEDSL_SHAPE_CASES:
            with self.subTest(case=name):
                run_jagged_dense_bmm_broadcast_add_test(
                    batch_size=batch_size,
                    max_seq_len=max_seq_len,
                    D=d,
                    K=k,
                    dtype=torch.bfloat16,
                    contiguous=True,
                    test_backward=test_backward,
                    atol=atol,
                    rtol=None if atol is None else 1.6e-2,
                    ref_kernel=HammerKernel.PYTORCH,
                    real_kernel=HammerKernel.CUTEDSL,
                    lengths=torch.tensor(lengths, dtype=torch.int64),
                )

    # tcgen05 MMA with cta_group=2 is Blackwell-only.
    @unittest.skipIf(not is_sm100_plus(), "CuTeDSL kernel requires sm_100+ (Blackwell)")
    @unittest.skipIf(*gpu_unavailable)
    def test_jagged_dense_bmm_broadcast_add_cutedsl_shapes(self) -> None:
        self._run_cutedsl_shape_cases(test_backward=False)

    # tcgen05 MMA with cta_group=2 is Blackwell-only.
    @unittest.skipIf(not is_sm100_plus(), "CuTeDSL kernel requires sm_100+ (Blackwell)")
    @unittest.skipIf(*gpu_unavailable)
    def test_jagged_dense_bmm_broadcast_add_cutedsl_backward_shapes(self) -> None:
        self._run_cutedsl_shape_cases(test_backward=True)

    @contextlib.contextmanager
    def _expect_triton_fallback(self, reasons: Tuple[str, ...]) -> Iterator[None]:
        """With `reasons`: the CUTEDSL request runs on Triton and logs one warning naming
        each reason. Without: it stays on CuTeDSL and logs nothing."""
        triton_fn = jagged_tensors.triton_jagged_dense_bmm_add
        with mock.patch.object(
            jagged_tensors, "triton_jagged_dense_bmm_add", wraps=triton_fn
        ) as triton_spy:
            if not reasons:
                with self.assertNoLogs(jagged_tensors.logger, level="WARNING"):
                    yield
                self.assertEqual(triton_spy.call_count, 0)
                return
            with self.assertLogs(jagged_tensors.logger, level="WARNING") as logs:
                yield
            self.assertEqual(triton_spy.call_count, 1)
        self.assertEqual(len(logs.records), 1)
        self.assertIn("falling back to the Triton kernel", logs.output[0])
        for reason in reasons:
            self.assertIn(reason, logs.output[0])

    # tcgen05 MMA with cta_group=2 is Blackwell-only.
    @unittest.skipIf(not is_sm100_plus(), "CuTeDSL kernel requires sm_100+ (Blackwell)")
    @unittest.skipIf(*gpu_unavailable)
    def test_jagged_dense_bmm_broadcast_add_cutedsl_unsupported_inputs(self) -> None:
        """Inputs the CuTeDSL kernels cannot compute (feature dims not divisible by 8,
        non-bf16) fall back to Triton; a non-contiguous dense is made contiguous. Each was
        silently wrong before. The dispatchers themselves refuse unsupported inputs."""
        bad_dims, bad_dtype = "not divisible by 8", "non-bf16 inputs"
        # (name, D, K, dtype, contiguous, reasons the fallback warning must name)
        cases = [
            ("D_not_multiple_of_8", 130, 200, torch.bfloat16, True, (bad_dims,)),
            ("K_not_multiple_of_8", 256, 130, torch.bfloat16, True, (bad_dims,)),
            ("fp16", 256, 200, torch.float16, True, (bad_dtype,)),
            (
                "fp16_and_D_not_multiple_of_8",
                130,
                200,
                torch.float16,
                True,
                (bad_dtype, bad_dims),
            ),
            # Supported (made contiguous by the dispatcher): CuTeDSL, no warning.
            ("noncontiguous_dense", 256, 200, torch.bfloat16, False, ()),
        ]
        # The fallback warning is logged once per distinct message; start clean.
        jagged_tensors._CUTEDSL_FALLBACK_WARNED.clear()
        for name, d, k, dtype, contiguous, reasons in cases:
            with self.subTest(case=name), self._expect_triton_fallback(reasons):
                run_jagged_dense_bmm_broadcast_add_test(
                    batch_size=4,
                    max_seq_len=300,
                    D=d,
                    K=k,
                    dtype=dtype,
                    contiguous=contiguous,
                    test_backward=True,
                    atol=None,
                    rtol=None,
                    ref_kernel=HammerKernel.PYTORCH,
                    real_kernel=HammerKernel.CUTEDSL,
                    lengths=torch.tensor([100, 37, 0, 260], dtype=torch.int64),
                )

        from generative_recommenders.ops.cutedsl.cutedsl_jagged import (
            _cutedsl_jagged_dense_bmm_broadcast_add_ddense_dbias,
        )

        device = torch.device("cuda")
        with self.assertRaises(ValueError):
            _cutedsl_jagged_dense_bmm_broadcast_add_ddense_dbias(
                torch.tensor([0, 4], device=device),
                torch.randn(4, 130, device=device, dtype=torch.bfloat16),
                torch.randn(4, 200, device=device, dtype=torch.bfloat16),
                4,
            )

    # tcgen05 MMA with cta_group=2 is Blackwell-only.
    @unittest.skipIf(not is_sm100_plus(), "CuTeDSL kernel requires sm_100+ (Blackwell)")
    @unittest.skipIf(*gpu_unavailable)
    def test_jagged_dense_bmm_broadcast_add_cutedsl_benchmark_shapes(self) -> None:
        """Forward and backward end to end at the benchmark's operating points: default
        dims (G=384, D=K=512) and production dims (G=40, D=2176, K=768), lengths drawn
        uniformly from [0, seq_len] so empty groups occur. seq_len stops at 16384: the
        PyTorch reference pads to (G, seq_len, D) and hits an int32 limit near 24K."""
        # (name, batch_size, seq_len, D, K)
        cases = [
            ("default_seq256", 384, 256, 512, 512),
            ("default_seq1024", 384, 1024, 512, 512),
            ("default_seq4096", 384, 4096, 512, 512),
            ("prod_seq256", 40, 256, 2176, 768),
            ("prod_seq1024", 40, 1024, 2176, 768),
            ("prod_seq8192", 40, 8192, 2176, 768),
            ("prod_seq16384", 40, 16384, 2176, 768),
        ]
        for name, batch_size, seq_len, d, k in cases:
            with self.subTest(case=name):
                torch.manual_seed(42)
                run_jagged_dense_bmm_broadcast_add_test(
                    batch_size=batch_size,
                    max_seq_len=seq_len,
                    D=d,
                    K=k,
                    dtype=torch.bfloat16,
                    test_backward=True,
                    atol=1e-2,
                    rtol=1e-2,
                    ref_kernel=HammerKernel.PYTORCH,
                    real_kernel=HammerKernel.CUTEDSL,
                    lengths=torch.randint(seq_len + 1, (batch_size,)),
                )

    # tcgen05 MMA with cta_group=2 is Blackwell-only.
    @unittest.skipIf(not is_sm100_plus(), "CuTeDSL kernel requires sm_100+ (Blackwell)")
    @unittest.skipIf(*gpu_unavailable)
    def test_jagged_dense_bmm_broadcast_add_cutedsl_bwd_kernels(self) -> None:
        """Each backward kernel on its own against an fp32 reference, so a failure points
        at one kernel: d_jagged (contraction K) and the fused d_dense + d_bias (output
        tiles over D x K). Dims cover single k-blocks, partial N/M tiles, K past the
        production 768, and production D."""
        from generative_recommenders.ops.cutedsl import cutedsl_jagged

        dims = [
            (64, 64),
            (256, 64),
            (64, 256),
            (256, 256),
            (256, 320),
            (256, 832),
            (2176, 768),
            (512, 512),
            (136, 200),
            (520, 512),
            (2176, 640),
            (2176, 896),
        ]
        lengths = [301, 0, 517, 129]
        seq_offsets = _seq_offsets(lengths)
        for d, k in dims:
            with self.subTest(D=d, K=k):
                torch.manual_seed(0)
                total = int(seq_offsets[-1])
                jagged = torch.randn(total, d, dtype=torch.bfloat16, device="cuda")
                dense = torch.randn(
                    len(lengths), d, k, dtype=torch.bfloat16, device="cuda"
                )
                d_out = torch.randn(total, k, dtype=torch.bfloat16, device="cuda")
                d_jagged = (
                    cutedsl_jagged._cutedsl_jagged_dense_bmm_broadcast_add_djagged(
                        seq_offsets, d_out, dense, max(lengths)
                    )
                )
                _assert_close_fp32(
                    d_jagged, _ref_djagged(seq_offsets, d_out, dense), "d_jagged"
                )
                d_dense, d_bias = (
                    cutedsl_jagged._cutedsl_jagged_dense_bmm_broadcast_add_ddense_dbias(
                        seq_offsets, jagged, d_out, max(lengths)
                    )
                )
                ref_dense, ref_bias = _ref_ddense_dbias(seq_offsets, jagged, d_out)
                _assert_close_fp32(d_dense, ref_dense, "d_dense")
                _assert_close_fp32(d_bias, ref_bias, "d_bias")

    # tcgen05 MMA with cta_group=2 is Blackwell-only.
    @unittest.skipIf(not is_sm100_plus(), "CuTeDSL kernel requires sm_100+ (Blackwell)")
    @unittest.skipIf(*gpu_unavailable)
    def test_jagged_dense_bmm_broadcast_add_cutedsl_ddense_dbias_cases(self) -> None:
        """Fused d_dense + d_bias against fp32 on both epilogue paths. After the operand
        swap D is the GEMM N (tile_n=256) and K the GEMM M (mma_tile_m=256); the direct
        st.global path needs both to divide, everything else takes the TMA-store path."""
        from generative_recommenders.ops.cutedsl import cutedsl_jagged

        torch.manual_seed(0)
        prod_lengths = torch.randint(0, 16385, (40,)).tolist()
        prod_lengths[3] = 0
        # (name, D, K, lengths, max_seq_len)
        cases = [
            ("production_random", 2176, 768, prod_lengths, 16384),
            ("fast_path", 2048, 768, [1000, 2048, 777, 1500], 2048),
            # One dim off its tile: a partial output M tile (K) or N tile (D).
            ("slow_path_partial_m", 2048, 640, [1000, 2048, 777, 1500], 2048),
            ("slow_path_partial_n", 2176, 640, [1000, 2048, 777, 1500], 2048),
            ("small_slow_path", 384, 256, [56, 130, 64], 130),
            # Neither dim divides its tile (multiples of 8 only: TMA row strides).
            ("ragged_2120x760", 2120, 760, [1311, 0, 2765, 7, 2999], 3000),
            ("ragged_200x96", 200, 96, [2048, 0, 513, 1], 3000),
            ("ragged_1000x520", 1000, 520, [600, 0, 2999], 3000),
        ]
        for name, d, k, lengths, max_seq_len in cases:
            with self.subTest(case=name):
                seq_offsets = _seq_offsets(lengths)
                total = int(seq_offsets[-1])
                jagged = torch.randn(total, d, dtype=torch.bfloat16, device="cuda")
                d_out = torch.randn(total, k, dtype=torch.bfloat16, device="cuda")
                d_dense, d_bias = (
                    cutedsl_jagged._cutedsl_jagged_dense_bmm_broadcast_add_ddense_dbias(
                        seq_offsets, jagged, d_out, max_seq_len
                    )
                )
                ref_dense, ref_bias = _ref_ddense_dbias(seq_offsets, jagged, d_out)
                _assert_close_fp32(d_dense, ref_dense, "d_dense")
                _assert_close_fp32(d_bias, ref_bias, "d_bias")

    # tcgen05 MMA with cta_group=2 is Blackwell-only.
    @unittest.skipIf(not is_sm100_plus(), "CuTeDSL kernel requires sm_100+ (Blackwell)")
    @unittest.skipIf(*gpu_unavailable)
    def test_jagged_dense_bmm_broadcast_add_cutedsl_ddense_dbias_int32_row_offsets(
        self,
    ) -> None:
        """Groups starting past row 986,895, where seq_start * D (D=2176) exceeds 2^31:
        element offsets must not wrap in 32-bit arithmetic. Needs ~6 GB of device memory."""
        from generative_recommenders.ops.cutedsl import cutedsl_jagged

        torch.manual_seed(3)
        d, k = 2176, 768
        lengths = [1_000_000, 4096, 3000]
        seq_offsets = _seq_offsets(lengths)
        total = int(seq_offsets[-1])
        jagged = torch.randn(total, d, dtype=torch.bfloat16, device="cuda")
        d_out = torch.randn(total, k, dtype=torch.bfloat16, device="cuda")
        d_dense, d_bias = (
            cutedsl_jagged._cutedsl_jagged_dense_bmm_broadcast_add_ddense_dbias(
                seq_offsets, jagged, d_out, lengths[0]
            )
        )
        o = seq_offsets.tolist()
        for g in (1, 2):  # the groups past the int32 boundary
            rows = slice(o[g], o[g + 1])
            ref = jagged[rows].float().T @ d_out[rows].float()
            _assert_close_fp32(d_dense[g], ref, f"d_dense[{g}]")
            _assert_close_fp32(d_bias[g], d_out[rows].float().sum(0), f"d_bias[{g}]")

    # tcgen05 MMA with cta_group=2 is Blackwell-only.
    @unittest.skipIf(not is_sm100_plus(), "CuTeDSL kernel requires sm_100+ (Blackwell)")
    @unittest.skipIf(*gpu_unavailable)
    def test_jagged_dense_bmm_broadcast_add_cutedsl_bwd_deterministic(self) -> None:
        """The backward kernels are bitwise run-to-run deterministic and correct on every
        run. Many short and empty groups (lengths 0..300) is the regime where per-cluster
        TMA descriptor slots raced and wrote rows into the next group; the production-dims
        case has long groups."""
        from generative_recommenders.ops.cutedsl import cutedsl_jagged

        ddense = cutedsl_jagged._cutedsl_jagged_dense_bmm_broadcast_add_ddense_dbias
        djagged = cutedsl_jagged._cutedsl_jagged_dense_bmm_broadcast_add_djagged
        # (name, G, max length, D, K, runs, check d_jagged too)
        cases = [
            ("short_groups", 160, 300, 2176, 768, 20, True),
            ("production_dims", 40, 4096, 2176, 768, 10, False),
        ]
        for name, num_groups, max_len, d, k, runs, with_djagged in cases:
            with self.subTest(case=name):
                torch.manual_seed(7)
                lengths = torch.randint(0, max_len + 1, (num_groups,)).tolist()
                seq_offsets = _seq_offsets(lengths)
                total = int(seq_offsets[-1])
                jagged = torch.randn(total, d, dtype=torch.bfloat16, device="cuda")
                d_out = torch.randn(total, k, dtype=torch.bfloat16, device="cuda")
                dense = torch.randn(
                    num_groups, d, k, dtype=torch.bfloat16, device="cuda"
                )
                ref_dense, ref_bias = _ref_ddense_dbias(seq_offsets, jagged, d_out)
                ref_djagged = (
                    _ref_djagged(seq_offsets, d_out, dense) if with_djagged else None
                )
                first = None
                for run in range(runs):
                    outs = ddense(seq_offsets, jagged, d_out, max_len)
                    if with_djagged:
                        outs += (djagged(seq_offsets, d_out, dense, max_len),)
                    if first is None:
                        first = outs
                        _assert_close_fp32(outs[0], ref_dense, "d_dense")
                        _assert_close_fp32(outs[1], ref_bias, "d_bias")
                        if ref_djagged is not None:
                            _assert_close_fp32(outs[2], ref_djagged, "d_jagged")
                        continue
                    for out_name, out, out0 in zip(
                        ("d_dense", "d_bias", "d_jagged"), outs, first
                    ):
                        self.assertTrue(
                            torch.equal(out, out0),
                            f"{out_name} of run {run} differs bitwise from run 0",
                        )

    # tcgen05 MMA with cta_group=2 is Blackwell-only.
    @unittest.skipIf(not is_sm100_plus(), "CuTeDSL kernel requires sm_100+ (Blackwell)")
    @unittest.skipIf(*gpu_unavailable)
    def test_jagged_dense_bmm_broadcast_add_cutedsl_ddense_dbias_one_compile_across_batches(
        self,
    ) -> None:
        """total_M and max_seq_len are not in the compile key, so one kernel per (D, K, G)
        must serve batches of any size -- including smaller ones than it was compiled for,
        which faulted while total_M was a static shape."""
        from generative_recommenders.ops.cutedsl import cutedsl_jagged

        for d, k, num_groups in [(2176, 768, 41), (2176, 640, 13)]:
            n_compiled = len(cutedsl_jagged._COMPILED_BWD_DDENSE_DBIAS)
            for seed, max_len in [(1, 4096), (2, 300), (3, 9000), (4, 64)]:
                with self.subTest(D=d, K=k, G=num_groups, max_len=max_len):
                    torch.manual_seed(seed)
                    lengths = torch.randint(0, max_len + 1, (num_groups,)).tolist()
                    seq_offsets = _seq_offsets(lengths)
                    total = int(seq_offsets[-1])
                    jagged = torch.randn(total, d, dtype=torch.bfloat16, device="cuda")
                    d_out = torch.randn(total, k, dtype=torch.bfloat16, device="cuda")
                    d_dense, d_bias = (
                        cutedsl_jagged._cutedsl_jagged_dense_bmm_broadcast_add_ddense_dbias(
                            seq_offsets, jagged, d_out, max_len
                        )
                    )
                    ref_dense, ref_bias = _ref_ddense_dbias(seq_offsets, jagged, d_out)
                    _assert_close_fp32(d_dense, ref_dense, "d_dense")
                    _assert_close_fp32(d_bias, ref_bias, "d_bias")
            self.assertEqual(
                len(cutedsl_jagged._COMPILED_BWD_DDENSE_DBIAS) - n_compiled,
                1,
                f"compiles for 4 batches of (D={d}, K={k}, G={num_groups})",
            )

    # tcgen05 MMA with cta_group=2 is Blackwell-only.
    @unittest.skipIf(not is_sm100_plus(), "CuTeDSL kernel requires sm_100+ (Blackwell)")
    @unittest.skipIf(*gpu_unavailable)
    def test_jagged_dense_bmm_broadcast_add_cutedsl_ddense_dbias_no_host_sync(
        self,
    ) -> None:
        """After its compile, the fused d_dense + d_bias dispatcher issues no synchronizing
        op, and it can be captured in a CUDA graph whose replay matches eager."""
        from generative_recommenders.ops.cutedsl import cutedsl_jagged

        torch.manual_seed(3)
        lengths = torch.randint(1, 4097, (40,)).tolist()
        seq_offsets = _seq_offsets(lengths)
        total = int(seq_offsets[-1])
        jagged = torch.randn(total, 2176, dtype=torch.bfloat16, device="cuda")
        d_out = torch.randn(total, 768, dtype=torch.bfloat16, device="cuda")

        def ddense() -> Tuple[torch.Tensor, torch.Tensor]:
            return cutedsl_jagged._cutedsl_jagged_dense_bmm_broadcast_add_ddense_dbias(
                seq_offsets, jagged, d_out, 4096
            )

        eager = ddense()  # compile outside the checks
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("error")
        try:
            ddense()
        finally:
            torch.cuda.set_sync_debug_mode("default")
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = ddense()
        graph.replay()
        torch.cuda.synchronize()
        self.assertTrue(
            torch.equal(captured[0], eager[0]), "d_dense: graph replay != eager"
        )
        self.assertTrue(
            torch.equal(captured[1], eager[1]), "d_bias: graph replay != eager"
        )

    # tcgen05 MMA with cta_group=2 is Blackwell-only.
    @unittest.skipIf(not is_sm100_plus(), "CuTeDSL kernel requires sm_100+ (Blackwell)")
    @unittest.skipIf(*gpu_unavailable)
    def test_jagged_dense_bmm_broadcast_add_cutedsl_djagged_one_compile_across_batches(
        self,
    ) -> None:
        """The offsets are not in the d_jagged compile key, so batches with different
        lengths (and totals) reuse one kernel per (D, K, G, max_seq_len)."""
        from generative_recommenders.ops.cutedsl import cutedsl_jagged

        max_len = 4096
        for d, k, num_groups in [(2176, 768, 41), (520, 200, 7)]:
            n_compiled = len(cutedsl_jagged._COMPILED_BWD_DJAGGED)
            for seed in range(4):
                with self.subTest(D=d, K=k, G=num_groups, seed=seed):
                    torch.manual_seed(seed)
                    lengths = torch.randint(0, max_len + 1, (num_groups,)).tolist()
                    seq_offsets = _seq_offsets(lengths)
                    total = int(seq_offsets[-1])
                    d_out = torch.randn(total, k, dtype=torch.bfloat16, device="cuda")
                    dense = torch.randn(
                        num_groups, d, k, dtype=torch.bfloat16, device="cuda"
                    )
                    d_jagged = (
                        cutedsl_jagged._cutedsl_jagged_dense_bmm_broadcast_add_djagged(
                            seq_offsets, d_out, dense, max_len
                        )
                    )
                    _assert_close_fp32(
                        d_jagged, _ref_djagged(seq_offsets, d_out, dense), "d_jagged"
                    )
            self.assertEqual(
                len(cutedsl_jagged._COMPILED_BWD_DJAGGED) - n_compiled,
                1,
                f"compiles for 4 batches of (D={d}, K={k}, G={num_groups})",
            )

    # tcgen05 MMA with cta_group=2 is Blackwell-only.
    @unittest.skipIf(not is_sm100_plus(), "CuTeDSL kernel requires sm_100+ (Blackwell)")
    @unittest.skipIf(*gpu_unavailable)
    def test_jagged_dense_bmm_broadcast_add_cutedsl_backward_no_host_sync(
        self,
    ) -> None:
        """After its compile, the d_jagged dispatcher issues no synchronizing op and can
        be captured in a CUDA graph whose replay matches eager; and a full CUTEDSL
        forward + backward through jagged_dense_bmm_broadcast_add issues none either."""
        from generative_recommenders.ops.cutedsl import cutedsl_jagged

        torch.manual_seed(5)
        lengths = torch.randint(1, 4097, (40,)).tolist()
        seq_offsets = _seq_offsets(lengths)
        total = int(seq_offsets[-1])
        d_out = torch.randn(total, 768, dtype=torch.bfloat16, device="cuda")
        dense = torch.randn(40, 2176, 768, dtype=torch.bfloat16, device="cuda")

        def djagged() -> torch.Tensor:
            return cutedsl_jagged._cutedsl_jagged_dense_bmm_broadcast_add_djagged(
                seq_offsets, d_out, dense, 4096
            )

        eager = djagged()  # compile outside the checks
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("error")
        try:
            djagged()
        finally:
            torch.cuda.set_sync_debug_mode("default")
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = djagged()
        graph.replay()
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(captured, eager), "d_jagged: graph replay != eager")

        jagged = torch.randn(total, 2176, dtype=torch.bfloat16, device="cuda")
        bias = torch.randn(40, 768, dtype=torch.bfloat16, device="cuda")

        def fwd_bwd() -> None:
            j, w, b = (t.detach().requires_grad_() for t in (jagged, dense, bias))
            out = jagged_tensors.jagged_dense_bmm_broadcast_add(
                4096, seq_offsets, j, w, b, kernel=HammerKernel.CUTEDSL
            )
            out.backward(d_out)

        fwd_bwd()  # compile all three kernels
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("error")
        try:
            fwd_bwd()
        finally:
            torch.cuda.set_sync_debug_mode("default")

    # tcgen05 MMA with cta_group=2 is Blackwell-only.
    @unittest.skipIf(not is_sm100_plus(), "CuTeDSL kernel requires sm_100+ (Blackwell)")
    @unittest.skipIf(*gpu_unavailable)
    def test_jagged_dense_bmm_broadcast_add_cutedsl_no_out_of_bounds_reads(
        self,
    ) -> None:
        """No CuTeDSL kernel reads past the end of its inputs. Each input in turn is placed
        against an unmapped guard page, so any such read is an illegal-address error rather
        than a silent read of a neighbouring allocation. Regression for the intermittent
        d_dense + d_bias illegal address: descriptors rebased to each group's first row let
        row-clipped TMA boxes read far past the group, and a total_M baked in at compile time
        over-read on later, smaller batches."""
        from generative_recommenders.ops.cutedsl import cutedsl_jagged

        guard = _GuardPage()
        ddense = cutedsl_jagged._cutedsl_jagged_dense_bmm_broadcast_add_ddense_dbias

        def check_ddense(
            guarded: str, d: int, k: int, lengths: List[int], tag: str
        ) -> None:
            seq_offsets = _seq_offsets(lengths, torch.int32)
            total = int(seq_offsets[-1])
            jagged = torch.empty(
                total, d, dtype=torch.bfloat16, device="cuda"
            ).uniform_(-1, 1)
            d_out = torch.randn(total, k, dtype=torch.bfloat16, device="cuda") * 0.01
            if guarded == "jagged":
                jagged = guard.place(jagged)
            elif guarded == "d_out":
                d_out = guard.place(d_out)
            else:
                seq_offsets = guard.place(seq_offsets)
            with self.subTest(case=tag):
                d_dense, d_bias = ddense(seq_offsets, jagged, d_out, max(lengths))
                torch.cuda.synchronize()  # surfaces an illegal address here
                ref_dense, ref_bias = _ref_ddense_dbias(seq_offsets, jagged, d_out)
                _assert_close_fp32(d_dense, ref_dense, f"d_dense ({tag})")
                _assert_close_fp32(d_bias, ref_bias, f"d_bias ({tag})")

        # Minimal deterministic repros (name, guarded input, D, K, lengths, warm-up lengths
        # run first with ordinary allocations, compiling for a larger total_M).
        repros = [
            # Rebased descriptor, boxes clipped by the row extent; faulted on the first call.
            (
                "rebased_rows_g8",
                "jagged",
                256,
                200,
                [30, 68, 90, 113, 7, 106, 20, 69],
                None,
            ),
            ("rebased_rows_g2", "jagged", 256, 200, [192, 69], None),
            # Boxes clipped in dim 0 (the last N tile of D=520; A halves past K=64).
            (
                "dim0_clip_jagged",
                "jagged",
                520,
                64,
                [421, 120, 18, 301, 60, 283, 403],
                None,
            ),
            (
                "dim0_clip_d_out",
                "d_out",
                520,
                64,
                [421, 120, 18, 301, 60, 283, 403],
                None,
            ),
            # Compiled on a larger total_M, then called on a smaller one.
            (
                "shrink_total_m_d_out",
                "d_out",
                136,
                64,
                [38, 55, 413, 25, 441],
                [404, 361, 424, 180, 338],
            ),
            (
                "shrink_total_m_jagged",
                "jagged",
                256,
                64,
                [4, 48, 7, 16, 7],
                [403, 64, 341, 47, 147],
            ),
        ]
        for name, guarded, d, k, lengths, warmup in repros:
            if warmup is not None:
                warm_offsets = _seq_offsets(warmup, torch.int32)
                warm_total = int(warm_offsets[-1])
                ddense(
                    warm_offsets,
                    torch.randn(warm_total, d, dtype=torch.bfloat16, device="cuda"),
                    torch.randn(warm_total, k, dtype=torch.bfloat16, device="cuda"),
                    max(warmup),
                )
            check_ddense(guarded, d, k, lengths, name)

        # Random small shapes like the hypothesis tests, each input against the guard.
        rng = random.Random(0)
        torch.manual_seed(0)
        for guarded in ("jagged", "d_out", "seq_offsets"):
            for it in range(30):
                batch_size, max_seq_len = rng.randint(4, 8), rng.randint(50, 500)
                d, k = rng.choice([64, 136, 256, 520]), rng.choice([64, 200, 256, 832])
                lengths = torch.randint(max_seq_len + 1, (batch_size,)).tolist()
                width = (
                    (batch_size + 1) * 4
                    if guarded == "seq_offsets"
                    else sum(lengths) * 2 * (d if guarded == "jagged" else k)
                )
                if width % 16:  # the guard-placed tensor must stay 16 B aligned
                    continue
                check_ddense(
                    guarded,
                    d,
                    k,
                    lengths,
                    f"random_{guarded}_{it} D={d} K={k} {lengths}",
                )

        # The forward (jagged guarded) and d_jagged (d_out guarded) kernels too.
        for it in range(15):
            batch_size, max_seq_len = rng.randint(4, 8), rng.randint(50, 500)
            d, k = rng.choice([64, 136, 256, 520]), rng.choice([64, 200, 256, 832])
            lengths = torch.randint(max_seq_len + 1, (batch_size,)).tolist()
            seq_offsets = _seq_offsets(lengths)
            total = int(seq_offsets[-1])
            dense = torch.randn(batch_size, d, k, dtype=torch.bfloat16, device="cuda")
            bias = torch.randn(batch_size, k, dtype=torch.bfloat16, device="cuda")
            with self.subTest(case=f"forward_{it} D={d} K={k} {lengths}"):
                if total * d % 8 == 0:
                    jagged = guard.place(
                        torch.randn(total, d, dtype=torch.bfloat16, device="cuda")
                    )
                    out = cutedsl_jagged.cutedsl_jagged_dense_bmm_broadcast_add(
                        seq_offsets, jagged, dense, bias, max(lengths)
                    )
                    torch.cuda.synchronize()
                    o = seq_offsets.tolist()
                    ref = torch.cat(
                        [
                            jagged[o[g] : o[g + 1]].float() @ dense[g].float()
                            + bias[g].float()
                            for g in range(batch_size)
                        ]
                    )
                    _assert_close_fp32(out, ref, "forward")
            with self.subTest(case=f"d_jagged_{it} D={d} K={k} {lengths}"):
                if total * k % 8 == 0:
                    d_out = guard.place(
                        torch.randn(total, k, dtype=torch.bfloat16, device="cuda")
                    )
                    d_jagged = (
                        cutedsl_jagged._cutedsl_jagged_dense_bmm_broadcast_add_djagged(
                            seq_offsets, d_out, dense, max(lengths)
                        )
                    )
                    torch.cuda.synchronize()
                    _assert_close_fp32(
                        d_jagged, _ref_djagged(seq_offsets, d_out, dense), "d_jagged"
                    )

    @unittest.skipIf(
        jagged_tensors._CUTEDSL_IMPORT_ERROR is not None,
        "cutedsl_jagged not importable",
    )
    def test_cutlass_dsl_version_check(self) -> None:
        """cutedsl_jagged refuses to import under an nvidia-cutlass-dsl it is not validated
        on, naming the BUCK fix, instead of failing inside the first JIT compile."""
        from generative_recommenders.ops.cutedsl.cutedsl_jagged import (
            _check_cutlass_dsl_version,
        )

        for version in ("4.5.2", "4.6.1", "5.0"):
            _check_cutlass_dsl_version(version)
        for version in ("4.4.2", "4.4.0.dev1", "0"):
            with self.subTest(version=version):
                with self.assertRaisesRegex(ImportError, "constraint_overrides"):
                    _check_cutlass_dsl_version(version)


if __name__ == "__main__":
    unittest.main()
