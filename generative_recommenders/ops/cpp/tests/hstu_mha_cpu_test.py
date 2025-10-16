# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

# cmd: buck2 run @//mode/opt -c fbcode.enable_gpu_sections=true -c fbcode.platform010_cuda_version=12.8 -c fbcode.nvcc_arch=b200a  //generative_recommenders/ops/cpp/tests:hstu_mha_cpu_test

import unittest

import torch

torch.ops.load_library(
    "//generative_recommenders/ops/cpp/hstu_attention:hstu_flash_attention"
)


class TestHstuMhaFwd(unittest.TestCase):
    def test_hstu_mha_fwd(self) -> None:
        q: torch.Tensor = torch.randn([100, 4, 64], dtype=torch.bfloat16, device="cpu")
        k: torch.Tensor = torch.randn([100, 4, 64], dtype=torch.bfloat16, device="cpu")
        v: torch.Tensor = torch.randn([100, 4, 64], dtype=torch.bfloat16, device="cpu")
        res = torch.ops.hstu.hstu_mha_fwd(
            10,
            0.25,
            q,
            k,
            v,
            torch.empty([0], dtype=torch.int32, device="cpu"),
            True,  # causal
            None,
            None,
            0,
            0,
            0,
            None,  # q_descale
            None,  # k_descale
            None,  # v_descale
            0,  # sm_margin
        )
        self.assertIsNotNone(res)
