/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.
 */
#include <torch/library.h> // @manual
#include <torch/nn/functional.h>
#include <torch/torch.h> // @manual

extern "C" {
/* Creates a dummy empty _C module that can be imported from Python.
    The import from Python will load the .so consisting of this file
    in this extension, so that the TORCH_LIBRARY static initializers
    below are run. */
PyObject* PyInit__ck_hstu_experimental_C(void) {
  static struct PyModuleDef module_def = {
      PyModuleDef_HEAD_INIT,
      "ck_hstu_experimental_C", /* name of module */
      NULL, /* module documentation, may be NULL */
      -1, /* size of per-interpreter state of the module,
              or -1 if the module keeps state in global variables. */
      NULL, /* methods */
  };
  return PyModule_Create(&module_def);
}
}

TORCH_LIBRARY_FRAGMENT(ck_hstu_experimental, m) {
  m.def(
      "ck_hstu_mha("
      "SymInt max_seq_len, "
      "float alpha, "
      "Tensor q, "
      "Tensor k, "
      "Tensor v, "
      "Tensor? seq_q_offsets, "
      "Tensor? seq_kv_offsets, "
      "bool causal, "
      "Tensor? num_targets, "
      "int max_attn_len, "
      "int min_full_attn_seq_len, "
      "int contextual_seq_len, "
      "Tensor? attn_scale, "
      "bool use_softmax, "
      "bool is_cross_attention, "
      "SymInt max_seq_len_kv=0, "
      "SymInt min_seq_len_q=0, "
      "SymInt min_seq_len_kv=0, "
      "int kv_truncate_len=0, "
      "bool kv_keep_last=False"
      ") -> Tensor");
  m.def(
      "ck_hstu_mha_fwd("
      "SymInt max_seq_len, "
      "float alpha, "
      "Tensor q, "
      "Tensor k, "
      "Tensor v, "
      "Tensor? seq_q_offsets, "
      "Tensor? seq_kv_offsets, "
      "bool causal, "
      "Tensor? num_targets, "
      "int max_attn_len, "
      "int min_full_attn_seq_len, "
      "int contextual_seq_len, "
      "Tensor? attn_scale, "
      "bool use_softmax, "
      "bool is_cross_attention, "
      "SymInt max_seq_len_kv=0, "
      "SymInt min_seq_len_q=0, "
      "SymInt min_seq_len_kv=0"
      ") -> (Tensor, Tensor)");
  m.def(
      "ck_hstu_mha_bwd("
      "SymInt max_seq_len, "
      "float alpha, "
      "Tensor grad_out, "
      "Tensor q, "
      "Tensor k, "
      "Tensor v, "
      "Tensor out, "
      "Tensor? lse, "
      "Tensor? seq_q_offsets, "
      "Tensor? seq_kv_offsets, "
      "bool causal, "
      "Tensor? num_targets, "
      "int max_attn_len, "
      "int min_full_attn_seq_len, "
      "int contextual_seq_len, "
      "Tensor? attn_scale, "
      "bool use_softmax, "
      "bool is_cross_attention, "
      "SymInt max_seq_len_kv=0, "
      "SymInt min_seq_len_q=0, "
      "SymInt min_seq_len_kv=0"
      ") -> (Tensor, Tensor, Tensor)");
}
