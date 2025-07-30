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

// Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar,
// Pradeep Ramani, Tri Dao. Splitting the different template instantiations to
// different files to speed up compilation. This file is auto-generated. See
// "generate_kernels.py"

#ifdef OSS_ENV
#include "hstu_attention/flash_fwd_launch_template.h"
#else
#include "flash_fwd_launch_template.h"
#endif

namespace hstu {
#ifndef FLASHATTENTION_DISABLE_HDIM256
template void run_mha_fwd_<90, cutlass::float_e4m3_t, 256, false>(
    Flash_fwd_params& params,
    cudaStream_t stream);
#endif
} // namespace hstu
