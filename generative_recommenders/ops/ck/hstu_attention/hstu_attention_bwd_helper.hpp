// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

// Workspace memory for the HSTU attention backward pass.
// This is allocated by the runtime/framework, not supplied by the API caller.
struct BwdWorkspace {
  // D[sq] = dO row(.) O per query position; written by kernel 1, consumed by
  // kernel 2. Only meaningful when use_softmax == true; nullptr otherwise.
  void* delta_ptr;
};
