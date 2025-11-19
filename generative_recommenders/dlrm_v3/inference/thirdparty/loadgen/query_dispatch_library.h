/* Copyright 2019 The MLPerf Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

/// \file
/// \brief Defines the QueryDispatchLibrary interface.

#ifndef MLPERF_LOADGEN_QUERY_DISPATCH_LIBRARY_H
#define MLPERF_LOADGEN_QUERY_DISPATCH_LIBRARY_H

#include <string>

#include "system_under_test.h"

namespace mlperf {

/// \addtogroup LoadgenAPI
/// @{

/// \brief The interface a client implements for the LoadGen over the network to
/// test. The API inherits the System_under_test.h API When working in LON mode
/// the QueryDispatchLibrary class is used and natively Upcasted to the
/// QueryDispatchLibrary class.

class QueryDispatchLibrary : public SystemUnderTest {
 public:
  virtual ~QueryDispatchLibrary() = default;
};

/// @}

}  // namespace mlperf

#endif  // MLPERF_LOADGEN_QUERY_DISPATCH_LIBRARY_H
