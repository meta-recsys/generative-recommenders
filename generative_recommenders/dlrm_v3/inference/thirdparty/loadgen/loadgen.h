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
/// \brief Provides the entry points for a SUT to start a test and respond
/// to issued queries.

#ifndef MLPERF_LOADGEN_LOADGEN_H_
#define MLPERF_LOADGEN_LOADGEN_H_

#include <cstddef>
#include <functional>
#include <numeric>
#include <string>

/// \brief Contains the loadgen API.
namespace mlperf {

struct QuerySampleResponse;
class QuerySampleLibrary;
class SystemUnderTest;
struct TestSettings;
struct LogSettings;

using ResponseCallback = std::function<void(QuerySampleResponse*)>;

/// \addtogroup LoadgenAPI Loadgen API
/// @{

///
/// \brief SUT calls this to notify loadgen of completed samples.
/// \details
/// * The samples may be from any combination of queries or partial queries as
///   issued by \link mlperf::SystemUnderTest::IssueQuery
///
///   SystemUnderTest::IssueQuery \endlink.
/// * The SUT is responsible for owning and allocating the reponse data. The
///   loadgen will copy the response data if needed (e.g. for accuracy mode).
///   + If no response callback is provided, the response data must remain valid
///     for the entire duration of this call.
///   + The response callback is untimed; it is called for each response in
///     responses after the loadgen records the completion time and before the
///     loadgen copies the response data. The response callback enables the
///     loadgen to simulate response data being stored in accelerator DRAM.
///     After the response callback is called, response data must reside on the
///     host so that the loadgen can copy it. Submitters must seek prior
///     approval to use this feature of loadgen (refer to
///     https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc#5-load-generator).
/// * All calls to QuerySampleComplete are thread-safe and wait-free bounded.
///   + Any number of threads can call QuerySampleComplete simultaneously.
///   + Regardless of where any other thread stalls, the current thread will
///     finish QuerySampleComplete in a bounded number of cycles.
///   + Note: If a callback is provided, the SUT must ensure that the callback
///     is also thread-safe and wait-free bounded for the above to hold.
void QuerySamplesComplete(QuerySampleResponse* responses, size_t response_count,
                          const ResponseCallback& response_cb = {});

void FirstTokenComplete(QuerySampleResponse* responses, size_t response_count,
                        const ResponseCallback& response_cb = {});

///
/// \brief Starts the test against SUT with the specified settings.
/// \details This is the C++ entry point. See mlperf::c::StartTest for the
/// C entry point.
///
void StartTest(SystemUnderTest* sut, QuerySampleLibrary* qsl,
               const TestSettings& requested_settings,
               const LogSettings& log_settings,
               const std::string audit_config_filename = "audit.config");

///
/// \brief Aborts the running test.
/// \details This function will stop issueing new samples to the SUT. StartTest
/// will return after the current inference finishes. Since StartTest is a
/// blocking function, this function can only be called in another thread.
void AbortTest();

///
/// \brief Register a thread for query issuing in Server scenario.
/// \details If a thread registers itself, the thread(s) is used to call SUT's
/// IssueQuery(). This function is blocking until the entire test is done. The
/// number of registered threads must match server_num_issue_query_threads in
/// TestSettings. This function only has effect in Server scenario.
/// This is the C++ entry point. See mlperf::c::RegisterIssueQueryThread for the
/// C entry point.
///
void RegisterIssueQueryThread();
// inline long long samples_overhead_acum;
// inline long long tokens_overhead_acum;
/// @}

}  // namespace mlperf

#endif  // MLPERF_LOADGEN_LOADGEN_H_
