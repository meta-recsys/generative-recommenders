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
/// \brief Defines PerformanceResult and PerformanceSummary.

#ifndef MLPERF_LOADGEN_RESULTS_H_
#define MLPERF_LOADGEN_RESULTS_H_

#include <string>
#include <vector>

#include "query_sample.h"
#include "test_settings_internal.h"

namespace mlperf {
namespace loadgen {

/// \brief Contains the performance results for benchmarks that have
/// token based metrics
struct TokenPerformanceResults {
  std::vector<QuerySampleLatency> first_token_latencies;
  std::vector<QuerySampleLatency> time_per_output_token_arr;
  std::vector<int64_t> tokens_per_sample;
};

/// \brief Provides performance results that are independent of scenario
/// and other context.
struct PerformanceResult {
  std::vector<QuerySampleLatency> sample_latencies;
  std::vector<QuerySampleLatency> query_latencies;
  size_t queries_issued;
  double max_latency;
  double final_query_scheduled_time;         // seconds from start.
  double final_query_issued_time;            // seconds from start.
  double final_query_all_samples_done_time;  // seconds from start.
  TokenPerformanceResults token_results;
};

/// \brief Wraps PerformanceResult with relevant context to change how
/// it's interpreted and reported.
struct PerformanceSummary {
  std::string sut_name;
  TestSettingsInternal settings;
  PerformanceResult pr;

  // Set by ProcessLatencies.
  size_t sample_count = 0;
  size_t query_count = 0;
  size_t overlatency_query_count = 0;
  QuerySampleLatency sample_latency_min = 0;
  QuerySampleLatency sample_latency_max = 0;
  QuerySampleLatency sample_latency_mean = 0;
  QuerySampleLatency query_latency_min = 0;
  QuerySampleLatency query_latency_max = 0;
  QuerySampleLatency query_latency_mean = 0;

  /// \brief The latency at a given percentile.
  struct PercentileEntry {
    const double percentile;
    QuerySampleLatency sample_latency = 0;
    QuerySampleLatency query_latency = 0;  // MultiStream only.
  };

  // Latency target percentile
  PercentileEntry target_latency_percentile{settings.target_latency_percentile};
  PercentileEntry latency_percentiles[6] = {{.50}, {.90}, {.95},
                                            {.97}, {.99}, {.999}};

  // Early stopping percentile estimates for SingleStream and MultiStream
  QuerySampleLatency early_stopping_latency_ss = 0;
  QuerySampleLatency early_stopping_latency_ms = 0;

  // Set by ProcessTokenLatencies
  size_t token_count = 0;
  size_t overlatency_first_token_count = 0;
  QuerySampleLatency first_token_latency_min = 0;
  QuerySampleLatency first_token_latency_max = 0;
  QuerySampleLatency first_token_latency_mean = 0;
  QuerySampleLatency time_per_output_token_min = 0;
  QuerySampleLatency time_per_output_token_max = 0;
  QuerySampleLatency time_per_output_token_mean = 0;

  // Latency token target percentile
  PercentileEntry token_target_latency_percentile{
      settings.target_latency_percentile};
  PercentileEntry token_latency_percentiles[6] = {{.50}, {.90}, {.95},
                                                  {.97}, {.99}, {.999}};
  PercentileEntry target_tpot_percentile{settings.target_latency_percentile};
  PercentileEntry tpot_percentiles[6] = {{.50}, {.90}, {.95},
                                         {.97}, {.99}, {.999}};

#if defined(_WIN32) || defined(WIN32) || defined(_WIN64) || defined(WIN64)
  // MSVC complains if there is no explicit constructor.
  // (target_latency_percentile above depends on construction with settings)
  PerformanceSummary(const std::string& sut_name_arg,
                     const TestSettingsInternal& settings_arg,
                     const PerformanceResult& pr_arg)
      : sut_name(sut_name_arg), settings(settings_arg), pr(pr_arg){};
#endif
  void ProcessLatencies();
  void ProcessTokenLatencies();

  bool MinDurationMet(std::string* recommendation);
  bool EarlyStopping(std::string* recommendation, int64_t queries_issued,
                     std::vector<QuerySampleLatency>* sample_latencies,
                     std::vector<QuerySampleLatency>* query_latencies,
                     std::chrono::nanoseconds target_latency);
  bool MinQueriesMet();
  bool MinSamplesMet();
  bool HasPerfConstraints();
  bool PerfConstraintsMet(std::string* recommendation);
  void LogSummary(AsyncSummary& summary);
  void LogDetail(AsyncDetail& detail);
};
}  // namespace loadgen
}  // namespace mlperf

#endif
