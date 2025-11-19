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

#include "results.h"

#include "early_stopping.h"
#include "utils.h"

namespace mlperf {
namespace loadgen {

void PerformanceSummary::ProcessLatencies() {
  if (pr.sample_latencies.empty()) {
    return;
  }

  sample_count = pr.sample_latencies.size();

  QuerySampleLatency accumulated_sample_latency = 0;
  for (auto latency : pr.sample_latencies) {
    accumulated_sample_latency += latency;
  }
  sample_latency_mean = accumulated_sample_latency / sample_count;

  std::sort(pr.sample_latencies.begin(), pr.sample_latencies.end());

  target_latency_percentile.sample_latency =
      pr.sample_latencies[sample_count * target_latency_percentile.percentile];
  sample_latency_min = pr.sample_latencies.front();
  sample_latency_max = pr.sample_latencies.back();
  for (auto& lp : latency_percentiles) {
    assert(lp.percentile >= 0.0);
    assert(lp.percentile < 1.0);
    lp.sample_latency = pr.sample_latencies[sample_count * lp.percentile];
  }

  query_count = pr.queries_issued;

  // Count the number of overlatency queries. Only for Server scenario. Since in
  // this scenario the number of samples per query is 1, sample_latencies are
  // used.
  if (settings.scenario == TestScenario::Server) {
    QuerySampleLatency max_latency = settings.target_latency.count() + 1;
    overlatency_query_count =
        pr.sample_latencies.end() -
        std::lower_bound(pr.sample_latencies.begin(), pr.sample_latencies.end(),
                         max_latency);
  }

  if (settings.use_token_latencies) {
    ProcessTokenLatencies();
  }

  // MultiStream only after this point.
  if (settings.scenario != TestScenario::MultiStream) {
    return;
  }

  // Calculate per-query stats.
  size_t query_count = pr.queries_issued;
  assert(pr.query_latencies.size() == query_count);
  std::sort(pr.query_latencies.begin(), pr.query_latencies.end());
  QuerySampleLatency accumulated_query_latency = 0;
  for (auto latency : pr.query_latencies) {
    accumulated_query_latency += latency;
  }
  query_latency_mean = accumulated_query_latency / query_count;
  query_latency_min = pr.query_latencies.front();
  query_latency_max = pr.query_latencies.back();
  target_latency_percentile.query_latency =
      pr.query_latencies[query_count * target_latency_percentile.percentile];
  for (auto& lp : latency_percentiles) {
    lp.query_latency = pr.query_latencies[query_count * lp.percentile];
  }
}

void PerformanceSummary::ProcessTokenLatencies() {
  constexpr auto nTokenInvalid = std::numeric_limits<int64_t>::min();
  token_count = 0;
  for (auto n_tokens : pr.token_results.tokens_per_sample) {
    if (n_tokens != nTokenInvalid) token_count += n_tokens;
  }
  if (pr.token_results.first_token_latencies.empty()) {
    return;
  }
  QuerySampleLatency accumulated_first_token_latency = 0;
  for (auto latency : pr.token_results.first_token_latencies) {
    accumulated_first_token_latency += latency;
  }
  first_token_latency_mean = accumulated_first_token_latency / sample_count;
  QuerySampleLatency accumulated_tpot = 0;
  for (auto latency : pr.token_results.time_per_output_token_arr) {
    accumulated_tpot += latency;
  }
  time_per_output_token_mean = accumulated_tpot / sample_count;
  std::sort(pr.token_results.first_token_latencies.begin(),
            pr.token_results.first_token_latencies.end());
  std::sort(pr.token_results.time_per_output_token_arr.begin(),
            pr.token_results.time_per_output_token_arr.end());

  token_target_latency_percentile.sample_latency =
      pr.token_results
          .first_token_latencies[sample_count *
                                 token_target_latency_percentile.percentile];
  first_token_latency_min = pr.token_results.first_token_latencies.front();
  first_token_latency_max = pr.token_results.first_token_latencies.back();
  for (auto& lp : token_latency_percentiles) {
    assert(lp.percentile >= 0.0);
    assert(lp.percentile < 1.0);
    lp.sample_latency =
        pr.token_results.first_token_latencies[sample_count * lp.percentile];
  }

  target_tpot_percentile.sample_latency =
      pr.token_results
          .time_per_output_token_arr[sample_count *
                                     target_tpot_percentile.percentile];
  time_per_output_token_min =
      pr.token_results.time_per_output_token_arr.front();
  time_per_output_token_max = pr.token_results.time_per_output_token_arr.back();
  for (auto& lp : tpot_percentiles) {
    assert(lp.percentile >= 0.0);
    assert(lp.percentile < 1.0);
    lp.sample_latency =
        pr.token_results
            .time_per_output_token_arr[sample_count * lp.percentile];
  }

  if (settings.scenario == TestScenario::Server) {
    // TODO: Maybe another target latency needs to be added?
    QuerySampleLatency max_latency = settings.target_latency.count() + 1;
    overlatency_first_token_count =
        pr.token_results.first_token_latencies.end() -
        std::lower_bound(pr.token_results.first_token_latencies.begin(),
                         pr.token_results.first_token_latencies.end(),
                         max_latency);
  }
}

bool PerformanceSummary::EarlyStopping(
    std::string* recommendation, int64_t queries_issued,
    std::vector<QuerySampleLatency>* sample_latencies,
    std::vector<QuerySampleLatency>* query_latencies,
    std::chrono::nanoseconds target_latency) {
  recommendation->clear();

  MinPassingQueriesFinder find_min_passing;
  double confidence = 0.99;
  double tolerance = 0.0;

  ProcessLatencies();
  switch (settings.scenario) {
    case TestScenario::SingleStream: {
      // TODO: Grab multistream percentile from settings, instead of hardcoding.
      double multi_stream_percentile = 0.99;
      int64_t t = 1;
      int64_t h_min = find_min_passing(1, target_latency_percentile.percentile,
                                       tolerance, confidence);
      int64_t h = h_min;
      if (queries_issued < h_min + 1) {
        *recommendation =
            " * Only processed " + std::to_string(queries_issued) +
            " queries.\n * Need to process at least " +
            std::to_string(h_min + 1) + " queries for early stopping.";
        return false;
      } else {
        for (int64_t i = 2; i < queries_issued + 1; ++i) {
          h = find_min_passing(i, target_latency_percentile.percentile,
                               tolerance, confidence);
          if (queries_issued < h + i) {
            t = i - 1;
            break;
          }
        }
      }
      QuerySampleLatency percentile_estimate =
          (*sample_latencies)[queries_issued - t];
      *recommendation =
          " * Processed at least " + std::to_string(h_min + 1) + " queries (" +
          std::to_string(queries_issued) + ").\n" + " * Would discard " +
          std::to_string(t - 1) + " highest latency queries.\n" +
          " * Early stopping " +
          DoubleToString(target_latency_percentile.percentile * 100, 1) +
          "th percentile estimate: " + std::to_string(percentile_estimate);
      early_stopping_latency_ss = percentile_estimate;

      // Early stopping estimate for 99%ile (used for infering multi-stream from
      // single-stream)
      t = 1;
      h_min =
          find_min_passing(1, multi_stream_percentile, tolerance, confidence);
      h = h_min;
      if (queries_issued < h_min + 1) {
        *recommendation +=
            "\n * Not enough queries processed for " +
            DoubleToString(multi_stream_percentile * 100, 1) +
            "th percentile\n" +
            " early stopping estimate (would need to process at\n least " +
            std::to_string(h_min + 1) + " total queries).";
      } else {
        for (int64_t i = 2; i < queries_issued + 1; ++i) {
          h = find_min_passing(i, multi_stream_percentile, tolerance,
                               confidence);
          if (queries_issued < h + i) {
            t = i - 1;
            break;
          }
        }
        percentile_estimate = (*sample_latencies)[queries_issued - t];
        *recommendation +=
            "\n * Early stopping " +
            DoubleToString(multi_stream_percentile * 100, 1) +
            "th percentile estimate: " + std::to_string(percentile_estimate);
        early_stopping_latency_ms = percentile_estimate;
      }
      break;
    }
    case TestScenario::Server: {
      int64_t t =
          std::count_if((*sample_latencies).begin(), (*sample_latencies).end(),
                        [=](auto const& latency) {
                          return latency > target_latency.count();
                        });
      int64_t h = find_min_passing(t, target_latency_percentile.percentile,
                                   tolerance, confidence);
      if (queries_issued >= h + t) {
        *recommendation = " * Run successful.";
      } else {
        *recommendation = " * Run unsuccessful.\n * Processed " +
                          std::to_string(queries_issued) + " queries.\n" +
                          " * Would need to run at least " +
                          std::to_string(h + t - queries_issued) +
                          " more queries,\n with the run being successful if "
                          "every additional\n query were under latency.";
        return false;
      }
      break;
    }
    case TestScenario::MultiStream: {
      int64_t t = 1;
      int64_t h_min = find_min_passing(1, target_latency_percentile.percentile,
                                       tolerance, confidence);
      int64_t h = h_min;
      if (queries_issued < h_min + 1) {
        *recommendation =
            " * Only processed " + std::to_string(queries_issued) +
            " queries.\n * Need to process at least " +
            std::to_string(h_min + 1) + " queries for early stopping.";
        return false;
      } else {
        for (int64_t i = 2; i < queries_issued + 1; ++i) {
          h = find_min_passing(i, target_latency_percentile.percentile,
                               tolerance, confidence);
          if (queries_issued < h + i) {
            t = i - 1;
            break;
          }
        }
      }
      QuerySampleLatency percentile_estimate =
          (*query_latencies)[queries_issued - t];
      *recommendation =
          " * Processed at least " + std::to_string(h_min + 1) + " queries (" +
          std::to_string(queries_issued) + ").\n" + " * Would discard " +
          std::to_string(t - 1) + " highest latency queries.\n" +
          " * Early stopping " +
          DoubleToString(target_latency_percentile.percentile * 100, 1) +
          "th percentile estimate: " + std::to_string(percentile_estimate);
      early_stopping_latency_ms = percentile_estimate;
      break;
    }
    case TestScenario::Offline:
      break;
  }
  return true;
}

bool PerformanceSummary::MinDurationMet(std::string* recommendation) {
  recommendation->clear();
  const double min_duration = DurationToSeconds(settings.min_duration);
  bool min_duration_met = false;
  switch (settings.scenario) {
    case TestScenario::Offline:
      min_duration_met = pr.max_latency >= min_duration;
      break;
    case TestScenario::Server:
      min_duration_met = pr.final_query_scheduled_time >= min_duration;
      break;
    case TestScenario::SingleStream:
    case TestScenario::MultiStream:
      min_duration_met = pr.final_query_issued_time >= min_duration;
      break;
  }
  if (min_duration_met) {
    return true;
  }

  switch (settings.scenario) {
    case TestScenario::SingleStream:
    case TestScenario::MultiStream:
      *recommendation =
          "Decrease the expected latency so the loadgen pre-generates more "
          "queries.";
      break;
    case TestScenario::Server:
      *recommendation =
          "Increase the target QPS so the loadgen pre-generates more queries.";
      break;
    case TestScenario::Offline:
      *recommendation =
          "Increase expected QPS so the loadgen pre-generates a larger "
          "(coalesced) query.";
      break;
  }
  return false;
}

bool PerformanceSummary::MinQueriesMet() {
  return pr.queries_issued >= settings.min_query_count;
}

bool PerformanceSummary::MinSamplesMet() {
  return sample_count >= settings.min_sample_count;
}

bool PerformanceSummary::HasPerfConstraints() {
  return settings.scenario == TestScenario::Server;
}

bool PerformanceSummary::PerfConstraintsMet(std::string* recommendation) {
  recommendation->clear();
  bool perf_constraints_met = true;
  switch (settings.scenario) {
    case TestScenario::SingleStream:
    case TestScenario::MultiStream:
      break;
    case TestScenario::Server:
      ProcessLatencies();
      if (!settings.use_token_latencies) {
        if (target_latency_percentile.sample_latency >
            settings.target_latency.count()) {
          *recommendation = "Reduce target QPS to improve latency.";
          perf_constraints_met = false;
        }
      } else {
        if (token_target_latency_percentile.sample_latency >
            settings.server_ttft_latency) {
          *recommendation =
              "TTFT constrain not met: Reduce target QPS to improve latency.";
          perf_constraints_met = false;
        }

        if (target_tpot_percentile.sample_latency >
            settings.server_tpot_latency) {
          if (recommendation->empty()) {
            *recommendation =
                "TPOT constrain not met: Reduce target QPS to improve latency.";
          } else {
            recommendation->append(
                "\n * TPOT constrain not met: Reduce target QPS to improve "
                "latency.");
          }
          perf_constraints_met = false;
        }
      }
      break;
    case TestScenario::Offline:
      break;
  }
  return perf_constraints_met;
}

void PerformanceSummary::LogSummary(AsyncSummary& summary) {
  ProcessLatencies();

  summary(
      "================================================\n"
      "MLPerf Results Summary\n"
      "================================================");
  summary("SUT name : ", sut_name);
  summary("Scenario : ", ToString(settings.scenario));
  summary("Mode     : ", ToString(settings.mode));

  switch (settings.scenario) {
    case TestScenario::SingleStream: {
      summary(DoubleToString(target_latency_percentile.percentile * 100, 1) +
                  "th percentile latency (ns) : ",
              target_latency_percentile.sample_latency);
      break;
    }
    case TestScenario::MultiStream: {
      summary(DoubleToString(target_latency_percentile.percentile * 100, 1) +
                  "th percentile latency (ns) : ",
              target_latency_percentile.query_latency);
      break;
    }
    case TestScenario::Server: {
      // Subtract 1 from sample count since the start of the final sample
      // represents the open end of the time range: i.e. [begin, end).
      // This makes sense since:
      // a) QPS doesn't apply if there's only one sample; it's pure latency.
      // b) If you have precisely 1k QPS, there will be a sample exactly on
      //    the 1 second time point; but that would be the 1001th sample in
      //    the stream. Given the first 1001 queries, the QPS is
      //    1000 queries / 1 second.
      // TODO: make a more permanent solution
      double qps_as_completed =
          (sample_count - 1) / pr.final_query_all_samples_done_time;
      summary("Completed samples per second    : ",
              DoubleToString(qps_as_completed));
      break;
    }
    case TestScenario::Offline: {
      double samples_per_second = sample_count / pr.max_latency;
      summary("Samples per second: ", samples_per_second);
      break;
    }
  }

  if (settings.use_token_latencies) {
    switch (settings.scenario) {
      case TestScenario::SingleStream: {
        summary(DoubleToString(token_target_latency_percentile.percentile * 100,
                               1) +
                    "th first token percentile latency (ns) : ",
                token_target_latency_percentile.sample_latency);
        break;
      }
      case TestScenario::MultiStream: {
        summary(DoubleToString(token_target_latency_percentile.percentile * 100,
                               1) +
                    "th first token percentile latency (ns) : ",
                token_target_latency_percentile.sample_latency);
        break;
      }
      case TestScenario::Offline: {
        double tokens_per_second = token_count / pr.max_latency;
        summary("Tokens per second: ", tokens_per_second);
        break;
      }
      case TestScenario::Server:
        double tps_as_completed =
            token_count / pr.final_query_all_samples_done_time;
        summary("Completed tokens per second: ",
                DoubleToString(tps_as_completed));
        break;
    }
  }

  if (settings.infer_token_latencies) {
    switch (settings.scenario) {
      case TestScenario::SingleStream: {
        break;
      }
      case TestScenario::MultiStream: {
        break;
      }
      case TestScenario::Offline: {
        double tokens_per_second = settings.token_latency_scaling_factor *
                                   sample_count / pr.max_latency;
        summary("Tokens per second (inferred): ", tokens_per_second);
        break;
      }
      case TestScenario::Server:
        double tps_as_completed = settings.token_latency_scaling_factor *
                                  (sample_count - 1) /
                                  pr.final_query_all_samples_done_time;
        summary("Completed tokens per second (inferred): ",
                DoubleToString(tps_as_completed));
        break;
    }
  }

  std::string min_duration_recommendation;
  std::string perf_constraints_recommendation;
  std::string early_stopping_recommendation;
  std::string early_stopping_ttft_recommendation;
  std::string early_stopping_tpot_recommendation;

  bool min_duration_met = MinDurationMet(&min_duration_recommendation);
  bool min_queries_met = MinQueriesMet() && MinSamplesMet();
  bool early_stopping_met = true;
  if (!settings.use_token_latencies) {
    early_stopping_met = EarlyStopping(
        &early_stopping_recommendation, pr.queries_issued, &pr.sample_latencies,
        &pr.query_latencies, settings.target_latency);
  } else {
    early_stopping_met =
        EarlyStopping(&early_stopping_tpot_recommendation, pr.queries_issued,
                      &pr.token_results.time_per_output_token_arr,
                      &pr.query_latencies,
                      std::chrono::nanoseconds(settings.server_tpot_latency)) &&
        EarlyStopping(&early_stopping_ttft_recommendation, pr.queries_issued,
                      &pr.token_results.first_token_latencies,
                      &pr.query_latencies,
                      std::chrono::nanoseconds(settings.server_ttft_latency));
  }
  bool perf_constraints_met =
      PerfConstraintsMet(&perf_constraints_recommendation);
  bool all_constraints_met = min_duration_met && min_queries_met &&
                             perf_constraints_met && early_stopping_met;
  summary("Result is : ", all_constraints_met ? "VALID" : "INVALID");
  if (HasPerfConstraints()) {
    summary("  Performance constraints satisfied : ",
            perf_constraints_met ? "Yes" : "NO");
  }
  summary("  Min duration satisfied : ", min_duration_met ? "Yes" : "NO");
  summary("  Min queries satisfied : ", min_queries_met ? "Yes" : "NO");
  summary("  Early stopping satisfied: ", early_stopping_met ? "Yes" : "NO");

  if (!all_constraints_met) {
    summary("Recommendations:");
    if (!perf_constraints_met) {
      summary(" * " + perf_constraints_recommendation);
    }
    if (!min_duration_met) {
      summary(" * " + min_duration_recommendation);
    }
    if (!min_queries_met) {
      summary(
          " * The test exited early, before enough queries were issued.\n"
          "   See the detailed log for why this may have occurred.");
    }
  }
  // Early stopping results
  if (settings.scenario == TestScenario::SingleStream ||
      settings.scenario == TestScenario::Server ||
      settings.scenario == TestScenario::MultiStream) {
    if (!settings.use_token_latencies) {
      summary("Early Stopping Result:");
      summary(early_stopping_recommendation);
    } else {
      summary("TTFT Early Stopping Result:");
      summary(early_stopping_ttft_recommendation);
      summary("TPOT Early Stopping Result:");
      summary(early_stopping_tpot_recommendation);
    }
  }

  summary(
      "\n"
      "================================================\n"
      "Additional Stats\n"
      "================================================");

  if (settings.scenario == TestScenario::SingleStream) {
    double qps_w_lg = (sample_count - 1) / pr.final_query_issued_time;
    double qps_wo_lg = 1 / QuerySampleLatencyToSeconds(sample_latency_mean);
    summary("QPS w/ loadgen overhead         : " + DoubleToString(qps_w_lg));
    summary("QPS w/o loadgen overhead        : " + DoubleToString(qps_wo_lg));
    summary("");
  } else if (settings.scenario == TestScenario::Server) {
    // Scheduled samples per second as an additional stat
    double qps_as_scheduled =
        (sample_count - 1) / pr.final_query_scheduled_time;
    summary("Scheduled samples per second : ",
            DoubleToString(qps_as_scheduled));
  } else if (settings.scenario == TestScenario::MultiStream) {
    summary("Per-query latency:  ");
    summary("Min latency (ns)                : ", query_latency_min);
    summary("Max latency (ns)                : ", query_latency_max);
    summary("Mean latency (ns)               : ", query_latency_mean);
    for (auto& lp : latency_percentiles) {
      summary(
          DoubleToString(lp.percentile * 100) + " percentile latency (ns)   : ",
          lp.query_latency);
    }
  }

  if (settings.scenario != TestScenario::MultiStream) {
    summary("Min latency (ns)                : ", sample_latency_min);
    summary("Max latency (ns)                : ", sample_latency_max);
    summary("Mean latency (ns)               : ", sample_latency_mean);
    for (auto& lp : latency_percentiles) {
      summary(
          DoubleToString(lp.percentile * 100) + " percentile latency (ns)   : ",
          lp.sample_latency);
    }
  }
  if (settings.use_token_latencies) {
    summary("");
    if (settings.scenario == TestScenario::SingleStream) {
      double tps_w_lg = token_count / pr.final_query_issued_time;
      double tps_wo_lg =
          ((double)token_count) /
          (QuerySampleLatencyToSeconds(sample_latency_mean) * sample_count);
      summary("TPS w/ loadgen overhead         : " + DoubleToString(tps_w_lg));
      summary("TPS w/o loadgen overhead        : " + DoubleToString(tps_wo_lg));

    } else if (settings.scenario == TestScenario::Server) {
      double tps_as_completed =
          token_count / pr.final_query_all_samples_done_time;
      summary("Completed tokens per second                 : ",
              DoubleToString(tps_as_completed));
    }

    if (settings.scenario != TestScenario::Offline) {
      summary("Min First Token latency (ns)                : ",
              first_token_latency_min);
      summary("Max First Token latency (ns)                : ",
              first_token_latency_max);
      summary("Mean First Token latency (ns)               : ",
              first_token_latency_mean);
      for (auto& lp : token_latency_percentiles) {
        summary(DoubleToString(lp.percentile * 100) +
                    " percentile first token latency (ns)   : ",
                lp.sample_latency);
      }
      summary("");
      summary("Min Time per Output Token (ns)                : ",
              time_per_output_token_min);
      summary("Max Time per Output Token (ns)                : ",
              time_per_output_token_max);
      summary("Mean Time per Output Token (ns)               : ",
              time_per_output_token_mean);
      for (auto& lp : tpot_percentiles) {
        summary(DoubleToString(lp.percentile * 100) +
                    " percentile time to output token (ns)   : ",
                lp.sample_latency);
      }
    }
  }

  summary(
      "\n"
      "================================================\n"
      "Test Parameters Used\n"
      "================================================");
  settings.LogSummary(summary);
}

void PerformanceSummary::LogDetail(AsyncDetail& detail) {
#if USE_NEW_LOGGING_FORMAT
  ProcessLatencies();

  // General validity checking
  std::string min_duration_recommendation;
  std::string perf_constraints_recommendation;
  std::string early_stopping_recommendation;
  std::string early_stopping_ttft_recommendation;
  std::string early_stopping_tpot_recommendation;
  bool min_duration_met = MinDurationMet(&min_duration_recommendation);
  bool min_queries_met = MinQueriesMet() && MinSamplesMet();
  bool perf_constraints_met =
      PerfConstraintsMet(&perf_constraints_recommendation);
  bool early_stopping_met = true;
  if (!settings.use_token_latencies) {
    early_stopping_met = EarlyStopping(
        &early_stopping_recommendation, pr.queries_issued, &pr.sample_latencies,
        &pr.query_latencies, settings.target_latency);
  } else {
    early_stopping_met =
        EarlyStopping(&early_stopping_tpot_recommendation, pr.queries_issued,
                      &pr.token_results.time_per_output_token_arr,
                      &pr.query_latencies,
                      std::chrono::nanoseconds(settings.server_tpot_latency)) &&
        EarlyStopping(&early_stopping_ttft_recommendation, pr.queries_issued,
                      &pr.token_results.first_token_latencies,
                      &pr.query_latencies,
                      std::chrono::nanoseconds(settings.server_ttft_latency));
  }
  bool all_constraints_met = min_duration_met && min_queries_met &&
                             perf_constraints_met && early_stopping_met;

  MLPERF_LOG(detail, "result_validity",
             all_constraints_met ? "VALID" : "INVALID");
  if (HasPerfConstraints()) {
    MLPERF_LOG(detail, "result_perf_constraints_met", perf_constraints_met);
  }
  MLPERF_LOG(detail, "result_min_duration_met", min_duration_met);
  MLPERF_LOG(detail, "result_min_queries_met", min_queries_met);
  MLPERF_LOG(detail, "early_stopping_met", early_stopping_met);
  if (!all_constraints_met) {
    std::string recommendation;
    if (!perf_constraints_met) {
      recommendation += perf_constraints_recommendation + " ";
    }
    if (!min_duration_met) {
      recommendation += min_duration_recommendation + " ";
    }
    if (!min_queries_met) {
      recommendation +=
          "The test exited early, before enough queries were issued.";
    }
    std::replace(recommendation.begin(), recommendation.end(), '\n', ' ');
    MLPERF_LOG(detail, "result_invalid_reason", recommendation);
  }
  std::replace(early_stopping_recommendation.begin(),
               early_stopping_recommendation.end(), '\n', ' ');
  if (!settings.use_token_latencies) {
    MLPERF_LOG(detail, "early_stopping_result", early_stopping_recommendation);
  } else {
    std::replace(early_stopping_ttft_recommendation.begin(),
                 early_stopping_ttft_recommendation.end(), '\n', ' ');
    std::replace(early_stopping_tpot_recommendation.begin(),
                 early_stopping_tpot_recommendation.end(), '\n', ' ');
    MLPERF_LOG(detail, "early_stopping_ttft_result",
               early_stopping_ttft_recommendation);
    MLPERF_LOG(detail, "early_stopping_tpot_result",
               early_stopping_tpot_recommendation);
  }
  // Report number of queries
  MLPERF_LOG(detail, "result_query_count", query_count);
  if (settings.scenario == TestScenario::Server) {
    MLPERF_LOG(detail, "result_overlatency_query_count",
               overlatency_query_count);
  }

  auto reportPerQueryLatencies = [&]() {
    MLPERF_LOG(detail, "result_min_query_latency_ns", query_latency_min);
    MLPERF_LOG(detail, "result_max_query_latency_ns", query_latency_max);
    MLPERF_LOG(detail, "result_mean_query_latency_ns", query_latency_mean);
    for (auto& lp : latency_percentiles) {
      std::string percentile = DoubleToString(lp.percentile * 100);
      MLPERF_LOG(detail,
                 "result_" + percentile + "_percentile_per_query_latency_ns",
                 lp.query_latency);
    }
  };

  // Per-scenario performance results.
  switch (settings.scenario) {
    case TestScenario::SingleStream: {
      double qps_w_lg = (sample_count - 1) / pr.final_query_issued_time;
      double qps_wo_lg = 1 / QuerySampleLatencyToSeconds(sample_latency_mean);
      MLPERF_LOG(detail, "result_qps_with_loadgen_overhead", qps_w_lg);
      MLPERF_LOG(detail, "result_qps_without_loadgen_overhead", qps_wo_lg);
      MLPERF_LOG(detail, "early_stopping_latency_ss",
                 early_stopping_latency_ss);
      MLPERF_LOG(detail, "early_stopping_latency_ms",
                 early_stopping_latency_ms);
      break;
    }
    case TestScenario::MultiStream: {
      reportPerQueryLatencies();
      MLPERF_LOG(detail, "early_stopping_latency_ms",
                 early_stopping_latency_ms);
      break;
    }
    case TestScenario::Server: {
      // Subtract 1 from sample count since the start of the final sample
      // represents the open end of the time range: i.e. [begin, end).
      // This makes sense since:
      // a) QPS doesn't apply if there's only one sample; it's pure latency.
      // b) If you have precisely 1k QPS, there will be a sample exactly on
      //    the 1 second time point; but that would be the 1001th sample in
      //    the stream. Given the first 1001 queries, the QPS is
      //    1000 queries / 1 second.
      double qps_as_scheduled =
          (sample_count - 1) / pr.final_query_scheduled_time;
      MLPERF_LOG(detail, "result_scheduled_samples_per_sec", qps_as_scheduled);
      double qps_as_completed =
          (sample_count - 1) / pr.final_query_all_samples_done_time;
      MLPERF_LOG(detail, "result_completed_samples_per_sec", qps_as_completed);
      break;
    }
    case TestScenario::Offline: {
      double samples_per_second = sample_count / pr.max_latency;
      MLPERF_LOG(detail, "result_samples_per_second", samples_per_second);
      break;
    }
  }

  // Detailed latencies
  MLPERF_LOG(detail, "result_min_latency_ns", sample_latency_min);
  MLPERF_LOG(detail, "result_max_latency_ns", sample_latency_max);
  MLPERF_LOG(detail, "result_mean_latency_ns", sample_latency_mean);
  for (auto& lp : latency_percentiles) {
    MLPERF_LOG(detail,
               "result_" + DoubleToString(lp.percentile * 100) +
                   "_percentile_latency_ns",
               lp.sample_latency);
  }
  // Detailed first token latencies
  if (settings.use_token_latencies) {
    if (settings.scenario != TestScenario::Offline) {
      MLPERF_LOG(detail, "result_first_token_min_latency_ns",
                 first_token_latency_min);
      MLPERF_LOG(detail, "result_first_token_max_latency_ns",
                 first_token_latency_max);
      MLPERF_LOG(detail, "result_first_token_mean_latency_ns",
                 first_token_latency_mean);
      for (auto& lp : token_latency_percentiles) {
        MLPERF_LOG(detail,
                   "result_first_token_" + DoubleToString(lp.percentile * 100) +
                       "_percentile_latency_ns",
                   lp.sample_latency);
      }
      double tps_w_lg = ((double)token_count) / pr.final_query_issued_time;
      double tps_wo_lg =
          ((double)token_count) / (sample_latency_mean * sample_count);
      MLPERF_LOG(detail, "result_token_throughput_with_loadgen_overhead",
                 tps_w_lg);
      MLPERF_LOG(detail, "result_token_throughput", tps_wo_lg);
      for (auto& lp : tpot_percentiles) {
        MLPERF_LOG(detail,
                   "result_time_per_output_token_" +
                       DoubleToString(lp.percentile * 100) + "_percentile_ns",
                   lp.sample_latency);
      }
      MLPERF_LOG(detail, "result_time_to_output_token_min",
                 time_per_output_token_min);
      MLPERF_LOG(detail, "result_time_to_output_token_max",
                 time_per_output_token_max);
      MLPERF_LOG(detail, "result_time_to_output_token_mean",
                 time_per_output_token_mean);
      double tps_as_completed =
          token_count / pr.final_query_all_samples_done_time;
      MLPERF_LOG(detail, "result_completed_tokens_per_second",
                 tps_as_completed);
    } else {
      double tokens_per_second = token_count / pr.max_latency;
      MLPERF_LOG(detail, "result_tokens_per_second", tokens_per_second);
    }
  }

  if (settings.infer_token_latencies) {
    switch (settings.scenario) {
      case TestScenario::Server: {
        double completed_tokens_per_second =
            (sample_count - 1) * settings.token_latency_scaling_factor /
            pr.final_query_all_samples_done_time;
        MLPERF_LOG(detail, "result_inferred_completed_tokens_per_second",
                   completed_tokens_per_second);
        break;
      }
      case TestScenario::Offline: {
        double tokens_per_second = sample_count *
                                   settings.token_latency_scaling_factor /
                                   pr.max_latency;
        MLPERF_LOG(detail, "result_inferred_tokens_per_second",
                   tokens_per_second);
        break;
      }
      case TestScenario::SingleStream: {
        break;
      }
      case TestScenario::MultiStream: {
        break;
      }
    }
  }
  MLPERF_LOG(detail, "num_errors", detail.async_log().GetErrorCount());
#endif
}
}  // namespace loadgen
}  // namespace mlperf
