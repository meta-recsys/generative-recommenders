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

#include "loadgen.h"

#include <stdint.h>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstring>
#include <ctime>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <queue>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "early_stopping.h"
#include "issue_query_controller.h"
#include "logging.h"
#include "query_sample.h"
#include "query_sample_library.h"
#include "results.h"
#include "system_under_test.h"
#include "test_settings.h"
#include "test_settings_internal.h"
#include "utils.h"
#include "version.h"

namespace mlperf {

/// \brief Loadgen implementation details.
namespace loadgen {

/// \brief A random set of samples in the QSL that should fit in RAM when
/// loaded together.
struct LoadableSampleSet {
  std::vector<QuerySampleIndex> set;
  const size_t sample_distribution_end;  // Excludes padding in MultiStream.
};

/// \brief Generates nanoseconds from a start time to multiple end times.
/// TODO: This isn't very useful anymore. Remove it.
struct DurationGeneratorNs {
  const PerfClock::time_point start;
  int64_t delta(PerfClock::time_point end) const {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
        .count();
  }
};

/// \brief ResponseDelegate implementation templated by scenario and mode.
template <TestScenario scenario, TestMode mode>
struct ResponseDelegateDetailed : public ResponseDelegate {
  double accuracy_log_offset = 0.0f;
  double accuracy_log_prob = 0.0f;

  void SampleComplete(SampleMetadata* sample, QuerySampleResponse* response,
                      PerfClock::time_point complete_begin_time,
                      const ResponseCallback& response_cb) override {
    // Using a raw pointer here should help us hit the std::function
    // small buffer optimization code path when we aren't copying data.
    // For some reason, using std::unique_ptr<std::vector> wasn't moving
    // into the lambda; even with C++14.
    std::vector<uint8_t>* sample_data_copy = nullptr;
    double accuracy_log_val =
        sample->accuracy_log_val + accuracy_log_offset < 1.0
            ? sample->accuracy_log_val + accuracy_log_offset
            : sample->accuracy_log_val + accuracy_log_offset - 1.0;
    if (mode == TestMode::AccuracyOnly ||
        accuracy_log_val <= accuracy_log_prob) {
      // if a response_cb callback is provided, data only needs to reside on the
      // host *after* calling it note that the callback is blocking and will
      // likely involve a memcpy from accelerator to host
      if (response_cb) {
        response_cb(response);
      }
      // TODO: Verify accuracy with the data copied here.
      uint8_t* src_begin = reinterpret_cast<uint8_t*>(response->data);
      uint8_t* src_end = src_begin + response->size;
      sample_data_copy = new std::vector<uint8_t>(src_begin, src_end);
    }
    int64_t n_tokens = response->n_tokens;
    Log([sample, complete_begin_time, sample_data_copy,
         n_tokens](AsyncLog& log) {
      QueryMetadata* query = sample->query_metadata;
      DurationGeneratorNs sched{query->scheduled_time};
      if (scenario == TestScenario::Server) {
        // Trace the server scenario as a stacked graph via counter events.
        DurationGeneratorNs issued{query->issued_start_time};
        log.TraceCounterEvent("Latency", query->scheduled_time, "issue_delay",
                              sched.delta(query->issued_start_time),
                              "issue_to_done",
                              issued.delta(complete_begin_time));
      }

      // While visualizing overlapping samples in offline mode is not
      // practical, sample completion is still recorded for auditing purposes.
      log.TraceSample("Sample", sample->sequence_id, query->scheduled_time,
                      complete_begin_time, "sample_seq", sample->sequence_id,
                      "query_seq", query->sequence_id, "sample_idx",
                      sample->sample_index, "issue_start_ns",
                      sched.delta(query->issued_start_time), "complete_ns",
                      sched.delta(complete_begin_time));

      if (sample_data_copy) {
        log.LogAccuracy(sample->sequence_id, sample->sample_index,
                        LogBinaryAsHexString{sample_data_copy}, n_tokens);
        delete sample_data_copy;
      }

      // Record the latency at the end, since it will unblock the issuing
      // thread and potentially destroy the metadata being used above.
      QuerySampleLatency latency = sched.delta(complete_begin_time);
      log.RecordSampleCompletion(sample->sequence_id, complete_begin_time,
                                 latency, n_tokens);
    });
  }

  void TokenComplete(SampleMetadata* sample, QuerySampleResponse* response,
                     PerfClock::time_point complete_begin_time,
                     const ResponseCallback& response_cb) override {
    // Using a raw pointer here should help us hit the std::function
    // small buffer optimization code path when we aren't copying data.
    // For some reason, using std::unique_ptr<std::vector> wasn't moving
    // into the lambda; even with C++14.
    std::vector<uint8_t>* token_data_copy = nullptr;
    double accuracy_log_val =
        sample->accuracy_log_val + accuracy_log_offset < 1.0
            ? sample->accuracy_log_val + accuracy_log_offset
            : sample->accuracy_log_val + accuracy_log_offset - 1.0;
    if (mode == TestMode::AccuracyOnly ||
        accuracy_log_val <= accuracy_log_prob) {
      uint8_t* src_begin = reinterpret_cast<uint8_t*>(response->data);
      uint8_t* src_end = src_begin + response->size;
      token_data_copy = new std::vector<uint8_t>(src_begin, src_end);
    }
    Log([sample, complete_begin_time, token_data_copy](AsyncLog& log) {
      QueryMetadata* query = sample->query_metadata;
      DurationGeneratorNs sched{query->scheduled_time};
      if (scenario == TestScenario::Server) {
        DurationGeneratorNs issued{query->issued_start_time};
        log.TraceCounterEvent(
            "Token_Latency", query->scheduled_time, "issue_delay",
            sched.delta(query->issued_start_time), "issue_to_done",
            issued.delta(complete_begin_time));
      } else {
        log.TraceSample("Token", sample->sequence_id, query->scheduled_time,
                        complete_begin_time, "sample_seq", sample->sequence_id,
                        "query_seq", query->sequence_id, "sample_idx",
                        sample->sample_index, "issue_start_ns",
                        sched.delta(query->issued_start_time), "complete_ns",
                        sched.delta(complete_begin_time));
      }
      if (token_data_copy) {
        log.CacheToken(sample->sequence_id,
                       LogBinaryAsHexString{token_data_copy});
      }
      QuerySampleLatency latency = sched.delta(complete_begin_time);
      log.RecordTokenCompletion(sample->sequence_id, complete_begin_time,
                                latency);
    });
  }

  void QueryComplete() override {
    // We only need to track outstanding queries in the server scenario to
    // detect when the SUT has fallen too far behind.
    if (scenario == TestScenario::Server) {
      queries_completed.fetch_add(1, std::memory_order_relaxed);
    }
  }
};

/// \brief Selects the query timestamps for all scenarios except Server.
template <TestScenario scenario>
auto ScheduleDistribution(double qps) {
  return [period = std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::duration<double>(1.0 / qps))](auto& /*gen*/) {
    return period;
  };
}

/// \brief Selects the query timestamps for the Server scenario.
template <>
auto ScheduleDistribution<TestScenario::Server>(double qps) {
  // Poisson arrival process corresponds to exponentially distributed
  // interarrival times.
  return [dist = std::exponential_distribution<>(qps)](auto& gen) mutable {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::duration<double>(dist(gen)));
  };
}

/// \brief Selects samples for the accuracy mode.
template <TestMode mode>
auto SampleDistribution(size_t sample_count, size_t stride, std::mt19937* rng) {
  std::vector<size_t> indices;
  for (size_t i = 0; i < sample_count; i += stride) {
    indices.push_back(i);
  }
  std::shuffle(indices.begin(), indices.end(), *rng);
  return [indices = std::move(indices), i = size_t(0)](auto& /*gen*/) mutable {
    return indices.at(i++);
  };
}

/// \brief Selects samples for the performance mode.
template <>
auto SampleDistribution<TestMode::PerformanceOnly>(size_t sample_count,
                                                   size_t /*stride*/,
                                                   std::mt19937* /*rng*/) {
  return [dist = std::uniform_int_distribution<>(0, sample_count - 1)](
             auto& gen) mutable { return dist(gen); };
}

/// \brief Sample across the dataset, and ensure coverage of each of the
/// samples.
// Useful for non-uniform dataset (e.g. Llama2, GPTJ, 3d-unet)
auto SampleDistributionEqualIssue(size_t sample_count, size_t set_size,
                                  std::mt19937* rng) {
  std::vector<size_t> indices;
  std::vector<size_t> shuffle_indices(set_size);
  std::iota(shuffle_indices.begin(), shuffle_indices.end(), 0);
  for (size_t j = 0; j < sample_count; j += set_size) {
    std::shuffle(shuffle_indices.begin(), shuffle_indices.end(), *rng);
    indices.insert(indices.end(), shuffle_indices.begin(),
                   shuffle_indices.end());
  }
  return [indices = std::move(indices), i = size_t(0)](auto& /*gen*/) mutable {
    return indices.at((i++) % indices.size());
  };
}

/// \brief Generates queries for the requested settings, templated by
/// scenario and mode.
/// \todo Make GenerateQueries faster.
/// QueryMetadata is expensive to move; either reserve queries in advance
/// so the queries vector doesn't need to grow. And/or parent samples to their
/// queries only after all queries have been generated.
/// \todo For the server scenario only, scale the query timeline at the end so
/// the QPS as scheduled is equal to the QPS as requested.
template <TestScenario scenario, TestMode mode>
std::vector<QueryMetadata> GenerateQueries(
    const TestSettingsInternal& settings,
    const LoadableSampleSet& loaded_sample_set, SequenceGen* sequence_gen,
    ResponseDelegate* response_delegate) {
  auto tracer =
      MakeScopedTracer([](AsyncTrace& trace) { trace("GenerateQueries"); });

  auto& loaded_samples = loaded_sample_set.set;

  // Generate 2x more samples than we think we'll need given the expected
  // QPS in case the SUT is faster than expected.
  // We should exit before issuing all queries.
  // Does not apply to the server scenario since the duration only
  // depends on the ideal scheduled time, not the actual issue time.
  const int duration_multiplier = scenario == TestScenario::Server ? 1 : 2;
  std::chrono::microseconds gen_duration =
      duration_multiplier * settings.target_duration;
  size_t min_queries = settings.min_query_count;

  size_t samples_per_query = settings.samples_per_query;
  if (mode == TestMode::AccuracyOnly && scenario == TestScenario::Offline) {
    samples_per_query = loaded_sample_set.sample_distribution_end;
  }

  // We should not exit early in accuracy mode.
  if (mode == TestMode::AccuracyOnly || settings.performance_issue_unique) {
    gen_duration = std::chrono::microseconds(0);
    // Integer truncation here is intentional.
    // For MultiStream, loaded samples is properly padded.
    // For Offline, we create a 'remainder' query at the end of this function.
    min_queries = loaded_samples.size() / samples_per_query;
  }

  std::vector<QueryMetadata> queries;

  // Using the std::mt19937 pseudo-random number generator ensures a modicum of
  // cross platform reproducibility for trace generation.
  std::mt19937 sample_rng(settings.sample_index_rng_seed);
  std::mt19937 schedule_rng(settings.schedule_rng_seed);

  constexpr bool kIsMultiStream = scenario == TestScenario::MultiStream;
  const size_t sample_stride = kIsMultiStream ? samples_per_query : 1;

  auto sample_distribution = SampleDistribution<mode>(
      loaded_sample_set.sample_distribution_end, sample_stride, &sample_rng);
  // Use the unique sample distribution same as in AccuracyMode to
  // to choose samples when either flag performance_issue_unique
  // or performance_issue_same is set.
  auto sample_distribution_unique = SampleDistribution<TestMode::AccuracyOnly>(
      loaded_sample_set.sample_distribution_end, sample_stride, &sample_rng);

  auto sample_distribution_equal_issue = SampleDistributionEqualIssue(
      min_queries, loaded_samples.size(), &sample_rng);

  auto schedule_distribution =
      ScheduleDistribution<scenario>(settings.target_qps);

  // When sample_concatenate_permutation is turned on, pad to a multiple of the
  // complete dataset to ensure fairness.
  auto enable_equal_issue = settings.sample_concatenate_permutation;
  if (mode != TestMode::AccuracyOnly && enable_equal_issue) {
    if (scenario == TestScenario::Offline &&
        samples_per_query % loaded_samples.size() != 0) {
      // In offline mode, we pad samples_per_query
      size_t pad_size =
          (loaded_samples.size() - samples_per_query % loaded_samples.size());
      samples_per_query += pad_size;
    } else if ((scenario != TestScenario::Offline) &&
               (min_queries % loaded_samples.size() != 0)) {
      // In Server, SingleStream, MultiStream mode, the min_queries should be
      // padded
      size_t pad_size =
          (loaded_samples.size() - min_queries % loaded_samples.size());
      min_queries += pad_size;
    }
  }

  std::vector<QuerySampleIndex> samples(samples_per_query);
  std::chrono::nanoseconds timestamp(0);
  std::chrono::nanoseconds prev_timestamp(0);
  // Choose a single sample to repeat when in performance_issue_same mode
  QuerySampleIndex same_sample = settings.performance_issue_same_index;

  while (prev_timestamp < gen_duration || queries.size() < min_queries) {
    if (kIsMultiStream) {
      QuerySampleIndex sample_i = settings.performance_issue_unique
                                      ? sample_distribution_unique(sample_rng)
                                  : settings.performance_issue_same
                                      ? same_sample
                                      : sample_distribution(sample_rng);
      for (auto& s : samples) {
        // Select contiguous samples in the MultiStream scenario.
        // This will not overflow, since GenerateLoadableSets adds padding at
        // the end of the loadable sets in the MultiStream scenario.
        // The padding allows the starting samples to be the same for each
        // query with respect to samples_per_query.
        s = loaded_samples[sample_i++];
      }
    } else if (scenario == TestScenario::Offline) {
      // For the Offline + Performance scenario, we also want to support
      // contiguous samples. In this scenario the query can be much larger than
      // what fits into memory. We simply repeat loaded_samples N times, plus a
      // remainder to ensure we fill up samples. Note that this eliminates
      // randomization.
      size_t num_loaded_samples = loaded_samples.size();
      size_t num_full_repeats = samples_per_query / num_loaded_samples;
      uint64_t remainder = samples_per_query % (num_loaded_samples);
      if (settings.performance_issue_same) {
        std::fill(samples.begin(), samples.begin() + samples_per_query,
                  loaded_samples[same_sample]);
      } else {
        for (size_t i = 0; i < num_full_repeats; ++i) {
          std::copy(loaded_samples.begin(), loaded_samples.end(),
                    samples.begin() + i * num_loaded_samples);

          if (settings.sample_concatenate_permutation) {
            std::shuffle(samples.begin() + i * num_loaded_samples,
                         samples.begin() + (i + 1) * num_loaded_samples,
                         sample_rng);
          }
        }

        std::copy(loaded_samples.begin(), loaded_samples.begin() + remainder,
                  samples.begin() + num_full_repeats * num_loaded_samples);

        if (settings.sample_concatenate_permutation) {
          assert(remainder == 0);
        }
      }
    } else {
      for (auto& s : samples) {
        s = loaded_samples[settings.performance_issue_unique
                               ? sample_distribution_unique(sample_rng)
                           : settings.performance_issue_same ? same_sample
                           : enable_equal_issue
                               ? sample_distribution_equal_issue(sample_rng)
                               : sample_distribution(sample_rng)];
      }
    }
    queries.emplace_back(samples, timestamp, response_delegate, sequence_gen);
    prev_timestamp = timestamp;
    timestamp += schedule_distribution(schedule_rng);
    // In equal_issue mode, the min_queries will be bumped up by a multiple of
    // the dataset size if the test time has not met the threshold.
    if (enable_equal_issue && (queries.size() >= min_queries) &&
        (prev_timestamp < gen_duration) &&
        (scenario != TestScenario::Offline)) {
      min_queries += loaded_samples.size();
    }
  }

  // See if we need to create a "remainder" query for offline+accuracy to
  // ensure we issue all samples in loaded_samples. Offline doesn't pad
  // loaded_samples like MultiStream does.
  if (scenario == TestScenario::Offline && mode == TestMode::AccuracyOnly) {
    size_t remaining_samples = loaded_samples.size() % samples_per_query;
    if (remaining_samples != 0) {
      samples.resize(remaining_samples);
      for (auto& s : samples) {
        s = loaded_samples[sample_distribution(sample_rng)];
      }
      queries.emplace_back(samples, timestamp, response_delegate, sequence_gen);
    }
  }

  LogDetail([count = queries.size(), spq = samples_per_query,
             duration = timestamp.count()](AsyncDetail& detail) {
#if USE_NEW_LOGGING_FORMAT
    MLPERF_LOG(detail, "generated_query_count", count);
    MLPERF_LOG(detail, "generated_samples_per_query", spq);
    MLPERF_LOG(detail, "generated_query_duration", duration);
#else
    detail("GeneratedQueries: ", "queries", count, "samples per query", spq,
           "duration", duration);
#endif
  });

  return queries;
}

/// \brief Issues a series of pre-generated queries.
// TODO: Templates for scenario and mode are overused, given the loadgen
//       no longer generates queries on the fly. Should we reduce the
//       use of templates?
template <TestScenario scenario, TestMode mode>
PerformanceResult IssueQueries(SystemUnderTest* sut,
                               const TestSettingsInternal& settings,
                               const LoadableSampleSet& loaded_sample_set,
                               SequenceGen* sequence_gen) {
  // Create reponse handler.
  ResponseDelegateDetailed<scenario, mode> response_logger;
  std::uniform_real_distribution<double> accuracy_log_offset_dist =
      std::uniform_real_distribution<double>(0.0, 1.0);
  std::mt19937 accuracy_log_offset_rng(settings.accuracy_log_rng_seed);
  response_logger.accuracy_log_offset =
      accuracy_log_offset_dist(accuracy_log_offset_rng);
  response_logger.accuracy_log_prob = settings.accuracy_log_probability;

  // Generate queries.
  auto sequence_id_start = sequence_gen->CurrentSampleId();
  std::vector<QueryMetadata> queries = GenerateQueries<scenario, mode>(
      settings, loaded_sample_set, sequence_gen, &response_logger);

  // Calculated expected number of queries
  uint64_t expected_queries =
      settings.target_qps * settings.min_duration.count() / 1000;
  uint64_t minimum_queries =
      settings.min_query_count * settings.samples_per_query;

  if (scenario != TestScenario::Offline) {
    expected_queries *= settings.samples_per_query;
  } else {
    minimum_queries = settings.min_sample_count;
  }

  expected_queries =
      expected_queries < minimum_queries ? minimum_queries : expected_queries;

  if (settings.accuracy_log_sampling_target > 0) {
    response_logger.accuracy_log_prob =
        (double)settings.accuracy_log_sampling_target / expected_queries;
  }
  auto sequence_id_end = sequence_gen->CurrentSampleId();
  size_t max_latencies_to_record = sequence_id_end - sequence_id_start;

  // Initialize logger for latency recording.
  GlobalLogger().RestartLatencyRecording(sequence_id_start,
                                         max_latencies_to_record);

  // Create and initialize an IssueQueryState.
  IssueQueryState state{
      sut, &queries, &response_logger, &settings, mode, {}, {}, false, 0,
      0,   {}};
  auto& controller = IssueQueryController::GetInstance();

  // Set number of IssueQueryThreads and wait for the threads to register.
  controller.SetNumThreads(settings.requested.server_num_issue_query_threads);

  // Start issuing the queries.
  controller.StartIssueQueries<scenario>(&state);

  // Gather query issuing statistics.
  const auto start_for_power = state.start_for_power;
  const auto start = state.start_time;
  const auto ran_out_of_generated_queries = state.ran_out_of_generated_queries;
  const auto queries_issued = state.queries_issued;
  const auto expected_latencies = state.expected_latencies;

  // Let the SUT know it should not expect any more queries.
  sut->FlushQueries();

  if (mode == TestMode::PerformanceOnly && ran_out_of_generated_queries) {
    LogDetail([](AsyncDetail& detail) {
#if USE_NEW_LOGGING_FORMAT
      MLPERF_LOG_ERROR(
          detail, "error_runtime",
          "Ending early: Ran out of generated queries to issue before the "
          "minimum query count and test duration were reached. "
          "Please update the relevant expected latency or target qps in the "
          "TestSettings so they are more accurate.");
#else
      detail.Error(
          "Ending early: Ran out of generated queries to issue before the "
          "minimum query count and test duration were reached.");
      detail(
          "Please update the relevant expected latency or target qps in the "
          "TestSettings so they are more accurate.");
#endif
    });
  }

  // Wait for tail queries to complete and collect all the latencies.
  // We have to keep the synchronization primitives alive until the SUT
  // is done with them.
  auto& final_query = queries[queries_issued - 1];
  std::vector<QuerySampleLatency> sample_latencies(
      GlobalLogger().GetLatenciesBlocking(expected_latencies));

  std::vector<QuerySampleLatency> first_token_latencies(
      GlobalLogger().GetTokenLatencies(expected_latencies));

  std::vector<QuerySampleLatency> time_per_output_token_arr(
      GlobalLogger().GetTimePerOutputToken(expected_latencies));

  std::vector<int64_t> tokens_per_sample(
      GlobalLogger().GetTokensPerSample(expected_latencies));

  // Log contention counters after every test as a sanity check.
  GlobalLogger().LogContentionAndAllocations();

  // This properly accounts for the fact that the max completion time may not
  // belong to the final query. It also excludes any time spent postprocessing
  // in the loadgen itself after final completion, which may be significant
  // in the offline scenario.
  PerfClock::time_point max_completion_time =
      GlobalLogger().GetMaxCompletionTime();
  auto sut_active_duration = max_completion_time - start;
  LogDetail([start_for_power, sut_active_duration](AsyncDetail& detail) {
    auto end_for_power =
        start_for_power +
        std::chrono::duration_cast<std::chrono::system_clock::duration>(
            sut_active_duration);
#if USE_NEW_LOGGING_FORMAT
    MLPERF_LOG_INTERVAL_START(detail, "power_begin",
                              DateTimeStringForPower(start_for_power));
    MLPERF_LOG_INTERVAL_END(detail, "power_end",
                            DateTimeStringForPower(end_for_power));
#else
    detail("POWER_BEGIN: ", "mode", ToString(mode), "time",
           DateTimeStringForPower(start_for_power));
    detail("POWER_END: ", "mode", ToString(mode), "time",
           DateTimeStringForPower(end_for_power));
#endif
  });

  double max_latency =
      QuerySampleLatencyToSeconds(GlobalLogger().GetMaxLatencySoFar());
  double final_query_scheduled_time =
      DurationToSeconds(final_query.scheduled_delta);
  double final_query_issued_time =
      DurationToSeconds(final_query.issued_start_time - start);
  double final_query_all_samples_done_time =
      DurationToSeconds(final_query.all_samples_done_time - start);

  std::vector<QuerySampleLatency> query_latencies;
  if (scenario == TestScenario::MultiStream) {
    query_latencies.resize(queries_issued);
    for (size_t i = 0; i < queries_issued; i++) {
      query_latencies[i] = DurationGeneratorNs{queries[i].scheduled_time}.delta(
          queries[i].all_samples_done_time);
    }
  }

  return PerformanceResult{
      std::move(sample_latencies),
      std::move(query_latencies),
      queries_issued,
      max_latency,
      final_query_scheduled_time,
      final_query_issued_time,
      final_query_all_samples_done_time,
      TokenPerformanceResults{first_token_latencies, time_per_output_token_arr,
                              tokens_per_sample}};
}

void LoadSamplesToRam(QuerySampleLibrary* qsl,
                      const std::vector<QuerySampleIndex>& samples) {
  LogDetail([&samples](AsyncDetail& detail) {
#if USE_NEW_LOGGING_FORMAT
    MLPERF_LOG(detail, "loaded_qsl_set", samples);
#else
    std::string set("\"[");
    for (auto i : samples) {
      set += std::to_string(i) + ",";
    }
    set.resize(set.size() - 1);
    set += "]\"";
    detail("Loading QSL : ", "set", set);
#endif
  });
  qsl->LoadSamplesToRam(samples);
}

/// \brief Generates random sets of samples in the QSL that we can load into
/// RAM at the same time.
std::vector<LoadableSampleSet> GenerateLoadableSets(
    QuerySampleLibrary* qsl, const TestSettingsInternal& settings) {
  auto tracer = MakeScopedTracer(
      [](AsyncTrace& trace) { trace("GenerateLoadableSets"); });

  std::vector<LoadableSampleSet> result;
  std::mt19937 qsl_rng(settings.qsl_rng_seed);

  // Generate indices for all available samples in the QSL.
  const size_t qsl_total_count = qsl->TotalSampleCount();
  std::vector<QuerySampleIndex> samples(qsl_total_count);
  for (size_t i = 0; i < qsl_total_count; i++) {
    samples[i] = static_cast<QuerySampleIndex>(i);
  }

  // Randomize the order of the samples.
  std::shuffle(samples.begin(), samples.end(), qsl_rng);

  // Partition the samples into loadable sets.
  const size_t set_size = settings.performance_sample_count;
  const size_t set_padding = (settings.scenario == TestScenario::MultiStream)
                                 ? settings.samples_per_query - 1
                                 : 0;
  std::vector<QuerySampleIndex> loadable_set;
  loadable_set.reserve(set_size + set_padding);

  for (auto s : samples) {
    loadable_set.push_back(s);
    if (loadable_set.size() == set_size) {
      result.push_back({std::move(loadable_set), set_size});
      loadable_set.clear();
      loadable_set.reserve(set_size + set_padding);
    }
  }

  if (!loadable_set.empty()) {
    // Copy the size since it will become invalid after the move.
    size_t loadable_set_size = loadable_set.size();
    result.push_back({std::move(loadable_set), loadable_set_size});
  }

  // Add padding for the multi stream scenario. Padding allows the
  // starting sample to be the same for all SUTs, independent of the value
  // of samples_per_query, while enabling samples in a query to be contiguous.
  for (auto& loadable_set : result) {
    auto& set = loadable_set.set;
    for (size_t i = 0; i < set_padding; i++) {
      // It's not clear in the spec if the STL deallocates the old container
      // before assigning, which would invalidate the source before the
      // assignment happens. Even though we should have reserved enough
      // elements above, copy the source first anyway since we are just moving
      // integers around.
      QuerySampleIndex p = set[i];
      set.push_back(p);
    }
  }

  return result;
}

/// \brief Opens and owns handles to all of the log files.
struct LogOutputs {
  LogOutputs(const LogOutputSettings& output_settings,
             const std::string& test_date_time) {
    std::string prefix = output_settings.outdir;
    prefix += "/" + output_settings.prefix;
    if (output_settings.prefix_with_datetime) {
      prefix += test_date_time + "_";
    }
    const std::string& suffix = output_settings.suffix;

    summary_out.open(prefix + "summary" + suffix + ".txt");
    detail_out.open(prefix + "detail" + suffix + ".txt");
    accuracy_out.open(prefix + "accuracy" + suffix + ".json");
    trace_out.open(prefix + "trace" + suffix + ".json");
  }

  bool CheckOutputs() {
    bool all_ofstreams_good = true;
    if (!summary_out.good()) {
      all_ofstreams_good = false;
      std::cerr << "LoadGen: Failed to open summary file.";
    }
    if (!detail_out.good()) {
      all_ofstreams_good = false;
      std::cerr << "LoadGen: Failed to open detailed log file.";
    }
    if (!accuracy_out.good()) {
      all_ofstreams_good = false;
      std::cerr << "LoadGen: Failed to open accuracy log file.";
    }
    if (!trace_out.good()) {
      all_ofstreams_good = false;
      std::cerr << "LoadGen: Failed to open trace file.";
    }
    return all_ofstreams_good;
  }

  std::ofstream summary_out;
  std::ofstream detail_out;
  std::ofstream accuracy_out;
  std::ofstream trace_out;
};

/// \brief Find boundaries of performance settings by widening bounds
/// exponentially.
/// \details To find an upper bound of performance, widen an
/// upper bound exponentially until finding a bound that can't satisfy
/// performance constraints. i.e. [1, 2) -> [2, 4) -> [4, 8) -> ...
template <TestScenario scenario>
std::pair<PerformanceSummary, PerformanceSummary> FindBoundaries(
    SystemUnderTest* sut, QuerySampleLibrary* qsl, SequenceGen* sequence_gen,
    PerformanceSummary l_perf_summary) {
  // Get upper bound
  TestSettingsInternal u_settings = l_perf_summary.settings;
  find_peak_performance::WidenPerformanceField<scenario>(&u_settings);

  LogDetail(
      [l_field = find_peak_performance::ToStringPerformanceField<scenario>(
           l_perf_summary.settings),
       u_field = find_peak_performance::ToStringPerformanceField<scenario>(
           u_settings)](AsyncDetail& detail) {
#if USE_NEW_LOGGING_FORMAT
        MLPERF_LOG(detail, "generic_message",
                   "FindBoundaries: Checking fields [" + l_field + ", " +
                       u_field + ")");
#else
        detail("FindBoundaries: Checking fields [" + l_field + ", " + u_field +
               ")");
#endif
      });

  std::vector<loadgen::LoadableSampleSet> loadable_sets(
      loadgen::GenerateLoadableSets(qsl, u_settings));
  const LoadableSampleSet& performance_set = loadable_sets.front();
  LoadSamplesToRam(qsl, performance_set.set);

  PerformanceResult u_pr(IssueQueries<scenario, TestMode::PerformanceOnly>(
      sut, u_settings, performance_set, sequence_gen));
  PerformanceSummary u_perf_summary{sut->Name(), u_settings, std::move(u_pr)};

  qsl->UnloadSamplesFromRam(performance_set.set);

  std::string tmp;
  if (!u_perf_summary.PerfConstraintsMet(&tmp)) {
    return std::make_pair(l_perf_summary, u_perf_summary);
  } else {
    return FindBoundaries<scenario>(sut, qsl, sequence_gen, u_perf_summary);
  }
}

/// \brief Find peak performance by binary search.
/// \details The found lower & upper bounds by the function 'FindBoundaries' are
/// used as initial bounds of binary search
template <TestScenario scenario>
PerformanceSummary FindPeakPerformanceBinarySearch(
    SystemUnderTest* sut, QuerySampleLibrary* qsl, SequenceGen* sequence_gen,
    const LoadableSampleSet& performance_set, PerformanceSummary l_perf_summary,
    PerformanceSummary u_perf_summary) {
  if (find_peak_performance::IsFinished<scenario>(l_perf_summary.settings,
                                                  u_perf_summary.settings)) {
    return l_perf_summary;
  }

  const TestSettingsInternal m_settings =
      find_peak_performance::MidOfBoundaries<scenario>(l_perf_summary.settings,
                                                       u_perf_summary.settings);

  LogDetail([l_field =
                 find_peak_performance::ToStringPerformanceField<scenario>(
                     l_perf_summary.settings),
             u_field =
                 find_peak_performance::ToStringPerformanceField<scenario>(
                     u_perf_summary.settings),
             m_field =
                 find_peak_performance::ToStringPerformanceField<scenario>(
                     m_settings)](AsyncDetail& detail) {
#if USE_NEW_LOGGING_FORMAT
    MLPERF_LOG(
        detail, "generic_message",
        "FindPeakPerformanceBinarySearch: Testing the mid value of bounds [" +
            l_field + ", " + u_field + "): " + m_field);
#else
    detail(
        "FindPeakPerformanceBinarySearch: Testing the mid value of bounds [" +
        l_field + ", " + u_field + "): " + m_field);
#endif
  });

  PerformanceResult m_pr(IssueQueries<scenario, TestMode::PerformanceOnly>(
      sut, m_settings, performance_set, sequence_gen));
  PerformanceSummary m_perf_summary{sut->Name(), m_settings, std::move(m_pr)};

  std::string tmp;
  if (m_perf_summary.PerfConstraintsMet(&tmp)) {
    return FindPeakPerformanceBinarySearch<scenario>(
        sut, qsl, sequence_gen, performance_set, m_perf_summary,
        u_perf_summary);
  } else {
    return FindPeakPerformanceBinarySearch<scenario>(
        sut, qsl, sequence_gen, performance_set, l_perf_summary,
        m_perf_summary);
  }
}

/// \brief Runs the performance mode, templated by scenario.
template <TestScenario scenario>
void RunPerformanceMode(SystemUnderTest* sut, QuerySampleLibrary* qsl,
                        const TestSettingsInternal& settings,
                        SequenceGen* sequence_gen) {
  LogDetail([](AsyncDetail& detail) {
#if USE_NEW_LOGGING_FORMAT
    MLPERF_LOG(detail, "generic_message", "Starting performance mode");
#else
    detail("Starting performance mode:");
#endif
  });

  // Use first loadable set as the performance set.
  std::vector<loadgen::LoadableSampleSet> loadable_sets(
      loadgen::GenerateLoadableSets(qsl, settings));
  const LoadableSampleSet& performance_set = loadable_sets.front();
  LoadSamplesToRam(qsl, performance_set.set);

  // Start PerfClock/system_clock timers for measuring performance interval
  // for comparison vs external timer.
  auto pc_start_ts = PerfClock::now();
  auto sc_start_ts = std::chrono::system_clock::now();
  if (settings.print_timestamps) {
    std::cout << "Loadgen :: Perf mode start. system_clock Timestamp = "
              << std::chrono::system_clock::to_time_t(sc_start_ts) << "\n"
              << std::flush;
  }

  PerformanceResult pr(IssueQueries<scenario, TestMode::PerformanceOnly>(
      sut, settings, performance_set, sequence_gen));

  // Measure PerfClock/system_clock timer durations for comparison vs
  // external timer.
  auto pc_stop_ts = PerfClock::now();
  auto sc_stop_ts = std::chrono::system_clock::now();
  auto pc_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                         pc_stop_ts - pc_start_ts)
                         .count();
  auto sc_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                         sc_stop_ts - sc_start_ts)
                         .count();
  float pc_sc_ratio = static_cast<float>(pc_duration) / sc_duration;
  if (settings.print_timestamps) {
    std::cout << "Loadgen :: Perf mode stop. systme_clock Timestamp = "
              << std::chrono::system_clock::to_time_t(sc_stop_ts) << "\n"
              << std::flush;
    std::cout << "Loadgen :: PerfClock Perf duration = " << pc_duration
              << "ms\n"
              << std::flush;
    std::cout << "Loadgen :: system_clock Perf duration = " << sc_duration
              << "ms\n"
              << std::flush;
    std::cout << "Loadgen :: PerfClock/system_clock ratio = " << std::fixed
              << std::setprecision(4) << pc_sc_ratio << "\n"
              << std::flush;
  }

  if (pc_sc_ratio > 1.01 || pc_sc_ratio < 0.99) {
    LogDetail([pc_sc_ratio](AsyncDetail& detail) {
#if USE_NEW_LOGGING_FORMAT
      std::stringstream ss;
      ss << "PerfClock and system_clock differ by more than 1%! "
         << " pc_sc_ratio: " << pc_sc_ratio;
      MLPERF_LOG_ERROR(detail, "error_runtime", ss.str());
#else
      detail.Error("PerfClock and system_clock differ by more than 1\%! ",
                   "pc_sc_ratio", pc_sc_ratio);
#endif
    });
  } else if (pc_sc_ratio > 1.001 || pc_sc_ratio < 0.999) {
    LogDetail([pc_sc_ratio](AsyncDetail& detail) {
#if USE_NEW_LOGGING_FORMAT
      std::stringstream ss;
      ss << "PerfClock and system_clock differ by more than 0.1%! "
         << " pc_sc_ratio: " << pc_sc_ratio;
      MLPERF_LOG_WARNING(detail, "warning_generic_message", ss.str());
#else
      detail.Warning("PerfClock and system_clock differ by more than 0.1\%. ",
                     "pc_sc_ratio", pc_sc_ratio);
#endif
    });
  }

  PerformanceSummary perf_summary{sut->Name(), settings, std::move(pr)};
  LogSummary([perf_summary](AsyncSummary& summary) mutable {
    perf_summary.LogSummary(summary);
  });
  // Create a copy to prevent thread hazard between LogSummary and LogDetail.
  PerformanceSummary perf_summary_detail{perf_summary};
  LogDetail([perf_summary_detail](AsyncDetail& detail) mutable {
    perf_summary_detail.LogDetail(detail);
  });

  qsl->UnloadSamplesFromRam(performance_set.set);
}

/// \brief Runs the binary search mode, templated by scenario.
/// \details 1. Check whether lower bound from user satisfies the performance
/// constraints, 2. Find an upper bound using the function 'FindBoundaries'
/// based on the lower bound, 3. Find peak performance settings using the
/// function 'FindPeakPerformanceBinarySearch'. note: Since we can't find a
/// lower bound programmatically because of the monotonicity issue of Server
/// scenario, rely on user's settings. After resolving this issue, we can
/// make the function 'FindBoundaries' find a lower bound as well from some
/// random initial settings.
template <TestScenario scenario>
void FindPeakPerformanceMode(SystemUnderTest* sut, QuerySampleLibrary* qsl,
                             const TestSettingsInternal& base_settings,
                             SequenceGen* sequence_gen) {
  LogDetail([](AsyncDetail& detail) {
#if USE_NEW_LOGGING_FORMAT
    MLPERF_LOG(detail, "generic_message", "Starting FindPeakPerformance mode");
#else
    detail("Starting FindPeakPerformance mode:");
#endif
  });

  if (scenario != TestScenario::Server) {
    LogDetail([unsupported_scenario = ToString(scenario)](AsyncDetail& detail) {
#if USE_NEW_LOGGING_FORMAT
      MLPERF_LOG_ERROR(detail, "error_invalid_config",
                       find_peak_performance::kNotSupportedMsg);
#else
      detail.Error(find_peak_performance::kNotSupportedMsg);
#endif
    });
    return;
  }

  LogDetail(
      [base_field = find_peak_performance::ToStringPerformanceField<scenario>(
           base_settings)](AsyncDetail& detail) {
#if USE_NEW_LOGGING_FORMAT
        MLPERF_LOG(
            detail, "generic_message",
            "FindPeakPerformance: Check validity of the base settings field: " +
                base_field);
#else
        detail(
            "FindPeakPerformance: Check validity of the base settings field: " +
            base_field);
#endif
      });

  // 1. Check whether the lower bound came from user satisfy performance
  // constraints or not.
  std::vector<loadgen::LoadableSampleSet> base_loadable_sets(
      loadgen::GenerateLoadableSets(qsl, base_settings));
  const LoadableSampleSet& base_performance_set = base_loadable_sets.front();
  LoadSamplesToRam(qsl, base_performance_set.set);

  PerformanceResult base_pr(IssueQueries<scenario, TestMode::PerformanceOnly>(
      sut, base_settings, base_performance_set, sequence_gen));
  PerformanceSummary base_perf_summary{sut->Name(), base_settings,
                                       std::move(base_pr)};

  // We can also use all_constraints_met to check performance constraints,
  // but to reduce searching time, leave it up to whether the settings satisfy
  // min duration & min queries or not to users.
  std::string msg;
  if (!base_perf_summary.PerfConstraintsMet(&msg)) {
    LogDetail([msg](AsyncDetail& detail) {
#if USE_NEW_LOGGING_FORMAT
      std::stringstream ss;
      ss << "FindPeakPerformance: Initial lower bound does not satisfy "
         << "performance constraints, msg: " << msg;
      MLPERF_LOG_ERROR(detail, "error_runtime", ss.str());
#else
      detail.Error(
          "FindPeakPerformance: Initial lower bound does not satisfy "
          "performance constraints, msg: " +
          msg);
#endif
    });

    PerformanceSummary perf_summary{sut->Name(), base_settings,
                                    std::move(base_perf_summary.pr)};
    LogSummary([perf_summary](AsyncSummary& summary) mutable {
      perf_summary.LogSummary(summary);
    });
    // Create a copy to prevent thread hazard between LogSummary and LogDetail.
    PerformanceSummary perf_summary_detail{perf_summary};
    LogDetail([perf_summary_detail](AsyncDetail& detail) mutable {
      perf_summary_detail.LogDetail(detail);
    });

    qsl->UnloadSamplesFromRam(base_performance_set.set);

    return;
  }

  // Clear loaded samples.
  qsl->UnloadSamplesFromRam(base_performance_set.set);

  // 2. Find an upper bound based on the lower bound.
  std::pair<PerformanceSummary, PerformanceSummary> boundaries =
      FindBoundaries<scenario>(sut, qsl, sequence_gen, base_perf_summary);
  PerformanceSummary l_perf_summary = boundaries.first;
  PerformanceSummary u_perf_summary = boundaries.second;

  LogDetail(
      [l_field = find_peak_performance::ToStringPerformanceField<scenario>(
           l_perf_summary.settings),
       u_field = find_peak_performance::ToStringPerformanceField<scenario>(
           u_perf_summary.settings)](AsyncDetail& detail) {
#if USE_NEW_LOGGING_FORMAT
        MLPERF_LOG(detail, "generic_message",
                   "FindPeakPerformance: Found boundaries: [" + l_field + ", " +
                       u_field + ")");
#else
        detail("FindPeakPerformance: Found boundaries: [" + l_field + ", " +
               u_field + ")");
#endif
      });

  // Reuse performance_set, u_perf_summary has the largest 'samples_per_query'.
  std::vector<loadgen::LoadableSampleSet> loadable_sets(
      loadgen::GenerateLoadableSets(qsl, u_perf_summary.settings));
  const LoadableSampleSet& performance_set = loadable_sets.front();
  LoadSamplesToRam(qsl, performance_set.set);

  // 3. Find peak performance settings using the found boundaries
  PerformanceSummary perf_summary = FindPeakPerformanceBinarySearch<scenario>(
      sut, qsl, sequence_gen, performance_set, l_perf_summary, u_perf_summary);

  // Print-out the peak performance test setting.
  LogDetail([field = find_peak_performance::ToStringPerformanceField<scenario>(
                 perf_summary.settings)](AsyncDetail& detail) {
#if USE_NEW_LOGGING_FORMAT
    MLPERF_LOG(detail, "generic_message",
               "FindPeakPerformance: Found peak performance field: " + field);
#else
    detail("FindPeakPerformance: Found peak performance field: " + field);
#endif
  });

  LogSummary([perf_summary](AsyncSummary& summary) mutable {
    perf_summary.LogSummary(summary);
  });
  // Create a copy to prevent thread hazard between LogSummary and LogDetail.
  PerformanceSummary perf_summary_detail{perf_summary};
  LogDetail([perf_summary_detail](AsyncDetail& detail) mutable {
    perf_summary_detail.LogDetail(detail);
  });

  qsl->UnloadSamplesFromRam(performance_set.set);
}

/// \brief Runs the accuracy mode, templated by scenario.
template <TestScenario scenario>
void RunAccuracyMode(SystemUnderTest* sut, QuerySampleLibrary* qsl,
                     const TestSettingsInternal& settings,
                     SequenceGen* sequence_gen) {
  LogDetail([](AsyncDetail& detail) {
#if USE_NEW_LOGGING_FORMAT
    MLPERF_LOG(detail, "generic_message", "Starting accuracy mode");
#else
    detail("Starting accuracy mode:");
#endif
  });

  std::vector<loadgen::LoadableSampleSet> loadable_sets(
      loadgen::GenerateLoadableSets(qsl, settings));

  for (auto& loadable_set : loadable_sets) {
    {
      auto tracer = MakeScopedTracer(
          [count = loadable_set.set.size()](AsyncTrace& trace) {
            trace("LoadSamples", "count", count);
          });
      LoadSamplesToRam(qsl, loadable_set.set);
    }

    PerformanceResult pr(IssueQueries<scenario, TestMode::AccuracyOnly>(
        sut, settings, loadable_set, sequence_gen));

    {
      auto tracer = MakeScopedTracer(
          [count = loadable_set.set.size()](AsyncTrace& trace) {
            trace("UnloadSampes", "count", count);
          });
      qsl->UnloadSamplesFromRam(loadable_set.set);
    }
  }
}

/// \brief Routes runtime scenario requests to the corresponding instances
/// of its templated mode functions.
struct RunFunctions {
  using Signature = void(SystemUnderTest* sut, QuerySampleLibrary* qsl,
                         const TestSettingsInternal& settings,
                         SequenceGen* sequence_gen);

  template <TestScenario compile_time_scenario>
  static RunFunctions GetCompileTime() {
    return {(RunAccuracyMode<compile_time_scenario>),
            (RunPerformanceMode<compile_time_scenario>),
            (FindPeakPerformanceMode<compile_time_scenario>)};
  }

  static RunFunctions Get(TestScenario run_time_scenario) {
    switch (run_time_scenario) {
      case TestScenario::SingleStream:
        return GetCompileTime<TestScenario::SingleStream>();
      case TestScenario::MultiStream:
        return GetCompileTime<TestScenario::MultiStream>();
      case TestScenario::Server:
        return GetCompileTime<TestScenario::Server>();
      case TestScenario::Offline:
        return GetCompileTime<TestScenario::Offline>();
    }
    // We should not reach this point.
    assert(false);
    return GetCompileTime<TestScenario::SingleStream>();
  }

  Signature& accuracy;
  Signature& performance;
  Signature& find_peak_performance;
};

}  // namespace loadgen

void StartTest(SystemUnderTest* sut, QuerySampleLibrary* qsl,
               const TestSettings& requested_settings,
               const LogSettings& log_settings,
               const std::string audit_config_filename) {
  GlobalLogger().StartIOThread();

  const std::string test_date_time = CurrentDateTimeISO8601();

  loadgen::LogOutputs log_outputs(log_settings.log_output, test_date_time);
  if (!log_outputs.CheckOutputs()) {
    return;
  }

  GlobalLogger().StartLogging(&log_outputs.summary_out, &log_outputs.detail_out,
                              &log_outputs.accuracy_out,
                              log_settings.log_output.copy_detail_to_stdout,
                              log_settings.log_output.copy_summary_to_stdout);

  GlobalLogger().SetUseTokens(requested_settings.use_token_latencies);
  bool needs_first_token =
      (requested_settings.scenario != TestScenario::Offline);
  GlobalLogger().SetNeedsFirstToken(needs_first_token);

  if (log_settings.enable_trace) {
    GlobalLogger().StartNewTrace(&log_outputs.trace_out, PerfClock::now());
  }

  // measure sut->Name() response time
  PerfClock::time_point pre_get_sut_name_ts = PerfClock::now();
  const std::string& sut_name = sut->Name();
  PerfClock::time_point post_get_sut_name_ts = PerfClock::now();

  auto get_sut_name_duration_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          post_get_sut_name_ts - pre_get_sut_name_ts)
          .count();

  LogLoadgenVersion();
  LogDetail([sut, qsl, test_date_time, &sut_name,
             &get_sut_name_duration_ns](AsyncDetail& detail) {
#if USE_NEW_LOGGING_FORMAT
    MLPERF_LOG(detail, "test_datetime", test_date_time);
    MLPERF_LOG(detail, "sut_name", sut_name);
    MLPERF_LOG(detail, "get_sut_name_duration_ns", get_sut_name_duration_ns);
    MLPERF_LOG(detail, "qsl_name", qsl->Name());
    MLPERF_LOG(detail, "qsl_reported_total_count", qsl->TotalSampleCount());
    MLPERF_LOG(detail, "qsl_reported_performance_count",
               qsl->PerformanceSampleCount());
#else
    detail("Date + time of test: ", test_date_time);
    detail("System Under Test (SUT) name: ", sut_name);
    detail("Get SUT name time [ns]: ", get_sut_name_duration_ns);
    detail("Query Sample Library (QSL) name: ", qsl->Name());
    detail("QSL total size: ", qsl->TotalSampleCount());
    detail("QSL performance size*: ", qsl->PerformanceSampleCount());
    detail("*TestSettings (performance_sample_count_override) can override");
    detail("*Refer to Effective Settings for actual value");
#endif
  });

  TestSettings test_settings = requested_settings;
  // Look for Audit Config file to override TestSettings during audit
  if (FileExists(audit_config_filename)) {
    LogDetail([](AsyncDetail& detail) {
#if USE_NEW_LOGGING_FORMAT
      MLPERF_LOG_WARNING(detail, "warning_generic_message",
                         "Found Audit Config file (audit.config)."
                         " Overriding TestSettings from audit.config file.");
#else
      detail(
          "Found Audit Config file (audit.config)."
          " Overriding TestSettings from audit.config file.");
#endif
    });
    std::string audit_scenario = loadgen::ToString(test_settings.scenario);
    // Remove Spaces from the string
    RemoveValue(&audit_scenario, ' ');
    const std::string generic_model = "*";
    test_settings.FromConfig(audit_config_filename, generic_model,
                             audit_scenario, 2);
  }
  if (test_settings.test05) {
    // If the configuration indicates we are running test05,
    // random seeds
    LogDetail([](AsyncDetail& detail) {
#if USE_NEW_LOGGING_FORMAT
      MLPERF_LOG_WARNING(detail, "warning_generic_message",
                         "Test05 flag detected"
                         " Overriding random seeds");
#else
      detail(
          "Test05 flag detected"
          " Overriding random seeds");
#endif
    });
    test_settings.mode = TestMode::PerformanceOnly;
    test_settings.qsl_rng_seed = requested_settings.test05_qsl_rng_seed;
    test_settings.sample_index_rng_seed =
        requested_settings.test05_sample_index_rng_seed;
    test_settings.schedule_rng_seed =
        requested_settings.test05_schedule_rng_seed;
  }

  loadgen::TestSettingsInternal sanitized_settings(
      test_settings, qsl->PerformanceSampleCount());
  sanitized_settings.LogAllSettings();

  auto run_funcs = loadgen::RunFunctions::Get(sanitized_settings.scenario);

  loadgen::SequenceGen sequence_gen;
  switch (sanitized_settings.mode) {
    case TestMode::SubmissionRun:
      run_funcs.accuracy(sut, qsl, sanitized_settings, &sequence_gen);
      run_funcs.performance(sut, qsl, sanitized_settings, &sequence_gen);
      break;
    case TestMode::AccuracyOnly:
      run_funcs.accuracy(sut, qsl, sanitized_settings, &sequence_gen);
      break;
    case TestMode::PerformanceOnly:
      run_funcs.performance(sut, qsl, sanitized_settings, &sequence_gen);
      break;
    case TestMode::FindPeakPerformance:
      run_funcs.find_peak_performance(sut, qsl, sanitized_settings,
                                      &sequence_gen);
      break;
  }

  loadgen::IssueQueryController::GetInstance().EndThreads();

  // Stop tracing after logging so all logs are captured in the trace.
  GlobalLogger().StopLogging();
  GlobalLogger().StopTracing();
  GlobalLogger().StopIOThread();
}

void AbortTest() {
  loadgen::IssueQueryController::GetInstance().EndThreads();
  GlobalLogger().StopLogging();
  GlobalLogger().StopTracing();
  GlobalLogger().StopIOThread();
}

void QuerySamplesComplete(QuerySampleResponse* responses, size_t response_count,
                          const ResponseCallback& response_cb) {
  PerfClock::time_point timestamp = PerfClock::now();

  auto tracer = MakeScopedTracer(
      [](AsyncTrace& trace) { trace("QuerySamplesComplete"); });

  const QuerySampleResponse* end = responses + response_count;

  // Notify first to unblock loadgen production ASAP.
  for (QuerySampleResponse* response = responses; response < end; response++) {
    loadgen::SampleMetadata* sample =
        reinterpret_cast<loadgen::SampleMetadata*>(response->id);
    loadgen::QueryMetadata* query = sample->query_metadata;
    query->NotifyOneSampleCompleted(timestamp);
  }

  // Log samples.
  for (QuerySampleResponse* response = responses; response < end; response++) {
    loadgen::SampleMetadata* sample =
        reinterpret_cast<loadgen::SampleMetadata*>(response->id);
    loadgen::QueryMetadata* query = sample->query_metadata;
    query->response_delegate->SampleComplete(sample, response, timestamp,
                                             response_cb);
  }
  // PerfClock::time_point end_timestamp = PerfClock::now();
  // mlperf::samples_overhead_acum += (end_timestamp - timestamp).count();
}

void FirstTokenComplete(QuerySampleResponse* responses, size_t response_count,
                        const ResponseCallback& response_cb) {
  PerfClock::time_point timestamp = PerfClock::now();

  auto tracer =
      MakeScopedTracer([](AsyncTrace& trace) { trace("FirstTokenComplete"); });

  const QuerySampleResponse* end = responses + response_count;

  // Log samples.
  for (QuerySampleResponse* response = responses; response < end; response++) {
    loadgen::SampleMetadata* sample =
        reinterpret_cast<loadgen::SampleMetadata*>(response->id);
    loadgen::QueryMetadata* query = sample->query_metadata;
    query->response_delegate->TokenComplete(sample, response, timestamp,
                                            response_cb);
  }
  // PerfClock::time_point end_timestamp = PerfClock::now();
  // mlperf::tokens_overhead_acum += (end_timestamp - timestamp).count();
}

}  // namespace mlperf
