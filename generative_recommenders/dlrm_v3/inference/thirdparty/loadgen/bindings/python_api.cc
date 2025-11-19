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
/// \brief Python bindings for the loadgen using pybind11.

#ifndef PYTHON_BINDINGS_H
#define PYTHON_BINDINGS_H

#include <functional>

#include "../loadgen.h"
#include "../query_dispatch_library.h"
#include "../query_sample.h"
#include "../query_sample_library.h"
#include "../system_under_test.h"
#include "../test_settings.h"
#include "pybind11/functional.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/stl_bind.h"

namespace mlperf {

namespace {

using IssueQueryCallback = std::function<void(std::vector<QuerySample>)>;
using FastIssueQueriesCallback =
    std::function<void(std::vector<ResponseId>, std::vector<QuerySampleIndex>)>;
using FlushQueriesCallback = std::function<void()>;
using NameCallback = std::function<std::string()>;

// Forwards SystemUnderTest calls to relevant callbacks.
class SystemUnderTestTrampoline : public SystemUnderTest {
 public:
  SystemUnderTestTrampoline(std::string name, IssueQueryCallback issue_cb,
                            FlushQueriesCallback flush_queries_cb)
      : name_(std::move(name)),
        issue_cb_(issue_cb),
        flush_queries_cb_(flush_queries_cb) {}
  ~SystemUnderTestTrampoline() override = default;

  const std::string& Name() override { return name_; }

  void IssueQuery(const std::vector<QuerySample>& samples) override {
    pybind11::gil_scoped_acquire gil_acquirer;
    issue_cb_(samples);
  }

  void FlushQueries() override { flush_queries_cb_(); }

 protected:
  std::string name_;
  IssueQueryCallback issue_cb_;
  FlushQueriesCallback flush_queries_cb_;
};

class FastSystemUnderTestTrampoline : public SystemUnderTestTrampoline {
 public:
  FastSystemUnderTestTrampoline(std::string name,
                                FastIssueQueriesCallback fast_issue_cb,
                                FlushQueriesCallback flush_queries_cb)
      : SystemUnderTestTrampoline(name, nullptr, flush_queries_cb),
        fast_issue_cb_(fast_issue_cb) {}
  ~FastSystemUnderTestTrampoline() override = default;

  void IssueQuery(const std::vector<QuerySample>& samples) override {
    pybind11::gil_scoped_acquire gil_acquirer;
    std::vector<ResponseId> responseIds;
    std::vector<QuerySampleIndex> querySampleIndices;
    for (auto& s : samples) {
      responseIds.push_back(s.id);
      querySampleIndices.push_back(s.index);
    }
    fast_issue_cb_(responseIds, querySampleIndices);
  }

 private:
  FastIssueQueriesCallback fast_issue_cb_;
};

using LoadSamplesToRamCallback =
    std::function<void(std::vector<QuerySampleIndex>)>;
using UnloadSamplesFromRamCallback =
    std::function<void(std::vector<QuerySampleIndex>)>;

// Forwards QuerySampleLibrary calls to relevant callbacks.
class QuerySampleLibraryTrampoline : public QuerySampleLibrary {
 public:
  QuerySampleLibraryTrampoline(
      std::string name, size_t total_sample_count,
      size_t performance_sample_count,
      LoadSamplesToRamCallback load_samples_to_ram_cb,
      UnloadSamplesFromRamCallback unload_samples_from_ram_cb)
      : name_(std::move(name)),
        total_sample_count_(total_sample_count),
        performance_sample_count_(performance_sample_count),
        load_samples_to_ram_cb_(load_samples_to_ram_cb),
        unload_samples_from_ram_cb_(unload_samples_from_ram_cb) {}
  ~QuerySampleLibraryTrampoline() override = default;

  const std::string& Name() override { return name_; }
  size_t TotalSampleCount() { return total_sample_count_; }
  size_t PerformanceSampleCount() { return performance_sample_count_; }

  void LoadSamplesToRam(const std::vector<QuerySampleIndex>& samples) override {
    pybind11::gil_scoped_acquire gil_acquirer;
    load_samples_to_ram_cb_(samples);
  }
  void UnloadSamplesFromRam(
      const std::vector<QuerySampleIndex>& samples) override {
    pybind11::gil_scoped_acquire gil_acquirer;
    unload_samples_from_ram_cb_(samples);
  }

 private:
  std::string name_;
  size_t total_sample_count_;
  size_t performance_sample_count_;
  LoadSamplesToRamCallback load_samples_to_ram_cb_;
  UnloadSamplesFromRamCallback unload_samples_from_ram_cb_;
};

// A QDL that allows defining callbacks for
// IssueQuery, FlushQueries, and Name methods.
class QueryDispatchLibraryTrampoline : public QueryDispatchLibrary {
 public:
  QueryDispatchLibraryTrampoline(IssueQueryCallback issue_query_callback,
                                 FlushQueriesCallback flush_queries_callback,
                                 NameCallback name_callback)
      : issue_query_callback_(issue_query_callback),
        flush_queries_callback_(flush_queries_callback),
        name_callback_(name_callback) {}

  // Returns the name of the SUT. Name shall be returned over the network
  // TODO: other bindings should also be fixed eventually to be used over the
  // network
  const std::string& Name() override {
    static std::string name;  // HACK: avoid returning a reference to temporary.
    pybind11::gil_scoped_acquire gil_acquirer;
    name = name_callback_();  // name_callback_() shall returned name over the
                              // network.
    return name;
  }

  void IssueQuery(const std::vector<QuerySample>& samples) override {
    pybind11::gil_scoped_acquire gil_acquirer;
    issue_query_callback_(samples);
  }

  void FlushQueries() override { flush_queries_callback_(); }

 protected:
  IssueQueryCallback issue_query_callback_;
  FlushQueriesCallback flush_queries_callback_;
  NameCallback name_callback_;
};

}  // namespace

/// \brief Python bindings.
namespace py {

uintptr_t ConstructSUT(IssueQueryCallback issue_cb,
                       FlushQueriesCallback flush_queries_cb) {
  SystemUnderTestTrampoline* sut =
      new SystemUnderTestTrampoline("PySUT", issue_cb, flush_queries_cb);
  return reinterpret_cast<uintptr_t>(sut);
}

void DestroySUT(uintptr_t sut) {
  SystemUnderTestTrampoline* sut_cast =
      reinterpret_cast<SystemUnderTestTrampoline*>(sut);
  delete sut_cast;
}

uintptr_t ConstructFastSUT(FastIssueQueriesCallback fast_issue_cb,
                           FlushQueriesCallback flush_queries_cb) {
  FastSystemUnderTestTrampoline* sut = new FastSystemUnderTestTrampoline(
      "PyFastSUT", fast_issue_cb, flush_queries_cb);
  return reinterpret_cast<uintptr_t>(sut);
}

void DestroyFastSUT(uintptr_t sut) {
  FastSystemUnderTestTrampoline* sut_cast =
      reinterpret_cast<FastSystemUnderTestTrampoline*>(sut);
  delete sut_cast;
}

uintptr_t ConstructQSL(
    size_t total_sample_count, size_t performance_sample_count,
    LoadSamplesToRamCallback load_samples_to_ram_cb,
    UnloadSamplesFromRamCallback unload_samples_from_ram_cb) {
  QuerySampleLibraryTrampoline* qsl = new QuerySampleLibraryTrampoline(
      "PyQSL", total_sample_count, performance_sample_count,
      load_samples_to_ram_cb, unload_samples_from_ram_cb);
  return reinterpret_cast<uintptr_t>(qsl);
}

void DestroyQSL(uintptr_t qsl) {
  QuerySampleLibraryTrampoline* qsl_cast =
      reinterpret_cast<QuerySampleLibraryTrampoline*>(qsl);
  delete qsl_cast;
}

uintptr_t ConstructQDL(IssueQueryCallback issue_cb,
                       FlushQueriesCallback flush_queries_cb,
                       NameCallback name_callback) {
  QueryDispatchLibraryTrampoline* qdl = new QueryDispatchLibraryTrampoline(
      issue_cb, flush_queries_cb, name_callback);
  return reinterpret_cast<uintptr_t>(qdl);
}

void DestroyQDL(uintptr_t qdl) {
  QueryDispatchLibraryTrampoline* qdl_cast =
      reinterpret_cast<QueryDispatchLibraryTrampoline*>(qdl);
  delete qdl_cast;
}

void StartTest(uintptr_t sut, uintptr_t qsl, mlperf::TestSettings test_settings,
               const std::string& audit_config_filename) {
  pybind11::gil_scoped_release gil_releaser;
  SystemUnderTestTrampoline* sut_cast =
      reinterpret_cast<SystemUnderTestTrampoline*>(sut);
  QuerySampleLibraryTrampoline* qsl_cast =
      reinterpret_cast<QuerySampleLibraryTrampoline*>(qsl);
  LogSettings default_log_settings;
  mlperf::StartTest(sut_cast, qsl_cast, test_settings, default_log_settings,
                    audit_config_filename);
}

void StartTestWithLogSettings(uintptr_t sut, uintptr_t qsl,
                              mlperf::TestSettings test_settings,
                              mlperf::LogSettings log_settings,
                              const std::string& audit_config_filename) {
  pybind11::gil_scoped_release gil_releaser;
  SystemUnderTestTrampoline* sut_cast =
      reinterpret_cast<SystemUnderTestTrampoline*>(sut);
  QuerySampleLibraryTrampoline* qsl_cast =
      reinterpret_cast<QuerySampleLibraryTrampoline*>(qsl);
  mlperf::StartTest(sut_cast, qsl_cast, test_settings, log_settings,
                    audit_config_filename);
}

using ResponseCallback = std::function<void(QuerySampleResponse*)>;

/// TODO: Get rid of copies.
void QuerySamplesComplete(std::vector<QuerySampleResponse> responses,
                          ResponseCallback response_cb = {}) {
  pybind11::gil_scoped_release gil_releaser;
  mlperf::QuerySamplesComplete(responses.data(), responses.size(), response_cb);
}

void FirstTokenComplete(std::vector<QuerySampleResponse> responses,
                        ResponseCallback response_cb = {}) {
  pybind11::gil_scoped_release gil_releaser;
  mlperf::FirstTokenComplete(responses.data(), responses.size(), response_cb);
}

PYBIND11_MODULE(mlperf_loadgen, m) {
  m.doc() = "MLPerf Inference load generator.";

  pybind11::enum_<TestScenario>(m, "TestScenario")
      .value("SingleStream", TestScenario::SingleStream)
      .value("MultiStream", TestScenario::MultiStream)
      .value("Server", TestScenario::Server)
      .value("Offline", TestScenario::Offline);

  pybind11::enum_<TestMode>(m, "TestMode")
      .value("SubmissionRun", TestMode::SubmissionRun)
      .value("AccuracyOnly", TestMode::AccuracyOnly)
      .value("PerformanceOnly", TestMode::PerformanceOnly)
      .value("FindPeakPerformance", TestMode::FindPeakPerformance);

  pybind11::class_<TestSettings>(m, "TestSettings")
      .def(pybind11::init<>())
      .def_readwrite("scenario", &TestSettings::scenario)
      .def_readwrite("mode", &TestSettings::mode)
      .def_readwrite("single_stream_expected_latency_ns",
                     &TestSettings::single_stream_expected_latency_ns)
      .def_readwrite("single_stream_target_latency_percentile",
                     &TestSettings::single_stream_target_latency_percentile)
      .def_readwrite("multi_stream_expected_latency_ns",
                     &TestSettings::multi_stream_expected_latency_ns)
      .def_readwrite("multi_stream_target_latency_percentile",
                     &TestSettings::multi_stream_target_latency_percentile)
      .def_readwrite("multi_stream_samples_per_query",
                     &TestSettings::multi_stream_samples_per_query)
      .def_readwrite("server_target_qps", &TestSettings::server_target_qps)
      .def_readwrite("server_target_latency_ns",
                     &TestSettings::server_target_latency_ns)
      .def_readwrite("server_target_latency_percentile",
                     &TestSettings::server_target_latency_percentile)
      .def_readwrite("server_coalesce_queries",
                     &TestSettings::server_coalesce_queries)
      .def_readwrite("server_find_peak_qps_decimals_of_precision",
                     &TestSettings::server_find_peak_qps_decimals_of_precision)
      .def_readwrite("server_find_peak_qps_boundary_step_size",
                     &TestSettings::server_find_peak_qps_boundary_step_size)
      .def_readwrite("server_max_async_queries",
                     &TestSettings::server_max_async_queries)
      .def_readwrite("server_num_issue_query_threads",
                     &TestSettings::server_num_issue_query_threads)
      .def_readwrite("offline_expected_qps",
                     &TestSettings::offline_expected_qps)
      .def_readwrite("min_duration_ms", &TestSettings::min_duration_ms)
      .def_readwrite("max_duration_ms", &TestSettings::max_duration_ms)
      .def_readwrite("min_query_count", &TestSettings::min_query_count)
      .def_readwrite("max_query_count", &TestSettings::max_query_count)
      .def_readwrite("qsl_rng_seed", &TestSettings::qsl_rng_seed)
      .def_readwrite("sample_index_rng_seed",
                     &TestSettings::sample_index_rng_seed)
      .def_readwrite("schedule_rng_seed", &TestSettings::schedule_rng_seed)
      .def_readwrite("accuracy_log_rng_seed",
                     &TestSettings::accuracy_log_rng_seed)
      .def_readwrite("accuracy_log_probability",
                     &TestSettings::accuracy_log_probability)
      .def_readwrite("print_timestamps", &TestSettings::print_timestamps)
      .def_readwrite("performance_issue_unique",
                     &TestSettings::performance_issue_unique)
      .def_readwrite("performance_issue_same",
                     &TestSettings::performance_issue_same)
      .def_readwrite("performance_issue_same_index",
                     &TestSettings::performance_issue_same_index)
      .def_readwrite("performance_sample_count_override",
                     &TestSettings::performance_sample_count_override)
      .def_readwrite("test05", &TestSettings::test05)
      .def_readwrite("test05_qsl_rng_seed", &TestSettings::test05_qsl_rng_seed)
      .def_readwrite("test05_sample_index_rng_seed",
                     &TestSettings::test05_sample_index_rng_seed)
      .def_readwrite("test05_schedule_rng_seed",
                     &TestSettings::test05_schedule_rng_seed)
      .def_readwrite("use_token_latencies", &TestSettings::use_token_latencies)
      .def_readwrite("ttft_latency", &TestSettings::server_ttft_latency)
      .def_readwrite("tpot_latency", &TestSettings::server_tpot_latency)
      .def_readwrite("infer_token_latencies",
                     &TestSettings::infer_token_latencies)
      .def_readwrite("token_latency_scaling_factor",
                     &TestSettings::token_latency_scaling_factor)
      .def("FromConfig", &TestSettings::FromConfig, pybind11::arg("path"),
           pybind11::arg("model"), pybind11::arg("scenario"),
           pybind11::arg("conf_type") = 1,
           "This function configures settings from the given user "
           "configuration file, model, and scenario. The conf_type flag "
           "should be set to 1 for loading user.conf or else only the default "
           "mlperf_conf file "
           "will be loaded by the loadgen.");

  pybind11::enum_<LoggingMode>(m, "LoggingMode")
      .value("AsyncPoll", LoggingMode::AsyncPoll)
      .value("EndOfTestOnly", LoggingMode::EndOfTestOnly)
      .value("Synchronous", LoggingMode::Synchronous);

  pybind11::class_<LogOutputSettings>(m, "LogOutputSettings")
      .def(pybind11::init<>())
      .def_readwrite("outdir", &LogOutputSettings::outdir)
      .def_readwrite("prefix", &LogOutputSettings::prefix)
      .def_readwrite("suffix", &LogOutputSettings::suffix)
      .def_readwrite("prefix_with_datetime",
                     &LogOutputSettings::prefix_with_datetime)
      .def_readwrite("copy_detail_to_stdout",
                     &LogOutputSettings::copy_detail_to_stdout)
      .def_readwrite("copy_summary_to_stdout",
                     &LogOutputSettings::copy_summary_to_stdout);

  pybind11::class_<LogSettings>(m, "LogSettings")
      .def(pybind11::init<>())
      .def_readwrite("log_output", &LogSettings::log_output)
      .def_readwrite("log_mode", &LogSettings::log_mode)
      .def_readwrite("log_mode_async_poll_interval_ms",
                     &LogSettings::log_mode_async_poll_interval_ms)
      .def_readwrite("enable_trace", &LogSettings::enable_trace);

  pybind11::class_<QuerySample>(m, "QuerySample")
      .def(pybind11::init<>())
      .def(pybind11::init<ResponseId, QuerySampleIndex>())
      .def_readwrite("id", &QuerySample::id)
      .def_readwrite("index", &QuerySample::index)
      .def(pybind11::pickle(
          [](const QuerySample& qs) {  // __getstate__
            /*Return a tuple that fully encodes state of object*/
            return pybind11::make_tuple(qs.id, qs.index);
          },
          [](pybind11::tuple t) {  // __setstate__
            if (t.size() != 2)
              throw std::runtime_error("Invalid state for QuerySample");
            /* Create a new C++ instance*/
            QuerySample q;
            q.id = t[0].cast<uintptr_t>();
            q.index = t[1].cast<size_t>();
            return q;
          }));

  pybind11::class_<QuerySampleResponse>(m, "QuerySampleResponse")
      .def(pybind11::init<>())
      .def(pybind11::init<ResponseId, uintptr_t, size_t>())
      .def(pybind11::init<ResponseId, uintptr_t, size_t, int64_t>())
      .def_readwrite("id", &QuerySampleResponse::id)
      .def_readwrite("data", &QuerySampleResponse::data)
      .def_readwrite("size", &QuerySampleResponse::size)
      .def_readwrite("n_tokens", &QuerySampleResponse::n_tokens)
      .def(pybind11::pickle(
          [](const QuerySampleResponse& qsr) {  // __getstate__
            /* Return a tuple that fully encodes state of object*/
            return pybind11::make_tuple(qsr.id, qsr.data, qsr.size);
          },
          [](pybind11::tuple t) {  // __setstate__
            if ((t.size() != 3) || (t.size() != 4))
              throw std::runtime_error("Invalid state for QuerySampleResponse");
            /* Create a new C++ instance*/
            QuerySampleResponse q;
            q.id = t[0].cast<uintptr_t>();
            q.data = t[1].cast<uintptr_t>();
            q.size = t[2].cast<size_t>();
            if (t.size() == 4) {
              q.n_tokens = t[3].cast<int64_t>();
            } else {
              q.n_tokens = 0;
            }
            return q;
          }));

  // TODO: Use PYBIND11_MAKE_OPAQUE for the following vector types.
  pybind11::bind_vector<std::vector<QuerySample>>(m, "VectorQuerySample");
  pybind11::bind_vector<std::vector<QuerySampleResponse>>(
      m, "VectorQuerySampleResponse");

  m.def("ConstructSUT", &py::ConstructSUT, "Construct the system under test.");
  m.def("DestroySUT", &py::DestroySUT,
        "Destroy the object created by ConstructSUT.");

  m.def("ConstructFastSUT", &py::ConstructFastSUT,
        "Construct the system under test, fast issue query");
  m.def("DestroyFastSUT", &py::DestroyFastSUT,
        "Destroy the object created by ConstructFastSUT.");

  m.def("ConstructQSL", &py::ConstructQSL,
        "Construct the query sample library.");
  m.def("DestroyQSL", &py::DestroyQSL,
        "Destroy the object created by ConstructQSL.");

  m.def("ConstructQDL", &py::ConstructQDL,
        "Construct the query sample library, communicating with the SUT over "
        "the network.");
  m.def("DestroyQDL", &py::DestroyQDL,
        "Destroy the object created by ConstructQDL.");

  m.def("StartTest", &py::StartTest,
        "Run tests on a SUT created by ConstructSUT() with the provided QSL. "
        "Uses default log settings.",
        pybind11::arg("sut"), pybind11::arg("qsl"),
        pybind11::arg("test_settings"),
        pybind11::arg("audit_config_filename") = "audit.config");
  m.def("StartTestWithLogSettings", &py::StartTestWithLogSettings,
        "Run tests on a SUT created by ConstructSUT() with the provided QSL. "
        "Accepts custom log settings.",
        pybind11::arg("sut"), pybind11::arg("qsl"),
        pybind11::arg("test_settings"), pybind11::arg("log_settings"),
        pybind11::arg("audit_config_filename") = "audit.config");
  m.def("QuerySamplesComplete", &py::QuerySamplesComplete,
        "Called by the SUT to indicate that samples from some combination of"
        "IssueQuery calls have finished.",
        pybind11::arg("responses"),
        pybind11::arg("response_cb") = ResponseCallback{});
  m.def("FirstTokenComplete", &py::FirstTokenComplete,
        "Called by the SUT to indicate that tokens from some combination of"
        "IssueQuery calls have finished.",
        pybind11::arg("responses"),
        pybind11::arg("response_cb") = ResponseCallback{});
}

}  // namespace py
}  // namespace mlperf

#endif  // PYTHON_BINDINGS_H
