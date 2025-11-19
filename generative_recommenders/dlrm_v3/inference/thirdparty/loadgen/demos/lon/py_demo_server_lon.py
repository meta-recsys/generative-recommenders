# Copyright 2019 The MLPerf Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

"""
Python demo showing how to use the MLPerf Inference LoadGen over the Network bindings.
This programs runs in the LON Node side.
It runs the demo in MLPerf server mode over the network.
It communicates over the network with a Network SUT node,
which is running the Network SUT demo based on a flask server, implemented in SUT_over_network.py
"""

import threading
import requests
import array
import time

from absl import app
from absl import flags
import mlperf_loadgen

FLAGS = flags.FLAGS

flags.DEFINE_list(
    "sut_server", "http://localhost:8000", "Address of the server(s) under test."
)


class QSL:
    """Demo QuerySampleLibrary with dummy features."""

    def __init__(self, total_sample_count, performance_sample_count):
        self.eval_features = {
            i: f"what_is_my_dummy_feature_{i}?" for i in range(total_sample_count)
        }
        self.qsl = mlperf_loadgen.ConstructQSL(
            total_sample_count,
            performance_sample_count,
            self.load_samples_to_ram,
            self.unload_samples_from_ram,
        )

    def get_features(self, sample_id):
        """Returns the feature for a given sample id."""
        return self.eval_features[sample_id]

    def load_samples_to_ram(self, query_samples):
        """Loads the features for the given query samples into RAM."""
        # Current implementation is not using this functionality.
        del query_samples
        return

    def unload_samples_from_ram(self, query_samples):
        """Unloads the features for the given query samples from RAM."""
        # Current implementation is not using this functionality.
        del query_samples
        return

    def __del__(self):
        mlperf_loadgen.DestroyQSL(self.qsl)


class QDL:
    """QDL acting as a proxy to the SUT.
    This QDL communicates with the SUT via HTTP.
    It uses two endpoints to communicate with the SUT:
    - /predict/ : Send a query to the SUT and get a response.
    - /getname/ : Get the name of the SUT. Send a getname to the SUT and get a response.
    """

    def __init__(self, qsl: QSL, sut_server_addr: list):
        """
        Constructor for the QDL.
        Args:
            qsl: The QSL to use.
            sut_server_addr: A list of addresses of the SUT.
        """
        self.qsl = qsl

        # Construct QDL from the python binding
        self.qdl = mlperf_loadgen.ConstructQDL(
            self.issue_query, self.flush_queries, self.client_get_name
        )
        self.sut_server_addr = sut_server_addr
        self.num_nodes = len(sut_server_addr)

        # For round robin between the SUTs:
        self.next_sut_id = 0
        self.lock = threading.Lock()

    def issue_query(self, query_samples):
        """Process the query to send to the SUT"""
        threading.Thread(
            target=self.process_query_async,
            args=[query_samples]).start()

    def flush_queries(self):
        """Flush the queries. Dummy implementation."""
        pass

    def process_query_async(self, query_samples):
        """
        This function is called by the Loadgen in a separate thread.
        It is responsible for
            1. Creating a query for the SUT, by reading the features from the QSL.
            2. Sending the query to the SUT.
            3. Waiting for the response from the SUT.
            4. Deserializing the response.
            5. Calling mlperf_loadgen.QuerySamplesComplete(query_samples, response)
        Args:
            query_samples: A list of QuerySample objects.
        """
        responses = []
        for s in query_samples:
            # Overall process:
            # QDL builds a real-world query and sends to SUT --> SUT processes --> SUT sends back to QDL
            # Read features from the QSL
            features = self.qsl.get_features(s.index)

            time.sleep(0.001)  # Ensure a maximal rate of queries to the SUT

            # Send the query to SUT in round robin
            # Wait for a response
            sut_result = self.client_predict(features, s.index)
            response_array = array.array("B", sut_result.encode("utf-8"))
            bi = response_array.buffer_info()
            responses.append(
                mlperf_loadgen.QuerySampleResponse(
                    s.id, bi[0], bi[1]))
        mlperf_loadgen.QuerySamplesComplete(responses)

    def get_sut_id_round_robin(self):
        """Get the SUT id in round robin."""
        with self.lock:
            res = self.next_sut_id
            self.next_sut_id = (self.next_sut_id + 1) % self.num_nodes
        return res

    def client_predict(self, query, id):
        """Serialize the query, send it to the SUT in round robin, and return the deserialized response."""
        url = "{}/predict/".format(
            self.sut_server_addr[self.get_sut_id_round_robin()])
        response = requests.post(url, json={"query": query, id: id})
        return response.json()["result"]

    def client_get_name(self):
        """Get the name of the SUT from ALL the SUTS."""
        if len(self.sut_server_addr) == 1:
            return requests.post(
                f"{self.sut_server_addr[0]}/getname/").json()["name"]

        sut_names = [
            requests.post(f"{addr}/getname/").json()["name"]
            for addr in self.sut_server_addr
        ]
        return "Multi-node SUT: " + ", ".join(sut_names)

    def __del__(self):
        mlperf_loadgen.DestroyQDL(self.qdl)


def main(argv):
    del argv
    settings = mlperf_loadgen.TestSettings()
    settings.scenario = mlperf_loadgen.TestScenario.Server
    settings.mode = mlperf_loadgen.TestMode.PerformanceOnly
    settings.server_target_qps = 100
    settings.server_target_latency_ns = 100000000
    settings.min_query_count = 100
    settings.min_duration_ms = 10000

    # QDL and QSL
    qsl = QSL(1024, 128)
    qdl = QDL(qsl, sut_server_addr=FLAGS.sut_server)

    mlperf_loadgen.StartTest(qdl.qdl, qsl.qsl, settings)


if __name__ == "__main__":
    app.run(main)
