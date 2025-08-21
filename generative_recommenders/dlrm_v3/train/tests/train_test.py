# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyre-strict
import unittest

from generative_recommenders.common import gpu_unavailable

from generative_recommenders.dlrm_v3.train.train_ranker import main


class DLRMV3TrainTest(unittest.TestCase):
    @unittest.skipIf(*gpu_unavailable)
    def test_e2e(self) -> None:
        main()


if __name__ == "__main__":
    unittest.main()
