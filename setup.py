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

from setuptools import find_packages, setup

setup(
    name="generative_recommenders",
    version="0.1.0",
    description="Library for generative recommendation algorithms.",
    packages=find_packages(exclude=["configs"]),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.6.0",
        "fbgemm_gpu>=1.1.0",
        "torchrec>=1.1.0",
        "gin_config>=0.5.0",
        "pandas>=2.2.0",
        "tensorboard>=2.19.0",
        "pybind11",
        "click",
        "pandas",
        "matplotlib",
    ],
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/meta-recsys/generative-recommenders",
    license="Apache-2.0",
)
