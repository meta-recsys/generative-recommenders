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

# \file
#  \brief MLPerf Inference LoadGen python module setup.
#  \details Creates a module that python can import.
#  All source files are compiled by python"s C++ toolchain  without depending
#  on a loadgen lib.
#
#  This setup.py can be used stand-alone, without the use of an external
#  build system. This will polute your source tree with output files
#  and binaries. Use one of the gn build targets instead if you want
#  to avoid poluting the source tree.

from setuptools import Extension, setup
from pathlib import Path
from pybind11 import get_include
from pybind11.setup_helpers import Pybind11Extension, build_ext
from version_generator import generate_loadgen_version_definitions
import subprocess

generated_version_source_filename = "generated/version_generated.cc"
generate_loadgen_version_definitions(generated_version_source_filename, ".")

public_headers = [
    "loadgen.h",
    "query_sample.h",
    "query_sample_library.h",
    "system_under_test.h",
    "test_settings.h",
    "issue_query_controller.h",
    "early_stopping.h",
    "query_dispatch_library.h"
]

lib_headers = [
    "logging.h",
    "test_settings_internal.h",
    "trace_generator.h",
    "utils.h",
    "version.h",
    "results.h",
    "bindings/c_api.h",
    "version_generator.py",
    "mlperf_conf.h"
]

lib_sources = [
    "early_stopping.cc",
    "issue_query_controller.cc",
    "loadgen.cc",
    "logging.cc",
    "test_settings_internal.cc",
    "utils.cc",
    "version.cc",
    "results.cc",
]

lib_bindings = [
    "bindings/c_api.cc",
    "bindings/python_api.cc",
]

this_directory = Path(__file__).parent
mlperf_loadgen_headers = public_headers + lib_headers
mlperf_loadgen_sources_no_gen = lib_sources + lib_bindings
mlperf_loadgen_sources = mlperf_loadgen_sources_no_gen + [
    generated_version_source_filename
]
mlperf_long_description = (
    this_directory /
    "README.md").read_text(
        encoding="utf-8")

with open("VERSION.txt", "r") as f:
    version = f.read()
version_split = version.split(".")

if len(version_split) < 2:
    print("Version is incomplete. Needs a format like 4.1.1 in VERSION file")


try:
    with open("mlperf.conf", 'r') as file:
        conf_contents = file.read()

    # Escape backslashes and double quotes
    conf_contents = conf_contents.replace('\\', '\\\\').replace('"', '\\"')

    # Convert newlines
    conf_contents = conf_contents.replace('\n', '\\n"\n"')

    formatted_content = f'const char* mlperf_conf =\n"{conf_contents}";\n'

    with open("mlperf_conf.h", 'w') as header_file:
        header_file.write(formatted_content)

except IOError as e:
    raise RuntimeError(f"Failed to generate header file: {e}")

mlperf_loadgen_module = Pybind11Extension(
    "mlperf_loadgen",
    define_macros=[
        ("MAJOR_VERSION",
         version_split[0]),
        ("MINOR_VERSION",
         version_split[1])
    ],
    include_dirs=[".", get_include()],
    sources=mlperf_loadgen_sources,
    depends=mlperf_loadgen_headers,
)

setup(name="mlcommons_loadgen",
      version=version,
      description="MLPerf Inference LoadGen python bindings",
      url="https://mlcommons.org/",
      cmdclass={"build_ext": build_ext},
      ext_modules=[mlperf_loadgen_module],
      packages=['mlcommons_loadgen'],
      package_dir={'mlcommons_loadgen': '.'},
      include_package_data=True,
      long_description=mlperf_long_description,
      long_description_content_type='text/markdown')
