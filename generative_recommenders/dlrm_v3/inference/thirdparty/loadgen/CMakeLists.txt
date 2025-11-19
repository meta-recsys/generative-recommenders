cmake_minimum_required(VERSION 3.12)

project(mlperf_loadgen)

# Read the version file
file(READ "${CMAKE_SOURCE_DIR}/VERSION.txt" VERSION_CONTENTS)

# Extract the major, minor, and patch versions from the VERSION file (assuming "MAJOR.MINOR.PATCH" format)
string(REGEX MATCH "^([0-9]+)\\.([0-9]+)\\.([0-9]+)" VERSION_MATCH ${VERSION_CONTENTS})

# Set the variables for the major, minor, and patch versions
set(mlperf_loadgen_VERSION_MAJOR "${CMAKE_MATCH_1}")
set(mlperf_loadgen_VERSION_MINOR "${CMAKE_MATCH_2}")
set(mlperf_loadgen_VERSION_PATCH "${CMAKE_MATCH_3}")

# Check if the version format was parsed correctly
if(NOT DEFINED mlperf_loadgen_VERSION_MAJOR OR NOT DEFINED mlperf_loadgen_VERSION_MINOR OR NOT DEFINED mlperf_loadgen_VERSION_PATCH)
    message(FATAL_ERROR "Version format in VERSION.txt is incorrect. Expected format: MAJOR.MINOR.PATCH")
endif()

# Print out the version
message("mlperf_loadgen v${mlperf_loadgen_VERSION_MAJOR}.${mlperf_loadgen_VERSION_MINOR}.${mlperf_loadgen_VERSION_PATCH}")

# Set build options. NB: CXX_STANDARD is supported since CMake 3.1.
if (NOT MSVC)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -W -Wall")
endif()
# Extra build options can be specified by setting the MLPERF_LOADGEN_CXX_FLAGS variable
if (MLPERF_LOADGEN_CXX_FLAGS)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MLPERF_LOADGEN_CXX_FLAGS}")
endif()
message(STATUS "Using C++ compiler flags: ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_STANDARD "14")
message(STATUS "Using C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Using static linker flags: ${CMAKE_STATIC_LINKER_FLAGS}")
message(STATUS "Using shared linker flags: ${CMAKE_SHARED_LINKER_FLAGS}")

# Output directory for libraries.
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})
message(STATUS "Using output path: ${LIBRARY_OUTPUT_PATH}")

# Detect Python to use for generating source file with version info.
# NB: PythonInterp has been deprecated since CMake 3.12
# but it works with earlier versions of CMake.
find_package(PythonInterp)
message(STATUS "Using Python interpreter: ${PYTHON_EXECUTABLE}")

# Specify the source and destination files
set(CONF_FILE "mlperf.conf")
set(HEADER_FILE "mlperf_conf.h")

# Read the content of the configuration file
file(READ ${CONF_FILE} CONF_CONTENTS)

# Escape all double quotes and backslashes
string(REPLACE "\\" "\\\\" CONF_CONTENTS "${CONF_CONTENTS}")
string(REPLACE "\"" "\\\"" CONF_CONTENTS "${CONF_CONTENTS}")

# Handle new lines
string(REPLACE "\n" "\\n\"\n\"" CONF_CONTENTS "${CONF_CONTENTS}")

# Wrap the content in a C++ string declaration
set(FORMATTED_CONTENT "const char* mlperf_conf =\n\"${CONF_CONTENTS}\";\n")

# Write the formatted content to the header file
file(WRITE ${HEADER_FILE} "${FORMATTED_CONTENT}")

message(STATUS "Output config:  ${CMAKE_BINARY_DIR}/mlperf_conf.h")

# Generate source file with version info.
execute_process(COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/version_generator.py ${CMAKE_BINARY_DIR}/version_generated.cc ${CMAKE_CURRENT_SOURCE_DIR})

# Add source files.
set(SOURCE
  ${CMAKE_CURRENT_SOURCE_DIR}/bindings/c_api.h
  ${CMAKE_CURRENT_SOURCE_DIR}/bindings/c_api.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/early_stopping.cc  
  ${CMAKE_CURRENT_SOURCE_DIR}/issue_query_controller.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/loadgen.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/logging.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/logging.h
  ${CMAKE_CURRENT_SOURCE_DIR}/test_settings_internal.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/test_settings_internal.h
  ${CMAKE_CURRENT_SOURCE_DIR}/utils.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/utils.h
  ${CMAKE_CURRENT_SOURCE_DIR}/results.h
  ${CMAKE_CURRENT_SOURCE_DIR}/results.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/version.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/version.h
  ${CMAKE_CURRENT_SOURCE_DIR}/mlperf_conf.h
  ${CMAKE_CURRENT_SOURCE_DIR}/VERSION.txt
  ${CMAKE_BINARY_DIR}/version_generated.cc
)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_library(mlperf_loadgen STATIC ${SOURCE})
target_link_libraries(mlperf_loadgen)

if(WIN32)
set (LIBS "")
else()
set (LIBS pthread)
endif()

add_executable(benchmark benchmark/repro.cpp)
target_link_libraries(benchmark PUBLIC mlperf_loadgen ${LIBS})

# Install library and headers.
install(TARGETS mlperf_loadgen
	DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
	DESTINATION ${CMAKE_INSTALL_PREFIX}/include FILES_MATCHING PATTERN "*.h")
