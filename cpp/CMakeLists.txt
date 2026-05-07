cmake_minimum_required(VERSION 3.20)
project(export_validator_cpp CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-Wall -Wextra -Wpedantic -Wno-error)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Find Torch via the bundled cmake config in the python torch wheel.
# The caller passes -DTORCH_DIR=$(python -c 'import torch,os; print(os.path.dirname(torch.__file__))')
# and we wire CMAKE_PREFIX_PATH to torch/share/cmake.
if(DEFINED TORCH_DIR)
  list(APPEND CMAKE_PREFIX_PATH "${TORCH_DIR}/share/cmake")
endif()
find_package(Torch QUIET)

# Find onnxruntime headers + library. Standard install layout: include/ + lib/.
# Caller passes -DONNXRUNTIME_DIR=/path/to/onnxruntime-release
if(DEFINED ONNXRUNTIME_DIR)
  set(ORT_INCLUDE_DIR "${ONNXRUNTIME_DIR}/include")
  find_library(ORT_LIB onnxruntime PATHS "${ONNXRUNTIME_DIR}/lib" NO_DEFAULT_PATH)
endif()

# The comparator core does not depend on libtorch or onnxruntime; that lets us
# build and test the math/JSON path even when those native deps are missing.
add_library(export_validator_core STATIC
  src/differ.cpp
  src/npz_loader.cpp
  src/json_writer.cpp
)
target_include_directories(export_validator_core PUBLIC include)

add_executable(export_validator_compare src/main.cpp)
target_link_libraries(export_validator_compare PRIVATE export_validator_core)

# Optional libtorch / onnxruntime smoke binary. Only built when both are
# present. The smoke binary loads the runtimes and prints version strings;
# this proves the linker line works on this host without requiring a real
# resnet18 round trip in CI.
if(Torch_FOUND AND ORT_LIB AND ORT_INCLUDE_DIR)
  add_executable(export_validator_runtimes src/runtimes_smoke.cpp)
  target_include_directories(export_validator_runtimes PRIVATE "${ORT_INCLUDE_DIR}")
  target_link_libraries(export_validator_runtimes PRIVATE ${TORCH_LIBRARIES} ${ORT_LIB})
  set_property(TARGET export_validator_runtimes PROPERTY CXX_STANDARD 20)
  message(STATUS "runtimes smoke target enabled (libtorch + onnxruntime)")
else()
  message(STATUS "runtimes smoke target skipped (Torch_FOUND=${Torch_FOUND} ORT_LIB=${ORT_LIB})")
endif()

enable_testing()
add_subdirectory(tests)
