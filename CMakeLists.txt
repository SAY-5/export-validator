# Top-level CMake. Forwards into cpp/ so projects opening the repo root still
# resolve a configurable build tree. Most callers should use cpp/ directly.
cmake_minimum_required(VERSION 3.20)
project(export_validator)

add_subdirectory(cpp)
