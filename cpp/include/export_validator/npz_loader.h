// Loader for the simple per-layer binary format emitted by the Python side.
//
// Format (little-endian):
//   magic[4]       = "EVL1"
//   version u32    = 1
//   n_layers u32
//   for each layer:
//     name_len u32, name bytes
//     ndim u32, dims[ndim] i64
//     dtype u32 (0 = float32; only float32 is supported)
//     element_count u64
//     data: element_count * sizeof(float)
//
// The format is deliberately minimal so the C++ comparator depends on nothing
// beyond the standard library.
#pragma once

#include <filesystem>
#include <vector>

#include "export_validator/types.h"

namespace export_validator {

// Throws std::runtime_error on malformed input or io error.
std::vector<Layer> load_layers(const std::filesystem::path& path);

}  // namespace export_validator
