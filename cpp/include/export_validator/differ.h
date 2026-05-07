// Core per-layer divergence computation. Pure stdlib.
#pragma once

#include <string>
#include <vector>

#include "export_validator/types.h"

namespace export_validator {

// Compares ``pt`` and ``ort`` layers in the order given by ``layer_order``.
// Layers absent from either side are skipped. The resulting Report mirrors
// the Python compare_python output to the byte.
Report compare(const std::vector<Layer>& pt,
               const std::vector<Layer>& ort_layers,
               const std::vector<std::string>& layer_order,
               const std::string& model,
               double tolerance);

}  // namespace export_validator
