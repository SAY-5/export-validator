// Shared C++ types for the comparator.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace export_validator {

struct Layer {
    std::string name;
    std::vector<int64_t> shape;
    std::vector<float> data;  // always float32 in our binary format
};

struct LayerStat {
    std::string name;
    std::vector<int64_t> shape;
    double max_abs_diff;
    double mean_abs_diff;
    bool exceeds_tol;
};

struct Report {
    std::string model;
    double tolerance;
    std::vector<LayerStat> layers;
    std::string drift_origin;  // empty == none
    int64_t layers_total;
    int64_t layers_exceeding;
};

}  // namespace export_validator
