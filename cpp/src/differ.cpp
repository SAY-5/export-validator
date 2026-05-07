#include "export_validator/differ.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <unordered_map>

namespace export_validator {

namespace {

// Mirror Python's _round() in compare.py: float -> string with %.12e then
// back to double. This makes the rounded value byte-identical between
// the Python report and the C++ report.
double round_to_precision(double value) {
    if (!std::isfinite(value)) {
        return value;
    }
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.12e", value);
    double rounded = 0.0;
    std::sscanf(buf, "%lf", &rounded);
    return rounded;
}

}  // namespace

Report compare(const std::vector<Layer>& pt,
               const std::vector<Layer>& ort_layers,
               const std::vector<std::string>& layer_order,
               const std::string& model,
               double tolerance) {
    std::unordered_map<std::string, const Layer*> pt_idx;
    std::unordered_map<std::string, const Layer*> ort_idx;
    for (const auto& l : pt) pt_idx[l.name] = &l;
    for (const auto& l : ort_layers) ort_idx[l.name] = &l;

    Report rpt;
    rpt.model = model;
    rpt.tolerance = tolerance;
    rpt.layers_total = 0;
    rpt.layers_exceeding = 0;

    for (const auto& name : layer_order) {
        auto pi = pt_idx.find(name);
        auto oi = ort_idx.find(name);
        if (pi == pt_idx.end() || oi == ort_idx.end()) {
            continue;
        }
        const auto& a = *pi->second;
        const auto& b = *oi->second;
        if (a.shape != b.shape) {
            throw std::runtime_error("shape mismatch for " + name);
        }
        if (a.data.size() != b.data.size()) {
            throw std::runtime_error("size mismatch for " + name);
        }
        double max_abs = 0.0;
        long double sum_abs = 0.0L;
        const auto n = a.data.size();
        for (size_t i = 0; i < n; ++i) {
            double d = std::fabs(static_cast<double>(a.data[i]) -
                                 static_cast<double>(b.data[i]));
            if (d > max_abs) max_abs = d;
            sum_abs += d;
        }
        double mean_abs = n == 0 ? 0.0 : static_cast<double>(sum_abs / static_cast<long double>(n));
        max_abs = round_to_precision(max_abs);
        mean_abs = round_to_precision(mean_abs);
        bool exceeds = max_abs > tolerance;

        LayerStat stat{name, a.shape, max_abs, mean_abs, exceeds};
        rpt.layers.push_back(std::move(stat));
        ++rpt.layers_total;
        if (exceeds) {
            ++rpt.layers_exceeding;
            if (rpt.drift_origin.empty()) {
                rpt.drift_origin = name;
            }
        }
    }
    return rpt;
}

}  // namespace export_validator
