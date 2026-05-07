#include "export_validator/json_writer.h"

#include <cmath>
#include <cstdio>
#include <sstream>
#include <string>

namespace export_validator {

namespace {

std::string escape_string(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 2);
    out.push_back('"');
    for (char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned>(c) & 0xff);
                    out += buf;
                } else {
                    out.push_back(c);
                }
        }
    }
    out.push_back('"');
    return out;
}

// Mirror Python's float repr() for finite values: it emits the shortest
// round-trippable decimal string. Our compare layer has already rounded to
// %.12e, so emitting via Python's float() round-trip is equivalent to using
// Python's repr() on a float that has at most 13 significant digits. Use
// %.17g (sufficient for double round-trip) and strip a trailing zero pad.
std::string format_double(double v) {
    if (std::isnan(v)) return "NaN";
    if (std::isinf(v)) return v < 0 ? "-Infinity" : "Infinity";
    if (v == 0.0) return "0.0";
    // Python's repr emits the shortest string that round-trips. Here, every
    // value we emit was first stringified through %.12e then parsed back, so
    // the IEEE bits are equivalent to a value with no more than 13 sig figs.
    // Format with %.13g, normalize exponent form to match Python's e+NN/e-NN.
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.13g", v);
    std::string s(buf);
    // If no decimal point, no 'e', no 'E', append '.0' to match Python.
    bool has_dot = s.find('.') != std::string::npos;
    bool has_exp = s.find('e') != std::string::npos || s.find('E') != std::string::npos;
    if (!has_dot && !has_exp) {
        s += ".0";
    }
    // Python uses e+NN with at least two exponent digits but no leading zeros
    // beyond that; %.13g typically emits e+05 etc which matches.
    return s;
}

std::string format_int(int64_t v) {
    return std::to_string(v);
}

std::string format_bool(bool v) { return v ? "true" : "false"; }

void emit_indent(std::ostringstream& os, int level) {
    for (int i = 0; i < level; ++i) os << "  ";
}

// Emit a single LayerStat row with sorted keys to match Python's
// json.dumps(sort_keys=True). Sorted: exceeds_tol, layer, max_abs_diff,
// mean_abs_diff, shape.
void emit_layer(std::ostringstream& os, const LayerStat& s, int level) {
    emit_indent(os, level);
    os << "{\n";
    emit_indent(os, level + 1);
    os << "\"exceeds_tol\": " << format_bool(s.exceeds_tol) << ",\n";
    emit_indent(os, level + 1);
    os << "\"layer\": " << escape_string(s.name) << ",\n";
    emit_indent(os, level + 1);
    os << "\"max_abs_diff\": " << format_double(s.max_abs_diff) << ",\n";
    emit_indent(os, level + 1);
    os << "\"mean_abs_diff\": " << format_double(s.mean_abs_diff) << ",\n";
    emit_indent(os, level + 1);
    os << "\"shape\": [";
    for (size_t i = 0; i < s.shape.size(); ++i) {
        if (i) os << ", ";
        os << format_int(s.shape[i]);
    }
    os << "]\n";
    emit_indent(os, level);
    os << "}";
}

}  // namespace

// Python serializes the Report dataclass via dataclasses.asdict + json.dumps
// with indent=2 and sort_keys=True. Top-level sorted keys: drift_origin,
// layers, layers_exceeding, layers_total, model, tolerance.
std::string serialize(const Report& report) {
    std::ostringstream os;
    os << "{\n";
    emit_indent(os, 1);
    if (report.drift_origin.empty()) {
        os << "\"drift_origin\": null,\n";
    } else {
        os << "\"drift_origin\": " << escape_string(report.drift_origin) << ",\n";
    }
    emit_indent(os, 1);
    os << "\"layers\": [";
    if (!report.layers.empty()) {
        os << "\n";
        for (size_t i = 0; i < report.layers.size(); ++i) {
            emit_layer(os, report.layers[i], 2);
            if (i + 1 < report.layers.size()) os << ",";
            os << "\n";
        }
        emit_indent(os, 1);
    }
    os << "],\n";
    emit_indent(os, 1);
    os << "\"layers_exceeding\": " << format_int(report.layers_exceeding) << ",\n";
    emit_indent(os, 1);
    os << "\"layers_total\": " << format_int(report.layers_total) << ",\n";
    emit_indent(os, 1);
    os << "\"model\": " << escape_string(report.model) << ",\n";
    emit_indent(os, 1);
    os << "\"tolerance\": " << format_double(report.tolerance) << "\n";
    os << "}\n";
    return os.str();
}

}  // namespace export_validator
