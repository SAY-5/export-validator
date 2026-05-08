// Randomised property tests for the JSON writer.
//
// We generate Report structs with varying drift_origin/layers/shapes, run the
// writer, then run a minimal in-test parser that knows exactly the shape the
// writer emits, reconstruct an equivalent Report, and re-serialize. Two
// contracts are asserted:
//
//   1. Every emitted document parses cleanly back into the same logical
//      Report (no trailing junk, no missing fields, key order respected).
//   2. Serializing the parsed-back Report is byte-identical to the original
//      output. This is the "random JSON serialization round-trip → byte-
//      identical output" contract: idempotence of (parse . serialize).
//
// The parser is intentionally minimal — it expects the exact format the
// writer produces (`json.dumps(..., indent=2, sort_keys=True)` shape). It
// is not a general-purpose JSON parser; if the writer changes its layout,
// these tests must be updated alongside the writer.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "export_validator/json_writer.h"
#include "export_validator/types.h"

using export_validator::Layer;
using export_validator::LayerStat;
using export_validator::Report;
using export_validator::serialize;

namespace {

// --- minimal parser tied to the writer's layout -----------------------------

struct Cursor {
    const std::string& s;
    size_t pos = 0;
    explicit Cursor(const std::string& str) : s(str) {}

    bool eof() const { return pos >= s.size(); }
    char peek() const { return s[pos]; }
    char advance() { return s[pos++]; }

    void skip_ws() {
        while (!eof() && (s[pos] == ' ' || s[pos] == '\n' || s[pos] == '\t' || s[pos] == '\r')) {
            ++pos;
        }
    }
    void expect(char c) {
        skip_ws();
        if (eof() || s[pos] != c) {
            throw std::runtime_error(std::string("expected '") + c + "' at pos " +
                                     std::to_string(pos));
        }
        ++pos;
    }
    void expect(const std::string& lit) {
        skip_ws();
        if (s.compare(pos, lit.size(), lit) != 0) {
            throw std::runtime_error("expected literal '" + lit + "' at pos " +
                                     std::to_string(pos));
        }
        pos += lit.size();
    }

    std::string parse_string() {
        skip_ws();
        if (advance() != '"') throw std::runtime_error("expected string");
        std::string out;
        while (!eof()) {
            char c = advance();
            if (c == '"') return out;
            if (c == '\\') {
                char e = advance();
                switch (e) {
                    case '"': out.push_back('"'); break;
                    case '\\': out.push_back('\\'); break;
                    case 'n': out.push_back('\n'); break;
                    case 'r': out.push_back('\r'); break;
                    case 't': out.push_back('\t'); break;
                    case 'b': out.push_back('\b'); break;
                    case 'f': out.push_back('\f'); break;
                    default: throw std::runtime_error("unsupported escape");
                }
            } else {
                out.push_back(c);
            }
        }
        throw std::runtime_error("unterminated string");
    }

    double parse_number() {
        skip_ws();
        size_t start = pos;
        if (peek() == '-') ++pos;
        while (!eof()) {
            char c = peek();
            if ((c >= '0' && c <= '9') || c == '.' || c == 'e' || c == 'E' ||
                c == '+' || c == '-') {
                ++pos;
            } else {
                break;
            }
        }
        std::string num = s.substr(start, pos - start);
        return std::strtod(num.c_str(), nullptr);
    }

    int64_t parse_int() {
        skip_ws();
        size_t start = pos;
        if (peek() == '-') ++pos;
        while (!eof() && peek() >= '0' && peek() <= '9') ++pos;
        std::string num = s.substr(start, pos - start);
        return std::strtoll(num.c_str(), nullptr, 10);
    }

    bool parse_bool() {
        skip_ws();
        if (s.compare(pos, 4, "true") == 0) {
            pos += 4;
            return true;
        }
        if (s.compare(pos, 5, "false") == 0) {
            pos += 5;
            return false;
        }
        throw std::runtime_error("expected bool");
    }

    bool consume_null_or(char delim) {
        // Returns true if the next non-ws token is `null`. Otherwise leaves
        // the cursor at the first non-ws byte.
        skip_ws();
        if (s.compare(pos, 4, "null") == 0) {
            pos += 4;
            return true;
        }
        return false;
        (void)delim;
    }
};

Report parse_report(const std::string& blob) {
    Cursor c(blob);
    Report rpt;
    c.expect('{');
    // drift_origin: null | string
    c.expect("\"drift_origin\":");
    if (!c.consume_null_or(',')) {
        rpt.drift_origin = c.parse_string();
    }
    c.expect(',');
    // layers: [ ... ]
    c.expect("\"layers\":");
    c.skip_ws();
    c.expect('[');
    c.skip_ws();
    while (c.peek() != ']') {
        c.expect('{');
        LayerStat stat;
        c.expect("\"exceeds_tol\":");
        stat.exceeds_tol = c.parse_bool();
        c.expect(',');
        c.expect("\"layer\":");
        stat.name = c.parse_string();
        c.expect(',');
        c.expect("\"max_abs_diff\":");
        stat.max_abs_diff = c.parse_number();
        c.expect(',');
        c.expect("\"mean_abs_diff\":");
        stat.mean_abs_diff = c.parse_number();
        c.expect(',');
        c.expect("\"shape\":");
        c.skip_ws();
        c.expect('[');
        c.skip_ws();
        while (c.peek() != ']') {
            stat.shape.push_back(c.parse_int());
            c.skip_ws();
            if (c.peek() == ',') {
                c.advance();
                c.skip_ws();
            }
        }
        c.expect(']');
        c.expect('}');
        rpt.layers.push_back(stat);
        c.skip_ws();
        if (c.peek() == ',') {
            c.advance();
            c.skip_ws();
        }
    }
    c.expect(']');
    c.expect(',');
    c.expect("\"layers_exceeding\":");
    rpt.layers_exceeding = c.parse_int();
    c.expect(',');
    c.expect("\"layers_total\":");
    rpt.layers_total = c.parse_int();
    c.expect(',');
    c.expect("\"model\":");
    rpt.model = c.parse_string();
    c.expect(',');
    c.expect("\"tolerance\":");
    rpt.tolerance = c.parse_number();
    c.skip_ws();
    c.expect('}');
    return rpt;
}

// --- generators -------------------------------------------------------------

LayerStat random_layer(std::mt19937& rng) {
    static const char* const layer_alphabet =
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._";
    static const size_t alphabet_n = std::strlen(layer_alphabet);
    LayerStat s;
    std::uniform_int_distribution<int> name_len(1, 14);
    std::uniform_int_distribution<size_t> name_idx(0, alphabet_n - 1);
    int n = name_len(rng);
    s.name.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        s.name.push_back(layer_alphabet[name_idx(rng)]);
    }
    std::uniform_int_distribution<int> ndim(1, 4);
    int nd = ndim(rng);
    std::uniform_int_distribution<int> dim(1, 64);
    for (int i = 0; i < nd; ++i) {
        s.shape.push_back(dim(rng));
    }
    // Use values within the "rounded by %.12e" envelope so the round-trip
    // is exact; 1e-12..1e3 covers typical activation drift magnitudes.
    std::uniform_real_distribution<double> mag_log(-12.0, 3.0);
    auto sample_pos = [&]() {
        double v = std::pow(10.0, mag_log(rng));
        // Run through the writer's same %.12e -> sscanf round trip so equality
        // after parse is exact (no surprises from the strtod side of double).
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%.12e", v);
        double r = 0.0;
        std::sscanf(buf, "%lf", &r);
        return r;
    };
    s.max_abs_diff = sample_pos();
    s.mean_abs_diff = sample_pos();
    if (s.mean_abs_diff > s.max_abs_diff) {
        std::swap(s.mean_abs_diff, s.max_abs_diff);
    }
    s.exceeds_tol = (rng() % 2) == 0;
    return s;
}

Report random_report(std::mt19937& rng) {
    Report rpt;
    rpt.model = "model_" + std::to_string(rng() % 1000);
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.12e",
                  std::pow(10.0, static_cast<double>(rng() % 12) - 8.0));
    double tol = 0.0;
    std::sscanf(buf, "%lf", &tol);
    rpt.tolerance = tol;
    std::uniform_int_distribution<int> n_layers(0, 6);
    int n = n_layers(rng);
    for (int i = 0; i < n; ++i) {
        rpt.layers.push_back(random_layer(rng));
    }
    rpt.layers_total = static_cast<int64_t>(rpt.layers.size());
    rpt.layers_exceeding = std::count_if(
        rpt.layers.begin(), rpt.layers.end(),
        [](const LayerStat& s) { return s.exceeds_tol; });
    for (const auto& s : rpt.layers) {
        if (s.exceeds_tol) {
            rpt.drift_origin = s.name;
            break;
        }
    }
    return rpt;
}

}  // namespace

TEST(JsonWriterRandom, RoundTripIsByteIdentical) {
    constexpr int kCases = 200;
    std::mt19937 rng(0xC0FFEEu);
    for (int i = 0; i < kCases; ++i) {
        Report rpt = random_report(rng);
        std::string first = serialize(rpt);
        Report parsed = parse_report(first);
        std::string second = serialize(parsed);
        ASSERT_EQ(first, second) << "round-trip diverged on case " << i << "\n--- first ---\n"
                                 << first << "--- second ---\n"
                                 << second;
    }
}

TEST(JsonWriterRandom, ParsedReportPreservesAllFields) {
    std::mt19937 rng(0xBADBEEFu);
    for (int i = 0; i < 100; ++i) {
        Report rpt = random_report(rng);
        std::string blob = serialize(rpt);
        Report parsed = parse_report(blob);
        EXPECT_EQ(parsed.model, rpt.model);
        EXPECT_EQ(parsed.tolerance, rpt.tolerance);
        EXPECT_EQ(parsed.drift_origin, rpt.drift_origin);
        EXPECT_EQ(parsed.layers_total, rpt.layers_total);
        EXPECT_EQ(parsed.layers_exceeding, rpt.layers_exceeding);
        ASSERT_EQ(parsed.layers.size(), rpt.layers.size());
        for (size_t j = 0; j < parsed.layers.size(); ++j) {
            EXPECT_EQ(parsed.layers[j].name, rpt.layers[j].name);
            EXPECT_EQ(parsed.layers[j].shape, rpt.layers[j].shape);
            EXPECT_EQ(parsed.layers[j].max_abs_diff, rpt.layers[j].max_abs_diff);
            EXPECT_EQ(parsed.layers[j].mean_abs_diff, rpt.layers[j].mean_abs_diff);
            EXPECT_EQ(parsed.layers[j].exceeds_tol, rpt.layers[j].exceeds_tol);
        }
    }
}

TEST(JsonWriterRandom, EmptyAndSingleLayerExtremes) {
    {
        Report rpt;
        rpt.model = "edge";
        rpt.tolerance = 1e-12;
        rpt.layers_total = 0;
        rpt.layers_exceeding = 0;
        std::string blob = serialize(rpt);
        Report parsed = parse_report(blob);
        EXPECT_EQ(parsed.layers.size(), 0u);
        EXPECT_TRUE(parsed.drift_origin.empty());
        EXPECT_EQ(serialize(parsed), blob);
    }
    {
        Report rpt;
        rpt.model = "edge";
        rpt.tolerance = 1.0;
        rpt.layers_total = 1;
        rpt.layers_exceeding = 1;
        rpt.drift_origin = "only";
        rpt.layers.push_back({"only", {1}, 2.0, 1.5, true});
        std::string blob = serialize(rpt);
        Report parsed = parse_report(blob);
        ASSERT_EQ(parsed.layers.size(), 1u);
        EXPECT_EQ(parsed.layers[0].name, "only");
        EXPECT_EQ(serialize(parsed), blob);
    }
}
