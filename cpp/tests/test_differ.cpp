#include <gtest/gtest.h>

#include "export_validator/differ.h"

using export_validator::compare;
using export_validator::Layer;
using export_validator::Report;

namespace {

Layer make(const std::string& name, std::vector<int64_t> shape, std::vector<float> data) {
    return Layer{name, std::move(shape), std::move(data)};
}

}  // namespace

TEST(Differ, IdenticalInputsHaveZeroDiff) {
    auto pt = std::vector<Layer>{make("a", {1, 3}, {1.0f, 2.0f, 3.0f})};
    auto ort = std::vector<Layer>{make("a", {1, 3}, {1.0f, 2.0f, 3.0f})};
    auto rpt = compare(pt, ort, {"a"}, "m", 1e-4);
    ASSERT_EQ(rpt.layers.size(), 1u);
    EXPECT_EQ(rpt.layers[0].max_abs_diff, 0.0);
    EXPECT_EQ(rpt.layers[0].mean_abs_diff, 0.0);
    EXPECT_FALSE(rpt.layers[0].exceeds_tol);
    EXPECT_EQ(rpt.layers_total, 1);
    EXPECT_EQ(rpt.layers_exceeding, 0);
    EXPECT_TRUE(rpt.drift_origin.empty());
}

TEST(Differ, FirstViolatorBecomesDriftOrigin) {
    auto pt = std::vector<Layer>{
        make("conv1", {2}, {0.0f, 0.0f}),
        make("bn1",   {2}, {1.0f, 1.0f}),
        make("relu",  {2}, {0.0f, 0.0f}),
    };
    auto ort = std::vector<Layer>{
        make("conv1", {2}, {0.0f, 0.00001f}),
        make("bn1",   {2}, {2.0f, 1.5f}),
        make("relu",  {2}, {0.0f, 0.5f}),
    };
    auto rpt = compare(pt, ort, {"conv1", "bn1", "relu"}, "m", 1e-4);
    EXPECT_EQ(rpt.drift_origin, "bn1");
    EXPECT_EQ(rpt.layers_exceeding, 2);
    EXPECT_FALSE(rpt.layers[0].exceeds_tol);
    EXPECT_TRUE(rpt.layers[1].exceeds_tol);
    EXPECT_TRUE(rpt.layers[2].exceeds_tol);
}

TEST(Differ, MissingLayerInOrtIsSkipped) {
    auto pt = std::vector<Layer>{
        make("a", {1}, {1.0f}),
        make("b", {1}, {2.0f}),
    };
    auto ort = std::vector<Layer>{make("a", {1}, {1.0f})};
    auto rpt = compare(pt, ort, {"a", "b"}, "m", 1e-4);
    EXPECT_EQ(rpt.layers_total, 1);
    EXPECT_EQ(rpt.layers[0].name, "a");
}

TEST(Differ, ShapeMismatchThrows) {
    auto pt = std::vector<Layer>{make("a", {1, 2}, {0.0f, 0.0f})};
    auto ort = std::vector<Layer>{make("a", {2}, {0.0f, 0.0f})};
    EXPECT_THROW(compare(pt, ort, {"a"}, "m", 1e-4), std::runtime_error);
}

TEST(Differ, ToleranceIsStrictlyGreater) {
    // Diff exactly == tolerance must NOT be flagged (matches Python semantics:
    // exceeds = max_abs_diff > tolerance).
    auto pt = std::vector<Layer>{make("a", {1}, {0.0f})};
    auto ort = std::vector<Layer>{make("a", {1}, {1e-4f})};
    auto rpt = compare(pt, ort, {"a"}, "m", 1e-4);
    EXPECT_FALSE(rpt.layers[0].exceeds_tol);
    EXPECT_EQ(rpt.layers_exceeding, 0);
}
