#include <gtest/gtest.h>

#include "export_validator/json_writer.h"

using export_validator::LayerStat;
using export_validator::Report;
using export_validator::serialize;

TEST(JsonWriter, EmptyReportIsStable) {
    Report rpt;
    rpt.model = "m";
    rpt.tolerance = 1e-4;
    rpt.layers_total = 0;
    rpt.layers_exceeding = 0;
    auto out = serialize(rpt);
    // drift_origin null, layers [], scalar fields sorted alphabetically.
    EXPECT_NE(out.find("\"drift_origin\": null"), std::string::npos);
    EXPECT_NE(out.find("\"layers\": []"), std::string::npos);
    EXPECT_NE(out.find("\"layers_exceeding\": 0"), std::string::npos);
    EXPECT_NE(out.find("\"layers_total\": 0"), std::string::npos);
    EXPECT_NE(out.find("\"model\": \"m\""), std::string::npos);
}

TEST(JsonWriter, KeysAreSortedAlphabeticallyAtTopAndPerRow) {
    Report rpt;
    rpt.model = "m";
    rpt.tolerance = 0.001;
    rpt.layers_total = 1;
    rpt.layers_exceeding = 1;
    rpt.drift_origin = "bn1";
    rpt.layers.push_back({"bn1", {1, 2}, 0.5, 0.25, true});
    auto out = serialize(rpt);
    // Top-level: drift_origin < layers < layers_exceeding < layers_total < model < tolerance.
    auto pos_drift = out.find("\"drift_origin\"");
    auto pos_layers = out.find("\"layers\"");
    auto pos_lex = out.find("\"layers_exceeding\"");
    auto pos_ltot = out.find("\"layers_total\"");
    auto pos_model = out.find("\"model\"");
    auto pos_tol = out.find("\"tolerance\"");
    EXPECT_LT(pos_drift, pos_layers);
    EXPECT_LT(pos_layers, pos_lex);
    EXPECT_LT(pos_lex, pos_ltot);
    EXPECT_LT(pos_ltot, pos_model);
    EXPECT_LT(pos_model, pos_tol);

    // Per-row: exceeds_tol < layer < max_abs_diff < mean_abs_diff < shape.
    auto pos_exceeds = out.find("\"exceeds_tol\"");
    auto pos_layer = out.find("\"layer\"");
    auto pos_max = out.find("\"max_abs_diff\"");
    auto pos_mean = out.find("\"mean_abs_diff\"");
    auto pos_shape = out.find("\"shape\"");
    EXPECT_LT(pos_exceeds, pos_layer);
    EXPECT_LT(pos_layer, pos_max);
    EXPECT_LT(pos_max, pos_mean);
    EXPECT_LT(pos_mean, pos_shape);
}

TEST(JsonWriter, DriftOriginIsStringWhenSet) {
    Report rpt;
    rpt.model = "m";
    rpt.tolerance = 1e-4;
    rpt.drift_origin = "layer4.0.bn2";
    auto out = serialize(rpt);
    EXPECT_NE(out.find("\"drift_origin\": \"layer4.0.bn2\""), std::string::npos);
}
