#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>

#include <gtest/gtest.h>

#include "export_validator/npz_loader.h"

namespace fs = std::filesystem;
using export_validator::load_layers;

namespace {

void write_le_u32(std::ostream& os, uint32_t v) { os.write(reinterpret_cast<const char*>(&v), 4); }
void write_le_u64(std::ostream& os, uint64_t v) { os.write(reinterpret_cast<const char*>(&v), 8); }
void write_le_i64(std::ostream& os, int64_t v)  { os.write(reinterpret_cast<const char*>(&v), 8); }

fs::path make_tmp(const std::string& tag) {
    auto p = fs::temp_directory_path() / ("evl1_" + tag + ".bin");
    return p;
}

}  // namespace

TEST(NpzLoader, RoundTripsTwoLayers) {
    auto path = make_tmp("rt");
    {
        std::ofstream out(path, std::ios::binary);
        out.write("EVL1", 4);
        write_le_u32(out, 1);  // version
        write_le_u32(out, 2);  // n_layers
        // layer 0: name "a", shape [2], data [1.0, 2.0]
        std::string n0 = "a";
        write_le_u32(out, static_cast<uint32_t>(n0.size()));
        out.write(n0.data(), n0.size());
        write_le_u32(out, 1);          // ndim
        write_le_i64(out, 2);          // dim 0
        write_le_u32(out, 0);          // dtype = float32
        write_le_u64(out, 2);          // count
        float d0[] = {1.0f, 2.0f};
        out.write(reinterpret_cast<const char*>(d0), sizeof(d0));
        // layer 1: name "bb", shape [1, 2], data [3, 4]
        std::string n1 = "bb";
        write_le_u32(out, static_cast<uint32_t>(n1.size()));
        out.write(n1.data(), n1.size());
        write_le_u32(out, 2);
        write_le_i64(out, 1);
        write_le_i64(out, 2);
        write_le_u32(out, 0);
        write_le_u64(out, 2);
        float d1[] = {3.0f, 4.0f};
        out.write(reinterpret_cast<const char*>(d1), sizeof(d1));
    }
    auto loaded = load_layers(path);
    ASSERT_EQ(loaded.size(), 2u);
    EXPECT_EQ(loaded[0].name, "a");
    EXPECT_EQ(loaded[0].shape, (std::vector<int64_t>{2}));
    EXPECT_FLOAT_EQ(loaded[0].data[0], 1.0f);
    EXPECT_FLOAT_EQ(loaded[0].data[1], 2.0f);
    EXPECT_EQ(loaded[1].name, "bb");
    EXPECT_EQ(loaded[1].shape, (std::vector<int64_t>{1, 2}));
    fs::remove(path);
}

TEST(NpzLoader, RejectsBadMagic) {
    auto path = make_tmp("bad");
    {
        std::ofstream out(path, std::ios::binary);
        out.write("XXXX", 4);
        write_le_u32(out, 1);
        write_le_u32(out, 0);
    }
    EXPECT_THROW(load_layers(path), std::runtime_error);
    fs::remove(path);
}
