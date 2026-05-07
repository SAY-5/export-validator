#include "export_validator/npz_loader.h"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace export_validator {

namespace {

template <typename T>
T read_le(std::istream& in) {
    T value{};
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!in) {
        throw std::runtime_error("npz_loader: short read");
    }
    return value;
}

}  // namespace

std::vector<Layer> load_layers(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("cannot open " + path.string());
    }
    char magic[4]{};
    in.read(magic, 4);
    if (std::memcmp(magic, "EVL1", 4) != 0) {
        throw std::runtime_error("bad magic in " + path.string());
    }
    auto version = read_le<uint32_t>(in);
    if (version != 1) {
        throw std::runtime_error("unsupported version");
    }
    auto n_layers = read_le<uint32_t>(in);
    std::vector<Layer> out;
    out.reserve(n_layers);
    for (uint32_t i = 0; i < n_layers; ++i) {
        Layer layer;
        auto name_len = read_le<uint32_t>(in);
        layer.name.resize(name_len);
        in.read(layer.name.data(), name_len);
        auto ndim = read_le<uint32_t>(in);
        layer.shape.resize(ndim);
        for (uint32_t d = 0; d < ndim; ++d) {
            layer.shape[d] = read_le<int64_t>(in);
        }
        auto dtype = read_le<uint32_t>(in);
        if (dtype != 0) {
            throw std::runtime_error("only float32 supported");
        }
        auto count = read_le<uint64_t>(in);
        layer.data.resize(count);
        in.read(reinterpret_cast<char*>(layer.data.data()),
                static_cast<std::streamsize>(count * sizeof(float)));
        if (!in) {
            throw std::runtime_error("short read in layer " + layer.name);
        }
        out.push_back(std::move(layer));
    }
    return out;
}

}  // namespace export_validator
