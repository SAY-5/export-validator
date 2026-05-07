// CLI entry point: reads two .evl1 files (one for PyTorch captures, one for
// ONNX Runtime captures), a layer-order text file, and emits the per-layer
// divergence report as JSON on stdout.
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "export_validator/differ.h"
#include "export_validator/json_writer.h"
#include "export_validator/npz_loader.h"

namespace {

struct Args {
    std::string pt_path;
    std::string ort_path;
    std::string layers_path;
    std::string model = "model";
    double tolerance = 1e-4;
};

[[noreturn]] void die(const std::string& msg) {
    std::cerr << "export_validator_compare: " << msg << "\n";
    std::cerr <<
        "usage: export_validator_compare --pt FILE --ort FILE --layers FILE "
        "[--model NAME] [--tolerance F]\n";
    std::exit(2);
}

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto need = [&](const std::string& flag) -> std::string {
            if (i + 1 >= argc) die("missing value for " + flag);
            return argv[++i];
        };
        if (k == "--pt") a.pt_path = need(k);
        else if (k == "--ort") a.ort_path = need(k);
        else if (k == "--layers") a.layers_path = need(k);
        else if (k == "--model") a.model = need(k);
        else if (k == "--tolerance") a.tolerance = std::stod(need(k));
        else die("unknown arg: " + k);
    }
    if (a.pt_path.empty() || a.ort_path.empty() || a.layers_path.empty()) {
        die("--pt, --ort, --layers are required");
    }
    return a;
}

std::vector<std::string> load_layer_order(const std::string& path) {
    std::ifstream in(path);
    if (!in) die("cannot open layer list: " + path);
    std::vector<std::string> out;
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty()) out.push_back(line);
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        auto args = parse_args(argc, argv);
        auto pt = export_validator::load_layers(args.pt_path);
        auto ort = export_validator::load_layers(args.ort_path);
        auto order = load_layer_order(args.layers_path);
        auto report = export_validator::compare(pt, ort, order, args.model, args.tolerance);
        std::cout << export_validator::serialize(report);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "export_validator_compare: " << e.what() << "\n";
        return 1;
    }
}
