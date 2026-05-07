// Smoke binary that loads libtorch and onnxruntime and prints their version
// strings. Built only when both deps are available, so the standard CMake
// configure proves the link line works on this host.
#include <iostream>

#include <onnxruntime_cxx_api.h>
#include <torch/torch.h>

int main() {
    std::cout << "libtorch: " << TORCH_VERSION << "\n";
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "export-validator-smoke");
    auto api = Ort::GetApi();
    const char* version = OrtGetApiBase()->GetVersionString();
    std::cout << "onnxruntime: " << version << "\n";
    (void)api;
    auto t = torch::ones({2, 2});
    std::cout << "torch::ones({2,2}).sum() = " << t.sum().item<float>() << "\n";
    return 0;
}
