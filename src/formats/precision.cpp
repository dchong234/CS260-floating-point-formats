#include "formats/precision.hpp"

#include <algorithm>
#include <cctype>
#include <unordered_map>

namespace fpstudy::formats {

namespace {

std::string to_lower(std::string_view name) {
    std::string result(name);
    std::transform(result.begin(), result.end(), result.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return result;
}

} // namespace

std::string precision_to_string(Precision p) {
    switch (p) {
        case Precision::FP64: return "fp64";
        case Precision::FP32: return "fp32";
        case Precision::TF32: return "tf32";
        case Precision::BF16: return "bf16";
        case Precision::P3109_8: return "p3109_8";
    }
    throw std::runtime_error("Unknown precision enum");
}

Precision precision_from_string(std::string_view name) {
    static const std::unordered_map<std::string, Precision> map = {
        {"fp64", Precision::FP64},
        {"float64", Precision::FP64},
        {"fp32", Precision::FP32},
        {"float32", Precision::FP32},
        {"tf32", Precision::TF32},
        {"tensorfloat32", Precision::TF32},
        {"bf16", Precision::BF16},
        {"bfloat16", Precision::BF16},
        {"p3109", Precision::P3109_8},
        {"p3109_8", Precision::P3109_8},
    };

    auto lower = to_lower(name);
    auto it = map.find(lower);
    if (it == map.end()) {
        throw std::runtime_error("Unknown precision string: " + std::string(name));
    }
    return it->second;
}

std::vector<Precision> all_precisions() {
    return {
        Precision::FP64,
        Precision::FP32,
        Precision::TF32,
        Precision::BF16,
        Precision::P3109_8
    };
}

} // namespace fpstudy::formats

