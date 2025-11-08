#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>
#include <stdexcept>
#include <type_traits>

#include <universal/number/cfloat/cfloat.hpp>

#include "formats/quantize.hpp"

namespace fpstudy::formats {

enum class Precision {
    FP64,
    FP32,
    TF32,
    BF16,
    P3109_8
};

std::string precision_to_string(Precision p);
Precision precision_from_string(std::string_view name);

using TF32 = sw::universal::cfloat<19, 8, uint32_t>;
using BF16 = sw::universal::cfloat<16, 8, uint16_t>;

class P3109Number {
public:
    P3109Number() = default;
    P3109Number(float v) { value_ = p3109_quantize(v); }
    P3109Number(double v) { value_ = p3109_quantize(static_cast<float>(v)); }
    P3109Number(int v) { value_ = p3109_quantize(static_cast<float>(v)); }

    explicit P3109Number(uint8_t raw) : value_(raw) {}

    operator float() const { return p3109_dequantize(value_); }
    operator double() const { return static_cast<double>(p3109_dequantize(value_)); }

    P3109Number& operator+=(const P3109Number& other) {
        float lhs = p3109_dequantize(value_);
        float rhs = p3109_dequantize(other.value_);
        float res = accumulate_in_fp32_ ? lhs + rhs : p3109_dequantize(p3109_quantize(lhs + rhs));
        value_ = p3109_quantize(res);
        return *this;
    }

    P3109Number& operator-=(const P3109Number& other) {
        float lhs = p3109_dequantize(value_);
        float rhs = p3109_dequantize(other.value_);
        float res = accumulate_in_fp32_ ? lhs - rhs : p3109_dequantize(p3109_quantize(lhs - rhs));
        value_ = p3109_quantize(res);
        return *this;
    }

    P3109Number& operator*=(const P3109Number& other) {
        float lhs = p3109_dequantize(value_);
        float rhs = p3109_dequantize(other.value_);
        float res = accumulate_in_fp32_ ? lhs * rhs : p3109_dequantize(p3109_quantize(lhs * rhs));
        value_ = p3109_quantize(res);
        return *this;
    }

    P3109Number& operator/=(const P3109Number& other) {
        float lhs = p3109_dequantize(value_);
        float rhs = p3109_dequantize(other.value_);
        float res = accumulate_in_fp32_ ? lhs / rhs : p3109_dequantize(p3109_quantize(lhs / rhs));
        value_ = p3109_quantize(res);
        return *this;
    }

    P3109Number operator-() const {
        P3109Number tmp(*this);
        float val = -p3109_dequantize(value_);
        tmp.value_ = p3109_quantize(val);
        return tmp;
    }

    [[nodiscard]] uint8_t raw() const { return value_; }

    static void set_accumulate_fp32(bool flag) { accumulate_in_fp32_ = flag; }
    static bool accumulate_fp32() { return accumulate_in_fp32_; }

private:
    uint8_t value_ = 0;
    inline static bool accumulate_in_fp32_ = true;
};

inline P3109Number operator+(P3109Number lhs, const P3109Number& rhs) {
    lhs += rhs;
    return lhs;
}

inline P3109Number operator-(P3109Number lhs, const P3109Number& rhs) {
    lhs -= rhs;
    return lhs;
}

inline P3109Number operator*(P3109Number lhs, const P3109Number& rhs) {
    lhs *= rhs;
    return lhs;
}

inline P3109Number operator/(P3109Number lhs, const P3109Number& rhs) {
    lhs /= rhs;
    return lhs;
}

template <Precision P>
struct PrecisionTraits;

template <>
struct PrecisionTraits<Precision::FP64> {
    using type = double;
};

template <>
struct PrecisionTraits<Precision::FP32> {
    using type = float;
};

template <>
struct PrecisionTraits<Precision::TF32> {
    using type = TF32;
};

template <>
struct PrecisionTraits<Precision::BF16> {
    using type = BF16;
};

template <>
struct PrecisionTraits<Precision::P3109_8> {
    using type = P3109Number;
};

std::vector<Precision> all_precisions();

template <typename T>
std::vector<T> cast_vector(const std::vector<double>& input) {
    std::vector<T> output;
    output.reserve(input.size());
    for (double v : input) {
        output.emplace_back(static_cast<T>(v));
    }
    return output;
}

template <typename T>
std::vector<double> to_double_vector(const std::vector<T>& input) {
    std::vector<double> output;
    output.reserve(input.size());
    for (const auto& v : input) {
        output.push_back(static_cast<double>(v));
    }
    return output;
}

template <>
inline std::vector<PrecisionTraits<Precision::TF32>::type> cast_vector(const std::vector<double>& input) {
    using T = PrecisionTraits<Precision::TF32>::type;
    std::vector<T> output;
    output.reserve(input.size());
    for (double v : input) {
        output.emplace_back(T(v));
    }
    return output;
}

template <>
inline std::vector<PrecisionTraits<Precision::BF16>::type> cast_vector(const std::vector<double>& input) {
    using T = PrecisionTraits<Precision::BF16>::type;
    std::vector<T> output;
    output.reserve(input.size());
    for (double v : input) {
        output.emplace_back(T(v));
    }
    return output;
}

template <>
inline std::vector<PrecisionTraits<Precision::P3109_8>::type> cast_vector(const std::vector<double>& input) {
    using T = PrecisionTraits<Precision::P3109_8>::type;
    std::vector<T> output;
    output.reserve(input.size());
    for (double v : input) {
        output.emplace_back(T(v));
    }
    return output;
}

} // namespace fpstudy::formats

