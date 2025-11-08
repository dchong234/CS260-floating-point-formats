#pragma once

#include <cstdint>
#include <cmath>
#include <limits>
#include <type_traits>

namespace fpstudy::formats {

struct P3109Layout {
    // 1 sign bit, 3 exponent bits (biased), 4 mantissa bits, 0 implicit?
    // We choose exponent bias = 3 to balance dynamic range (approx ~1e2).
    uint8_t exponent_bits = 3;
    uint8_t mantissa_bits = 4;
    int8_t exponent_bias = 3;
};

inline uint8_t p3109_quantize(float value,
                              const P3109Layout& layout = P3109Layout{}) {
    if (std::isnan(value)) {
        return 0xFF;
    }
    if (std::isinf(value)) {
        return (value > 0.f) ? 0x7F : 0xFE;
    }

    const float sign = std::signbit(value) ? -1.f : 1.f;
    float abs_v = std::fabs(value);
    if (abs_v == 0.0f) {
        return std::signbit(value) ? 0x80 : 0x00;
    }

    int exp;
    float mant = std::frexp(abs_v, &exp); // abs_v = mant * 2^exp, mant in [0.5,1)

    int bias = layout.exponent_bias;
    int exp_val = exp + bias - 1; // align with frexp
    const int max_exp = (1 << layout.exponent_bits) - 2; // reserve top for inf/nan
    const int min_exp = 1;

    if (exp_val > max_exp) {
        return (sign < 0 ? 0x80 : 0x00) | ((max_exp + 1) << layout.mantissa_bits) - 1;
    }
    if (exp_val < min_exp) {
        // flush to zero
        return std::signbit(value) ? 0x80 : 0x00;
    }

    const int mantissa_mask = (1 << layout.mantissa_bits) - 1;
    float scaled = (mant * 2.0f) - 1.0f; // convert [0.5,1) -> [0,1)
    int mantissa = static_cast<int>(std::round(scaled * (mantissa_mask + 1)));
    if (mantissa > mantissa_mask) {
        mantissa = 0;
        ++exp_val;
        if (exp_val > max_exp) {
            return (sign < 0 ? 0x80 : 0x00) | ((max_exp + 1) << layout.mantissa_bits) - 1;
        }
    }

    uint8_t sign_bit = sign < 0 ? 0x80 : 0x00;
    uint8_t exponent_field = static_cast<uint8_t>(exp_val << layout.mantissa_bits);
    return static_cast<uint8_t>(sign_bit | exponent_field | mantissa);
}

inline float p3109_dequantize(uint8_t code,
                              const P3109Layout& layout = P3109Layout{}) {
    if (code == 0xFF) { // NaN
        return std::numeric_limits<float>::quiet_NaN();
    }
    if (code == 0x7F) {
        return std::numeric_limits<float>::infinity();
    }
    if (code == 0xFE) {
        return -std::numeric_limits<float>::infinity();
    }

    const bool negative = (code & 0x80) != 0;
    const int mantissa_mask = (1 << layout.mantissa_bits) - 1;
    const int exponent_mask = (1 << layout.exponent_bits) - 1;
    int exponent = (code >> layout.mantissa_bits) & exponent_mask;
    int mantissa = code & mantissa_mask;

    if (exponent == 0) {
        return negative ? -0.0f : 0.0f;
    }

    float mant = 1.0f + static_cast<float>(mantissa) / static_cast<float>(mantissa_mask + 1);
    int exp = exponent - layout.exponent_bias;
    float value = std::ldexp(mant, exp);
    return negative ? -value : value;
}

template <typename T>
inline bool is_special(T value) {
    if constexpr (std::is_floating_point_v<T>) {
        return std::isnan(value) || std::isinf(value);
    } else {
        return false;
    }
}

} // namespace fpstudy::formats

