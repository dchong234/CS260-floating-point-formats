#pragma once

#include <vector>
#include <cstddef>
#include <type_traits>

namespace fpstudy::algorithms {

struct FIROptions {
    bool use_kahan = false;
    bool accumulate_in_fp32 = false;
};

template <typename T>
std::vector<T> fir_filter(const std::vector<T>& h,
                          const std::vector<T>& x,
                          FIROptions opts = {}) {
    const std::size_t M = h.size();  // Number of filter taps
    const std::size_t N = x.size();  // Number of input samples
    std::vector<T> y(N, T{});        // Output signal

    for (std::size_t n = 0; n < N; ++n) {
        if (opts.accumulate_in_fp32) {
            float sum = 0.0f;
            float compensation = 0.0f;
            for (std::size_t k = 0; k < M; ++k) {
                // For n < k, x[n-k] is zero-padded (assumed to be zero)
                if (n >= k) {
                    float hk = static_cast<float>(h[k]);
                    float xnk = static_cast<float>(x[n - k]);
                    float prod = hk * xnk;
                    if (opts.use_kahan) {
                        float y_val = prod - compensation;
                        float t = sum + y_val;
                        compensation = (t - sum) - y_val;
                        sum = t;
                    } else {
                        sum += prod;
                    }
                }
            }
            y[n] = T(sum);
        } else {
            T sum{};
            T compensation{};
            for (std::size_t k = 0; k < M; ++k) {
                // For n < k, x[n-k] is zero-padded (assumed to be zero)
                if (n >= k) {
                    T prod = h[k] * x[n - k];
                    if (opts.use_kahan) {
                        T y_val = prod - compensation;
                        T t = sum + y_val;
                        compensation = (t - sum) - y_val;
                        sum = t;
                    } else {
                        sum = sum + prod;
                    }
                }
            }
            y[n] = sum;
        }
    }
    return y;
}

} // namespace fpstudy::algorithms

