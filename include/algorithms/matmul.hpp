#pragma once

#include <vector>
#include <cstddef>
#include <type_traits>

namespace fpstudy::algorithms {

struct MatMulOptions {
    bool use_kahan = false;
    bool accumulate_in_fp32 = false;
};

template <typename T>
std::vector<T> matmul_square(const std::vector<T>& A,
                             const std::vector<T>& B,
                             std::size_t n,
                             MatMulOptions opts = {}) {
    std::vector<T> C(n * n, T{});
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            if (opts.accumulate_in_fp32) {
                float sum = 0.0f;
                float compensation = 0.0f;
                for (std::size_t k = 0; k < n; ++k) {
                    float prod = static_cast<float>(A[i * n + k]) * static_cast<float>(B[k * n + j]);
                    if (opts.use_kahan) {
                        float y = prod - compensation;
                        float t = sum + y;
                        compensation = (t - sum) - y;
                        sum = t;
                    } else {
                        sum += prod;
                    }
                }
                C[i * n + j] = T(sum);
            } else {
                T sum{};
                T compensation{};
                for (std::size_t k = 0; k < n; ++k) {
                    T prod = A[i * n + k] * B[k * n + j];
                    if (opts.use_kahan) {
                        T y = prod - compensation;
                        T t = sum + y;
                        compensation = (t - sum) - y;
                        sum = t;
                    } else {
                        sum = sum + prod;
                    }
                }
                C[i * n + j] = sum;
            }
        }
    }
    return C;
}

} // namespace fpstudy::algorithms

