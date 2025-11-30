#pragma once

#include <cstddef>
#include <vector>
#include <cmath>

namespace fpstudy::algorithms {

struct GradientDescentOptions {
    double step_size = 1e-2;
    std::size_t max_iters = 1000;
    double tol = 1e-6;
};

template <typename T>
struct GradientDescentResult {
    std::vector<T> x;
    std::size_t iterations = 0;
    bool converged = false;
};

template <typename T>
GradientDescentResult<T> gradient_descent_quadratic(const std::vector<T>& Q,
                                                    const std::vector<T>& b,
                                                    const std::vector<T>& initial,
                                                    std::size_t dim,
                                                    const GradientDescentOptions& opts) {
    std::vector<T> x = initial;
    std::vector<T> gradient(dim, T{});
    for (std::size_t iter = 0; iter < opts.max_iters; ++iter) {
        for (std::size_t i = 0; i < dim; ++i) {
            T acc{};
            for (std::size_t j = 0; j < dim; ++j) {
                acc = acc + Q[i * dim + j] * x[j];
            }
            gradient[i] = acc + b[i];
        }
        double grad_norm = 0.0;
        for (const T& g : gradient) {
            double gd = static_cast<double>(g);
            grad_norm += gd * gd;
        }
        grad_norm = std::sqrt(grad_norm);
        if (grad_norm < opts.tol) {
            return {x, iter, true};
        }
        for (std::size_t i = 0; i < dim; ++i) {
            x[i] = x[i] - T(opts.step_size) * gradient[i];
        }
    }
    return {x, opts.max_iters, false};
}

} // namespace fpstudy::algorithms

