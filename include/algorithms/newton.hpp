#pragma once

#include <cstddef>
#include <cmath>

namespace fpstudy::algorithms {

struct NewtonOptions {
    std::size_t max_iters = 100;
    double tol = 1e-8;
};

template <typename T>
struct NewtonResult {
    T root;
    std::size_t iterations = 0;
    bool converged = false;
};

template <typename T, typename Func, typename Deriv>
NewtonResult<T> newton_raphson(T initial,
                               Func f,
                               Deriv df,
                               const NewtonOptions& opts) {
    T x = initial;
    for (std::size_t iter = 0; iter < opts.max_iters; ++iter) {
        T fx = f(x);
        T dfx = df(x);
        double abs_fx = std::fabs(static_cast<double>(fx));
        if (abs_fx < opts.tol) {
            return {x, iter, true};
        }
        if (static_cast<double>(dfx) == 0.0) {
            return {x, iter, false};
        }
        x = x - fx / dfx;
    }
    return {x, opts.max_iters, false};
}

} // namespace fpstudy::algorithms

