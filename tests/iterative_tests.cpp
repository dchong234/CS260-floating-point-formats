#include <cmath>
#include <iostream>
#include <vector>

#include "algorithms/gradient_descent.hpp"
#include "algorithms/newton.hpp"

bool run_iterative_tests() {
    // Gradient descent test
    std::vector<double> Q = {
        4.0, 1.0,
        1.0, 3.0
    };
    std::vector<double> b = {-1.0, 2.0};
    std::vector<double> x0 = {0.0, 0.0};
    fpstudy::algorithms::GradientDescentOptions opts;
    opts.step_size = 0.05;
    opts.max_iters = 200;
    opts.tol = 1e-8;

    auto gd_result = fpstudy::algorithms::gradient_descent_quadratic(Q, b, x0, 2, opts);
    if (!gd_result.converged || gd_result.iterations >= 200) {
        std::cerr << "Gradient descent did not converge within expected iterations\n";
        return false;
    }
    std::vector<double> expected = {5.0 / 11.0, -9.0 / 11.0};
    for (std::size_t i = 0; i < expected.size(); ++i) {
        if (std::fabs(static_cast<double>(gd_result.x[i]) - expected[i]) > 1e-3) {
            std::cerr << "Gradient descent mismatch at index " << i << "\n";
            return false;
        }
    }

    // Newton-Raphson test for x^3 - 2 = 0
    fpstudy::algorithms::NewtonOptions newton_opts;
    newton_opts.max_iters = 30;
    newton_opts.tol = 1e-10;
    auto newton_result = fpstudy::algorithms::newton_raphson<double>(
        1.0,
        [](double x) { return x * x * x - 2.0; },
        [](double x) { return 3.0 * x * x; },
        newton_opts);
    double root = static_cast<double>(newton_result.root);
    if (!newton_result.converged || std::fabs(root - std::cbrt(2.0)) > 1e-8) {
        std::cerr << "Newton method failed to converge to cube root of 2\n";
        return false;
    }
    return true;
}

