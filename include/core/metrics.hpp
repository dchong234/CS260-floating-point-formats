#pragma once

#include <chrono>
#include <cmath>
#include <stdexcept>
#include <limits>
#include <vector>
#include <algorithm>

namespace fpstudy::core {

struct RunMetrics {
    double relative_error = 0.0;
    int iterations = 0;
    bool converged = false;
    int nan_count = 0;
    int inf_count = 0;
    double elapsed_ms = 0.0;
};

inline double vector_norm(const std::vector<double>& v) {
    double accum = 0.0;
    for (double x : v) {
        accum += x * x;
    }
    return std::sqrt(accum);
}

inline double relative_error(const std::vector<double>& truth,
                             const std::vector<double>& approx,
                             double eps = 1e-12) {
    if (truth.size() != approx.size()) {
        throw std::runtime_error("Vector size mismatch in relative_error");
    }
    std::vector<double> diff(truth.size());
    for (size_t i = 0; i < truth.size(); ++i) {
        diff[i] = truth[i] - approx[i];
    }
    double norm_truth = vector_norm(truth);
    double norm_diff = vector_norm(diff);
    return norm_diff / std::max(norm_truth, eps);
}

template <typename T>
inline int count_nan(const std::vector<T>& data) {
    return static_cast<int>(std::count_if(data.begin(), data.end(), [](const T& x) {
        return std::isnan(static_cast<double>(x));
    }));
}

template <typename T>
inline int count_inf(const std::vector<T>& data) {
    return static_cast<int>(std::count_if(data.begin(), data.end(), [](const T& x) {
        double v = static_cast<double>(x);
        return std::isinf(v);
    }));
}

class ScopedTimer {
public:
    ScopedTimer() : start_(Clock::now()) {}
    double elapsed_ms() const {
        auto delta = Clock::now() - start_;
        return std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(delta).count();
    }

private:
    using Clock = std::chrono::steady_clock;
    Clock::time_point start_;
};

} // namespace fpstudy::core

