#pragma once

#include <random>
#include <vector>
#include <cstddef>

namespace fpstudy::core {

class Random {
public:
    explicit Random(uint32_t seed) : engine_(seed) {}

    std::mt19937& engine() { return engine_; }

    template <typename T>
    T uniform(T min, T max) {
        std::uniform_real_distribution<double> dist(static_cast<double>(min), static_cast<double>(max));
        return static_cast<T>(dist(engine_));
    }

private:
    std::mt19937 engine_;
};

inline std::vector<double> random_vector(size_t n, Random& rng, double scale = 1.0) {
    std::normal_distribution<double> dist(0.0, scale);
    std::vector<double> data(n);
    for (double& v : data) {
        v = dist(rng.engine());
    }
    return data;
}

inline std::vector<double> random_matrix(size_t rows, size_t cols, Random& rng, bool ill_conditioned = false) {
    std::normal_distribution<double> dist(0.0, 1.0);
    std::vector<double> mat(rows * cols);
    for (double& v : mat) {
        v = dist(rng.engine());
    }
    if (ill_conditioned && cols > 0) {
        for (size_t r = 0; r < rows; ++r) {
            mat[r * cols] *= 1e-6;
        }
    }
    return mat;
}

} // namespace fpstudy::core

