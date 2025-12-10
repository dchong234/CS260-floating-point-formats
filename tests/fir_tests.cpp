#include <cmath>
#include <iostream>
#include <vector>

#include "algorithms/fir.hpp"

bool run_fir_tests() {
    // Test case: Simple 2-tap moving average filter
    // Filter coefficients: h = [0.5, 0.5] (normalized)
    // Input signal: x = [1.0, 2.0, 3.0, 4.0]
    // Expected output (with zero-padding):
    //   y[0] = 0.5×1.0 + 0.5×0   = 0.5
    //   y[1] = 0.5×2.0 + 0.5×1.0 = 1.5
    //   y[2] = 0.5×3.0 + 0.5×2.0 = 2.5
    //   y[3] = 0.5×4.0 + 0.5×3.0 = 3.5
    
    std::vector<double> h = {0.5, 0.5};
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0};
    
    auto y = fpstudy::algorithms::fir_filter<double>(h, x);
    
    std::vector<double> expected = {0.5, 1.5, 2.5, 3.5};
    
    if (y.size() != expected.size()) {
        std::cerr << "FIR test failed: output size mismatch\n";
        return false;
    }
    
    for (std::size_t i = 0; i < expected.size(); ++i) {
        if (std::fabs(y[i] - expected[i]) > 1e-9) {
            std::cerr << "FIR test failed at index " << i 
                      << ": expected " << expected[i] 
                      << ", got " << y[i] << "\n";
            return false;
        }
    }
    
    // Test case 2: Single-tap filter (identity with scaling)
    // h = [1.0], x = [1.0, 2.0, 3.0]
    // Expected: y = [1.0, 2.0, 3.0]
    std::vector<double> h2 = {1.0};
    std::vector<double> x2 = {1.0, 2.0, 3.0};
    auto y2 = fpstudy::algorithms::fir_filter<double>(h2, x2);
    
    if (y2.size() != x2.size()) {
        std::cerr << "FIR test 2 failed: output size mismatch\n";
        return false;
    }
    
    for (std::size_t i = 0; i < x2.size(); ++i) {
        if (std::fabs(y2[i] - x2[i]) > 1e-9) {
            std::cerr << "FIR test 2 failed at index " << i << "\n";
            return false;
        }
    }
    
    return true;
}

