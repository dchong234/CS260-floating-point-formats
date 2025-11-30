#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include <cassert>

#include "algorithms/matmul.hpp"
#include "formats/precision.hpp"

bool run_matmul_tests() {
    std::vector<double> A = {1.0, 2.0, 3.0, 4.0};
    std::vector<double> B = {5.0, 6.0, 7.0, 8.0};
    auto C = fpstudy::algorithms::matmul_square<double>(A, B, 2);
    std::vector<double> expected = {19.0, 22.0, 43.0, 50.0};
    for (std::size_t i = 0; i < expected.size(); ++i) {
        if (std::fabs(C[i] - expected[i]) > 1e-9) {
            std::cerr << "MatMul test failed at index " << i << "\n";
            return false;
        }
    }
    return true;
}

bool run_iterative_tests();
bool run_io_tests();

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Specify test suite name\n";
        return 1;
    }
    std::string suite = argv[1];
    bool ok = false;
    if (suite == "MatMul") {
        ok = run_matmul_tests();
    } else if (suite == "Iterative") {
        ok = run_iterative_tests();
    } else if (suite == "IO") {
        ok = run_io_tests();
    } else {
        std::cerr << "Unknown test suite: " << suite << "\n";
        return 1;
    }
    return ok ? 0 : 1;
}

