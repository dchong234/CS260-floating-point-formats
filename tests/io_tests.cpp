#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "core/io.hpp"

bool run_io_tests() {
    auto temp_dir = std::filesystem::temp_directory_path();
    auto csv_path = temp_dir / "fpstudy_csv_smoke.csv";
    {
        fpstudy::core::CsvWriter writer(csv_path, false);
        std::vector<std::string> header = {
            "algo", "size", "precision", "seed",
            "params_json", "rel_error", "iters",
            "converged", "n_nan", "n_inf", "elapsed_ms"
        };
        writer.write_header(header);
        writer.write_row({
            "matmul", "2", "fp64", "123",
            "{}", "0.0", "0", "1", "0", "0", "0.1"
        });
    }
    std::ifstream ifs(csv_path);
    std::string line;
    std::getline(ifs, line);
    std::string expected_header = "algo,size,precision,seed,params_json,rel_error,iters,converged,n_nan,n_inf,elapsed_ms";
    bool ok = line == expected_header;
    std::filesystem::remove(csv_path);
    return ok;
}

