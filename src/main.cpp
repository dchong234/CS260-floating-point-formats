#include <filesystem>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>

#include "core/io.hpp"
#include "core/metrics.hpp"
#include "core/random.hpp"
#include "algorithms/matmul.hpp"
#include "algorithms/gradient_descent.hpp"
#include "algorithms/newton.hpp"
#include "formats/precision.hpp"

using fpstudy::core::CsvWriter;
namespace json = fpstudy::core::json;
namespace alg = fpstudy::algorithms;
namespace fmt = fpstudy::formats;
namespace core = fpstudy::core;

namespace {

const std::vector<std::string> kCsvHeader = {
    "algo", "size", "precision", "seed",
    "params_json", "rel_error", "iters",
    "converged", "n_nan", "n_inf", "elapsed_ms"
};

const json::Value& require_field(const json::Object& obj, const std::string& key) {
    auto it = obj.find(key);
    if (it == obj.end()) {
        throw std::runtime_error("Missing required field: " + key);
    }
    return it->second;
}

std::vector<fmt::Precision> parse_precisions(const json::Value& value) {
    std::vector<fmt::Precision> result;
    for (const auto& entry : value.as_array()) {
        result.push_back(fmt::precision_from_string(entry.as_string()));
    }
    return result;
}

std::vector<bool> parse_bool_list(const json::Value* value) {
    if (!value) return {false};
    std::vector<bool> flags;
    for (const auto& entry : value->as_array()) {
        flags.push_back(entry.as_bool());
    }
    if (flags.empty()) flags.push_back(false);
    return flags;
}

std::vector<int> parse_int_list(const json::Value& value) {
    std::vector<int> result;
    for (const auto& entry : value.as_array()) {
        result.push_back(static_cast<int>(entry.as_number()));
    }
    return result;
}

std::vector<double> parse_double_list(const json::Value& value) {
    std::vector<double> result;
    for (const auto& entry : value.as_array()) {
        result.push_back(entry.as_number());
    }
    return result;
}

std::vector<std::vector<double>> build_spd_cases(std::size_t dim,
                                                 std::size_t trials,
                                                 uint32_t base_seed,
                                                 bool ill_conditioned) {
    std::vector<std::vector<double>> cases;
    cases.reserve(trials);
    for (std::size_t t = 0; t < trials; ++t) {
        core::Random rng(base_seed + static_cast<uint32_t>(t * 17 + dim * 13));
        auto M = core::random_matrix(dim, dim, rng, ill_conditioned);
        std::vector<double> Q(dim * dim, 0.0);
        for (std::size_t i = 0; i < dim; ++i) {
            for (std::size_t j = 0; j < dim; ++j) {
                double acc = 0.0;
                for (std::size_t k = 0; k < dim; ++k) {
                    acc += M[k * dim + i] * M[k * dim + j];
                }
                if (i == j) acc += dim * 0.1;
                Q[i * dim + j] = acc;
            }
        }
        cases.push_back(std::move(Q));
    }
    return cases;
}

template <typename T>
core::RunMetrics emit_run(const json::Object& params,
                          const std::string& algo_name,
                          const std::string& size_str,
                          fmt::Precision precision,
                          uint32_t seed,
                          CsvWriter& writer,
                          const std::vector<double>& truth,
                          const std::vector<T>& result,
                          std::size_t iterations,
                          bool converged,
                          double elapsed_ms) {
    core::RunMetrics metrics;
    metrics.relative_error = core::relative_error(truth, fmt::to_double_vector(result));
    metrics.iterations = static_cast<int>(iterations);
    metrics.converged = converged;
    metrics.nan_count = core::count_nan(result);
    metrics.inf_count = core::count_inf(result);
    metrics.elapsed_ms = elapsed_ms;

    json::Object params_obj = params;
    params_obj.emplace("precision", json::Value(fmt::precision_to_string(precision)));
    auto params_json = json::serialize_compact(json::Value(params_obj));

    writer.write_row({
        algo_name,
        size_str,
        fmt::precision_to_string(precision),
        std::to_string(seed),
        params_json,
        std::to_string(metrics.relative_error),
        std::to_string(metrics.iterations),
        metrics.converged ? "1" : "0",
        std::to_string(metrics.nan_count),
        std::to_string(metrics.inf_count),
        std::to_string(metrics.elapsed_ms)
    });
    return metrics;
}

} // namespace

int main(int argc, char** argv) {
    std::optional<std::filesystem::path> config_path;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "--config" || arg == "-c") && i + 1 < argc) {
            config_path = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: fpstudy --config path/to/config.json\n";
            return 0;
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            return 1;
        }
    }

    if (!config_path) {
        std::cerr << "Config path required. Use --config <path>.\n";
        return 1;
    }

    auto config_value = json::load_file(*config_path);
    const auto& root = config_value.as_object();

    uint32_t base_seed = static_cast<uint32_t>(require_field(root, "seed").as_number());
    auto out_csv_path = std::filesystem::path(require_field(root, "out_csv").as_string());
    const auto& experiments = require_field(root, "experiments").as_array();

    CsvWriter writer(out_csv_path, false);
    writer.write_header(kCsvHeader);

    for (const auto& exp_value : experiments) {
        const auto& exp = exp_value.as_object();
        const std::string algo = require_field(exp, "algo").as_string();

        if (algo == "matmul") {
            auto sizes = parse_int_list(require_field(exp, "sizes"));
            auto precisions = parse_precisions(require_field(exp, "precisions"));
            auto trials = exp.contains("trials") ? static_cast<std::size_t>(require_field(exp, "trials").as_number()) : std::size_t(1);
            auto accumulate_flags = exp.contains("accumulate_in_fp32")
                ? parse_bool_list(&require_field(exp, "accumulate_in_fp32"))
                : std::vector<bool>{false};
            bool use_kahan = exp.contains("kahan") && require_field(exp, "kahan").as_bool();

            for (int size : sizes) {
                for (std::size_t trial = 0; trial < trials; ++trial) {
                    uint32_t trial_seed = base_seed + static_cast<uint32_t>(size * 997 + trial);
                    fpstudy::core::Random rng(trial_seed);
                    auto A = core::random_matrix(size, size, rng);
                    auto B = core::random_matrix(size, size, rng);
                    auto truth = alg::matmul_square<double>(A, B, size, {use_kahan, false});
                    for (bool accumulate : accumulate_flags) {
                        for (auto precision : precisions) {
                            alg::MatMulOptions opts{use_kahan, accumulate};
                            std::string size_str = std::to_string(size);
                            json::Object params;
                            params.emplace("size", json::Value(static_cast<double>(size)));
                            params.emplace("trial", json::Value(static_cast<double>(trial)));
                            params.emplace("accumulate_in_fp32", json::Value(accumulate));
                            params.emplace("kahan", json::Value(use_kahan));

                            switch (precision) {
                                case fmt::Precision::FP64: {
                                    core::ScopedTimer timer;
                                    auto result = alg::matmul_square<double>(A, B, size, {use_kahan, false});
                                    auto elapsed = timer.elapsed_ms();
                                    emit_run(params, algo, size_str, precision, trial_seed, writer,
                                             truth, result, 0, true, elapsed);
                                    break;
                                }
                                case fmt::Precision::FP32: {
                                    auto A32 = fmt::cast_vector<float>(A);
                                    auto B32 = fmt::cast_vector<float>(B);
                                    core::ScopedTimer timer;
                                    auto result = alg::matmul_square<float>(A32, B32, size, opts);
                                    auto elapsed = timer.elapsed_ms();
                                    emit_run(params, algo, size_str, precision, trial_seed, writer,
                                             truth, result, 0, true, elapsed);
                                    break;
                                }
                                case fmt::Precision::TF32: {
                                    auto A19 = fmt::cast_vector<fmt::TF32>(A);
                                    auto B19 = fmt::cast_vector<fmt::TF32>(B);
                                    core::ScopedTimer timer;
                                    auto result = alg::matmul_square<fmt::TF32>(A19, B19, size, opts);
                                    auto elapsed = timer.elapsed_ms();
                                    emit_run(params, algo, size_str, precision, trial_seed, writer,
                                             truth, result, 0, true, elapsed);
                                    break;
                                }
                                case fmt::Precision::BF16: {
                                    auto A16 = fmt::cast_vector<fmt::BF16>(A);
                                    auto B16 = fmt::cast_vector<fmt::BF16>(B);
                                    core::ScopedTimer timer;
                                    auto result = alg::matmul_square<fmt::BF16>(A16, B16, size, opts);
                                    auto elapsed = timer.elapsed_ms();
                                    emit_run(params, algo, size_str, precision, trial_seed, writer,
                                             truth, result, 0, true, elapsed);
                                    break;
                                }
                                case fmt::Precision::P3109_8: {
                                    fmt::P3109Number::set_accumulate_fp32(accumulate);
                                    auto A8 = fmt::cast_vector<fmt::P3109Number>(A);
                                    auto B8 = fmt::cast_vector<fmt::P3109Number>(B);
                                    core::ScopedTimer timer;
                                    auto result = alg::matmul_square<fmt::P3109Number>(A8, B8, size, opts);
                                    auto elapsed = timer.elapsed_ms();
                                    emit_run(params, algo, size_str, precision, trial_seed, writer,
                                             truth, result, 0, true, elapsed);
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        } else if (algo == "gd_quadratic") {
            std::size_t dim = static_cast<std::size_t>(require_field(exp, "dim").as_number());
            auto precisions = parse_precisions(require_field(exp, "precisions"));
            std::size_t trials = exp.contains("trials") ? static_cast<std::size_t>(require_field(exp, "trials").as_number()) : 1;
            alg::GradientDescentOptions opts;
            opts.step_size = exp.contains("step_size") ? require_field(exp, "step_size").as_number() : 1e-2;
            opts.max_iters = exp.contains("max_iters") ? static_cast<std::size_t>(require_field(exp, "max_iters").as_number()) : 1000;
            opts.tol = exp.contains("tol") ? require_field(exp, "tol").as_number() : 1e-6;
            bool ill_conditioned = exp.contains("ill_conditioned") && require_field(exp, "ill_conditioned").as_bool();

            for (std::size_t trial = 0; trial < trials; ++trial) {
                uint32_t trial_seed = base_seed + static_cast<uint32_t>(dim * 577 + trial * 31);
                fpstudy::core::Random rng(trial_seed);
                auto Q_cases = build_spd_cases(dim, 1, trial_seed, ill_conditioned);
                auto Q = Q_cases.front();
                auto b = fpstudy::core::random_vector(dim, rng);
                auto x0 = std::vector<double>(dim, 0.0);

                core::ScopedTimer baseline_timer;
                alg::GradientDescentResult<double> truth_result = alg::gradient_descent_quadratic<double>(
                    Q, b, x0, dim, opts);
                double baseline_elapsed = baseline_timer.elapsed_ms();
                auto truth_vec = truth_result.x;

                json::Object params;
                params.emplace("dim", json::Value(static_cast<double>(dim)));
                params.emplace("trial", json::Value(static_cast<double>(trial)));
                params.emplace("step_size", json::Value(opts.step_size));
                params.emplace("tol", json::Value(opts.tol));
                params.emplace("max_iters", json::Value(static_cast<double>(opts.max_iters)));
                params.emplace("ill_conditioned", json::Value(ill_conditioned));

                for (auto precision : precisions) {
                    switch (precision) {
                        case fmt::Precision::FP64: {
                            auto result = truth_result;
                            emit_run(params, algo, std::to_string(dim), precision, trial_seed, writer,
                                     truth_vec, result.x, result.iterations, result.converged, baseline_elapsed);
                            break;
                        }
                        case fmt::Precision::FP32: {
                            auto Q32 = fmt::cast_vector<float>(Q);
                            auto b32 = fmt::cast_vector<float>(b);
                            auto x32 = fmt::cast_vector<float>(x0);
                            core::ScopedTimer timer;
                            auto result = alg::gradient_descent_quadratic<float>(Q32, b32, x32, dim, opts);
                            auto elapsed = timer.elapsed_ms();
                            emit_run(params, algo, std::to_string(dim), precision, trial_seed, writer,
                                     truth_vec, result.x, result.iterations, result.converged, elapsed);
                            break;
                        }
                        case fmt::Precision::TF32: {
                            auto Q19 = fmt::cast_vector<fmt::TF32>(Q);
                            auto b19 = fmt::cast_vector<fmt::TF32>(b);
                            auto x19 = fmt::cast_vector<fmt::TF32>(x0);
                            core::ScopedTimer timer;
                            auto result = alg::gradient_descent_quadratic<fmt::TF32>(Q19, b19, x19, dim, opts);
                            auto elapsed = timer.elapsed_ms();
                            emit_run(params, algo, std::to_string(dim), precision, trial_seed, writer,
                                     truth_vec, result.x, result.iterations, result.converged, elapsed);
                            break;
                        }
                        case fmt::Precision::BF16: {
                            auto Q16 = fmt::cast_vector<fmt::BF16>(Q);
                            auto b16 = fmt::cast_vector<fmt::BF16>(b);
                            auto x16 = fmt::cast_vector<fmt::BF16>(x0);
                            core::ScopedTimer timer;
                            auto result = alg::gradient_descent_quadratic<fmt::BF16>(Q16, b16, x16, dim, opts);
                            auto elapsed = timer.elapsed_ms();
                            emit_run(params, algo, std::to_string(dim), precision, trial_seed, writer,
                                     truth_vec, result.x, result.iterations, result.converged, elapsed);
                            break;
                        }
                        case fmt::Precision::P3109_8: {
                            fmt::P3109Number::set_accumulate_fp32(true);
                            auto Q8 = fmt::cast_vector<fmt::P3109Number>(Q);
                            auto b8 = fmt::cast_vector<fmt::P3109Number>(b);
                            auto x8 = fmt::cast_vector<fmt::P3109Number>(x0);
                            core::ScopedTimer timer;
                            auto result = alg::gradient_descent_quadratic<fmt::P3109Number>(Q8, b8, x8, dim, opts);
                            auto elapsed = timer.elapsed_ms();
                            emit_run(params, algo, std::to_string(dim), precision, trial_seed, writer,
                                     truth_vec, result.x, result.iterations, result.converged, elapsed);
                            break;
                        }
                    }
                }
            }
        } else if (algo == "newton") {
            const std::string function_name = require_field(exp, "function").as_string();
            auto initials = parse_double_list(require_field(exp, "initials"));
            auto precisions = parse_precisions(require_field(exp, "precisions"));
            alg::NewtonOptions opts;
            opts.max_iters = exp.contains("max_iters") ? static_cast<std::size_t>(require_field(exp, "max_iters").as_number()) : 100;
            opts.tol = exp.contains("tol") ? require_field(exp, "tol").as_number() : 1e-8;

            auto function = [&](double x) {
                if (function_name == "x3_minus_2") {
                    return x * x * x - 2.0;
                }
                throw std::runtime_error("Unknown Newton function: " + function_name);
            };
            auto derivative = [&](double x) {
                if (function_name == "x3_minus_2") {
                    return 3.0 * x * x;
                }
                throw std::runtime_error("Unknown Newton derivative: " + function_name);
            };

            for (double initial : initials) {
                core::ScopedTimer baseline_timer;
                auto truth_result = alg::newton_raphson<double>(
                    initial,
                    [&](double x) { return function(x); },
                    [&](double x) { return derivative(x); },
                    opts);
                double baseline_elapsed = baseline_timer.elapsed_ms();
                std::vector<double> truth_vec = {static_cast<double>(truth_result.root)};
                uint32_t trial_seed = base_seed + static_cast<uint32_t>(initial * 101);

                json::Object params;
                params.emplace("function", json::Value(function_name));
                params.emplace("initial", json::Value(initial));
                params.emplace("tol", json::Value(opts.tol));
                params.emplace("max_iters", json::Value(static_cast<double>(opts.max_iters)));

                for (auto precision : precisions) {
                    switch (precision) {
                        case fmt::Precision::FP64: {
                            emit_run(params, algo, "1", precision, trial_seed, writer,
                                     truth_vec, std::vector<double>{truth_result.root},
                                     truth_result.iterations, truth_result.converged, baseline_elapsed);
                            break;
                        }
                        case fmt::Precision::FP32: {
                            float init = static_cast<float>(initial);
                            core::ScopedTimer timer;
                            auto result = alg::newton_raphson<float>(
                                init,
                                [&](float x) { return static_cast<float>(function(x)); },
                                [&](float x) { return static_cast<float>(derivative(x)); },
                                opts);
                            auto elapsed = timer.elapsed_ms();
                            emit_run(params, algo, "1", precision, trial_seed, writer,
                                     truth_vec, std::vector<float>{result.root},
                                     result.iterations, result.converged, elapsed);
                            break;
                        }
                        case fmt::Precision::TF32: {
                            fmt::TF32 init(initial);
                            core::ScopedTimer timer;
                            auto result = alg::newton_raphson<fmt::TF32>(
                                init,
                                [&](fmt::TF32 x) { return fmt::TF32(function(static_cast<double>(x))); },
                                [&](fmt::TF32 x) { return fmt::TF32(derivative(static_cast<double>(x))); },
                                opts);
                            auto elapsed = timer.elapsed_ms();
                            emit_run(params, algo, "1", precision, trial_seed, writer,
                                     truth_vec, std::vector<fmt::TF32>{result.root},
                                     result.iterations, result.converged, elapsed);
                            break;
                        }
                        case fmt::Precision::BF16: {
                            fmt::BF16 init(initial);
                            core::ScopedTimer timer;
                            auto result = alg::newton_raphson<fmt::BF16>(
                                init,
                                [&](fmt::BF16 x) { return fmt::BF16(function(static_cast<double>(x))); },
                                [&](fmt::BF16 x) { return fmt::BF16(derivative(static_cast<double>(x))); },
                                opts);
                            auto elapsed = timer.elapsed_ms();
                            emit_run(params, algo, "1", precision, trial_seed, writer,
                                     truth_vec, std::vector<fmt::BF16>{result.root},
                                     result.iterations, result.converged, elapsed);
                            break;
                        }
                        case fmt::Precision::P3109_8: {
                            fmt::P3109Number::set_accumulate_fp32(true);
                            fmt::P3109Number init(initial);
                            core::ScopedTimer timer;
                            auto result = alg::newton_raphson<fmt::P3109Number>(
                                init,
                                [&](fmt::P3109Number x) {
                                    double xd = static_cast<double>(x);
                                    return fmt::P3109Number(function(xd));
                                },
                                [&](fmt::P3109Number x) {
                                    double xd = static_cast<double>(x);
                                    return fmt::P3109Number(derivative(xd));
                                },
                                opts);
                            auto elapsed = timer.elapsed_ms();
                            emit_run(params, algo, "1", precision, trial_seed, writer,
                                     truth_vec, std::vector<fmt::P3109Number>{result.root},
                                     result.iterations, result.converged, elapsed);
                            break;
                        }
                    }
                }
            }
        } else {
            throw std::runtime_error("Unsupported algo: " + algo);
        }
    }

    return 0;
}

