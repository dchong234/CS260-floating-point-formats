# Floating-Point Precision Study

`fpstudy` is a C++20 command-line tool for comparing reduced-precision floating-point formats against FP64 on multiple numerical workloads:

- Square matrix multiplication
- Gradient descent on positive definite quadratics
- Newton–Raphson root finding
- FIR (Finite Impulse Response) filtering

Metrics (relative error, convergence behaviour, NaN/Inf counts, runtime) are exported to CSV for downstream analysis.

## Build Dependencies

- CMake ≥ 3.20
- A C++20-compatible compiler (Clang 15+, GCC 11+, MSVC 2022)
- Git (required so CMake’s FetchContent can pull Stillwater Universal)

No Python or additional scripting is needed; Universal and all project code build as part of the standard configure step.

## Stillwater Universal

The Stillwater **Universal** library is fetched automatically at configure time through `FetchContent`. If your machine lacks outbound network access, download Universal manually and either vendor it into `third_party/universal` or set `UNIVERSAL_EXTERN_DIR` when invoking CMake to point at a local checkout.

## Build & Test

```bash
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . -j
ctest --output-on-failure
```

The build produces:

- `fpstudy` — CLI for running precision experiments
- `fpstudy_tests` — minimal test harness invoked by `ctest`

## Run an Experiment

```bash
./fpstudy --config ../configs/small_sanity.json
```

Each config specifies experiments, precisions to sweep, and seeds. Results are written to `results/*.csv` (the directory is auto-created). The provided `configs/small_sanity.json` runs quickly (<5 s) while `configs/sweep_example.json` performs a larger sweep. For comprehensive evaluation, use `configs/full_sweep.json` which includes all algorithms across a wide parameter range (~655 experiments, 4-8 hours runtime).

## Precision Formats

| Label | Backing type | Notes |
|-------|--------------|-------|
| `fp64` | `double` | Baseline truth |
| `fp32` | `float` | Native single precision |
| `tf32` | `sw::universal::cfloat<19,8>` | TensorFloat-32 emulation |
| `bf16` | `sw::universal::cfloat<16,8>` | bfloat16 emulation |
| `p3109_8` | custom wrapper | 8-bit (1 sign, 3 exponent, 4 mantissa bits) quantizer |

`p3109_8` uses explicit quantize/dequantize helpers with a configurable accumulation mode:

- `accumulate_in_fp32=true` keeps partial sums in FP32 before re-quantizing
- `accumulate_in_fp32=false` re-quantizes every multiply-add

Switching the flag highlights why mixed-precision accumulation dramatically improves accuracy, especially in long dot products such as matmul inners.

## Algorithms

### Matrix Multiplication
Square matrix multiplication (`matmul`) tests precision effects in large dot product accumulations. Supports optional Kahan summation and FP32 accumulation modes for P3109_8.

### Gradient Descent
Gradient descent on positive definite quadratics (`gd_quadratic`) evaluates convergence behavior across precisions. Configurable step size, tolerance, and iteration limits. Supports both well-conditioned and ill-conditioned problem instances.

### Newton-Raphson
Root finding via Newton-Raphson iteration (`newton`) tests precision impact on iterative convergence. Configurable function, initial guesses, tolerance, and iteration limits.

### FIR Filtering
Finite Impulse Response (FIR) filtering (`fir`) performs convolution: `y[n] = Σ h[k] * x[n-k]` where `h[k]` are normalized filter coefficients and `x[n]` is the input signal. Tests precision effects in signal processing applications with configurable filter order (M taps) and signal length (N samples). Supports optional Kahan summation and FP32 accumulation modes. Filter coefficients are normalized to sum to 1 for each trial.

## CSV Schema

One row is produced per algorithm/size/precision/trial combination:

```
algo,size,precision,seed,params_json,rel_error,iters,converged,n_nan,n_inf,elapsed_ms
```

`params_json` captures algorithm-specific knobs:
- **matmul**: size, trial, accumulate_in_fp32, kahan
- **gd_quadratic**: dim, trial, step_size, tol, max_iters, ill_conditioned
- **newton**: function, initial, tol, max_iters
- **fir**: filter_order, signal_length, trial, accumulate_in_fp32, kahan

This allows downstream tooling to regroup results by any parameter.

## Determinism

All randomness flows from the root `seed` in the config plus loop indices, ensuring reproducible experiments across platforms. FP64 truth is computed once per problem instance and reused when evaluating other precisions.

## Accuracy Trends

- **Matrix Multiplication**: FP32 relative error grows with matrix size as longer accumulations magnify rounding. TF32 and BF16 typically land between FP64 and FP32, while `p3109_8` benefits substantially from FP32 accumulation when enabled.
- **Gradient Descent**: Reduced precision can slow or prevent convergence, especially for ill-conditioned problems. Lower precision formats may require more iterations or fail to converge entirely.
- **Newton-Raphson**: Precision limits the achievable accuracy of the root estimate. Lower precision formats may converge to less accurate solutions or fail to converge for poorly-chosen initial guesses.
- **FIR Filtering**: Error accumulates over long signal sequences. Filter order and signal length both affect precision sensitivity, with longer filters and signals showing larger relative errors in reduced precision formats.

## Repository Layout

```
include/         Headers (algorithms, formats, utilities)
src/             CLI entry point and shared sources
configs/         Example experiment descriptions
tests/           Minimal regression tests (used by ctest)
third_party/     Placeholder for manual Universal checkout if needed
results/         Output CSVs (gitignored)
```

## Extending

- Add new formats by specializing `PrecisionTraits` in `formats/precision.hpp` and updating the switch in `main.cpp`.
- Introduce algorithms as additional templated headers under `include/algorithms`.
- Extend the configuration schema by enhancing the lightweight JSON parser in `core/io`.