# Floating-Point Precision Study

`fpstudy` is a C++20 command-line tool for comparing reduced-precision floating-point formats against FP64 on multiple numerical workloads:

- Square matrix multiplication
- Gradient descent on positive definite quadratics
- Newton–Raphson root finding

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

Each config specifies experiments, precisions to sweep, and seeds. Results are written to `results/*.csv` (the directory is auto-created). The provided `configs/small_sanity.json` runs quickly (<5 s) while `configs/sweep_example.json` performs a larger sweep.

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

## CSV Schema

One row is produced per algorithm/size/precision/trial combination:

```
algo,size,precision,seed,params_json,rel_error,iters,converged,n_nan,n_inf,elapsed_ms
```

`params_json` captures algorithm-specific knobs (size, tolerance, accumulation mode, etc.) so downstream tooling can regroup results.

## Determinism

All randomness flows from the root `seed` in the config plus loop indices, ensuring reproducible experiments across platforms. FP64 truth is computed once per problem instance and reused when evaluating other precisions.

## Accuracy Trends

Expect FP32 relative error for matmul to grow with matrix size as longer accumulations magnify rounding. TF32 and BF16 typically land between FP64 and FP32, while `p3109_8` benefits substantially from FP32 accumulation when enabled.

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