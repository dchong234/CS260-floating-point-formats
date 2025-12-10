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

### Running Tests

The test suite includes:

```bash
ctest --output-on-failure          # Run all tests
ctest -R MatMulTests               # Run specific test suite
```

**Test suites:**
- **MatMulTests**: Validates matrix multiplication correctness with known 2×2 example
- **IterativeTests**: Tests gradient descent convergence and Newton-Raphson root finding
- **IOTests**: Verifies CSV file writing functionality
- **FIRTests**: Tests FIR filter convolution with known filter coefficients and signals

Tests use FP64 (double precision) and verify algorithms produce correct results, not precision comparisons.

## Run an Experiment

```bash
./fpstudy --config ../configs/small_sanity.json
```

Each config specifies experiments, precisions to sweep, and seeds. Results are written to `results/*.csv` (the directory is auto-created). The provided `configs/small_sanity.json` runs quickly (<5 s) while `configs/sweep_example.json` performs a larger sweep. For comprehensive evaluation, use `configs/full_sweep.json` which includes all algorithms across a wide parameter range (~655 experiments, 4-8 hours runtime).

### Command-Line Options

```bash
./fpstudy --config <path>     # Run experiments from JSON config
./fpstudy -c <path>           # Short form
./fpstudy --help              # Show usage information
```

### Configuration File Format

Configuration files are JSON with the following structure:

```json
{
  "seed": 123,                    // Base seed for reproducibility
  "out_csv": "results/output.csv", // Output CSV file path
  "experiments": [                 // Array of experiment configurations
    {
      "algo": "matmul",            // Algorithm name
      "sizes": [32, 64, 128],      // Problem sizes
      "precisions": ["fp64", "fp32", "tf32", "bf16", "p3109_8"],
      "accumulate_in_fp32": [true, false], // For P3109_8
      "kahan": false,              // Enable Kahan summation
      "trials": 5                  // Number of random trials
    }
  ]
}
```

**Available algorithms:**
- `matmul`: Matrix multiplication (requires `sizes` array)
- `gd_quadratic`: Gradient descent (requires `dim`, `step_size`, `max_iters`, `tol`, optional `ill_conditioned`)
- `newton`: Newton-Raphson (requires `function`, `initials` array, `max_iters`, `tol`)
- `fir`: FIR filtering (requires `filter_order`, `signal_length`, optional `trials`, `kahan`)

### Example Configurations

- **`small_sanity.json`**: Quick validation test (<5 seconds)
- **`sweep_example.json`**: Medium-sized parameter sweep
- **`full_sweep.json`**: Comprehensive evaluation (~655 experiments, 4-8 hours)
- **`fir_test.json`**: FIR filter specific test

## Precision Formats

| Label | Backing type | Notes |
|-------|--------------|-------|
| `fp64` | `double` | Baseline truth |
| `fp32` | `float` | Native single precision |
| `tf32` | `sw::universal::cfloat<19,8>` | TensorFloat-32 emulation |
| `bf16` | `sw::universal::cfloat<16,8>` | bfloat16 emulation |
| `p3109_8` | custom wrapper | 8-bit (1 sign, 3 exponent, 4 mantissa bits) quantizer |

`p3109_8` is a custom ultra-low precision format implementing the IEEE P3109 proposal. It uses explicit quantize/dequantize helpers with a configurable accumulation mode:

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

The seeding strategy ensures:
- Same seed → same random inputs
- Deterministic trial generation based on problem parameters
- Reproducible results across different runs and machines

## Accuracy Trends

- **Matrix Multiplication**: FP32 relative error grows with matrix size as longer accumulations magnify rounding. TF32 and BF16 typically land between FP64 and FP32, while `p3109_8` benefits substantially from FP32 accumulation when enabled.
- **Gradient Descent**: Reduced precision can slow or prevent convergence, especially for ill-conditioned problems. Lower precision formats may require more iterations or fail to converge entirely.
- **Newton-Raphson**: Precision limits the achievable accuracy of the root estimate. Lower precision formats may converge to less accurate solutions or fail to converge for poorly-chosen initial guesses.
- **FIR Filtering**: Error accumulates over long signal sequences. Filter order and signal length both affect precision sensitivity, with longer filters and signals showing larger relative errors in reduced precision formats.

## Interpreting Results

### CSV Output Columns

- **algo**: Algorithm name (matmul, gd_quadratic, newton, fir)
- **size**: Problem size (matrix dimension, signal length, etc.)
- **precision**: Floating-point format used
- **seed**: Random seed for this experiment
- **params_json**: JSON string with algorithm-specific parameters
- **rel_error**: Relative error vs FP64 ground truth (0.0 = perfect)
- **iters**: Number of iterations (0 for non-iterative, actual count for iterative)
- **converged**: 1 if converged, 0 if failed
- **n_nan**: Count of NaN values in result
- **n_inf**: Count of Infinity values in result
- **elapsed_ms**: Runtime in milliseconds

### Understanding Relative Error

Relative error is computed as:
```
error = ||result - truth|| / ||truth||
```

Where `||·||` is the L2 (Euclidean) norm:
```
||v|| = sqrt(Σ v[i]²)
```

**Mathematical formula:**
```
            sqrt(Σ (truth[i] - result[i])²)
error = ─────────────────────────────────────
              sqrt(Σ truth[i]²)
```

**Interpretation:**
- `0.0` = Perfect match (only FP64 matches itself exactly)
- `0.001` = 0.1% error (excellent)
- `0.01` = 1% error (good)
- `0.1` = 10% error (acceptable for some applications)
- `1.0` = 100% error (poor)
- `> 1.0` = Very poor or divergent

### Example Analysis

To compare formats for a specific algorithm:
```bash
# Filter results by algorithm
grep "matmul" results/full_sweep.csv > matmul_results.csv

# Compare precisions
grep "fp32" matmul_results.csv | awk -F',' '{print $6}' | sort -n
```

## Repository Layout

```
include/
  algorithms/     Algorithm implementations (matmul, gradient_descent, newton, fir)
  core/           Utilities (io, metrics, random)
  formats/        Precision format definitions (precision, quantize)
src/
  core/           IO implementation
  formats/        Precision format implementation
  main.cpp        CLI entry point and experiment orchestration
configs/          Example JSON experiment configurations
tests/            Unit tests (matmul, iterative, io)
results/          CSV output files (gitignored, auto-created)
third_party/      Placeholder for manual Universal checkout if needed
```

## Project Information

This project implements a comprehensive floating-point precision evaluation framework for analyzing reduced-precision formats (TF32, BF16, P3109_8) across multiple numerical algorithms.

**Key Features:**
- ✅ All 4 algorithms implemented (matmul, gradient descent, Newton-Raphson, FIR)
- ✅ All 5 precision formats supported (FP64, FP32, TF32, BF16, P3109_8)
- ✅ Optional Kahan summation for improved accuracy
- ✅ Mixed-precision accumulation modes for ultra-low precision
- ✅ Deterministic, reproducible experiments
- ✅ Comprehensive CSV output for analysis
- ✅ Extensible architecture for adding new algorithms/formats

## Troubleshooting

### Build Issues

**CMake can't find compiler:**
- Ensure C++20-compatible compiler is installed and in PATH
- On macOS: `xcode-select --install`
- On Linux: Install `g++` or `clang++` (version 11+)

**Stillwater Universal fetch fails:**
- Check internet connection (FetchContent requires git access)
- Manual workaround: Clone Universal to `third_party/universal/` and set `UNIVERSAL_EXTERN_DIR`

**Tests fail:**
- Ensure build type is Release: `cmake -DCMAKE_BUILD_TYPE=Release ..`
- Rebuild cleanly: `rm -rf build && mkdir build && cd build && cmake ..`

### Runtime Issues

**Config file not found:**
- Use absolute paths or paths relative to current directory
- Check JSON syntax with `python3 -m json.tool config.json`

**CSV output directory:**
- The `results/` directory is created automatically
- Ensure write permissions in the build directory

**Out of memory for large experiments:**
- Reduce problem sizes or number of trials
- Use `full_sweep.json` settings as maximum recommended values

## License

This project is for academic/research purposes as part of CS260 coursework.

## Extending

### Adding New Precision Formats

1. Implement the format type in `include/formats/precision.hpp`
2. Add to `Precision` enum
3. Specialize `PrecisionTraits<Precision::NEW_FORMAT>`
4. Update `precision_to_string()` and `precision_from_string()`
5. Add switch case in `main.cpp` for all algorithms

### Adding New Algorithms

1. Create templated header in `include/algorithms/`
2. Follow existing pattern (template on type T)
3. Support optional features (Kahan summation, FP32 accumulation)
4. Add algorithm case in `main.cpp`:
   - Parse config parameters
   - Generate problem instances
   - Compute FP64 ground truth
   - Test all precision formats
   - Emit results via `emit_run()`
5. Update README documentation

### Extending Configuration Schema

The lightweight JSON parser in `core/io.hpp` supports:
- Objects, arrays, numbers, strings, booleans
- Add new fields by parsing in the algorithm-specific section of `main.cpp`
- All parameters are automatically logged in `params_json` column