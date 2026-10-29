# Verification record (2026-10-29)

## Environment
- Linux x86_64, GCC 14.2.0, 2 CPU cores, ~1.9 GiB RAM.
- No NVIDIA GPU or driver. CUDA 13.0.88 compiler/runtime wheels used for
  offline compilation/link only.
- Upstream baseline: `37576c82d198c20fca65b14da74138ae6153a446`.

## Commands and results
| Command / check | Result |
|---|---|
| `make -j2` (GCC 14) | Pass |
| `make CXX=clang++` (Clang 19) and regression suite | Pass |
| `python3 tests/run_tests.py --binary ./kangaroo` | ALL TESTS PASSED |
| ASan+UBSan build and tests (`ASAN_OPTIONS=detect_leaks=0`, `UBSAN_OPTIONS=halt_on_error=1`) | ALL TESTS PASSED |
| GPU-enabled sm_86 binary, CPU regression suite | ALL TESTS PASSED |
| CUDA compile/link `ccap=86` | Pass |
| CUDA compile/link `ccap=90` | Pass |
| `kangaroo-cuda86 -l` without device | Graceful CUDA error |
| `kangaroo-cuda86 -gpu -check` without device | Exit 1, no hang |
| CPU A/B benchmark harness | Ran; ratio ≈0.997x (see JSON) |
| YAML workflow parse | Pass |

## Regression suite coverage
- Independent secp256k1 oracle known-answer check.
- GPU batch-policy unit tests (CUDA-free).
- Upstream arithmetic self-check.
- `-gpuBatch` CLI validation.
- 24/28/32-bit synthetic interval solves with independent public-key verification.
- Checkpoint save, resume, and `-wcheck` integrity.

## Artifacts
- `docs/benchmarks/cpu-ab-2026-10-29.json`
- `docs/benchmarks/cuda86-compile-resources.json`

## Explicitly not verified
- Execution of CUDA kernels on physical hardware.
- CPU/GPU parity on physical hardware.
- GPU throughput speedup or time-to-solution improvement.
- Windows / legacy CUDA 8 builds.
- Network/server distributed mode.
