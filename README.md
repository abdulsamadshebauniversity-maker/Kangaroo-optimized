# Kangaroo Optimized

Independent engineering fork of [JeanLucPons/Kangaroo](https://github.com/JeanLucPons/Kangaroo),
a Pollard-kangaroo interval ECDLP solver for secp256k1.

- **Upstream baseline:** `37576c82d198c20fca65b14da74138ae6153a446`
- **License:** GPL-3.0-only (see `LICENSE.txt` and `NOTICE`)
- **Fork version:** `2.2-opt.1`
- **Original usage/documentation:** [`DOC/UPSTREAM_README.md`](DOC/UPSTREAM_README.md)

> **Status: engineering preview.** CPU correctness, sanitizers, checkpointing, and
> offline CUDA compilation are verified in this workspace. **GPU runtime parity and
> GPU speedup are not yet measured** because the build environment has no NVIDIA
> device. Run the GPU validation steps below on your target GPU before relying on
> the kernel changes for long searches.

## What changed

### GPU engine
- **Output-safe kernel batching:** the host selects a batch length from the walker
  count, DP mask, and output-buffer capacity, targeting ≤50% expected occupancy.
  This reduces lost distinguished points when `-d` is small or the grid is large.
- **Optional removal of block barriers:** the kernel has no shared-memory
  producer/consumer relationship, so `GPU_BARRIERS=0` (default) removes
  `__syncthreads()` straggler waits. Restore with `GPU_BARRIERS=1` for A/B testing.
- **Persistent CUDA event** instead of create/destroy per launch.
- **Cached pinned staging memory** for checkpoint readback (upstream used
  write-combined memory that the CPU also had to read).
- **Spin-wait path no longer copies the full output buffer** before copying the
  populated prefix. The default async path already transferred only the count first.
- **CUDA errors abort the search with a nonzero exit code** instead of continuing
  with invalid state. Early GPU failure releases the startup barrier.
- **CUDA 13 portability** (`cudaDeviceProp::computeMode` removal) and aligned-safe
  DP payload parsing.

### Correctness / portability
- Fixed GCC 14 build (`<cstdint>` in `Timer.h`).
- Removed undefined behavior found by UBSan/ASan: signed overflow in Montgomery
  setup and misaligned 64-bit accesses in public-key serialization.
- `-check` and search now return meaningful exit codes.
- Checkpoint save/resume/integrity regression tests.

### Engineering
- Reproducible Makefile with dependency tracking, configurable CUDA architecture,
  separate build directories/targets, and compile-time tuning knobs.
- CPU regression suite, ASan/UBSan CI, CUDA compile CI, and a **manual-only**
  self-hosted GPU validation workflow.
- Benchmark harness and recorded verification artifacts under `docs/`.

## Build

### CPU-only
```bash
make -j2
make test
```

### NVIDIA GPU
Use the compute capability of your GPU (`ccap`), e.g. 86 for RTX 30xx, 89 for
RTX 40xx, 90 for H100. CUDA 12.8+ is recommended for the CI matrix.

```bash
make -j2 gpu=1 ccap=86 CUDA=/usr/local/cuda
./kangaroo -l
./kangaroo -gpu -gpuId 0 -check       # CPU/GPU parity; required before long runs
```

Offline compile/link without a GPU is possible with the pinned wheels in
`tools/requirements-cuda-compile.txt`; this cannot run kernels.

## Usage

Upstream flags are preserved. New flag:

```text
-gpuBatch maxSteps   Maximum GPU steps per launch, 1..4096 (default 64).
                     The engine may choose a smaller value to protect the DP buffer.
```

Example:
```bash
./kangaroo -gpu -gpuId 0 -t 0 -d 16 -gpuBatch 64 -o result.txt in.txt
```

### Compile-time GPU tuning knobs
Defaults preserve upstream walker group size and step cap; barriers are the one
kernel-default change and must be parity-checked on your GPU.

```bash
# A/B variants (use separate BUILD_DIR and TARGET)
make gpu=1 ccap=86 GPU_GRP_SIZE=128 GPU_BARRIERS=1 BUILD_DIR=obj-b1 TARGET=kangaroo-b1 -j2
make gpu=1 ccap=86 GPU_GRP_SIZE=64  GPU_BARRIERS=0 BUILD_DIR=obj-g64 TARGET=kangaroo-g64 -j2
```

- `GPU_GRP_SIZE`: walkers per CUDA thread (upstream 128). Smaller values reduce
  local-memory stack but can increase inversion overhead; 32 spilled heavily in
  offline sm_86 compilation. Benchmark 64 vs 128 on your device.
- `NB_RUN`: default maximum steps per launch (upstream 64).
- `GPU_BARRIERS`: 0 removes block barriers (default), 1 restores them.

## Tests and benchmarks

```bash
python3 tests/run_tests.py --binary ./kangaroo
python3 scripts/benchmark.py --binary ./kangaroo --threads 2 --dp 8 --duration 6

# GPU A/B on the target machine
python3 scripts/benchmark.py --baseline ./kangaroo-baseline --candidate ./kangaroo \
  --gpu --threads 0 --gpu-id 0 --grid 56,256 --dp 16 \
  --duration 30 --repeats 5 --json gpu-ab.json
```

See:
- [`docs/ENGINEERING.md`](docs/ENGINEERING.md) — requirements, design, risks, release checklist
- [`docs/BENCHMARKING.md`](docs/BENCHMARKING.md) — measurement protocol and tuning matrix
- [`docs/VERIFICATION.md`](docs/VERIFICATION.md) — what was actually run and observed
- [`docs/benchmarks/`](docs/benchmarks/) — CPU A/B JSON and CUDA compile-resource counts

## Responsible use

Use this software only on keys/challenges you own or are explicitly authorized to
test. It is a research/engineering tool, not a service for recovering third-party
keys. Do not commit private keys, work files containing sensitive state, or tokens.

## License and attribution

Original code copyright Jean-Luc PONS and contributors, GPL-3.0-only.
This fork preserves the license and attribution; see `NOTICE` and `LICENSE.txt`.
