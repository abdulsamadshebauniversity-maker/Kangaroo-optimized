# Engineering cycle record

## 1. Scope and requirements
- Target: NVIDIA GPU throughput and result reliability for the upstream
  Pollard-kangaroo solver, while preserving CPU behavior and work-file formats.
- Preserve GPL-3.0-only licensing and upstream attribution.
- Publish to the user-owned public repository
  `abdulsamadshebauniversity-maker/Kangaroo-optimized`, not upstream.
- Do not claim a GPU speedup without same-hardware A/B measurements.

## 2. Baseline
- Upstream commit: `37576c82d198c20fca65b14da74138ae6153a446`.
- Environment: Debian/Linux x86_64, GCC 14.2, 2 CPU cores, ~1.9 GiB RAM,
  **no NVIDIA GPU/driver**. CUDA 13.0.88 compiler installed for offline checks.
- Baseline CPU build initially failed with GCC 14 due to missing `<cstdint>`.
  A compatibility flag (`-include cstdint`) produced the baseline binary used for
  CPU A/B; the fork fixes the header directly.

## 3. Hot-path review
Findings in `GPU/GPUEngine.cu`, `GPU/GPUCompute.h`, `GPU/GPUMath.h`:
1. Fixed `NB_RUN` batches can overflow the DP output buffer for small DP masks /
   large grids, losing distinguished points and therefore collisions.
2. Block-wide `__syncthreads()` exists although the kernel uses no shared memory
   and threads do not exchange walker state.
3. Spin-wait mode copied the entire output buffer and then the populated prefix.
4. CUDA event created/destroyed every launch; input staging memory was
   write-combined despite being read back during checkpointing.
5. CUDA errors were often logged but not propagated; early GPU failure could leave
   the process in an invalid state.
6. DP payload parsing cast 4-byte-aligned data to `uint64_t*`.

## 4. Design decisions
- **Batch policy:** pure host-side function, unit-testable without CUDA. Expected
  DP occupancy ≤50% of `maxFound`; clamp to configured max steps. This changes
  launch granularity, not the walk sequence.
- **Barriers:** compile-time `GPU_BARRIERS`, default 0. Rationale: no shared-memory
  producer/consumer relationship. Escape hatch retained for architecture-specific
  A/B. **Must pass `-gpu -check` on target hardware.**
- **Group size / step cap:** exposed as compile-time knobs, defaults unchanged
  (128 / 64) because offline resource counts alone cannot select a runtime winner.
- **Error handling:** GPU methods return bool; worker aborts search and releases
  the startup barrier; `Run`/`Check` return status to `main`.
- **No algorithmic shortcut:** no change to jump distribution, DP definition,
  symmetry handling, or work-file format.

## 5. Verification matrix
| Check | Status |
|---|---|
| CPU build (GCC 14) | Pass |
| CPU regression suite | Pass |
| ASan + UBSan regression suite | Pass (leak detection disabled) |
| GPU-enabled binary CPU regression | Pass |
| CUDA compile/link sm_86, sm_90 | Pass |
| CUDA compile variants (group 64/32, barriers on/off) | Pass; resource counts recorded |
| No-device GPU failure path | Nonzero exit, no hang |
| GPU runtime parity (`-gpu -check`) | **Pending target hardware** |
| GPU throughput A/B | **Pending target hardware** |
| Work-file format compatibility | Save/resume/integrity tests pass; upstream merge compatibility not exhaustively tested |

## 6. Release checklist
- [x] License/attribution preserved
- [x] CPU tests and sanitizers
- [x] CUDA compile matrix
- [x] Benchmark harness and documentation
- [ ] Run `./kangaroo -gpu -gpuId 0 -check` on target GPU
- [ ] Run GPU A/B benchmark with fixed grid/DP/duration
- [ ] Confirm no `items lost` warnings on target workload
- [ ] Tag release only after GPU parity and benchmark review

## 7. Known risks / follow-ups
- Barrier removal and batch sizing need physical-GPU validation.
- `GPU_GRP_SIZE=32` spilled heavily in offline sm_86 compilation; do not assume
  smaller is faster.
- Legacy Windows/CUDA 8 project files are preserved but not modernized.
- Network/server mode is not covered by the new automated tests.
- Leak detection was disabled for sanitizer runs; legacy global allocations remain.
