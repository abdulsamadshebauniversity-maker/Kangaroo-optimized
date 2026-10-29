# Changelog

## 2.2-opt.1 (engineering preview)

### GPU
- Add output-capacity-aware GPU batch sizing (`GPU/GPUBatch.h`) and `-gpuBatch`.
- Make kernel step count runtime-configurable; account actual steps in counters.
- Add `GPU_BARRIERS` compile switch; default 0 removes unnecessary block barriers.
- Reuse a CUDA event; use cached pinned staging memory; avoid redundant full
  output-buffer copy in spin-wait mode.
- Propagate CUDA errors to a nonzero process exit; fix early-failure startup barrier.
- CUDA 13 portability; aligned-safe DP payload parsing.

### CPU / correctness
- Fix GCC 14 `<cstdint>` portability.
- Fix UBSan signed overflow in Montgomery setup and misaligned public-key
  serialization accesses.
- Meaningful exit codes for `-check` and search.

### Engineering
- Reproducible Makefile, dependency tracking, configurable `ccap`, `BUILD_DIR`,
  `TARGET`, `GPU_GRP_SIZE`, `NB_RUN`, `GPU_BARRIERS`.
- CPU regression suite, benchmark harness, ASan/UBSan + CUDA compile CI,
  manual self-hosted GPU validation workflow.
- Verification and benchmarking documentation.

### Not yet verified
- GPU runtime parity and throughput on physical NVIDIA hardware.
