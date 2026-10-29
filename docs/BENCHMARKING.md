# Benchmarking protocol

## Principles
1. Compare on the **same GPU**, driver, CUDA version, input interval, DP bits,
   grid, and duration.
2. Measure **sustained throughput**, not lucky time-to-solution. The harness runs
   an interval too large to solve during the window.
3. Interleave or repeat runs; report median and spread.
4. Treat `items lost` as a correctness/reliability regression, not just noise.
5. Compile-time resource counts (registers/stack/spills) are diagnostic only.

## Build baseline and candidate
```bash
# Baseline (upstream commit). GCC 14 needs the compatibility include.
git clone https://github.com/JeanLucPons/Kangaroo.git kangaroo-upstream
cd kangaroo-upstream && git checkout 37576c82d198c20fca65b14da74138ae6153a446
make -j2 gpu=1 ccap=86 CUDA=/usr/local/cuda \
  CXXFLAGS='-DWITHGPU -m64 -mssse3 -O2 -I. -I/usr/local/cuda/include -include cstdint'
cp kangaroo ../kangaroo-baseline
cd ..

# Candidate
make -j2 gpu=1 ccap=86 CUDA=/usr/local/cuda TARGET=kangaroo
```

## GPU parity first
```bash
./kangaroo -gpu -gpuId 0 -check
```
Do not proceed to long searches if this fails.

## A/B throughput
```bash
python3 scripts/benchmark.py \
  --baseline ./kangaroo-baseline --candidate ./kangaroo \
  --gpu --threads 0 --gpu-id 0 --grid 56,256 --dp 16 \
  --duration 30 --repeats 5 --json gpu-ab.json
```
Adjust `--grid` and `--dp` to your device/workload. The JSON records binary
SHA-256, GPU/driver info, per-run medians, and lost-item warnings.

## Tuning matrix
Build separate targets and compare:
```bash
make gpu=1 ccap=86 GPU_GRP_SIZE=128 GPU_BARRIERS=1 BUILD_DIR=obj-b1 TARGET=kangaroo-b1 -j2
make gpu=1 ccap=86 GPU_GRP_SIZE=128 GPU_BARRIERS=0 BUILD_DIR=obj-b0 TARGET=kangaroo-b0 -j2
make gpu=1 ccap=86 GPU_GRP_SIZE=64  GPU_BARRIERS=0 BUILD_DIR=obj-g64 TARGET=kangaroo-g64 -j2
```
Suggested order:
1. barriers on vs off at group 128;
2. group 64 vs 128 with the better barrier setting;
3. `-gpuBatch` 32/64/128 only after parity and lost-item checks.

## Offline compile-resource observations (sm_86, nvcc 13.0.88)
See `docs/benchmarks/cuda86-compile-resources.json`.

| Variant | Registers | Barriers | Stack frame | Spill stores/loads |
|---|---:|---:|---:|---:|
| Upstream baseline* | 98 | 1 | 18560 B | 0 / 0 |
| Candidate group 128, barriers 0 | 96 | 0 | 18560 B | 0 / 0 |
| Candidate group 128, barriers 1 | 96 | 1 | 18560 B | 0 / 0 |
| Candidate group 64, barriers 0 | 96 | 0 | 9344 B | 0 / 0 |
| Candidate group 32, barriers 0 | 255 | 0 | 9472 B | 7244 / 6588 |

\*Baseline compiled with a CUDA 13 `computeMode` compatibility guard only.
**These are not runtime speed measurements.**

## CPU reference run in this workspace
`docs/benchmarks/cpu-ab-2026-10-29.json`: candidate/baseline throughput ratio
≈0.997x on a 56-bit synthetic interval, 2 CPU threads, DP 8, 6 s × 2 repeats.
This is consistent with no intended CPU speedup; the fork targets GPU behavior.
