// SPDX-License-Identifier: GPL-3.0-only
// Host-side unit tests for the CUDA-free GPU batch sizing policy.
#include "../GPU/GPUBatch.h"
#include <cassert>
#include <cstdio>

static uint64_t mask(int bits) {
  if(bits <= 0) return 0;
  if(bits >= 64) return ~0ULL;
  return ~((1ULL << (64 - bits)) - 1);
}

int main() {
  using namespace kangaroo;
  assert(CountDpBits(0) == 0);
  assert(CountDpBits(mask(8)) == 8);
  assert(CountDpBits(mask(64)) == 64);
  assert(CountDpBits(0xF0F0F0F0F0F0F0F0ULL) == 32);

  // Large headroom: configured cap wins.
  assert(SelectGpuBatchSteps(1, 0, 131072, 64) == 64);
  assert(SelectGpuBatchSteps(1024, mask(10), 1000, 64) == 64);
  assert(SelectGpuBatchSteps(1ULL << 20, mask(16), 131072, 64) == 64);

  // Exact 50% occupancy boundary.
  assert(SelectGpuBatchSteps(1ULL << 20, mask(10), 131072, 64) == 64);

  // High DP rate: shrink the batch instead of overflowing the output buffer.
  assert(SelectGpuBatchSteps(3ULL << 20, mask(10), 131072, 64) == 21);
  assert(SelectGpuBatchSteps(1024, 0, 1000, 64) == 1);

  // Degenerate inputs never return zero.
  assert(SelectGpuBatchSteps(0, 0, 0, 0) == 1);
  assert(SelectGpuBatchSteps(1, 0, 10, 0) == 1);

  // Configured cap is respected.
  assert(SelectGpuBatchSteps(1, 0, 131072, 7) == 7);
  std::printf("batch policy: OK\n");
  return 0;
}
