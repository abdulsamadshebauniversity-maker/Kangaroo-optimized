/*
* This file is part of the Kangaroo distribution (https://github.com/JeanLucPons/Kangaroo).
* SPDX-License-Identifier: GPL-3.0-only
*
* Host-side GPU batch sizing policy. Kept CUDA-free so it can be unit tested
* on machines without a GPU.
*/
#ifndef KANGAROO_GPU_BATCH_H
#define KANGAROO_GPU_BATCH_H

#include <cstdint>
#include <cmath>

namespace kangaroo {

// Number of bits constrained by the distinguished-point mask.
inline uint32_t CountDpBits(uint64_t mask) {
  uint32_t n = 0;
  while(mask) { mask &= mask - 1; ++n; }
  return n;
}

// Choose a batch length whose *expected* DP output occupies at most half of
// the device output buffer. The 50% headroom absorbs the random fluctuation
// of a Poisson-like DP arrival count. This does not change the walk sequence;
// it only changes how many steps are executed before the host drains output.
//
// nbKangaroo : total tame+wild walkers launched by this engine
// dpMask     : distinguished-point mask (uniform x => probability 2^-popcount)
// capacity   : maxFound, the number of ITEM slots in the output buffer
// maxSteps   : configured upper bound (NB_RUN by default)
inline uint32_t SelectGpuBatchSteps(uint64_t nbKangaroo, uint64_t dpMask,
                                    uint32_t capacity, uint32_t maxSteps) {
  if(nbKangaroo == 0 || capacity == 0 || maxSteps == 0) return 1;
  const uint32_t dpBits = CountDpBits(dpMask);
  const long double expectedPerStep =
      static_cast<long double>(nbKangaroo) * std::ldexp(1.0L, -static_cast<int>(dpBits));
  const long double safeSteps =
      std::floor((static_cast<long double>(capacity) * 0.5L) / expectedPerStep);
  if(safeSteps < 1.0L) return 1;
  if(safeSteps > static_cast<long double>(maxSteps)) return maxSteps;
  return static_cast<uint32_t>(safeSteps);
}

} // namespace kangaroo

#endif // KANGAROO_GPU_BATCH_H
