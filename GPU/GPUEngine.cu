/*
* This file is part of the BTCCollider distribution (https://github.com/JeanLucPons/Kangaroo).
* Copyright (c) 2020 Jean Luc PONS.
*
* This program is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, version 3.
*
* This program is distributed in the hope that it will be useful, but
* WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
* General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef WIN64
#include <unistd.h>
#include <stdio.h>
#endif

#include "GPUEngine.h"
#include <cuda.h>
#include <cuda_runtime.h>

#include <stdint.h>
#include <climits>
#include <cstdio>
#include <cstring>
#include "../Timer.h"

#include "GPUMath.h"
#include "GPUCompute.h"
#include "GPUBatch.h"

static bool CudaOk(cudaError_t err, const char *what) {
  if(err != cudaSuccess) {
    fprintf(stderr, "GPUEngine: %s: %s\n", what, cudaGetErrorString(err));
    return false;
  }
  return true;
}

// ---------------------------------------------------------------------------------------

__global__ void comp_kangaroos(uint64_t *kangaroos,uint32_t maxFound,uint32_t *found,uint64_t dpMask,uint32_t nbRun) {

  int xPtr = (blockIdx.x*blockDim.x*GPU_GRP_SIZE) * KSIZE; // x[4] , y[4] , d[2], lastJump
  ComputeKangaroos(kangaroos + xPtr,maxFound,found,dpMask,nbRun);

}

// ---------------------------------------------------------------------------------------
//#define GPU_CHECK
#ifdef GPU_CHECK
__global__ void check_gpu() {

  // Check ModInv
  uint64_t N[5] = { 0x0BE3D7593BE1147CULL,0x4952AAF512875655ULL,0x08884CCAACCB9B53ULL,0x9EAE2E2225044292ULL,0ULL };
  uint64_t I[5];
  uint64_t R[5];
  bool ok = true;

  /*
  for(uint64_t i=0;i<10000 && ok;i++) {

    Load(R,N);
    _ModInv(R);
    Load(I,R);
    _ModMult(R,N);
    SubP(R);
    if(!_IsOne(R)) {
      ok = false;
      printf("ModInv wrong %d\n",(int)i);
      printf("N = %016llx %016llx %016llx %016llx %016llx\n",N[4],N[3],N[2],N[1],N[0]);
      printf("I = %016llx %016llx %016llx %016llx %016llx\n",I[4],I[3],I[2],I[1],I[0]);
      printf("R = %016llx %016llx %016llx %016llx %016llx\n",R[4],R[3],R[2],R[1],R[0]);
    }

    N[0]++;

  }
  */
  I[4] = 0;
  R[4] = 0;
  for(uint64_t i = 0; i < 100000 && ok; i++) {

    _ModSqr(I,N);
    _ModMult(R,N,N);
    if(!_IsEqual(I,R)) {
      ok = false;
      printf("_ModSqr wrong %d\n",(int)i);
      printf("N = %016llx %016llx %016llx %016llx %016llx\n",N[4],N[3],N[2],N[1],N[0]);
      printf("I = %016llx %016llx %016llx %016llx %016llx\n",I[4],I[3],I[2],I[1],I[0]);
      printf("R = %016llx %016llx %016llx %016llx %016llx\n",R[4],R[3],R[2],R[1],R[0]);
    }

    N[0]++;

  }

}
#endif

// ---------------------------------------------------------------------------------------

using namespace std;

int _ConvertSMVer2Cores(int major,int minor) {

  // Defines for GPU Architecture types (using the SM version to determine
  // the # of cores per SM
  typedef struct {
    int SM;  // 0xMm (hexidecimal notation), M = SM Major version,
             // and m = SM minor version
    int Cores;
  } sSMtoCores;

  sSMtoCores nGpuArchCoresPerSM[] = {
    { 0x20, 32 }, // Fermi Generation (SM 2.0) GF100 class
    { 0x21, 48 }, // Fermi Generation (SM 2.1) GF10x class
    { 0x30, 192 },
    { 0x32, 192 },
    { 0x35, 192 },
    { 0x37, 192 },
    { 0x50, 128 },
    { 0x52, 128 },
    { 0x53, 128 },
    { 0x60,  64 },
    { 0x61, 128 },
    { 0x62, 128 },
    { 0x70,  64 },
    { 0x72,  64 },
    { 0x75,  64 },
    { -1, -1 } };

  int index = 0;

  while(nGpuArchCoresPerSM[index].SM != -1) {
    if(nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
      return nGpuArchCoresPerSM[index].Cores;
    }

    index++;
  }

  return 0;

}

void GPUEngine::SetWildOffset(Int* offset) {
  wildOffset.Set(offset);
}

GPUEngine::GPUEngine(int nbThreadGroup,int nbThreadPerGroup,int gpuId,uint32_t maxFound,uint32_t maxSteps)
  : nbThread(0), nbThreadPerGroup(0), gpuId(-1),
    inputKangaroo(NULL), inputKangarooPinned(NULL), outputItem(NULL),
    outputItemPinned(NULL), jumpPinned(NULL), completionEvent(NULL),
    initialised(false), lostWarning(false), maxFound(0), maxSteps(0),
    stepsPerLaunch(1), lostItemsTotal(0), outputSize(0), kangarooSize(0),
    kangarooSizePinned(0), jumpSize(0), dpMask(0) {

  this->gpuId = gpuId;
  int deviceCount = 0;
  if(!CudaOk(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount")) return;
  if(deviceCount == 0) {
    fprintf(stderr, "GPUEngine: no CUDA-capable device is available\n");
    return;
  }
  if(gpuId < 0 || gpuId >= deviceCount) {
    fprintf(stderr, "GPUEngine: invalid GPU id %d (device count %d)\n", gpuId, deviceCount);
    return;
  }
  if(!CudaOk(cudaSetDevice(gpuId), "cudaSetDevice")) return;

  cudaDeviceProp deviceProp;
  if(!CudaOk(cudaGetDeviceProperties(&deviceProp, gpuId), "cudaGetDeviceProperties")) return;

  if(nbThreadGroup <= 0 || nbThreadPerGroup <= 0 ||
     nbThreadPerGroup > deviceProp.maxThreadsPerBlock ||
     nbThreadGroup > deviceProp.maxGridSize[0]) {
    fprintf(stderr, "GPUEngine: invalid grid %dx%d for this device (max block %d, max grid %d)\n",
            nbThreadGroup, nbThreadPerGroup, deviceProp.maxThreadsPerBlock, deviceProp.maxGridSize[0]);
    return;
  }
  const uint64_t totalThreads = static_cast<uint64_t>(nbThreadGroup) * static_cast<uint64_t>(nbThreadPerGroup);
  if(totalThreads == 0 || totalThreads > static_cast<uint64_t>(INT_MAX)) {
    fprintf(stderr, "GPUEngine: thread count out of range\n");
    return;
  }
  if(maxFound == 0 || maxSteps == 0) {
    fprintf(stderr, "GPUEngine: maxFound and maxSteps must be positive\n");
    return;
  }

  this->nbThreadPerGroup = nbThreadPerGroup;
  this->nbThread = static_cast<int>(totalThreads);
  this->maxFound = maxFound;
  this->maxSteps = maxSteps;
  this->stepsPerLaunch = maxSteps;
  this->outputSize = static_cast<size_t>(maxFound) * ITEM_SIZE + 4;

  char tmp[512];
  snprintf(tmp, sizeof(tmp), "GPU #%d %s (%dx%d cores) Grid(%dx%d)",
           gpuId, deviceProp.name, deviceProp.multiProcessorCount,
           _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
           nbThreadGroup, nbThreadPerGroup);
  deviceName = std::string(tmp);

  // Prefer L1 (the kernel does not use __shared__ memory).
  if(!CudaOk(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1), "cudaDeviceSetCacheConfig")) return;
  if(!CudaOk(cudaEventCreateWithFlags(reinterpret_cast<cudaEvent_t *>(&completionEvent),
                                      cudaEventDisableTiming), "cudaEventCreate")) return;

  kangarooSize = static_cast<size_t>(nbThread) * GPU_GRP_SIZE * KSIZE * 8;
  if(!CudaOk(cudaMalloc(reinterpret_cast<void **>(&inputKangaroo), kangarooSize), "allocate input memory")) return;

  kangarooSizePinned = static_cast<size_t>(nbThreadPerGroup) * GPU_GRP_SIZE * KSIZE * 8;
  // This staging buffer is read back during checkpointing, so it must be cached
  // host memory (the upstream write-combined flag made CPU reads very slow).
  if(!CudaOk(cudaHostAlloc(reinterpret_cast<void **>(&inputKangarooPinned), kangarooSizePinned,
                           cudaHostAllocDefault), "allocate input pinned memory")) return;

  if(!CudaOk(cudaMalloc(reinterpret_cast<void **>(&outputItem), outputSize), "allocate output memory")) return;
  if(!CudaOk(cudaHostAlloc(reinterpret_cast<void **>(&outputItemPinned), outputSize,
                           cudaHostAllocDefault), "allocate output pinned memory")) return;

  jumpSize = NB_JUMP * 8 * 4;
  if(!CudaOk(cudaHostAlloc(reinterpret_cast<void **>(&jumpPinned), jumpSize,
                           cudaHostAllocDefault), "allocate jump pinned memory")) return;

  wildOffset.SetInt32(0);
  initialised = true;

#ifdef GPU_CHECK
  double minT = 1e9;
  for(int i=0;i<5;i++) {
    double t0 = Timer::get_tick();
    check_gpu<<<1,1>>>();
    cudaDeviceSynchronize();
    double t1 = Timer::get_tick();
    if((t1-t0)<minT) minT = (t1-t0);
  }
  printf("Cuda: %.3f ms\n",minT*1000.0);
  exit(0);
#endif
}

GPUEngine::~GPUEngine() {
  if(gpuId >= 0) cudaSetDevice(gpuId);
  if(completionEvent) cudaEventDestroy(reinterpret_cast<cudaEvent_t>(completionEvent));
  if(inputKangaroo) cudaFree(inputKangaroo);
  if(outputItem) cudaFree(outputItem);
  if(inputKangarooPinned) cudaFreeHost(inputKangarooPinned);
  if(outputItemPinned) cudaFreeHost(outputItemPinned);
  if(jumpPinned) cudaFreeHost(jumpPinned);
}

size_t GPUEngine::GetMemory() {
  return kangarooSize + outputSize + jumpSize;
}

int GPUEngine::GetGroupSize() {
  return GPU_GRP_SIZE;
}

bool GPUEngine::GetGridSize(int gpuId,int *x,int *y) {

  if(*x <= 0 || *y <= 0) {

    int deviceCount = 0;
    if(!CudaOk(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount")) return false;
    if(deviceCount == 0) {
      fprintf(stderr, "GPUEngine: no CUDA-capable device is available\n");
      return false;
    }
    if(gpuId < 0 || gpuId >= deviceCount) {
      fprintf(stderr, "GPUEngine::GetGridSize() invalid GPU id %d\n", gpuId);
      return false;
    }

    cudaDeviceProp deviceProp;
    if(!CudaOk(cudaGetDeviceProperties(&deviceProp, gpuId), "cudaGetDeviceProperties")) return false;

    if(*x <= 0) *x = 2 * deviceProp.multiProcessorCount;
    if(*y <= 0) *y = 2 * _ConvertSMVer2Cores(deviceProp.major,deviceProp.minor);
    if(*y <= 0) *y = 128;
    if(*y > deviceProp.maxThreadsPerBlock) *y = deviceProp.maxThreadsPerBlock;
    if(*x > deviceProp.maxGridSize[0]) *x = deviceProp.maxGridSize[0];
  }

  return (*x > 0 && *y > 0);
}

void *GPUEngine::AllocatePinnedMemory(size_t size) {

  void *buff;

  cudaError_t err = cudaHostAlloc(&buff,size,cudaHostAllocPortable);
  if(err != cudaSuccess) {
    printf("GPUEngine: AllocatePinnedMemory: %s\n",cudaGetErrorString(err));
    return NULL;
  }

  return buff;

}

void GPUEngine::FreePinnedMemory(void *buff) {
  cudaFreeHost(buff);
}

void GPUEngine::PrintCudaInfo() {

  cudaError_t err;

#if CUDART_VERSION < 13000
  const char *sComputeMode[] =
  {
    "Multiple host threads",
    "Only one host thread",
    "No host thread",
    "Multiple process threads",
    "Unknown",
    NULL
  };
#endif

  int deviceCount = 0;
  cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

  if(error_id != cudaSuccess) {
    printf("GPUEngine: CudaGetDeviceCount %s\n",cudaGetErrorString(error_id));
    return;
  }

  // This function call returns 0 if there are no CUDA capable devices.
  if(deviceCount == 0) {
    printf("GPUEngine: There are no available device(s) that support CUDA\n");
    return;
  }

  for(int i = 0; i<deviceCount; i++) {

    err = cudaSetDevice(i);
    if(err != cudaSuccess) {
      printf("GPUEngine: cudaSetDevice(%d) %s\n",i,cudaGetErrorString(err));
      return;
    }

    cudaDeviceProp deviceProp;
    if(!CudaOk(cudaGetDeviceProperties(&deviceProp,i), "cudaGetDeviceProperties")) return;
#if CUDART_VERSION >= 13000
    // CUDA 13 removed cudaDeviceProp::computeMode.
    printf("GPU #%d %s (%dx%d cores) (Cap %d.%d) (%.1f MB)\n",
      i,deviceProp.name,deviceProp.multiProcessorCount,
      _ConvertSMVer2Cores(deviceProp.major,deviceProp.minor),
      deviceProp.major,deviceProp.minor,(double)deviceProp.totalGlobalMem / 1048576.0);
#else
    printf("GPU #%d %s (%dx%d cores) (Cap %d.%d) (%.1f MB) (%s)\n",
      i,deviceProp.name,deviceProp.multiProcessorCount,
      _ConvertSMVer2Cores(deviceProp.major,deviceProp.minor),
      deviceProp.major,deviceProp.minor,(double)deviceProp.totalGlobalMem / 1048576.0,
      sComputeMode[deviceProp.computeMode]);
#endif

  }

}

int GPUEngine::GetNbThread() {
  return nbThread;
}

bool GPUEngine::SetKangaroos(Int *px,Int *py,Int *d) {

  if(!initialised) return false;
  // Sets the kangaroos of each thread
  int gSize = KSIZE * GPU_GRP_SIZE;
  int strideSize = nbThreadPerGroup * KSIZE;
  int nbBlock = nbThread / nbThreadPerGroup;
  int blockSize = nbThreadPerGroup * gSize;
  int idx = 0;

  for(int b = 0; b < nbBlock; b++) {
    for(int g = 0; g < GPU_GRP_SIZE; g++) {
      for(int t = 0; t < nbThreadPerGroup; t++) {

        // X
        inputKangarooPinned[g * strideSize + t + 0 * nbThreadPerGroup] = px[idx].bits64[0];
        inputKangarooPinned[g * strideSize + t + 1 * nbThreadPerGroup] = px[idx].bits64[1];
        inputKangarooPinned[g * strideSize + t + 2 * nbThreadPerGroup] = px[idx].bits64[2];
        inputKangarooPinned[g * strideSize + t + 3 * nbThreadPerGroup] = px[idx].bits64[3];

        // Y
        inputKangarooPinned[g * strideSize + t + 4 * nbThreadPerGroup] = py[idx].bits64[0];
        inputKangarooPinned[g * strideSize + t + 5 * nbThreadPerGroup] = py[idx].bits64[1];
        inputKangarooPinned[g * strideSize + t + 6 * nbThreadPerGroup] = py[idx].bits64[2];
        inputKangarooPinned[g * strideSize + t + 7 * nbThreadPerGroup] = py[idx].bits64[3];

        // Distance
        Int dOff;
        dOff.Set(&d[idx]);
        if(idx % 2 == WILD) dOff.ModAddK1order(&wildOffset);
        inputKangarooPinned[g * strideSize + t + 8 * nbThreadPerGroup] = dOff.bits64[0];
        inputKangarooPinned[g * strideSize + t + 9 * nbThreadPerGroup] = dOff.bits64[1];

#ifdef USE_SYMMETRY
        // Last jump
        inputKangarooPinned[t + 10 * nbThreadPerGroup] = (uint64_t)NB_JUMP;
#endif

        idx++;
      }

    }

    size_t offset = static_cast<size_t>(b) * blockSize;
    if(!CudaOk(cudaMemcpy(inputKangaroo + offset,inputKangarooPinned,kangarooSizePinned,
                          cudaMemcpyHostToDevice), "SetKangaroos")) return false;

  }

  return CudaOk(cudaGetLastError(), "SetKangaroos");
}

bool GPUEngine::GetKangaroos(Int *px,Int *py,Int *d) {

  if(!initialised || inputKangarooPinned==NULL) {
    fprintf(stderr, "GPUEngine: GetKangaroos: engine is not initialised\n");
    return false;
  }

  // Sets the kangaroos of each thread
  int gSize = KSIZE * GPU_GRP_SIZE;
  int strideSize = nbThreadPerGroup * KSIZE;
  int nbBlock = nbThread / nbThreadPerGroup;
  int blockSize = nbThreadPerGroup * gSize;
  int idx = 0;

  for(int b = 0; b < nbBlock; b++) {

    size_t offset = static_cast<size_t>(b) * blockSize;
    if(!CudaOk(cudaMemcpy(inputKangarooPinned,inputKangaroo + offset,kangarooSizePinned,
                          cudaMemcpyDeviceToHost), "GetKangaroos")) return false;

    for(int g = 0; g < GPU_GRP_SIZE; g++) {

      for(int t = 0; t < nbThreadPerGroup; t++) {

        // X
        px[idx].bits64[0] = inputKangarooPinned[g * strideSize + t + 0 * nbThreadPerGroup];
        px[idx].bits64[1] = inputKangarooPinned[g * strideSize + t + 1 * nbThreadPerGroup];
        px[idx].bits64[2] = inputKangarooPinned[g * strideSize + t + 2 * nbThreadPerGroup];
        px[idx].bits64[3] = inputKangarooPinned[g * strideSize + t + 3 * nbThreadPerGroup];
        px[idx].bits64[4] = 0;

        // Y
        py[idx].bits64[0] = inputKangarooPinned[g * strideSize + t + 4 * nbThreadPerGroup];
        py[idx].bits64[1] = inputKangarooPinned[g * strideSize + t + 5 * nbThreadPerGroup];
        py[idx].bits64[2] = inputKangarooPinned[g * strideSize + t + 6 * nbThreadPerGroup];
        py[idx].bits64[3] = inputKangarooPinned[g * strideSize + t + 7 * nbThreadPerGroup];
        py[idx].bits64[4] = 0;

        // Distance
        Int dOff;
        dOff.SetInt32(0);
        dOff.bits64[0] = inputKangarooPinned[g * strideSize + t + 8 * nbThreadPerGroup];
        dOff.bits64[1] = inputKangarooPinned[g * strideSize + t + 9 * nbThreadPerGroup];
        if(idx % 2 == WILD) dOff.ModSubK1order(&wildOffset);
        d[idx].Set(&dOff);

        idx++;
      }

    }
  }

  return CudaOk(cudaGetLastError(), "GetKangaroos");
}

bool GPUEngine::SetKangaroo(uint64_t kIdx,Int *px,Int *py,Int *d) {

  if(!initialised) return false;
  if(kIdx >= static_cast<uint64_t>(nbThread) * GPU_GRP_SIZE) {
    fprintf(stderr, "GPUEngine: SetKangaroo: index out of range\n");
    return false;
  }
  int gSize = KSIZE * GPU_GRP_SIZE;
  int strideSize = nbThreadPerGroup * KSIZE;
  int blockSize = nbThreadPerGroup * gSize;

  uint64_t t = kIdx % nbThreadPerGroup;
  uint64_t g = (kIdx / nbThreadPerGroup) % GPU_GRP_SIZE;
  uint64_t b = kIdx / (nbThreadPerGroup*GPU_GRP_SIZE);

  // X
  inputKangarooPinned[0] = px->bits64[0];
  if(!CudaOk(cudaMemcpy(inputKangaroo + (b * blockSize + g * strideSize + t + 0 * nbThreadPerGroup),inputKangarooPinned,8,cudaMemcpyHostToDevice), "SetKangaroo")) return false;
  inputKangarooPinned[0] = px->bits64[1];
  if(!CudaOk(cudaMemcpy(inputKangaroo + (b * blockSize + g * strideSize + t + 1 * nbThreadPerGroup),inputKangarooPinned,8,cudaMemcpyHostToDevice), "SetKangaroo")) return false;
  inputKangarooPinned[0] = px->bits64[2];
  if(!CudaOk(cudaMemcpy(inputKangaroo + (b * blockSize + g * strideSize + t + 2 * nbThreadPerGroup),inputKangarooPinned,8,cudaMemcpyHostToDevice), "SetKangaroo")) return false;
  inputKangarooPinned[0] = px->bits64[3];
  if(!CudaOk(cudaMemcpy(inputKangaroo + (b * blockSize + g * strideSize + t + 3 * nbThreadPerGroup),inputKangarooPinned,8,cudaMemcpyHostToDevice), "SetKangaroo")) return false;

  // Y
  inputKangarooPinned[0] = py->bits64[0];
  if(!CudaOk(cudaMemcpy(inputKangaroo + (b * blockSize + g * strideSize + t + 4 * nbThreadPerGroup),inputKangarooPinned,8,cudaMemcpyHostToDevice), "SetKangaroo")) return false;
  inputKangarooPinned[0] = py->bits64[1];
  if(!CudaOk(cudaMemcpy(inputKangaroo + (b * blockSize + g * strideSize + t + 5 * nbThreadPerGroup),inputKangarooPinned,8,cudaMemcpyHostToDevice), "SetKangaroo")) return false;
  inputKangarooPinned[0] = py->bits64[2];
  if(!CudaOk(cudaMemcpy(inputKangaroo + (b * blockSize + g * strideSize + t + 6 * nbThreadPerGroup),inputKangarooPinned,8,cudaMemcpyHostToDevice), "SetKangaroo")) return false;
  inputKangarooPinned[0] = py->bits64[3];
  if(!CudaOk(cudaMemcpy(inputKangaroo + (b * blockSize + g * strideSize + t + 7 * nbThreadPerGroup),inputKangarooPinned,8,cudaMemcpyHostToDevice), "SetKangaroo")) return false;

  // D
  Int dOff;
  dOff.Set(d);
  if(kIdx % 2 == WILD) dOff.ModAddK1order(&wildOffset);
  inputKangarooPinned[0] = dOff.bits64[0];
  if(!CudaOk(cudaMemcpy(inputKangaroo + (b * blockSize + g * strideSize + t + 8 * nbThreadPerGroup),inputKangarooPinned,8,cudaMemcpyHostToDevice), "SetKangaroo")) return false;
  inputKangarooPinned[0] = dOff.bits64[1];
  if(!CudaOk(cudaMemcpy(inputKangaroo + (b * blockSize + g * strideSize + t + 9 * nbThreadPerGroup),inputKangarooPinned,8,cudaMemcpyHostToDevice), "SetKangaroo")) return false;

#ifdef USE_SYMMETRY
  // Last jump
  inputKangarooPinned[0] = (uint64_t)NB_JUMP;
  if(!CudaOk(cudaMemcpy(inputKangaroo + (b * blockSize + g * strideSize + t + 10 * nbThreadPerGroup),inputKangarooPinned,8,cudaMemcpyHostToDevice), "SetKangaroo")) return false;
#endif

  return true;
}

bool GPUEngine::callKernel() {

  if(!initialised || stepsPerLaunch == 0) return false;
  if(!CudaOk(cudaMemset(outputItem,0,4), "reset DP counter")) return false;

  comp_kangaroos << < nbThread / nbThreadPerGroup,nbThreadPerGroup >> >
      (inputKangaroo,maxFound,outputItem,dpMask,stepsPerLaunch);

  return CudaOk(cudaGetLastError(), "kernel launch");
}

bool GPUEngine::SetParams(uint64_t dpMask,Int *distance,Int *px,Int *py) {

  if(!initialised) return false;
  this->dpMask = dpMask;
  this->stepsPerLaunch = kangaroo::SelectGpuBatchSteps(
      static_cast<uint64_t>(nbThread) * GPU_GRP_SIZE, dpMask, maxFound, maxSteps);

  const long double expectedPerStep =
      static_cast<long double>(nbThread) * GPU_GRP_SIZE *
      std::ldexp(1.0L, -static_cast<int>(kangaroo::CountDpBits(dpMask)));
  if(expectedPerStep > static_cast<long double>(maxFound)) {
    fprintf(stderr, "GPUEngine: warning: expected DP rate exceeds output capacity even at one step; increase -d or reduce -g\n");
  }

  for(int i=0;i< NB_JUMP;i++)
    memcpy(jumpPinned + 2*i,distance[i].bits64,16);
  if(!CudaOk(cudaMemcpyToSymbol(jD,jumpPinned,jumpSize/2), "SetParams jD")) return false;

  for(int i = 0; i < NB_JUMP; i++)
    memcpy(jumpPinned + 4 * i,px[i].bits64,32);
  if(!CudaOk(cudaMemcpyToSymbol(jPx,jumpPinned,jumpSize), "SetParams jPx")) return false;

  for(int i = 0; i < NB_JUMP; i++)
    memcpy(jumpPinned + 4 * i,py[i].bits64,32);
  if(!CudaOk(cudaMemcpyToSymbol(jPy,jumpPinned,jumpSize), "SetParams jPy")) return false;
  return true;
}

bool GPUEngine::callKernelAndWait() {

  // Debug function
  if(!callKernel()) return false;
  return CudaOk(cudaMemcpy(outputItemPinned,outputItem,outputSize,cudaMemcpyDeviceToHost),
                "callKernelAndWait");

}

bool GPUEngine::Launch(std::vector<ITEM> &hashFound,bool spinWait) {

  hashFound.clear();
  if(!initialised) return false;

  // Wait for the previous kernel by transferring only the 4-byte DP count.
  // In spin-wait mode upstream copied the entire multi-megabyte output buffer
  // here and then copied the populated prefix again; the full-buffer copy was
  // redundant. The default async path already used a 4-byte count transfer.
  if(spinWait) {
    if(!CudaOk(cudaMemcpy(outputItemPinned,outputItem,4,cudaMemcpyDeviceToHost),
                "Launch wait")) return false;
  } else {
    if(!CudaOk(cudaMemcpyAsync(outputItemPinned,outputItem,4,cudaMemcpyDeviceToHost,0),
                "Launch async count")) return false;
    cudaEvent_t evt = reinterpret_cast<cudaEvent_t>(completionEvent);
    if(!CudaOk(cudaEventRecord(evt,0), "Launch event record")) return false;
    cudaError_t status;
    while((status = cudaEventQuery(evt)) == cudaErrorNotReady) {
      Timer::SleepMillis(1);
    }
    if(!CudaOk(status, "Launch event query")) return false;
  }

  uint32_t nbFound = outputItemPinned[0];
  if(nbFound > maxFound) {
    const uint32_t lost = nbFound - maxFound;
    lostItemsTotal += lost;
    if(!lostWarning) {
      fprintf(stderr, "\nWarning, %u items lost (total %llu)\nHint: increase -d or reduce -g\n",
              lost, static_cast<unsigned long long>(lostItemsTotal));
      lostWarning = true;
    }
    nbFound = maxFound;
  }

  if(nbFound > 0) {
    const size_t bytes = static_cast<size_t>(nbFound) * ITEM_SIZE + 4;
    if(!CudaOk(cudaMemcpy(outputItemPinned,outputItem,bytes,cudaMemcpyDeviceToHost),
                "Launch payload")) return false;
  }

  hashFound.reserve(nbFound);
  for(uint32_t i = 0; i < nbFound; i++) {
    uint32_t *itemPtr = outputItemPinned + (i*ITEM_SIZE32 + 1);
    ITEM it;

    // Use memcpy instead of casting uint32_t* to uint64_t*: the payload is
    // 4-byte aligned, and unaligned loads are undefined behavior in C++.
    uint64_t x[4];
    uint64_t d[4];
    uint64_t kIdx;
    memcpy(x, itemPtr, sizeof(x));
    memcpy(d, itemPtr + 8, sizeof(d));
    memcpy(&kIdx, itemPtr + 12, sizeof(kIdx));
    it.kIdx = kIdx;

    it.x.bits64[0] = x[0];
    it.x.bits64[1] = x[1];
    it.x.bits64[2] = x[2];
    it.x.bits64[3] = x[3];
    it.x.bits64[4] = 0;

    it.d.bits64[0] = d[0];
    it.d.bits64[1] = d[1];
    it.d.bits64[2] = 0;
    it.d.bits64[3] = 0;
    it.d.bits64[4] = 0;
    if(it.kIdx % 2 == WILD) it.d.ModSubK1order(&wildOffset);

    hashFound.push_back(it);
  }

  return callKernel();
}
