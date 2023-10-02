#include <assert.h>
#include <stdio.h>

#include "tileorders.h"
#include "policies.h"
#include "device-functions.h"
#include "wait-kernel.h"

#pragma once

#define HOST_FUNC __host__
#define DEVICE_FUNC __device__

#define CUDA_CHECK(cmd) do {                        \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",       \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0);

template<typename T>
T divup(T x, T y) {
  return (x + y - 1)/y;
}

enum CuStageType {
  Producer = 1,
  Consumer = 1 << 2,
  LLaMAMiddle = 1 << 3,
};

template<int stageType, typename Sched, typename Sync>
struct CuStage {
  dim3 grid_;
  dim3 prodGrid_;
  dim3 tileSize_;
  uint* tileCounter;
  dim3* tileOrder;
  volatile uint* tileStatusWrite_;
  volatile uint* tileStatusRead_;
  int* kernelExecuted_;
  int iter;
  Sync syncPolicy_;
  bool canPrint;

  __device__ __host__ CuStage(): iter(0) {}

  CuStage(dim3 grid, dim3 tileSize, Sync syncPolicy) : 
    grid_(grid), tileSize_(tileSize), iter(0), prodGrid_(0), 
    syncPolicy_(syncPolicy), canPrint(false) {
      buildScheduleBuffer();

      if (isProducer()) {
        volatile uint* tileStatus;
        CUDA_CHECK(cudaMalloc(&tileStatus, numTiles() * sizeof(int) * 8));
        CUDA_CHECK(cudaMemset((uint*)tileStatus, 0, numTiles() * sizeof(int) * 8));
        tileStatusWrite_ = tileStatus;
      }
  }

  __device__ __host__ size_t numTiles() {return grid_.x * grid_.y * grid_.z;}
  // __host__ size_t numTiles() {return grid_.x * grid_.y *;}

  void buildScheduleBuffer() {
    CUDA_CHECK(cudaMalloc(&tileCounter, sizeof(int)));
    CUDA_CHECK(cudaMemset(tileCounter, 0, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&tileOrder, sizeof(*tileOrder) * numTiles()));
    dim3* hTileOrder = new dim3[numTiles()];

    if (grid_.z > 1) {
      for (uint z = 0; z < grid_.z; z++) {
      for (uint x = 0; x < grid_.x; x++) {
      for (uint y = 0; y < grid_.y; y++) {
        size_t id = RowMajor().order(grid_, {x, y, z});
        hTileOrder[id] = {x, y, z};
      }}}
    } else {
      for (uint x = 0; x < grid_.x; x++) {
      for (uint y = 0; y < grid_.y; y++) {
      for (uint z = 0; z < grid_.z; z++) {
        size_t id = RowMajor().order(grid_, {x, y, z});
        hTileOrder[id] = {x, y, z};
      }}}
    }

    CUDA_CHECK(cudaMemcpy(tileOrder, hTileOrder, 
                          sizeof(*tileOrder) * numTiles(),
                          cudaMemcpyHostToDevice));
    delete[] hTileOrder;
  }

  void setTileStatusToPost(volatile uint* tileStatus) {
    tileStatusWrite_ = tileStatus;
  }

  volatile uint* getTileStatusToPost() {
    return tileStatusWrite_;
  }

  void setTileStatusToWait(volatile uint* tileStatus) {
    tileStatusRead_ = tileStatus;
  }

 __device__ __host__ volatile uint* getTileStatusToWait() {
    return tileStatusRead_;
  }

  __device__ void wait(dim3& tile, uint waitingThread = 0, bool callSync = true) {
    if (!isConsumer() && !isLLaMAMiddle()) return;
    if (!syncPolicy_.isSync(tile, prodGrid_)) return;
    // if (prodGrid_.y == grid_.y) return;
    if (threadIdx.x == waitingThread && threadIdx.y == 0 && threadIdx.z == 0) {
      if (std::is_same<Sync, Conv2DTileSync<1,9>>::value) {
        tile.y = tile.y/9;
      }
      uint w = syncPolicy_.waitValue(tile, prodGrid_);
      uint idx = syncPolicy_.tileIndex(tile, prodGrid_);
      auto v = glLoad(&tileStatusRead_[idx]);
      while(v < iter * w) {
        v = volatileLoad(&tileStatusRead_[idx]);
      }      
      // // asm volatile("prefetch.global.L1 [%0];" :: "l"(&tileStatusRead_[idx]));
      // // v = glLoad(&tileStatusRead_[idx]);
      // // 
      // if (true) {
      //   uint val = 0;
      //   int* addr = (int*)&tileStatusRead_[idx];
      //   while(val < iter * w) {
      //     asm volatile ("ld.global.cg.u32 {%0}, [%1];" : "=r"(val) : "l"(addr) : "memory");
      //     if (val >= iter * w) break;
      //   }
      // }
      // //   v = volatileLoad(&tileStatusRead_[idx]);
      // //   // if (v < iter * w) break;
      // // }
    }

    if (callSync)
      __syncthreads();
  }

  __device__ void post(const dim3& tile, uint postThread = 0) {
    if (!isProducer()) return;
    __syncthreads();
    // printf("407\n");
    if (threadIdx.x == postThread && threadIdx.y == 0 && threadIdx.z == 0) {
      __threadfence_system();
      uint idx = syncPolicy_.tileIndex(tile, grid_);
      // printf("tileStatusWrite_ %p\n", tileStatusWrite_);
      #ifndef NO_ATOMIC_ADD
      atomicAdd((int*)&tileStatusWrite_[idx],
                syncPolicy_.postValue(tile, grid_));
      #else
      uint val = syncPolicy_.postValue(tile, grid_) * iter;
      asm volatile ("st.global.cg.u32 [%0], {%1};" :: "l"((int*)&tileStatusWrite_[idx]), "r"(val));
      #endif
    }

    __syncwarp();
  }

  __device__ __host__ bool isProducer() {
    return stageType & CuStageType::Producer;
  }

  __device__ __host__ bool isConsumer() {
    return stageType & CuStageType::Consumer;
  }

  __device__ __host__ bool isLLaMAMiddle() {
    return stageType & CuStageType::LLaMAMiddle;
  }

  __device__ dim3 init() {}

  __forceinline__ __device__ dim3 tile(dim3* shared_storage) {
     #ifndef AVOID_WAIT_KERNEL
      if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && 
          blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && isProducer()) {
        *kernelExecuted_ = iter;
      }
      #endif
    #ifndef AVOID_CUSTOM_ORDER
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
      if (shared_storage != nullptr) {
        uint linear_id = atomicAdd(tileCounter, 1);
        if (linear_id == numTiles() - 1) {
          *tileCounter = 0;
        }
        *shared_storage = tileOrder[linear_id];
      }
    }    

    if (shared_storage != nullptr) {
      __syncthreads();
      return *shared_storage;
    }
    return blockIdx;
    #else
    return blockIdx;
    #endif
  }

  void invokeWaitKernel(cudaStream_t stream) {
    assert(isProducer());
    waitKernel<<<1,1,0,stream>>>((uint*)kernelExecuted_, iter);
  }
};

template<typename Stage1, typename Stage2>
void initProducerConsumer(Stage1& prod, Stage2& cons) {
  assert(prod.isProducer());
  assert(cons.isConsumer());

  if (prod.getTileStatusToPost() == nullptr) {
    printf("tileStatusToPost is null\n");
    abort();
  }
  cons.prodGrid_ = prod.grid_;
  cons.setTileStatusToWait(prod.getTileStatusToPost());
  CUDA_CHECK(cudaMalloc(&prod.kernelExecuted_, sizeof(int)));
  CUDA_CHECK(cudaMemset(prod.kernelExecuted_, 0, sizeof(int)));
}