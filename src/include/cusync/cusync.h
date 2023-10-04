#include <assert.h>
#include <stdio.h>
#include <type_traits>

#include "tile-orders.h"
#include "policies.h"
#include "device-functions.h"
#include "wait-kernel.h"

#pragma once

#define CUDA_CHECK(cmd) do {                        \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",       \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0);

#define DIVUP(x, y) (((x) + (y) - 1)/(y));

/*
 * A test class to access private members of CuStage 
 */
class CuSyncTest;

/*
 * CuStageType enum describes if the CuStage is 
 * a producer or a consumer or both.  
 */
enum CuStageType {
  CuStageNone =      0,
  Producer    = 1 << 0,
  Consumer    = 1 << 1,
};

/*
 * List of optimizations for a CuStage that avoids certain operations
 * performed by a CuStage for a specific scenario.
 * NoOptimizations : No optimization is performed
 * NoAtomicAdd     : Use memory write instead of atomic add. Useful when each
 *                   tile is associated with a distinct semaphore                 
 * AvoidWaitKernel : Avoid calling wait kernel. Useful when thread blocks of all 
 *                   CuStages can be allocated within a single wave
 * AvoidCustomOrder: Avoid assigning tiles in the specific order but use CUDA's 
 *                   arbitrary order. Useful when thread blocks of N dependent 
 *                   CuStages can be allocated within (N - 1) waves 
 * ReorderTileLoads: Reorder tile loads to overlap computation of one input's tile 
 *                   with loading of other inputs tile
 */
enum Optimizations {
  NoOptimization   =      0,
  NoAtomicAdd      = 1 << 0,
  AvoidWaitKernel  = 1 << 1,
  AvoidCustomOrder = 1 << 2,
  ReorderTileLoads = 1 << 3
};

/*
 * A CuStage is associated with a single kernel. A CuStage contains following
 * information about its kernel:
 * 1. grid and tile size of the kernel
 * 2. grid size of its producer kernel
 * 3. synchronization policy for the kernel
 * 
 * Moreover, CuStage contains pointers to the tile order and array of semaphore 
 * for tile synchronization policies. 
 */
template<int stageType,             //Type of stage constructed using CuStageType enum
         typename TileOrder,        //Tile processing order see tile-orders.h 
         typename Sync,             //Synchronization policy see policies.h
         int Opts = NoOptimization  //Optimizations for CuStage using Optimizations enum
        >
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
  friend class CuSyncTest;

  __device__ __host__ 
  CuStage(): iter(0) {}

  __host__
  CuStage(dim3 grid, dim3 tileSize, Sync syncPolicy) : 
    grid_(grid), tileSize_(tileSize), iter(0), prodGrid_(0), 
    syncPolicy_(syncPolicy), canPrint(false) {
      buildScheduleBuffer();

      if (isProducer()) {
        volatile uint* tileStatus;
        CUDA_CHECK(cudaMalloc(&tileStatus, numTiles() * sizeof(int)));
        CUDA_CHECK(cudaMemset((uint*)tileStatus, 0, numTiles() * sizeof(int)));
        tileStatusWrite_ = tileStatus;
      }
  }

  //Optimization Flags
  __device__ __host__ bool getNoAtomicAdd     () {return Opts & NoAtomicAdd;     }
  __device__ __host__ bool getAvoidWaitKernel () {return Opts & AvoidWaitKernel; }
  __device__ __host__ bool getReorderTileLoads() {return Opts & ReorderTileLoads;}
  __device__ __host__ bool getAvoidCustomOrder() {return Opts & AvoidCustomOrder;}

  __device__ __host__ size_t numTiles() {return grid_.x * grid_.y * grid_.z;}

  void buildScheduleBuffer() {
    CUDA_CHECK(cudaMalloc(&tileCounter, sizeof(int)));
    CUDA_CHECK(cudaMemset(tileCounter, 0, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&tileOrder, sizeof(*tileOrder) * numTiles()));
    dim3* hTileOrder = new dim3[numTiles()];

    for (uint z = 0; z < grid_.z; z++) {
    for (uint x = 0; x < grid_.x; x++) {
    for (uint y = 0; y < grid_.y; y++) {
      size_t id = RowMajorZYX().order(grid_, {x, y, z});
      hTileOrder[id] = {x, y, z};
    }}}

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

 __device__ __host__
 volatile uint* getTileStatusToWait() {
    return tileStatusRead_;
  }

  __device__
  void wait(dim3& tile, uint waitingThread = 0, bool callSync = true) {
    if (!isConsumer()) return;
    if (!syncPolicy_.isSync(tile, prodGrid_)) return;
    // if (prodGrid_.y == grid_.y) return;
    if (threadIdx.x == waitingThread && threadIdx.y == 0 && threadIdx.z == 0) {
      if (std::is_same<Sync, Conv2DTileSync<1,9>>::value) {
        tile.y = tile.y/9;
      }
      uint w = syncPolicy_.waitValue(tile, prodGrid_);
      uint idx = syncPolicy_.tileIndex(tile, prodGrid_);
      auto v = globalLoad(&tileStatusRead_[idx]);
      while(v < iter * w) {
        v = globalVolatileLoad(&tileStatusRead_[idx]);
      }
    }

    if (callSync)
      __syncthreads();
  }

  __device__
  uint waitTileIndex(dim3 tile) {
    if (std::is_same<Sync, Conv2DTileSync<1,9>>::value) {
      tile.y = tile.y/9;
    }
    return syncPolicy_.tileIndex(tile, grid_);;
  }

  __device__
  uint waitSemValue(uint tileIndex) {
    return globalVolatileLoad(&tileStatusRead_[tileIndex]);
  }

  __device__
  uint expectedWaitValue(dim3 tile) {
    return syncPolicy_.waitValue(tile, prodGrid_);
  }

  __device__
  void post(const dim3& tile, uint postThread = 0) {
    if (!isProducer()) return;
    __syncthreads();
    // printf("407\n");
    if (threadIdx.x == postThread && threadIdx.y == 0 && threadIdx.z == 0) {
      __threadfence_system();
      uint idx = syncPolicy_.tileIndex(tile, grid_);
      if (!getNoAtomicAdd()) {
        atomicAdd((int*)&tileStatusWrite_[idx],
                  syncPolicy_.postValue(tile, grid_));
      } else {
        uint val = syncPolicy_.postValue(tile, grid_) * iter;
        asm volatile ("st.global.cg.u32 [%0], {%1};" :: "l"((int*)&tileStatusWrite_[idx]), "r"(val));
      }
    }

    __syncwarp();
  }

  __device__ __host__ 
  bool isProducer() {
    return stageType & CuStageType::Producer;
  }

  __device__ __host__ 
  bool isConsumer() {
    return stageType & CuStageType::Consumer;
  }

  __device__
  dim3 init() {}

  __forceinline__ __device__
  dim3 tile(dim3* shared_storage) {
    if (!getAvoidWaitKernel()) {
      if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && 
          blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && isProducer()) {
        *kernelExecuted_ = iter;
      }
    }
    if (!getAvoidCustomOrder()) {
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
    } else {
      return blockIdx;
    }
  }

  void invokeWaitKernel(cudaStream_t stream) {
    assert(isProducer());
    if (!getAvoidWaitKernel()) {
      waitKernel<<<1,1,0,stream>>>((uint*)kernelExecuted_, iter);
    }
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