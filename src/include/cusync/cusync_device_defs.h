// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

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

namespace cusync {
#define CUSTAGE_METHOD_DEF(RET_TYPE) \
    template<typename TileOrder, \
             typename InputSyncPolicy, \
             typename OutputSyncPolicy, \
             int Opts> \
    CUSYNC_DEVICE RET_TYPE CuStage<TileOrder, InputSyncPolicy, OutputSyncPolicy, Opts>::

  /*
   * Wait until the semaphore of the tile reaches the wait value
   */
  CUSTAGE_METHOD_DEF(CuSyncError) wait(dim3& tile, uint32_t waitingThread, bool callSync) {
    if (!isConsumer()) return CuSyncErrorNotConsumer;
    if (!inputPolicy_.isSync(tile, prodGrid_)) return CuSyncSuccess;
    
    if (threadIdx.x == waitingThread && threadIdx.y == 0 && threadIdx.z == 0) {
      uint32_t w = inputPolicy_.waitValue(tile, prodGrid_);
      uint32_t idx = inputPolicy_.tileIndex(tile, prodGrid_);
      auto v = globalLoad(&tileStatusRead_[idx]);
      while(v < iter * w) {
        v = globalVolatileLoad(&tileStatusRead_[idx]);
      }
    }

    if (callSync)
      __syncthreads();
    
    return CuSyncSuccess;
  }

  /*
   * Post the status of completion of tile.
  */
  CUSTAGE_METHOD_DEF(CuSyncError) post(const dim3& tile, uint32_t postThread) {
    if (!isProducer()) return CuSyncErrorNotProducer;
    __syncthreads();
    if (threadIdx.x == postThread && threadIdx.y == 0 && threadIdx.z == 0) {
      __threadfence_system();
      uint32_t idx = outputPolicy_.tileIndex(tile, grid_);
      if (!getNoAtomicAdd()) {
        atomicAdd((int*)&tileStatusWrite_[idx],
                  outputPolicy_.postValue(tile, grid_));
      } else {
        uint32_t val = outputPolicy_.postValue(tile, grid_) * iter;
        asm volatile ("st.global.release.gpu.u32 [%0], {%1};" :: "l"((int*)&tileStatusWrite_[idx]), "r"(val));
      }
    }

    __syncwarp();
    return CuSyncSuccess;
  }

  /*
   * Returns the next tile process and set the waitkernel's semaphore if valid
   */  
  CUSTAGE_METHOD_DEF(dim3) tile(dim3* shared_storage) {
    if (!getAvoidWaitKernel()) {
      if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && 
          blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && isProducer()) {
        *kernelExecuted_ = iter;
      }
    }
    if (!getAvoidCustomOrder()) {
      if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        if (shared_storage != nullptr) {
          uint32_t linear_id = atomicAdd(tileCounter, 1);
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
}
