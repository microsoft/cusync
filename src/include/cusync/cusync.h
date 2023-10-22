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

namespace cusync {
/*
 * A test class to access private members of CuStage 
 */
class CuSyncTest;
class CuSync;

enum CuSyncError {
  CuSyncErrorNotProducer,
  CuSyncErrorNotConsumer,
  CuSyncErrorNotInitialized,
  CuSyncErrorCUDAError,
  CuSyncSuccess
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
template<typename TileOrder,        //Tile processing order (see tile-orders.h) 
         typename InputSyncPolicy,  //Policy for synchronizing on the input (see policies.h)
         typename OutputSyncPolicy, //Policy for synchronizing on the output (see policies.h)
         int Opts = NoOptimization  //Optimizations for CuStage using Optimizations enum
        >
class CuStage {
private:
  //grid size of this stage
  dim3 grid_;
  //grid size of the producer stage
  dim3 prodGrid_;
  //tile size of this stage
  dim3 tileSize_;
  
  //Number of runs of stage kernels invoked
  int iter;
  //Producer Sync policy of the stage
  InputSyncPolicy inputPolicy_;
  //Consumer Sync policy of the stage
  OutputSyncPolicy outputPolicy_;

  //GPU pointer to array of order of tiles
  dim3* tileOrder;
  //GPU pointer to counter of tile for index in tile order
  uint* tileCounter;

  //GPU pointer to wait kernel semaphore
  int* kernelExecuted_;
  
  volatile uint* tileStatusWrite_;
  volatile uint* tileStatusRead_;

  //CuSyncTest and CuSync can access private members
  friend class CuSyncTest;
  friend class CuSync;

  //Get tile status semaphore arrays
  __device__ __host__ __forceinline__
  volatile uint* getTileStatusToPost()         {return tileStatusWrite_;}
  __device__ __host__ __forceinline__
  volatile uint* getTileStatusToWait()         {return tileStatusRead_;}

  //Set tile status semaphore arrays
  __host__ __forceinline__
  void setTileStatusToPost(volatile uint* ptr) {tileStatusWrite_ = ptr ;}
  __host__ __forceinline__
  void setTileStatusToWait(volatile uint* ptr) {tileStatusRead_  = ptr ;}

  //Call TileOrder parameter to generate tile order and store 
  //it in tileOrder
  void buildScheduleBuffer() {
    dim3* hTileOrder = new dim3[numTiles()];

    CUDA_CHECK(cudaMalloc(&tileCounter, sizeof(int)));
    CUDA_CHECK(cudaMemset(tileCounter, 0, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&tileOrder, sizeof(*tileOrder) * numTiles()));

    for (uint z = 0; z < grid_.z; z++) {
    for (uint y = 0; y < grid_.y; y++) {
    for (uint x = 0; x < grid_.x; x++) {
      size_t id = TileOrder()(grid_, {x, y, z});
      hTileOrder[id] = {x, y, z};
    }}}

    CUDA_CHECK(cudaMemcpy(tileOrder, hTileOrder, 
                          sizeof(*tileOrder) * numTiles(),
                          cudaMemcpyHostToDevice));
    delete[] hTileOrder;
  }
public:
  __device__ __host__ 
  CuStage(): iter(1) {}

  __host__
  CuStage(dim3 grid, dim3 tileSize, InputSyncPolicy inputPolicy, OutputSyncPolicy outputPolicy) : 
    grid_(grid), 
    prodGrid_(0), //set by CuSync::set* methods 
    tileSize_(tileSize),
    iter(1),     //run counter starts from 1 
    inputPolicy_(inputPolicy),
    outputPolicy_(outputPolicy) {
    
    buildScheduleBuffer();

    if (isProducer()) {
      //Allocate tile status semaphore array for all tiles
      //CuSync::set* methods set this array to consumer stages
      CUDA_CHECK(cudaMalloc(&tileStatusWrite_, numTiles() * sizeof(int)));
      CUDA_CHECK(cudaMemset((uint*)tileStatusWrite_, 0, numTiles() * sizeof(int)));

      //Allocate wait kernel semaphore
      if (!getAvoidWaitKernel()) {
        CUDA_CHECK(cudaMalloc(&kernelExecuted_, sizeof(int)));
        CUDA_CHECK(cudaMemset(kernelExecuted_, 0, sizeof(int)));
      }
    }
  } 

  /*
   * Getters and setters for private variables.
   */
  //Getters for optimizations
  __device__ __host__ __forceinline__
  bool getNoAtomicAdd     () {return Opts & NoAtomicAdd;     }
  __device__ __host__ __forceinline__
  bool getAvoidWaitKernel () {return Opts & AvoidWaitKernel; }
  __device__ __host__ __forceinline__
  bool getReorderTileLoads() {return Opts & ReorderTileLoads;}
  __device__ __host__ __forceinline__
  bool getAvoidCustomOrder() {return Opts & AvoidCustomOrder;}

  //A producer does have a policy for its output 
  __device__ __host__ __forceinline__
  bool isProducer() {return !std::is_same<OutputSyncPolicy, NoSync>::value;}

  //A consumer does have a policy for its input 
  __device__ __host__ __forceinline__
  bool isConsumer() {return !std::is_same<InputSyncPolicy, NoSync>::value;}

  //Return grid size of this stage
  dim3 grid() {return grid_;}

  //Set producer grid
  template<typename ProdCuStage>
  __host__
  void setProdGrid(ProdCuStage& prod) {prodGrid_ = prod.grid();}

  //Get total number of tiles
  __device__ __host__
  size_t numTiles() {return grid_.x *grid_.y*grid_.z;}

  __device__ __forceinline__
  void wait(dim3& tile, uint waitingThread = 0, bool callSync = true) {
    if (!isConsumer()) return;
    if (!inputPolicy_.isSync(tile, prodGrid_)) return;
    // if (prodGrid_.y == grid_.y) return;
    if (threadIdx.x == waitingThread && threadIdx.y == 0 && threadIdx.z == 0) {
      if (std::is_same<InputSyncPolicy, Conv2DTileSync<3,3>>::value) {
        tile.y = tile.y/9;
      }
      uint w = inputPolicy_.waitValue(tile, prodGrid_);
      uint idx = inputPolicy_.tileIndex(tile, prodGrid_);
      auto v = globalLoad(&tileStatusRead_[idx]);
      while(v < iter * w) {
        // if (threadIdx.x == 0) printf("w %d tile {%d, %d, %d} v %d\n", w, tile.x, tile.y, tile.z, v);
        v = globalVolatileLoad(&tileStatusRead_[idx]);
      }
    }

    if (callSync)
      __syncthreads();
  }

  __device__ __forceinline__
  uint waitTileIndex(dim3 tile) {
    if (std::is_same<InputSyncPolicy, Conv2DTileSync<3,3>>::value) {
      tile.y = tile.y/9;
    }
    return inputPolicy_.tileIndex(tile, grid_);;
  }

  __device__ __forceinline__
  uint waitSemValue(uint tileIndex) {
    return globalVolatileLoad(&tileStatusRead_[tileIndex]);
  }

  __device__ __forceinline__
  uint expectedWaitValue(dim3 tile) {
    return inputPolicy_.waitValue(tile, prodGrid_);
  }

  __device__ __forceinline__
  void post(const dim3& tile, uint postThread = 0) {
    if (!isProducer()) return;
    __syncthreads();
    // printf("407\n");
    if (threadIdx.x == postThread && threadIdx.y == 0 && threadIdx.z == 0) {
      __threadfence_system();
      uint idx = outputPolicy_.tileIndex(tile, grid_);
      if (!getNoAtomicAdd()) {
        atomicAdd((int*)&tileStatusWrite_[idx],
                  outputPolicy_.postValue(tile, grid_));
      } else {
        uint val = outputPolicy_.postValue(tile, grid_) * iter;
        asm volatile ("st.global.cg.u32 [%0], {%1};" :: "l"((int*)&tileStatusWrite_[idx]), "r"(val));
      }
    }

    __syncwarp();
  }

  __device__ __forceinline__
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
          dim3 t = *shared_storage;
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

  __host__
  CuSyncError invokeWaitKernel(cudaStream_t stream) {
    if (!isProducer()) return CuSyncErrorNotProducer;
    if (!getAvoidWaitKernel()) {
      waitKernel<<<1,1,0,stream>>>((uint*)kernelExecuted_, iter);
    }
    
    if (cudaGetLastError() == cudaSuccess) return CuSyncErrorCUDAError;
    return CuSyncSuccess;
  }

  __host__
  void incrementIter() {iter += 1;}  
};

struct CuSync {
  template<typename Stage1, typename Stage2>
  static CuSyncError setProducerConsumerPair(Stage1& prod, Stage2& cons) {
    if (!prod.isProducer()) return CuSyncErrorNotProducer;
    if (!cons.isConsumer()) return CuSyncErrorNotConsumer;
    if (prod.getTileStatusToPost() == nullptr)
      return CuSyncErrorNotInitialized;

    cons.setProdGrid(prod);
    cons.setTileStatusToWait(prod.getTileStatusToPost());
    return CuSyncSuccess;
  }
};
}
