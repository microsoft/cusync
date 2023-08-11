#include <assert.h>
#include <stdio.h>

#ifndef __CUSYNC__
#define __CUSYNC__

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

struct RowMajor {
  //overload call operator ()
  size_t order(dim3 grid, dim3 currTile) {
    return currTile.x * grid.y * grid.z + currTile.y * grid.z + currTile.z;
  }
};

// template<typename Sched, typename Sync> struct CuStage;
//todo: make args constant

struct RowSync {
  uint waitValue_;
  uint postValue_;
  __device__ __host__ RowSync()  : waitValue_(0), postValue_(0) {}
  __device__ __host__ RowSync(uint waitValue) : waitValue_(waitValue), postValue_(1) {}
  __device__ __host__ RowSync(uint waitValue, uint postValue) : 
    waitValue_(waitValue), postValue_(postValue) {}
  
  __device__ uint waitValue(const dim3& tile, const dim3& grid) {
    return waitValue_;
  }

  __device__ uint tileIndex(const dim3& tile, const dim3& grid) {
    return tile.x;
  }

  __device__ bool isSync(const dim3& tile) {
    return tile.y == 0;
  }

  __device__ uint postValue(const dim3& tile, const dim3& grid) {
    return postValue_;
  }
};

#define BatchedRows 2

struct BatchedRowSync {
  uint waitValue_;
  uint postValue_;
  __device__ __host__ BatchedRowSync()  : waitValue_(0), postValue_(0) {}
  __device__ __host__ BatchedRowSync(uint waitValue) : waitValue_(waitValue), postValue_(1) {}
  __device__ __host__ BatchedRowSync(uint waitValue, uint postValue) : 
    waitValue_(waitValue), postValue_(postValue) {}
  
  __device__ bool canBatch(const dim3& tile) {
    return true;
  }
  
  __device__ uint waitValue(const dim3& tile, const dim3& grid) {
    return waitValue_ * BatchedRows;
  }

  __device__ uint tileIndex(const dim3& tile, const dim3& grid) {
    return tile.x/BatchedRows;
  }

  __device__ bool isSync(const dim3& tile) {
    return tile.y == 0;
  }

  __device__ uint postValue(const dim3& tile, const dim3& grid) {
    return postValue_;
  }
};

struct BatchedRowSync2 {
  uint waitValue_;
  uint postValue_;
  __device__ __host__ BatchedRowSync2()  : waitValue_(0), postValue_(0) {}
  __device__ __host__ BatchedRowSync2(uint waitValue) : waitValue_(waitValue), postValue_(1) {}
  __device__ __host__ BatchedRowSync2(uint waitValue, uint postValue) : 
    waitValue_(waitValue), postValue_(postValue) {}
  
  __device__ bool canBatch(const dim3& tile) {
    if (tile.x >= BatchedRows)
      return true;
    return false;
  }
  
  __device__ uint waitValue(const dim3& tile, const dim3& grid) {
    if (canBatch(tile))
      return waitValue_ * (grid.x - BatchedRows);
    return waitValue_;
  }

  __device__ uint tileIndex(const dim3& tile, const dim3& grid) {
    if (canBatch(tile)) {
      return BatchedRows;
    }
    return tile.x;
  }

  __device__ bool isSync(const dim3& tile) {
    return tile.y == 0;
  }

  __device__ uint postValue(const dim3& tile, const dim3& grid) {
    return postValue_;
  }
};

struct TileFirstAndRowSync {
  uint waitTileValue_;
  uint postTileValue_;
  uint waitRowValue_;
  uint postRowValue_;
  
  __device__ __host__ TileFirstAndRowSync() {}
  __device__ __host__ TileFirstAndRowSync(uint waitTileValue, uint postTileValue, 
                                          uint waitRowValue) : 
    waitTileValue_(waitTileValue), postTileValue_(postTileValue), waitRowValue_(waitRowValue), postRowValue_(1) {}
  // __device__ __host__ TileFirstAndRowSync(uint waitValue, uint postValue) : 
  //   waitValue_(waitValue), postValue_(postValue) {}
  
  __device__ int tileBatch(const dim3& tile) {
    if (isTileSync(tile))
      return 8;
    return 1;
  }

  __device__ bool isTileSync(const dim3& tile) {
    if (tile.x < 1) {
      return true;
    }
    return false;
  }

  __device__ bool isRowSync(const dim3& tile) {
    return !isTileSync(tile);
  }

  __device__ uint waitValue(const dim3& tile, const dim3& grid) {
    if (isTileSync(tile)) {
      return waitTileValue_ * tileBatch(tile);
    }

    return waitRowValue_;
  }

  __device__ uint tileIndex(const dim3& tile, const dim3& grid) {
    if (isTileSync(tile)) {
      return (tile.x * 48 + tile.y)/tileBatch(tile);
    } 
    return 1 * 48 + tile.x;
  }

  __device__ bool isSync(const dim3& tile) {
    if (isTileSync(tile))
      return true;
    else
      return tile.y == 0;
  }

  __device__ uint postValue(const dim3& tile, const dim3& grid) {
    if (isTileSync(tile)) {
      return postTileValue_;
    }

    return postRowValue_;
  }
};

template<uint batch>
struct TileSync {
  uint waitValue_;
  uint postValue_;

  __device__ __host__ TileSync(): waitValue_(1), postValue_(1) {}
  __device__ __host__ TileSync(uint waitValue, uint postValue): 
    waitValue_(waitValue), postValue_(postValue) {}
  
  __device__ __host__ uint waitValue(const dim3& tile, const dim3& grid) {
    if (batch > 1) {
      if (grid.y % batch != 0 && tile.y >= (grid.y/batch) * batch)
        return waitValue_ * (grid.y - (grid.y/batch) * batch);
    }
    return waitValue_ * batch;
  }

  __device__ __host__ uint postValue(const dim3& tile, const dim3& grid) 
    {return postValue_;}

  __device__ constexpr uint tileIndex(const dim3& tile, const dim3& grid) {
    return (tile.x * grid.y + tile.y)/batch;
  }

  __device__ bool isSync(const dim3& tile) {
    return true;
  }
};

enum CuStageType {
  Producer = 1,
  Consumer = 1 << 2,
};

__forceinline__ __device__ 
uint semaphoreLoad(volatile uint* semaphore) {
  uint state;
  asm volatile ("ld.global.acquire.gpu.u32 %0, [%1];" : "=r"(state) : "l"(semaphore));
  return state;
}

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
    grid_(grid), tileSize_(tileSize), iter(0), prodGrid_(0), syncPolicy_(syncPolicy), canPrint(false) {
      buildScheduleBuffer();
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

  volatile uint* getTileStatusToWait() {
    return tileStatusRead_;
  }

  __device__ void wait(const dim3& tile, uint waitingThread = 0, bool callSync = true) {
    if (!isConsumer()) return;
    if (!syncPolicy_.isSync(tile)) return;
    
    if (threadIdx.x == waitingThread && threadIdx.y == 0 && threadIdx.z == 0) {
      uint w = syncPolicy_.waitValue(tile, prodGrid_);
      uint idx = syncPolicy_.tileIndex(tile, prodGrid_);
      // printf("302: tileStatusRead_[%d] %d {%d, %d, %d} w %d iter %d\n", 
      // idx, tileStatusRead_[idx], tile.x, tile.y, tile.z, w, iter);
      while(tileStatusRead_[idx] < iter * w);
      // printf("305: tileStatusRead_[%d] %d {%d, %d, %d} w %d iter %d\n", 
      // idx, tileStatusRead_[idx], tile.x, tile.y, tile.z, w, iter);
    }

    if (callSync)
      __syncthreads();
  }

  __device__ void post(const dim3& tile, uint postThread = 0) {
    if (!isProducer()) return;
    __syncthreads();
  
    if (threadIdx.x == postThread && threadIdx.y == 0 && threadIdx.z == 0) {
      __threadfence_system();
      uint idx = syncPolicy_.tileIndex(tile, grid_);
      atomicAdd((int*)&tileStatusWrite_[idx],
                syncPolicy_.postValue(tile, grid_));
      // printf("tileStatusWrite_[%d] %d\n", tileStatusWrite_[idx], idx);
    }

    __syncwarp();
  }

  __device__ __host__ bool isProducer() {
    return stageType & CuStageType::Producer;
  }

  __device__ __host__ bool isConsumer() {
    return stageType & CuStageType::Consumer;
  }

  __device__ dim3 init() {}

  __forceinline__ __device__ dim3 tile(dim3* shared_storage) {
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
      if (isProducer()) {
        if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
          *kernelExecuted_ = iter;
        }
      }

      uint linear_id = atomicAdd(tileCounter, 1);
      if (linear_id == numTiles() - 1) {
        *tileCounter = 0;
      }
      *shared_storage = tileOrder[linear_id];
    }

    __syncthreads();
    dim3 i = *shared_storage;

    return i;
    // return isProducer() ? dim3{0, blockIdx.x%24, blockIdx.x/24} : 
    //                       dim3{0, blockIdx.x%48, blockIdx.x/48};
  }
};

__device__ inline uint glLoad(volatile uint* addr) {
  uint val;
  asm ("ld.volatile.global.u32 {%0}, [%1];" : "=r"(val) : "l"(addr));
  return val;
}

__global__ void waitKernel(volatile uint* kernelExecuted, uint expectedValue) {
  if (threadIdx.x == 0) {
    uint v = glLoad(kernelExecuted);
    while(v < expectedValue) {
      v = glLoad(kernelExecuted);
    }
  }
}

template<typename Stage1, typename Stage2>
struct CuSync {
  Stage1 prod_;
  __host__ Stage1& prod() {return prod_;}
  Stage2 cons_;
  __host__ Stage2& cons() {return cons_;}

  volatile uint* tileStatus;
  int* kernelExecuted;
  int iter;

  __device__ __host__ CuSync() {}

  void invokeWaitKernel(cudaStream_t stream) {
    waitKernel<<<1,1,0,stream>>>((uint*)kernelExecuted, prod().iter);
  }

  CuSync(Stage1 prod, Stage2 cons): prod_(prod), cons_(cons) {
    CUDA_CHECK(cudaMalloc(&tileStatus, prod.numTiles() * sizeof(int)));
    CUDA_CHECK(cudaMemset((uint*)tileStatus, 0, prod.numTiles() * sizeof(int)));
    iter = 0;
    cons_.prodGrid_ = prod.grid_;
    prod_.setTileStatusToPost(tileStatus);
    cons_.setTileStatusToWait(tileStatus);
    CUDA_CHECK(cudaMalloc(&kernelExecuted, sizeof(int)));
    CUDA_CHECK(cudaMemset(kernelExecuted, 0, sizeof(int)));
    prod_.kernelExecuted_ = kernelExecuted;
  }
};

#endif