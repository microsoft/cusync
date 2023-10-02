#pragma once

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

  __device__ bool isSync(const dim3& tile, const dim3& grid) {
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

  __device__ bool isSync(const dim3& tile, const dim3& grid) {
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
    return waitValue_ * batch;
  }

  __device__ __host__ uint postValue(const dim3& tile, const dim3& grid) 
    {return postValue_;}

  __device__ constexpr uint tileIndex(const dim3& tile, const dim3& grid) {
    return (tile.x * grid.y + tile.y)/batch;
  }

  __device__ __forceinline__ bool isSync(const dim3& tile, const dim3& grid) {
    return tile.y < grid.y; //for self-attention
  }
};


template<uint batch, uint convKernelSize>
struct Conv2DTileSync {
  uint waitValue_;
  uint postValue_;

  __device__ __host__ Conv2DTileSync(): waitValue_(1), postValue_(1) {}
  __device__ __host__ Conv2DTileSync(uint waitValue, uint postValue): 
    waitValue_(waitValue), postValue_(postValue) {}
  
  __device__ __host__ uint waitValue(const dim3& tile, const dim3& grid) {
    return waitValue_ * batch;
  }

  __device__ __host__ uint postValue(const dim3& tile, const dim3& grid) 
    {return postValue_;}

  __device__ constexpr uint tileIndex(const dim3& tile, const dim3& grid) {
    return tile.x * grid.y + tile.y;
  }

  __device__ __forceinline__ bool isSync(const dim3& tile, const dim3& grid) {
    return tile.y % 9 == 0;
  }
};

struct FirstTileSync {
  uint waitValue_;
  uint postValue_;

  __device__ __host__ FirstTileSync(): waitValue_(1), postValue_(1) {}
  __device__ __host__ FirstTileSync(uint waitValue, uint postValue): 
    waitValue_(waitValue), postValue_(postValue) {}
  
  __device__ __host__ uint waitValue(const dim3& tile, const dim3& grid) {
    return waitValue_;
  }

  __device__ __host__ uint postValue(const dim3& tile, const dim3& grid) 
    {return postValue_;}

  __device__ constexpr uint tileIndex(const dim3& tile, const dim3& grid) {
    return (tile.x * grid.y + tile.y);
  }

  __device__ bool isSync(const dim3& tile, const dim3& grid) {
    return tile.y == 0;
  }
};