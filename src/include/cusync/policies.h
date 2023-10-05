#pragma once

namespace cusync {
/*
 * A synchronization policy (SyncPolicy) is a struct of four functions:
 * uint waitValue(const dim3& tile, const dim3& grid) returns completed value for the tile
 * uint tileIndex(const dim3& tile, const dim3& grid) returns semaphore index for the tile
 * uint isSync   (const dim3& tile, const dim3& grid) returns if semaphore should be sync for the tile
 * uint postValue(const dim3& tile, const dim3& grid) returns the value of semaphore when tile is processed
 */

/*
 * RowSync policy assigns same semaphore for tiles sharing the same row (x index of tile)
 */
struct RowSync {
  //Value to wait on for each row
  uint waitValue_;
  //Value to post when the tile is computed
  uint postValue_;

  /*
   * Default constructor for RowSync initializes wait and post value to 0
   */
  __device__ __host__
  RowSync()  : waitValue_(0), postValue_(0) {}
  
  /*
   * Initializes post value to 1 and wait value to the given value
   */
  __device__ __host__
  RowSync(uint waitValue) : waitValue_(waitValue), postValue_(1) {}

  /*
   * Initializes post value and wait value
   */
  __device__ __host__
  RowSync(uint waitValue, uint postValue) : 
    waitValue_(waitValue), postValue_(postValue) {}
  
  /*
   * Returns the wait value
   */
  __device__ uint waitValue(const dim3& tile, const dim3& grid) {
    return waitValue_;
  }

  /*
   * Returns the tile index as the x index of tile
   */
  __device__ uint tileIndex(const dim3& tile, const dim3& grid) {
    return tile.x;
  }

  /*
   * Returns true only when the tile is the first tile of column,
   * i.e., y index is 0
   */
  __device__ bool isSync(const dim3& tile, const dim3& grid) {
    return tile.y == 0;
  }

  /*
   * Returns the post value
   */
  __device__ uint postValue(const dim3& tile, const dim3& grid) {
    return postValue_;
  }
};

/*
 * TileSync assigns distinct semaphore to each tile
 */
struct TileSync {
  uint waitValue_;
  uint postValue_;

  /*
   * Initializes both wait and post value to 1
   */
  __device__ __host__ 
  TileSync(): waitValue_(1), postValue_(1) {}
  
  /*
   * Initializes both wait and post values to given values
   */
  __device__ __host__ 
  TileSync(uint waitValue, uint postValue): 
    waitValue_(waitValue), postValue_(postValue) {}
  
  /*
   * Return the wait value
   */
  __device__ __host__ uint waitValue(const dim3& tile, const dim3& grid) {
    return waitValue_;
  }

  /*
   * Return the post value
   */
  __device__ __host__ uint postValue(const dim3& tile, const dim3& grid) 
    {return postValue_;}

  /*
   * Return the linear tile index for the grid
   */
  __device__ constexpr uint tileIndex(const dim3& tile, const dim3& grid) {
    return (tile.x * grid.y + tile.y);
  }

  /*
   * Synchronizes on a tile if it is within the grid
   */
  __device__ __forceinline__ bool isSync(const dim3& tile, const dim3& grid) {
    return tile.y < grid.y and tile.x < grid.x;
  }
};

/*
 * Synchronizes tiles of 2D Implicit GeMM Convolution for given values of
 * 2D Convolution kernel size (R x S).
 * 
 * The implict GeMM algorithm converts a Conv2D of B input images of size [P, Q, C] 
 * with a kernel matrix of size [R, S] into a GeMM of matrices 
 * [B∗P∗Q, C∗R∗S] x [C∗R∗S, C]. Therefore, a tile {x,y} of the second Conv2D
 * synchronizes on the tile {x, y/(R*S)} of first Conv2D.
 */
template<uint R, uint S>
struct Conv2DTileSync {
  uint waitValue_;
  uint postValue_;

  /*
   * Initializes both wait and post value to 1
   */
  __device__ __host__
  Conv2DTileSync(): waitValue_(1), postValue_(1) {}
  
  /*
   * Initializes both wait and post value to given values
   */
  __device__ __host__
  Conv2DTileSync(uint waitValue, uint postValue): 
    waitValue_(waitValue), postValue_(postValue) {}
  
  /*
   * Returns the wait value 
   */
  __device__ __host__ __forceinline__
  uint waitValue(const dim3& tile, const dim3& grid) {
    return waitValue_;
  }

  /*
   * Returns the post value 
   */
  __device__ __host__ __forceinline__
  uint postValue(const dim3& tile, const dim3& grid) 
    {return postValue_;}

  /*
   * Returns the wait value 
   */
  __device__ __forceinline__
  uint tileIndex(const dim3& tile, const dim3& grid) {
    return tile.x * grid.y + tile.y;
  }

  /*
   * Synchronizes tiles only when it is a multiple of 
   * the conv kernel size
   */
  __device__ __forceinline__
  bool isSync(const dim3& tile, const dim3& grid) {
    return tile.y % (R * S) == 0;
  }
};

/*
 * Other sync policies
 */
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
}