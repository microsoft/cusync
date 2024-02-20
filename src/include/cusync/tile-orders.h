// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#pragma once

namespace cusync {
/*
 * A tile order generates processing order of tiles and maps tiles to thread blocks.
 * A tile order is declared as follows and must subclass GenericTileOrder:
 * struct TileOrder : public GenericTileOrder<TileOrder> {
 *  size_t operator()(dim3& grid, dim3& tile);
 *  dim3 tileToBlock(const dim3& tile); 
 * }
 * GenericTileOrder assume its subclass should not have a state. 
 * If a stateful order is needed the subclass should override 
 * the GenericTileOrder::tileIndex method.
 */

template<typename Child>
struct GenericTileOrder {
  /*
   * Returns a linear index of the thread block in the grid.
   */
  CUSYNC_DEVICE_HOST
  uint32_t blockIndex(const dim3& grid, const dim3& block)
  {return 0;}

  /*
   * Maps a tile to a thread block.
   */
  CUSYNC_DEVICE
  dim3 tileToBlock(const dim3& tile)
  {return {0,0,0};}

  /*
   * Returns a linear tile index.
   */
  CUSYNC_DEVICE
  uint32_t tileIndex(const dim3& tile, const dim3& grid) {
    dim3 block = Child().tileToBlock(tile);
    return Child().blockIndex(grid, block);
  }
};

/*
 * TransposeXYOrder order that generates tile indices first for X-dimension, then 
 * Y-dimension, and finally Z-dimension.
 * It maps a tile {x, y, z} to threadblock {y, x, z}
 */
struct TransposeXYOrder : public GenericTileOrder<TransposeXYOrder> {
  CUSYNC_DEVICE_HOST
  uint32_t blockIndex(const dim3& grid, const dim3& block) {
    return block.x + block.y * grid.x + block.z * grid.x * grid.y;
  }

  CUSYNC_DEVICE
  dim3 tileToBlock(const dim3& tile) {
    return dim3{tile.y, tile.x, tile.z};
  }
};

/*
 * IdentityOrder orders tile indices first for X-dimension, then Y-dim, and finally Z-dim. 
 * It maps a tile {x,y,z} to threadblock {x,y,z}.
 */
struct IdentityOrder : public GenericTileOrder<IdentityOrder> {
  CUSYNC_DEVICE_HOST
  uint32_t blockIndex(const dim3& grid, const dim3& block) {
    return block.x + block.y * grid.x + block.z * grid.x * grid.y;
  }

  CUSYNC_DEVICE
  dim3 tileToBlock(const dim3& tile) {
    return dim3{tile.x, tile.y, tile.z};
  }
};

#if 0
//Experimental Orders
struct OrderZXY {
  __device__ __host__ __forceinline__
  uint32_t operator()(const dim3& grid, const dim3& tile) {
    return tile.z + tile.x * grid.z + tile.y * grid.x * grid.z;
  }

  __device__ __host__ __forceinline__
  dim3 tileToBlock(const dim3& tile) {
    return dim3{tile.y, tile.x, tile.z};
  }

  __device__ __host__ __forceinline__
  uint32_t tileIndex(const dim3& tile, const dim3& grid) {
    dim3 block = tileToBlock(tile);
    return this->operator()(grid, block);
  }
};

struct OrderZXY2 {
  __device__ __host__ __forceinline__
  uint32_t operator()(const dim3& grid, const dim3& tile) {
    return tile.x + tile.y * grid.x + tile.z * grid.x * grid.y;
  }

  __device__ __host__ __forceinline__
  dim3 tileToBlock(const dim3& tile) {
    return dim3{tile.y, tile.x, tile.z};
  }

  __device__ __host__ __forceinline__
  uint32_t tileIndex(const dim3& tile, const dim3& grid) {
    dim3 block = tileToBlock(tile);
    return OrderZXY()(grid, block);
  }
};
#endif
}
