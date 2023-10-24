#pragma once

namespace cusync {
/*TODO:
 * A tile processing order is a functor that takes grid dimensions and tile index 
 * and return a linear index. A tile order is declared as:
 * struct TileOrder {
 *  size_t operator()(dim3& grid, dim3& tile);
 *  dim3 tileToBlock(const dim3& tile);
 *  uint tileIndex(const dim3& tile, const dim3& grid); 
 * }
 */

/*
 * TransposeXYOrder order that generates tile indices first for X-dimension, then 
 * Y-dimension, and finally Z-dimension.
 * It maps a tile {x, y, z} to threadblock {y, x, z}
 */
struct TransposeXYOrder {
  __device__ __host__ __forceinline__
  uint operator()(const dim3& grid, const dim3& tile) {
    return tile.x + tile.y * grid.x + tile.z * grid.x * grid.y;
  }

  __device__ __host__ __forceinline__
  dim3 tileToBlock(const dim3& tile) {
    return dim3{tile.y, tile.x, tile.z};
  }

  __device__ __host__ __forceinline__
  uint tileIndex(const dim3& tile, const dim3& grid) {
    dim3 block = tileToBlock(tile);
    return this->operator()(grid, block);
  }
};

/*
 * IdentityOrder orders tile indices first for X-dimension, then Y-dim, and finally Z-dim. 
 * It maps a tile {x,y,z} to threadblock {x,y,z}.
 */
struct IdentityOrder {
  __device__ __host__ __forceinline__
  uint operator()(const dim3& grid, const dim3& tile) {
    return tile.x + tile.y * grid.x + tile.z * grid.x * grid.y;
  }

  __device__ __host__ __forceinline__
  dim3 tileToBlock(const dim3& tile) {
    return dim3{tile.x, tile.y, tile.z};
  }

  __device__ __host__ __forceinline__
  uint tileIndex(const dim3& tile, const dim3& grid) {
    dim3 block = tileToBlock(tile);
    return this->operator()(grid, block);
  }
};
}