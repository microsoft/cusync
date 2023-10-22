#pragma once

namespace cusync {
/*
 * A tile processing order is a functor that takes grid dimensions and tile index 
 * and return a linear index. A tile order is declared as:
 * struct TileOrder {
 *  size_t operator()(dim3 grid, dim3 tile);
 * }
 */

/*
 * OrderZYX order that generates tile indices first for Z-dimension, then 
 * Y-dimension, and finally X-dimension.
 */
struct OrderZYX {
  size_t operator()(const dim3& grid, const dim3& tile) {
    return tile.z + tile.y * grid.z + tile.x * grid.y * grid.z;
  }
};

/*
 * OrderXYZ order that generates tile indices first for X-dimension, then 
 * Y-dimension, and finally Z-dimension.
 */
struct OrderXYZ {
  size_t operator()(const dim3& grid, const dim3& tile) {
    return tile.x + tile.y * grid.x + tile.z * grid.x * grid.y;
  }
};
}