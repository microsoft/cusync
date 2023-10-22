/////////////////////////////////////////////////////////////////////////////////////////////////

#include <cutlass/gemm/threadblock/threadblock_swizzle.h>

namespace cutlass {
namespace gemm {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Adds two methods over GemmHorizontalThreadblockSwizzle
struct CuSyncGemmHorizontalThreadblockSwizzle : public GemmHorizontalThreadblockSwizzle {
  /// Obtains the threadblock offset (in units of threadblock-scoped tiles)
  /// block_idx are already transposed in the kernel grid, 
  /// so x is column dim and y is row dim
  CUTLASS_DEVICE
  GemmCoord get_tile_offset(int log_tile) const {
    int block_idx_x = RematerializeBlockIdxX();
    int block_idx_y = RematerializeBlockIdxY();
    int block_idx_z = RematerializeBlockIdxZ();

    return GemmCoord{(block_idx_y >> log_tile),  //
                     (block_idx_x << log_tile) + ((block_idx_y) & ((1 << (log_tile)) - 1)),
                     block_idx_z};
  }

  CUTLASS_DEVICE
  GemmCoord get_tile_offset(int log_tile, int block_idx_x, int block_idx_y, int block_idx_z) const {
    return GemmCoord{(block_idx_y >> log_tile),  //
                    (block_idx_x << log_tile) + ((block_idx_y) & ((1 << (log_tile)) - 1)),
                    block_idx_z};
  }

  CUTLASS_HOST_DEVICE
  GemmCoord get_tiled_shape(
    GemmCoord problem_size,
    GemmCoord tile_size,
    int split_k_slices) const {
      return GemmHorizontalThreadblockSwizzle::get_tiled_shape(problem_size, tile_size, split_k_slices);
  }
  CUTLASS_HOST_DEVICE
  GemmCoord get_tiled_shape(
    cutlass::conv::Operator conv_operator,
    cutlass::conv::Conv2dProblemSize const &problem_size,
    GemmCoord tile_size,
    int split_k_slices) const {

    gemm::GemmCoord implicit_gemm_problem_size = 
    cutlass::conv::implicit_gemm_problem_size(conv_operator, problem_size);

    return GemmHorizontalThreadblockSwizzle::get_tiled_shape(
      implicit_gemm_problem_size, tile_size, split_k_slices);
  }
};

template <int N = 1>
struct CuSyncGemmIdentityThreadblockSwizzle : public GemmIdentityThreadblockSwizzle<N> {
  /// get_tile_offset based on custom block indices
  CUTLASS_DEVICE
  GemmCoord get_tile_offset(int log_tile, int block_idx_x, int block_idx_y, int block_idx_z) const {
    return GemmCoord{(block_idx_x >> log_tile),  //
                    (block_idx_y << log_tile) + ((block_idx_x) & ((1 << (log_tile)) - 1)),
                    block_idx_z};
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace gemm
} // namespace cutlass