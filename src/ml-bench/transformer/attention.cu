// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

//<OPTIMIZATIONS>
//</OPTIMIZATIONS>

#if defined(TILESYNC)
#define NO_ATOMIC_ADD
#endif

#if defined(TILESYNC) || defined(STRIDEDSYNC)
// #define AVOID_CUSTOM_ORDER
#define REORDER_TILE_LOADS
// #define AVOID_WAIT_KERNEL
#endif

//Always AVOID for batch <= 512

// #define AVOID_CUSTOM_ORDER
// #define AVOID_WAIT_KERNEL

#include<cusync/cusync.h>

#include "common.h"

#ifndef EVAL_TILE_SIZES
//Tile sizes of all GeMMs
typedef cutlass::gemm::GemmShape<256, 256, 32> ShapeThreadBlock1;
typedef cutlass::gemm::GemmShape<128, 64, 32> ShapeWarp1;
const int SoftmaxRowTile = 1;
using ShapeThreadBlock2 = cutlass::gemm::GemmShape<128, 128, 32>;
using ShapeWarp2 = cutlass::gemm::GemmShape<64, 32, 32>;
#else
//<eval tiles>
const int SoftmaxRowTile = 1;
using ShapeThreadBlock = cutlass::gemm::GemmShape<256, 128, 32>;
using ShapeWarp = cutlass::gemm::GemmShape<128, 64, 32>;
//</eval tiles>
#endif

struct TileSizeLinearLayers {
  using ShapeThreadBlock = ::ShapeThreadBlock1;
  using ShapeWarp = ::ShapeWarp1;
};
struct TileSizeAttention {
  using ShapeThreadBlock = ::ShapeThreadBlock2;
  using ShapeWarp = ::ShapeWarp2;
};


template<typename TileOrder, uint H, uint TileM, uint TileN>
struct ConsecutiveSync {  
  __device__ __host__ ConsecutiveSync() {}

  __device__ __host__ uint waitValue(const dim3& tile, const dim3& grid) {
    return ((H/8)/TileN);
  }

  __device__ __host__ uint postValue(const dim3& tile, const dim3& grid) 
    {return 1;}

  __device__ constexpr uint tileIndex(const dim3& tile, const dim3& grid) {
    return TileOrder().tileIndex({tile.x/TileM, (tile.y/TileN)/((H/8)/TileN), 0},
                                 {(grid.x/((H/8)/TileN)), grid.y, grid.z});
  }

  __device__ bool isSync(const dim3& tile, const dim3& grid) {
    return tile.z == 1;
  }
};

// const int SoftmaxThreads = ShapeThreadBlock::kN;
using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;  
using namespace cusync;

const uint Opts = 
#ifdef AVOID_CUSTOM_ORDER
  Optimizations::AvoidCustomOrder |
#endif
#ifdef AVOID_WAIT_KERNEL
  Optimizations::AvoidWaitKernel  |
#endif
#ifdef NO_ATOMIC_ADD
  Optimizations::NoAtomicAdd      |
#endif
#ifdef REORDER_TILE_LOADS
  Optimizations::ReorderTileLoads |
#endif
  Optimizations::NoOptimization;

#ifdef ROWSYNC 
  using Sync1       = RowSync<TileSizeLinearLayers::ShapeThreadBlock::kM>;
  using Sync2       = RowSync<TileSizeAttention   ::ShapeThreadBlock::kM>;
  using XQKVCuStage = CuStage<TransposeXYOrder, NoSync, Sync1,  Opts>;
  using SCuStage    = CuStage<TransposeXYOrder, Sync1,  Sync2,  Opts | Optimizations::AvoidCustomOrder>;
  using OCuStage    = CuStage<TransposeXYOrder, Sync2,  Sync1,  Opts | Optimizations::AvoidCustomOrder>;
  using XW12CuStage = CuStage<TransposeXYOrder, Sync1,  NoSync, Opts>;
#elif defined(TILESYNC)
  using Sync1 = TileSync<TransposeXYOrder, TileSizeLinearLayers::ShapeThreadBlock::kM, TileSizeLinearLayers::ShapeThreadBlock::kN>;
  using Sync2 = TileSync<TransposeXYOrder, TileSizeAttention::ShapeThreadBlock::kM, TileSizeLinearLayers::ShapeThreadBlock::kN>;
  using XQKVCuStage = CuStage<TransposeXYOrder, NoSync, Sync1, Opts>;
  using SCuStage    = CuStage<TransposeXYOrder, Sync1, Sync2,  Opts | Optimizations::AvoidCustomOrder>;
  using OCuStage    = CuStage<TransposeXYOrder, Sync2, Sync1,  Opts | Optimizations::AvoidCustomOrder>;
  using XW12CuStage = CuStage<TransposeXYOrder, Sync1, NoSync, Opts>;
#elif defined(STRIDEDSYNC)
  #if defined(GPT3)
    using StridedSyncImpl = ConsecutiveSync<TransposeXYOrder, 12288, TileSizeLinearLayers::ShapeThreadBlock::kM, TileSizeLinearLayers::ShapeThreadBlock::kN>;
  #elif defined(LLaMA)
    using StridedSyncImpl = ConsecutiveSync<TransposeXYOrder,  8192, TileSizeLinearLayers::ShapeThreadBlock::kM, TileSizeLinearLayers::ShapeThreadBlock::kN>;
  #else
    #error "GPT3 or LLaMA"
  #endif
  using Sync1 = RowSync<TileSizeLinearLayers::ShapeThreadBlock::kM>;
  using Sync2 = RowSync<TileSizeAttention::ShapeThreadBlock::kM>;
  using XQKVCuStage = CuStage<TransposeXYOrder, NoSync, StridedSyncImpl, Opts>;
  using SCuStage    = CuStage<TransposeXYOrder, StridedSyncImpl, Sync2, Opts | Optimizations::AvoidCustomOrder>;
  using OCuStage    = CuStage<TransposeXYOrder, Sync2, Sync1, Opts | Optimizations::AvoidCustomOrder>;
  using XW12CuStage = CuStage<TransposeXYOrder, Sync1, NoSync, Opts>;
#else
  #error "Unknown Synchronization"
#endif 

//Element types of A, B, and C
using ElementAccumulator = float;
using ElementInputA = cutlass::half_t;
using ElementInputB = cutlass::half_t;
using ElementOutput = cutlass::half_t;
using ElementComputeEpilogue = cutlass::half_t;

//All matrices are in RowMajor
using LayoutInputA = cutlass::layout::RowMajor;
using LayoutInputB = cutlass::layout::RowMajor;
using LayoutOutput = cutlass::layout::RowMajor;

//Use FP-16 Tensor Cores
using MMAOp = cutlass::arch::OpClassTensorOp;

using SmArch = cutlass::arch::Sm70;

//Second GeMM in MLP performs no extra fused computations 
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,                                        
    128 / cutlass::sizeof_bits<ElementOutput>::value,     
    ElementAccumulator,
    ElementComputeEpilogue>;

template<typename TileSize, bool splitK>
class BaseLinearGemm : public cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, 
                                                        ElementInputB, LayoutInputB,
                                                        ElementOutput, LayoutOutput,
                                                        ElementAccumulator, MMAOp,
                                                        SmArch, 
                                                        typename TileSize::ShapeThreadBlock,
                                                        typename TileSize::ShapeWarp, 
                                                        ShapeMMAOp,
                                                        EpilogueOp, 
                                                        cutlass::gemm::threadblock::CuSyncGemmHorizontalThreadblockSwizzle, 
                                                        2, 8, 8, splitK> {};

// Baseline GeMMs
using LinearLayerGemm = BaseLinearGemm<TileSizeLinearLayers, false>;
using AttnGemm2 = BaseLinearGemm<TileSizeAttention, false>;

//Baseline GeMMs with SplitK enabled
using LinearLayerGemmSplitK = BaseLinearGemm<TileSizeLinearLayers, true>;
using AttnGemm2SplitK = BaseLinearGemm<TileSizeAttention, true>;

using LayoutK = cutlass::layout::ColumnMajor;
template<bool splitK>
class BColumnMajorGemm : public cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, 
                                                     ElementInputB, LayoutK,
                                                     ElementOutput, LayoutOutput,
                                                     ElementAccumulator, MMAOp,
                                                     SmArch, 
                                                     TileSizeAttention::ShapeThreadBlock,
                                                     TileSizeAttention::ShapeWarp,
                                                     ShapeMMAOp,
                                                     EpilogueOp, 
                                                     cutlass::gemm::threadblock::CuSyncGemmHorizontalThreadblockSwizzle, 
                                                     2, 8, 8, splitK> {};

// Baseline GeMMs
using AttnColumnMajorGemm1 = BColumnMajorGemm<false>;
using AttnColumnMajorGemmSplitK1 = BColumnMajorGemm<true>;

using CuSyncGeMMSwizzle = cutlass::gemm::threadblock::CuSyncGemmHorizontalThreadblockSwizzle;

//CuSync GeMMs
template<typename CuStage, typename TileSize, bool splitK>
class CuSyncLinearLayerGemm : public cutlass::gemm::device::CuSyncGemm<CuStage, 
                                                    ElementInputA, LayoutInputA, 
                                                    ElementInputB, LayoutInputB,
                                                    ElementOutput, LayoutOutput,
                                                    ElementAccumulator, MMAOp,
                                                    SmArch, 
                                                    typename TileSize::ShapeThreadBlock,
                                                    typename TileSize::ShapeWarp, ShapeMMAOp,
                                                    EpilogueOp, 
                                                    CuSyncGeMMSwizzle,
                                                    2, 8, 8, splitK> {};

template<typename CuStage, bool splitK>
class CuSyncBColumnMajorGemm : public cutlass::gemm::device::CuSyncGemm<CuStage, ElementInputA, LayoutInputA, 
                                                     ElementInputB, LayoutK,
                                                     ElementOutput, LayoutOutput,
                                                     ElementAccumulator, MMAOp,
                                                     SmArch, TileSizeAttention::ShapeThreadBlock,
                                                     TileSizeAttention::ShapeWarp, ShapeMMAOp,
                                                     EpilogueOp,
                                                     CuSyncGeMMSwizzle,
                                                     2, 8, 8, splitK> {};

// CuSync GeMMs
using XQKVCuSyncGemm = CuSyncLinearLayerGemm<XQKVCuStage, TileSizeLinearLayers, false>;
using SCuSyncGemm = CuSyncBColumnMajorGemm<SCuStage, false>;
using OCuSyncGemm = CuSyncLinearLayerGemm<OCuStage, TileSizeAttention, false>;
using XW12CuSyncGemm = CuSyncLinearLayerGemm<XW12CuStage, TileSizeLinearLayers, false>;

using XQKVCuSyncGemmSplitK = CuSyncLinearLayerGemm<XQKVCuStage, TileSizeLinearLayers, true>;
using SCuSyncGemmSplitK = CuSyncBColumnMajorGemm<SCuStage, true>;
using OCuSyncGemmSplitK = CuSyncLinearLayerGemm<OCuStage, TileSizeAttention, true>;
using XW12CuSyncGemmSplitK = CuSyncLinearLayerGemm<XW12CuStage, TileSizeLinearLayers, true>;

using HostTensor = cutlass::HostTensor<ElementInputA, LayoutInputA>;

struct AttentionParams {
  //Attention does following computations:
  //XQKV = X * QKV
  //Q, K, V = QKV[:,0:H/3], QKV[:,H/3:2H/3] , QKV[:,2H/3]
  //S = Q * K^T
  //P = softmax(S)
  //O = P * V
  //XW12 = O * W2  
  std::string model;
  int seqlen;
  int hidden_dim;
  int batch;

  HostTensor x;
  HostTensor qkv;
  HostTensor xqkv_storage;
  HostTensor s;
  HostTensor p;
  HostTensor o;
  HostTensor w2;
  HostTensor xw12;

  cutlass::TensorRef<ElementInputA, LayoutInputA> xqkv;

  HostTensor ref_xqkv;
  HostTensor ref_xqkv_storage;
  HostTensor ref_s;
  HostTensor ref_p;
  HostTensor ref_o;
  HostTensor ref_xw12;

  cutlass::gemm::GemmCoord gemm_size_xqkv, gemm_size_s, gemm_size_o, gemm_size_xw12, size_xqkv_storage;
  curandState* randStates;
  bool refCheck;
  ElementComputeEpilogue alpha;
  ElementComputeEpilogue beta;

  uint model_parallel_H;
  
  AttentionParams(std::string model, int batch, int seqlen, int hidden_dim, bool check) {
    this->model = model;
    this->seqlen = seqlen;
    this->hidden_dim = hidden_dim;
    this->batch = batch;
    int numGPUs = 8;
    model_parallel_H = hidden_dim/numGPUs;
    gemm_size_xqkv = cutlass::gemm::GemmCoord(batch, model_parallel_H * 3, hidden_dim);
    size_xqkv_storage = cutlass::gemm::GemmCoord(batch + seqlen, model_parallel_H * 3, hidden_dim);
    gemm_size_s = cutlass::gemm::GemmCoord(batch, batch + seqlen, model_parallel_H);
    gemm_size_o = cutlass::gemm::GemmCoord(batch, model_parallel_H, batch+seqlen);
    gemm_size_xw12 = cutlass::gemm::GemmCoord(batch, hidden_dim, model_parallel_H);
    
    alpha = ElementComputeEpilogue(1);
    beta = ElementComputeEpilogue(0);
  
    x    = HostTensor(gemm_size_xqkv.mk());
    qkv  = HostTensor(gemm_size_xqkv.kn());
    xqkv_storage = HostTensor(size_xqkv_storage.mn());
    s = HostTensor(gemm_size_s.mn());
    o = HostTensor(gemm_size_o.mn());
    w2   = HostTensor(gemm_size_xw12.kn());
    xw12 = HostTensor(gemm_size_xw12.mn());

    xqkv = {xqkv_storage.device_data() + seqlen*model_parallel_H, LayoutInputA(3*model_parallel_H)};
    printf("309: %p %p\n", xqkv.data(), xqkv_storage.device_data());
    ref_xqkv = HostTensor(gemm_size_xqkv.mn());
    ref_xqkv_storage = HostTensor(size_xqkv_storage.mn());
    ref_s = HostTensor(gemm_size_s.mn());
    ref_o = HostTensor(gemm_size_o.mn());
    ref_xw12 = HostTensor(gemm_size_xw12.mn());

    size_t numRandStates = gemm_size_xqkv.m() * 1024;
    CUDA_CHECK(cudaMalloc(&randStates, sizeof(curandState)*(numRandStates)));
    init_curand_states<<<numRandStates/128, 128>>>(randStates, numRandStates);
    CUDA_CHECK(cudaDeviceSynchronize());
    refCheck = check;
  }

  void initIns() {
    if (refCheck) {
      memset_random2(x.host_data(), ElementOutput(0.005), 
                     ElementOutput(0.01), x.size());
      memset_random2(xqkv_storage.host_data(), ElementOutput(0.005), 
                     ElementOutput(0.01), xqkv_storage.size());
      memcpy(ref_xqkv_storage.host_data(), xqkv_storage.host_data(), ref_xqkv_storage.size() * sizeof(ElementInputA));
      memset_random2(qkv.host_data(), ElementOutput(0.005), 
                     ElementOutput(0.01), qkv.size());
      memset_random2(w2.host_data(), ElementOutput(0.01),
                     ElementOutput(0.05), w2.size());
    } else {
      cutlass::reference::host::TensorFill(x.host_view(),
                                           ElementOutput(0.05));
      cutlass::reference::host::TensorFill(xqkv_storage.host_view(),
                                           ElementOutput(0.05));
      cutlass::reference::host::TensorFill(qkv.host_view(),
                                           ElementOutput(0.5));
      cutlass::reference::host::TensorFill(w2.host_view(),
                                           ElementOutput(0.01));
    }

    // Copy data from host to GPU
    xqkv_storage.sync_device();
    ref_xqkv_storage.sync_device();
    x.sync_device();
    qkv.sync_device();
    w2.sync_device();
  }

  void initOuts() {
    //Zeros all output tensors
    cutlass::reference::host::TensorFill(s.host_view());
    cutlass::reference::host::TensorFill(p.host_view());
    cutlass::reference::host::TensorFill(o.host_view());
    cutlass::reference::host::TensorFill(xw12.host_view());
  }

  void initRefs() {
    cutlass::reference::host::TensorFill(ref_xqkv.host_view());
    cutlass::reference::host::TensorFill(ref_s.host_view());
    cutlass::reference::host::TensorFill(ref_p.host_view());
    cutlass::reference::host::TensorFill(ref_o.host_view());
    cutlass::reference::host::TensorFill(ref_xw12.host_view());
  }
};

// template<uint NTHREADS, typename T, typename AT, uint TileM, uint TileN, uint RowTile, bool enableOverlap>
// __global__ void selfAttnDotProdSoftmaxDropout(uint32_t M, uint32_t N,
//                                               T* XQKV, T* out, float p,
//                                               curandState* randStates,
//                                               MiddleCuStage cons1, MiddleCuStage prod2) {
//   extern __shared__ half xqkRows[];

//   __shared__ AT sum;
//   if (enableOverlap)
//     prod2.tile(nullptr);
//   int linearThreadId = blockIdx.x * blockDim.x + threadIdx.x;
//   curandState* localRandState = &randStates[linearThreadId];
//   // __shared__ shRandStates[sizeof(curandState) * NTHREADS];
//   uint ROW = blockIdx.x * RowTile;
//   const uint tileRow = blockIdx.x;
//   const uint tileM = ROW/TileM;
//   if (enableOverlap) {
//     // && tileM == 0) printf("TileM %d TileN %d ROW %d\n", TileM, TileN, ROW);
//     // handle1.waitOnTilesWithSyncValue(tileM, 0, 0, 1);
//     // if (tileM < M/TileM) {
//     //   {tileM + 1, 0, 0};
//     //   handle1.waitOnTile();
//     // }
//   }

//   for (uint ti = 0; ti < RowTile && ROW < M; ti++) {
//     if (threadIdx.x == 0) {
//       sum = 0;
//     }

//     AT threadSum = (AT)0.0f;

//     for (uint COL = threadIdx.x; COL < N; COL += blockDim.x) {
//       if (enableOverlap) {
//         if (ti == 0 && ROW % TileM == 0) {
//           dim3 tile = {tileM, COL/TileN, 0};
//           cons1.wait(tile, (COL/TileN)%NTHREADS);
//         }
//       }
//       T xq = XQKV[ROW * 3 * N + COL];
//       if (enableOverlap  && ti == 0 && ROW % TileM == 0) {
//         dim3 tile = {tileM, N/TileN + COL/TileN, 0};
//         #ifdef TILESYNC
//         cons1.wait(tile, (COL/TileN)%NTHREADS);
//         #endif
//       }
//       T xk = XQKV[ROW * 3 * N + (COL + N)];
//       T xqk = xq * xk;
//       threadSum += (AT)exp((AT)xqk);
//       xqkRows[COL] = xqk;
//     }
//     __syncthreads();
//     atomicAdd(&sum, (AT)threadSum);
//     __syncthreads();
//     for (uint COL = threadIdx.x; COL < N; COL += blockDim.x) {
//       float r = curand_uniform(localRandState);
//       // if (enableOverlap && ti == 0) {
//       //   if (rowSyncOrTileSync) {

//       //   } else {
//       if (enableOverlap && ti == 0 && ROW % TileM == 0) {
//         dim3 tile = {tileM, N/TileN*2 + COL/TileN, 0};
//         #ifndef TILESYNC
//         cons1.wait(tile, (COL/TileN)%NTHREADS);
//         #endif
//       }
//       __half v = (r <= p) ? (__half)(((float)(exp((AT)xqkRows[COL]) * 
//                                      (float)XQKV[ROW* 3 * N + (COL + 2 * N)]))/sum) : (__half)0.0f;
//       out[ROW * N + COL] = v;
//       if (enableOverlap && ti == SoftmaxRowTile - 1) {
//         dim3 tile = {tileM, COL/TileN, 0};
//         prod2.post(tile, ((COL/TileN)*TileN)%NTHREADS);
//       }
//     }
//     __syncthreads();

//     ROW++;
//   }

//   // if (enableOverlap) {
//   //   if (rowSyncOrTileSync) {
//   //     // tileM = ROW/TileM;
//   //     handle2.setRowStatus(tileM, 0, 0, RowTile);
//   //   } else {
      
//   //   }
//   // }
// }

void attnRefMatmul(cutlass::gemm::GemmCoord size, ElementOutput* a, ElementOutput* b, ElementOutput* c) {
  ref_matmul<ElementOutput, ElementAccumulator>(size.m(), size.n(), 
                                                size.k(), a, b, c);
}

cudaError_t host_attention(AttentionParams& attnParams) {
  attnRefMatmul(attnParams.gemm_size_xqkv, attnParams.x.device_data(), 
                attnParams.qkv.device_data(), attnParams.ref_xqkv.host_data());
  
  //assert(attnParams.ref_xdot.size() == attnParams.gemm_size1.m() * attnParams.gemm_size1.n()/3);
  size_t N = attnParams.model_parallel_H;
  size_t B = attnParams.batch;
  size_t SEQ = attnParams.seqlen;
  memcpy(attnParams.ref_xqkv_storage.host_data() + SEQ*N, attnParams.ref_xqkv.host_data(),
         attnParams.ref_xqkv.size() * sizeof(ElementInputA));
  
  ElementOutput* host_xq = attnParams.ref_xqkv.host_data();
  ElementOutput* host_xqkv = attnParams.ref_xqkv_storage.host_data();
  ElementOutput* host_s = attnParams.ref_s.host_data();

  for (size_t i = 0; i < B; i++) {
    for (size_t j = 0; j < B + SEQ; j++) {
      ElementAccumulator result = 0.0f;
      ElementOutput r1 = (ElementOutput)0.0f;

      for (size_t k = 0; k < N; k++) {
        ElementOutput xq = host_xq[i * 3 * N + k];
        ElementOutput xk = host_xqkv[j * 3 * N + k + N];
        result += xq * xk;
        r1 += xq * xk;
      }
      host_s[i * (B + SEQ) + j] = (ElementOutput)result;
    }
  }

  ElementOutput* host_p = host_s; //new ElementOutput[B * B];

  // for (size_t i = 0; i < B; i++) {
  //   float sum = 0.0f;
  //   for (size_t j = 0; j < B; j++) {
  //     sum += exp((float)host_s[i*B + j]);
  //   }
    
  //   for (size_t j = 0; j < B; j++) {
  //     //Assume dropout probability is 1.0
  //     host_p[i*B + j] = exp(host_s[i*B + j])/sum;
  //   }
  // }
  
  ElementOutput* host_o = attnParams.ref_o.host_data();
  
  for (size_t i = 0; i < B; i++) {
    for (size_t j = 0; j < N; j++) {
      ElementAccumulator result = 0.0f;
      
      for (size_t k = 0; k < B + SEQ; k++) {
        ElementOutput host_xv = host_xqkv[k * 3 * N + j + N * 2];
        
        result += host_xv * host_p[i*(B+SEQ) + k];
      }
      host_o[i * N + j] = (ElementOutput)result;
    }
  }

  attnParams.ref_o.sync_device();

  attnRefMatmul(attnParams.gemm_size_xw12, attnParams.ref_o.device_data(), 
                attnParams.w2.device_data(), attnParams.ref_xw12.host_data());
  return cudaSuccess;
}

cudaError_t check_results(AttentionParams& attnParams) {
  if (attnParams.model == "gpt3") {
    printf("No point in checking for GPT-3 because fp16 elements will overflow. Check with llama.\n");
    return cudaSuccess;
  }
  attnParams.xqkv_storage.sync_host();
  printf("Checking XQKV=X*QKV\n");

  bool eq = equals(attnParams.ref_xqkv.size(), 
                   attnParams.ref_xqkv.host_data(), 
                   attnParams.xqkv_storage.host_data() + attnParams.model_parallel_H * attnParams.seqlen, 1e-1f);
  if (eq == false) {
    printf("not correct\n");
    return cudaErrorUnknown;
  }

  attnParams.s.sync_host();
  printf("Checking S=Q*K.T\n");
  eq = equals(attnParams.ref_s.size(), attnParams.ref_s.host_data(),
              attnParams.s.host_data(), 1e-1f);
  if (eq == false) {
    printf("not correct\n");
    return cudaErrorUnknown;
  }

  attnParams.o.sync_host();
  printf("Checking O=S*V\n");
  eq = equals(attnParams.ref_o.size(), attnParams.ref_o.host_data(),
              attnParams.o.host_data(), 1e-1f);
  if (eq == false) {
    printf("not correct\n");
    return cudaErrorUnknown;
  }

  printf("Passed\n");

  return cudaSuccess;
}

__global__ void print_kernel(ElementOutput* data) {
  if (threadIdx.x < 10) {
    printf("%p %f\n", data, (float)data[threadIdx.x]);
  }
}

//Run our baseline of Self-Attention
template<typename GemmTy1, typename GemmTy2, typename GemmTy3, typename GemmTy4>
cudaError_t runAttentionBaseline(int split_k1, int split_k2, int split_k3, int split_k4,
                                 AttentionParams& attnParams,
                                 cudaStream_t streams[],
                                 double& execTime,
                                 double& matmul1Time,
                                 double& matmul2Time,
                                 double& matmul3Time,
                                 double& matmul4Time,
                                 int iters = 100) {  
  // ElementOutput* device_xqkv = tensor_xqkv.device_data();
  cutlass::Status status;
  //Setup First GeMM
  typename GemmTy1::Arguments args1{attnParams.gemm_size_xqkv,
                                    attnParams.x.device_ref(),
                                    attnParams.qkv.device_ref(),
                                    attnParams.xqkv, attnParams.xqkv,
                                    {attnParams.alpha, attnParams.beta},
                                    split_k1};
  size_t workspace_size = GemmTy1::get_workspace_size(args1);
  cutlass::device_memory::allocation<uint8_t> workspace1(workspace_size);
  GemmTy1 gemm_op1;
  status = gemm_op1.can_implement(args1);
  CUTLASS_CHECK(status);
  status = gemm_op1.initialize(args1, workspace1.get());
  CUTLASS_CHECK(status);

  size_t N = attnParams.gemm_size_xqkv.n()/3;

  ElementOutput* device_xk = attnParams.xqkv_storage.device_data() + N;
  cutlass::TensorRef xk{device_xk, LayoutK(3*N)};

  //Setup S=Q*K.T GeMM
  typename GemmTy2::Arguments args2{attnParams.gemm_size_s,
                                    attnParams.xqkv, xk,
                                    attnParams.s.device_ref(),
                                    attnParams.s.device_ref(),
                                    {attnParams.alpha, attnParams.beta},
                                    split_k2};
  workspace_size = GemmTy2::get_workspace_size(args2);
  cutlass::device_memory::allocation<uint8_t> workspace2(workspace_size);
  GemmTy2 gemm_op2;
  status = gemm_op2.can_implement(args2);
  CUTLASS_CHECK(status);
  status = gemm_op2.initialize(args2, workspace2.get());
  CUTLASS_CHECK(status);

  ElementInputA* device_xv = attnParams.xqkv_storage.device_data() + N;
  cutlass::TensorRef xv{device_xv, LayoutInputB(3*N)};

  //Setup O=S*V GeMM
  typename GemmTy3::Arguments args3{attnParams.gemm_size_o,
                                    attnParams.s.device_ref(),
                                    xv,
                                    attnParams.o.device_ref(),
                                    attnParams.o.device_ref(),
                                    {attnParams.alpha, attnParams.beta},
                                    split_k3};
  workspace_size = GemmTy3::get_workspace_size(args3);
  cutlass::device_memory::allocation<uint8_t> workspace3(workspace_size);
  GemmTy3 gemm_op3;
  status = gemm_op3.can_implement(args3);
  CUTLASS_CHECK(status);
  status = gemm_op3.initialize(args3, workspace3.get());
  CUTLASS_CHECK(status);

  //Setup XW12=O*W12 GeMM
  typename GemmTy4::Arguments args4{attnParams.gemm_size_xw12,
                                    attnParams.o.device_ref(),
                                    attnParams.w2.device_ref(),
                                    attnParams.xw12.device_ref(),
                                    attnParams.xw12.device_ref(),
                                    {attnParams.alpha, attnParams.beta},
                                    split_k4};
  workspace_size = GemmTy4::get_workspace_size(args4);
  cutlass::device_memory::allocation<uint8_t> workspace4(workspace_size);
  GemmTy4 gemm_op4;
  status = gemm_op4.can_implement(args4);
  CUTLASS_CHECK(status);
  status = gemm_op4.initialize(args4, workspace4.get());
  CUTLASS_CHECK(status);

  execTime = 0;
  
  //Launch kernels
  for (int r = 0; r < iters; r++) {
    double start = timeInMicroSeconds();
    status = gemm_op1(streams[0]);
    CUTLASS_CHECK(status);
    CUDA_CHECK(cudaStreamSynchronize(streams[0]));
    double middle1 = timeInMicroSeconds();
    double iterMatMul1 = middle1-start;
    matmul1Time += iterMatMul1;
    
    status = gemm_op2(streams[0]);
    CUTLASS_CHECK(status);
    CUDA_CHECK(cudaDeviceSynchronize());
    double middle2 = timeInMicroSeconds();
    double iterMatmul2 = middle2-middle1;
    matmul2Time += iterMatmul2;
    
    status = gemm_op3(streams[0]);
    CUTLASS_CHECK(status);
    CUDA_CHECK(cudaDeviceSynchronize());
    double middle3 = timeInMicroSeconds();
    double iterMatmul3 = middle3-middle2;
    matmul3Time += iterMatmul3;

    status = gemm_op4(streams[0]);
    CUTLASS_CHECK(status);
    CUDA_CHECK(cudaDeviceSynchronize());
    double middle4 = timeInMicroSeconds();
    double iterMatmul4 = middle4-middle3;
    matmul4Time += iterMatmul4;
  
    double end = timeInMicroSeconds();
    if (iters > 10)
      printf("{\"Total\": %lf, \"matmul1Time\": %lf, \"matmul2Time\": %lf, \"matmul3Time\": %lf, \"matmul4Time\": %lf}\n",
             end-start, iterMatMul1, iterMatmul2, iterMatmul3, iterMatmul4);
    execTime += end-start;
  }

  return cudaSuccess;
}

cudaError_t runAttentionBaseline(int split_k1, int split_k2, int split_k3, int split_k4,
                                 AttentionParams& attnParams, 
                                 cudaStream_t streams[],
                                 double& execTime,
                                 double& matmul1Time,
                                 double& matmul2Time,
                                 double& matmul3Time,
                                 double& matmul4Time,
                                 int iters = 100) {
  cudaError_t result;
  if (split_k1 == 1 && split_k4 == 1) {
    result = runAttentionBaseline<LinearLayerGemm, AttnColumnMajorGemmSplitK1, AttnGemm2SplitK, LinearLayerGemm>(split_k1, split_k2, split_k3, split_k4, attnParams, streams, execTime, matmul1Time, matmul2Time, matmul3Time, matmul4Time, iters);
  } else if (split_k1 > 1 && split_k4 == 1) {
     result = runAttentionBaseline<LinearLayerGemmSplitK, AttnColumnMajorGemmSplitK1, AttnGemm2SplitK, LinearLayerGemm>(split_k1, split_k2, split_k3, split_k4, attnParams, streams, execTime, matmul1Time, matmul2Time, matmul3Time, matmul4Time, iters);
  } else if (split_k1 == 1 && split_k4 > 1) {
    result = runAttentionBaseline<LinearLayerGemm, AttnColumnMajorGemmSplitK1, AttnGemm2SplitK, LinearLayerGemmSplitK>(split_k1, split_k2, split_k3, split_k4, attnParams, streams, execTime, matmul1Time, matmul2Time, matmul3Time, matmul4Time, iters);
  } else if (split_k1 > 1 && split_k4 > 1) {
    result = runAttentionBaseline<LinearLayerGemmSplitK, AttnColumnMajorGemmSplitK1, AttnGemm2SplitK, LinearLayerGemmSplitK>(split_k1, split_k2, split_k3, split_k4, attnParams, streams, execTime, matmul1Time, matmul2Time, matmul3Time, matmul4Time, iters);
  }

  return result;
}

//Self-Attention using CuSync
template<typename XQKVCuSyncGemmTy, typename SCuSyncGemmTy, typename OCuSyncGemmTy, typename XW12CuSyncGemmTy>
cudaError_t runAttentionCuSync(int split_k1, int split_k2, int split_k3, int split_k4,
                               AttentionParams& attnParams,
                               XQKVCuStage& xqkvstage, SCuStage& scustage, OCuStage& ocustage, XW12CuStage& xw12custage,
                               cudaStream_t streams[],
                               double& execTime,
                               int iters = 100) {
  //Setup XQKV = X * QKV GeMM
  typename XQKVCuSyncGemmTy::Arguments args1{xqkvstage,
                                            attnParams.gemm_size_xqkv,
                                            attnParams.x.device_ref(),
                                            attnParams.qkv.device_ref(),
                                            attnParams.xqkv,
                                            attnParams.xqkv,
                                            {attnParams.alpha, attnParams.beta},
                                            split_k1};
  size_t workspace_size = XQKVCuSyncGemmTy::get_workspace_size(args1);
  cutlass::device_memory::allocation<uint8_t> workspace1(workspace_size);
  XQKVCuSyncGemmTy gemm_op1;
  cutlass::Status status;
  status = gemm_op1.can_implement(args1);
  CUTLASS_CHECK(status);
  status = gemm_op1.initialize(args1, workspace1.get());
  CUTLASS_CHECK(status);

  size_t N = attnParams.gemm_size_xqkv.n()/3;
  ElementOutput* device_xk = attnParams.xqkv_storage.device_data() + N;
  cutlass::TensorRef xk{device_xk, LayoutK(3*N)};

  //Setup S = Q * K.T GeMM
  typename SCuSyncGemmTy::Arguments args2{scustage,
                                    attnParams.gemm_size_s,
                                    attnParams.xqkv, xk,
                                    attnParams.s.device_ref(),
                                    attnParams.s.device_ref(),
                                    {attnParams.alpha, attnParams.beta},
                                    split_k2};
  workspace_size = SCuSyncGemmTy::get_workspace_size(args2);
  cutlass::device_memory::allocation<uint8_t> workspace2(workspace_size);
  SCuSyncGemmTy gemm_op2;
  status = gemm_op2.can_implement(args2);
  CUTLASS_CHECK(status);
  status = gemm_op2.initialize(args2, workspace2.get());
  CUTLASS_CHECK(status);
  
  ElementOutput* device_xv = attnParams.xqkv_storage.device_data() + N;
  cutlass::TensorRef xv{device_xv, LayoutInputB(3*N)};

  //Setup O=S*V GeMM
  typename OCuSyncGemmTy::Arguments args3{ocustage,
                                    attnParams.gemm_size_o,
                                    attnParams.s.device_ref(),
                                    xv,
                                    attnParams.o.device_ref(),
                                    attnParams.o.device_ref(),
                                    {attnParams.alpha, attnParams.beta},
                                    split_k3};
  workspace_size = OCuSyncGemmTy::get_workspace_size(args3);
  cutlass::device_memory::allocation<uint8_t> workspace3(workspace_size);
  OCuSyncGemmTy gemm_op3;
  status = gemm_op3.can_implement(args3);
  CUTLASS_CHECK(status);
  status = gemm_op3.initialize(args3, workspace3.get());
  CUTLASS_CHECK(status);

  //Setup XW12=O*W12 GeMM
  typename XW12CuSyncGemmTy::Arguments args4{xw12custage,
                                    attnParams.gemm_size_xw12,
                                    attnParams.o.device_ref(),
                                    attnParams.w2.device_ref(),
                                    attnParams.xw12.device_ref(),
                                    attnParams.xw12.device_ref(),
                                    {attnParams.alpha, attnParams.beta},
                                    split_k4};
  workspace_size = XW12CuSyncGemmTy::get_workspace_size(args4);
  cutlass::device_memory::allocation<uint8_t> workspace4(workspace_size);
  XW12CuSyncGemmTy gemm_op4;
  status = gemm_op4.can_implement(args4);
  CUTLASS_CHECK(status);
  status = gemm_op4.initialize(args4, workspace4.get());
  CUTLASS_CHECK(status);

  execTime = 0;
  
  //Run Kernels in Self-Attention
  for (int r = 0; r < iters; r++) {
    double start = timeInMicroSeconds();
    status = gemm_op1.run(true, NULL, streams[0]);
    CUTLASS_CHECK(status);

    xqkvstage.invokeWaitKernel(streams[1]);
    status = gemm_op2.run(true, NULL, streams[1]);
    CUTLASS_CHECK(status);
    
    // double e = timeInMicroSeconds();
    // // if (iters > 10) printf("%f\n", (e-start));
    scustage.invokeWaitKernel(streams[2]);
    status = gemm_op3.run(true, NULL, streams[2]);
    CUTLASS_CHECK(status);
    // // CUDA_CHECK(cudaStreamSynchronize(streams[2]));
    ocustage.invokeWaitKernel(streams[3]);
    status = gemm_op4.run(true, NULL, streams[3]);
    CUTLASS_CHECK(status);

    CUDA_CHECK(cudaDeviceSynchronize());
    double end = timeInMicroSeconds();

    if (status != cutlass::Status::kSuccess) {
      return cudaErrorUnknown;
    }

    xqkvstage.incrementIter();
    scustage.incrementIter();
    ocustage.incrementIter();
    xw12custage.incrementIter();

    gemm_op1.params_.custage.incrementIter();
    gemm_op2.params_.custage.incrementIter();
    gemm_op3.params_.custage.incrementIter();
    gemm_op4.params_.custage.incrementIter();

    if (iters > 10)
      printf("{\"Total\": %lf}\n",end-start);
    execTime += end-start;
  }

  return cudaSuccess;
}

cudaError_t runAttentionCuSync(int split_k1, int split_k2, int split_k3, int split_k4,
                               AttentionParams& attnParams,
                               XQKVCuStage& xqkvstage, SCuStage& scustage, OCuStage& ocustage, XW12CuStage& xw12custage,
                               cudaStream_t streams[],
                               double& execTime,
                               int iters = 100) {
  cudaError_t result;
  if (split_k1 == 1 && split_k4 == 1) {
    result = runAttentionCuSync<XQKVCuSyncGemm, SCuSyncGemmSplitK, OCuSyncGemmSplitK, XW12CuSyncGemm>(split_k1, split_k2, split_k3, split_k4, attnParams, 
                                              xqkvstage, scustage, ocustage, xw12custage, streams, execTime, iters);
  } else if (split_k1 == 1 && split_k4 > 1) {
    result = runAttentionCuSync<XQKVCuSyncGemm, SCuSyncGemmSplitK, OCuSyncGemmSplitK, XW12CuSyncGemmSplitK>(split_k1, split_k2, split_k3, split_k4, attnParams, 
                                              xqkvstage, scustage, ocustage, xw12custage, streams, execTime, iters);
  } else if (split_k1 > 1 && split_k4 == 1) {
    result = runAttentionCuSync<XQKVCuSyncGemmSplitK, SCuSyncGemmSplitK, OCuSyncGemmSplitK, XW12CuSyncGemm>(split_k1, split_k2, split_k3, split_k4, attnParams, 
                                              xqkvstage, scustage, ocustage, xw12custage, streams, execTime, iters);
  } else if (split_k1 > 1 && split_k4 > 1) {
    result = runAttentionCuSync<XQKVCuSyncGemmSplitK, SCuSyncGemmSplitK, OCuSyncGemmSplitK, XW12CuSyncGemmSplitK>(split_k1, split_k2, split_k3, split_k4, attnParams, 
                                              xqkvstage, scustage, ocustage, xw12custage, streams, execTime, iters);
  }

  return result;
}

int run(int argc, char* argv[]) {
  cudaDeviceProp props;

  cudaError_t error = cudaGetDeviceProperties(&props, 0);
  if (error != cudaSuccess) {
    std::cerr << "cudaGetDeviceProperties() returned an error: " << cudaGetErrorString(error) << std::endl;
    return -1;
  }

  if (props.major != 7) {
    std::cerr << "Volta Tensor Ops must be run on a machine with compute capability of 70, 72, or 75."
              << std::endl;

    // Return 0 so tests are considered passing if run on unsupported architectures or CUDA Toolkits.
    return 0;
  }
  const uint NUM_ARGS = 8;
  std::string argNames[NUM_ARGS] = {"--model", "--batch", "--check", "--seqlen", "--split-k1", "--split-k2", "--split-k3", "--split-k4"};
  std::string argHelp[NUM_ARGS] = {"GPT-3 or LLaMa", "Batch size", "Sequence Length", "Check results", 
                                   "Split K for XQKV = X*QKV GeMM", "Split K for P=Q*K.T GeMM",
                                   "Split K for O=P*V GeMM", "Split K for XW12=O*W12 GeMM"};
  
  if (argc < NUM_ARGS+1) {
    std::cout << "usage: " << std::endl
              << argNames[0] << " gpt3|llama " << argHelp[0] << std::endl 
              << argNames[1] << " <int>" << argHelp[1] << std::endl
              << argNames[2] << " true|false" << argHelp[2] << std::endl
              << argNames[3] << " <int> " << argHelp[3] << std::endl
              << argNames[4] << " <int> " << argHelp[4] << std::endl
              << argNames[5] << " <int> " << argHelp[5] << std::endl
              << argNames[6] << " <int> " << argHelp[6] << std::endl
              << argNames[7] << " <int> " << argHelp[7] << std::endl;
    return 0;
  }

  std::string model = "";
  uint batch = 0;
  uint seqlen = 0;
  bool doChecking = false;
  uint split_k1 = 1;
  uint split_k2 = 1;
  uint split_k3 = 1;
  uint split_k4 = 1;

  for (int i = 1; i < argc; ++i) {
    std::string arg = std::string(argv[i]);
    if (arg.find(argNames[0]) == 0) {
      model = std::string(argv[i+1]);
      i = i + 1;
    } else if (arg.find(argNames[1]) == 0) {
      std::stringstream ss(argv[i+1]);
      ss >> batch;
      i = i + 1;
    } else if (arg.find(argNames[2]) == 0) {
      std::string val = std::string(argv[i+1]);
      if (val == "true") {
        doChecking = true;
      } else if (val == "false") {
        doChecking = false;
      } else {
        std::cout << "Invalid value for check " << val << std::endl;
      }
      i = i + 1;
    } else if (arg.find(argNames[3]) == 0) {
      seqlen = atoi(argv[i+1]);
      i=i+1;
    } else if (arg.find(argNames[4]) == 0) {
      split_k1 = atoi(argv[i+1]);
      i=i+1;
    } else if (arg.find(argNames[5]) == 0) {
      split_k2 = atoi(argv[i+1]);
      i=i+1;
    } else if (arg.find(argNames[6]) == 0) {
      split_k3 = atoi(argv[i+1]);
      i=i+1;
    } else if (arg.find(argNames[7]) == 0) {
      split_k4 = atoi(argv[i+1]);
      i=i+1;
    }
  }

  if (model == "" || batch == 0) {
    std::cout<<"invalid model or batch" <<std::endl;
    return 0;
  }
  
  std::cout << "model=" << model << " batch=" << batch << " seqlen=" << seqlen <<  " check="<<doChecking <<std::endl;
  int hidden_dim = 0;  
  if (model=="gpt3") {
    hidden_dim = 12288;
  } else if (model=="llama") {
    hidden_dim = 8192;
  }

  //
  // Run the CUTLASS GEMM test.
  //

  int highestPriority;
  int lowestPriority;
  
  CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&lowestPriority, &highestPriority));
  if (highestPriority >= lowestPriority) {
    printf("Wrong priorites: Lowest %d highest %d\n", lowestPriority, highestPriority);
  }
  cudaStream_t streams[(lowestPriority - highestPriority + 1)];
  for (int i = highestPriority; i <= lowestPriority; i++) {
    CUDA_CHECK(cudaStreamCreateWithPriority(&streams[i - highestPriority], 0, i));
  }
  
  // Create and initialize attention tensors
  AttentionParams attnParams(model, batch, seqlen, hidden_dim, doChecking);
  attnParams.initIns();
  attnParams.initOuts();
  attnParams.initRefs();
  
  cudaError_t result;
  int epochs = 20;
  int warmup = 10;

  if (doChecking) {
    result = host_attention(attnParams);
    CUDA_CHECK(result);
  }
  
  double baselineTime = 0;
  double matmul1Time = 0;
  double matmul2Time = 0;
  double matmul3Time = 0;
  double matmul4Time = 0;

  if (true) {
    result = runAttentionBaseline(split_k1, split_k2, split_k3, split_k4, attnParams, streams, baselineTime, matmul1Time, matmul2Time, matmul3Time, matmul4Time, 1);
    CUDA_CHECK(cudaDeviceSynchronize());
    if (doChecking) {
      result = check_results(attnParams);
      CUDA_CHECK(result);
    }

    result = runAttentionBaseline(split_k1, split_k2, split_k3, split_k4, attnParams, streams, baselineTime, matmul1Time, matmul2Time, matmul3Time, matmul4Time, warmup);

    CUDA_CHECK(cudaDeviceSynchronize());
    matmul1Time = 0;
    matmul2Time = 0;
    matmul3Time = 0;
    matmul4Time = 0;
    printf("START-BASELINE:\n");
    result = runAttentionBaseline(split_k1, split_k2, split_k3, split_k4, attnParams, streams, baselineTime, matmul1Time, matmul2Time, matmul3Time, matmul4Time, epochs);

    CUDA_CHECK(result);
  
    printf("END-BASELINE: {\"Total\": %lf, \"matmul1Time\": %lf, \"matmul2Time\": %lf, \"matmul3Time\": %lf, \"matmul4Time\": %lf} microseconds\n", baselineTime/(float)epochs, matmul1Time/(float)epochs, matmul2Time/(float)epochs, matmul3Time/(float)epochs, matmul4Time/(float)epochs);
  }
  
  attnParams.initOuts();

  cutlass::gemm::GemmCoord tileSizeCoord1{TileSizeLinearLayers::ShapeThreadBlock::kM, TileSizeLinearLayers::ShapeThreadBlock::kN, 1};
  cutlass::gemm::GemmCoord tileSizeCoord2{TileSizeAttention::ShapeThreadBlock::kM, TileSizeAttention::ShapeThreadBlock::kN, 1};


  cutlass::gemm::GemmCoord gridDim1 = CuSyncGeMMSwizzle().get_tiled_shape(attnParams.gemm_size_xqkv, tileSizeCoord1, split_k1);
  cutlass::gemm::GemmCoord gridDim2 = CuSyncGeMMSwizzle().get_tiled_shape(attnParams.gemm_size_s,    tileSizeCoord2, split_k2);
  cutlass::gemm::GemmCoord gridDim3 = CuSyncGeMMSwizzle().get_tiled_shape(attnParams.gemm_size_o,    tileSizeCoord2, split_k3);
  cutlass::gemm::GemmCoord gridDim4 = CuSyncGeMMSwizzle().get_tiled_shape(attnParams.gemm_size_xw12, tileSizeCoord1, split_k4);

#ifdef ROWSYNC
  Sync1 sync1(gridDim1.n());
  Sync2 sync2(gridDim2.n());
  Sync1 sync3(gridDim3.n());
#elif defined(TILESYNC)
  Sync1 sync1(1,1);
  Sync2 sync2(1,1);
  Sync1 sync3(1,1);
#elif defined(STRIDEDSYNC)
  StridedSyncImpl sync1;
  Sync2 sync2(gridDim2.n());
  Sync1 sync3(gridDim3.n());
#else
  #error "Unknown Policy"
#endif
  
  XQKVCuStage xqkvStage(CuSyncGeMMSwizzle().get_grid_shape(gridDim1), {1,1,1}, NoSync(), sync1);
  SCuStage    sStage   (CuSyncGeMMSwizzle().get_grid_shape(gridDim2), {1,1,1}, sync1,    sync2);
  OCuStage    oStage   (CuSyncGeMMSwizzle().get_grid_shape(gridDim3), {1,1,1}, sync2,    sync3);
  XW12CuStage xw12Stage(CuSyncGeMMSwizzle().get_grid_shape(gridDim4), {1,1,1}, sync3,    NoSync());
  
  CuSync::setProducerConsumerPair(xqkvStage, sStage);
  CuSync::setProducerConsumerPair(sStage, oStage);
  CuSync::setProducerConsumerPair(oStage, xw12Stage);

  double overlapTime = 0;
  matmul1Time = 0;
  matmul2Time = 0;
  if (true) {
    result = runAttentionCuSync(split_k1, split_k2, split_k3, split_k4, attnParams, xqkvStage, sStage, oStage, xw12Stage, streams, overlapTime, 1);

    CUDA_CHECK(cudaDeviceSynchronize());
    if (doChecking) {
      result = check_results(attnParams);
      if (result != cudaSuccess) {
        return 1;
      }
    }
    // //warmup
    result = runAttentionCuSync(split_k1, split_k2, split_k3, split_k4, attnParams, xqkvStage, sStage, oStage, xw12Stage, streams, overlapTime, warmup);

    CUDA_CHECK(cudaDeviceSynchronize());
    printf("START-OVERLAPPED\n");
    result = runAttentionCuSync(split_k1, split_k2, split_k3, split_k4, attnParams, xqkvStage, sStage, oStage, xw12Stage, streams, overlapTime, epochs);
    
    printf("END-OVERLAPPED: {\"Total\": %lf} microseconds\n", overlapTime/(float)epochs);
  }

  return 0;
}
