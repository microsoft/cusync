/***************************************************************************************************
 * Copyright (c) 2017 - 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

//<OPTIMIZATIONS>
//</OPTIMIZATIONS>

// #if defined(TILESYNC)
// #define NO_ATOMIC_ADD
// #endif

// #if defined(TILESYNC) || defined(TILEBATCH) || defined(STRIDEDSYNC)
// #define AVOID_CUSTOM_ORDER
// #define REORDER_TILE_LOADS
// #define AVOID_WAIT_KERNEL
// #endif

#include<cuSync.h>

#include "common.h"

#ifndef EVAL_TILE_SIZES
//Tile sizes of all GeMMs
using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<256, 128, 32>;  
using ShapeMMAWarp = cutlass::gemm::GemmShape<128, 64, 32>;
const int SoftmaxRowTile = 1;
#else
//<eval tiles>
const int SoftmaxRowTile = 1;
using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<256, 128, 32>;  
using ShapeMMAWarp = cutlass::gemm::GemmShape<128, 64, 32>;
//</eval tiles>
#endif


template<uint H, uint Tile, uint stride>
struct StridedSync {
  uint waitValue_;
  uint postValue_;

  __device__ __host__ StridedSync(): waitValue_(stride), postValue_(1) {}
  
  __device__ __host__ uint waitValue(const dim3& tile, const dim3& grid) {
    return stride;
  }

  __device__ __host__ uint postValue(const dim3& tile, const dim3& grid) 
    {return 1;}

  __device__ constexpr uint tileIndex(const dim3& tile, const dim3& grid) {
    if (grid.y > ((H/8)/Tile))
      return tile.x * (grid.y/((H/8)/Tile)) + tile.y%((H/8)/Tile);
    else
    return tile.x * grid.y + tile.y;
  }

  __device__ bool isSync(const dim3& tile, const dim3& grid) {
    return true; //tile.y < (H/8)/ShapeMMAThreadBlock::kN;
  }
};

const int SoftmaxThreads = ShapeMMAThreadBlock::kN;
using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;  


#ifdef ROWSYNC 
  using ProdCuStage = CuStage<CuStageType::Producer, RowMajor, RowSync>;
  using MiddleCuStage = CuStage<CuStageType::Producer|CuStageType::Consumer, RowMajor, RowSync>;
  using ConsCuStage = CuStage<CuStageType::Consumer, RowMajor, RowSync>;
  using Sync = RowSync;
#elif defined(TILESYNC)
  using ProdCuStage = CuStage<CuStageType::Producer, RowMajor, TileSync<1>>;
  using MiddleCuStage = CuStage<CuStageType::Producer|CuStageType::Consumer, RowMajor, TileSync<1>>;
  using ConsCuStage = CuStage<CuStageType::Consumer, RowMajor, TileSync<1>>;
  using Sync = TileSync<1>;
#elif defined(STRIDEDSYNC)
  #if defined(GPT3)
    using StridedSyncImpl = StridedSync<12288, ShapeMMAThreadBlock::kN, 3>;
  #elif defined(LLaMA)
    using StridedSyncImpl = StridedSync<8192, ShapeMMAThreadBlock::kN, 3>;
  #else
    #error "GPT3 or LLaMA"
  #endif
  using ProdCuStage = CuStage<CuStageType::Producer, RowMajor, StridedSyncImpl>;
  using MiddleCuStage = CuStage<CuStageType::Producer|CuStageType::Consumer, RowMajor, StridedSyncImpl>;
  using ConsCuStage = CuStage<CuStageType::Consumer, RowMajor, TileSync<1>>;
  using Sync = TileSync<1>;
#else
  #error "Unknown Synchronization"
#endif 

using CuSyncImpl1 = CuSync<ProdCuStage, MiddleCuStage>;
using CuSyncImpl2 = CuSync<MiddleCuStage, ConsCuStage>;


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

template<bool splitK>
class BaseMLPGemm : public cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, 
                                                        ElementInputB, LayoutInputB,
                                                        ElementOutput, LayoutOutput,
                                                        ElementAccumulator, MMAOp,
                                                        SmArch, ShapeMMAThreadBlock,
                                                        ShapeMMAWarp, ShapeMMAOp,
                                                        EpilogueOp, 
                                                        cutlass::gemm::threadblock::GemmHorizontalThreadblockSwizzle, 
                                                        2, 8, 8, splitK> {};

// Baseline GeMMs
using Gemm1 = BaseMLPGemm<false>;
using Gemm2 = BaseMLPGemm<false>;

//Baseline GeMMs with SplitK enabled
using GemmSplitK1 = BaseMLPGemm<true>;
using GemmSplitK2 = BaseMLPGemm<true>;

//CuSync GeMMs
template<typename CuStage, bool splitK>
class CuSyncAttentionGemm : public cutlass::gemm::device::CuSyncGemm<CuStage, 
                                                        ElementInputA, LayoutInputA, 
                                                        ElementInputB, LayoutInputB,
                                                        ElementOutput, LayoutOutput,
                                                        ElementAccumulator, MMAOp,
                                                        SmArch, ShapeMMAThreadBlock,
                                                        ShapeMMAWarp, ShapeMMAOp,
                                                        EpilogueOp, 
                                                        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 
                                                        2, 8, 8, splitK> {};

using CuSyncGemm1 = CuSyncAttentionGemm<ProdCuStage, false>;
using CuSyncGemm2 = CuSyncAttentionGemm<ConsCuStage, false>;

using CuSyncGemmSplitK1 = CuSyncAttentionGemm<ProdCuStage, true>;
using CuSyncGemmSplitK2 = CuSyncAttentionGemm<ConsCuStage, true>;

using HostTensor = cutlass::HostTensor<ElementInputA, LayoutInputA>;

struct AttentionParams {
  HostTensor x;
  HostTensor qkv;
  HostTensor xqkv;
  HostTensor xdot;
  HostTensor w2;
  HostTensor xw12;

  HostTensor ref_xqkv;
  HostTensor ref_xdot;
  HostTensor ref_xw12;

  cutlass::gemm::GemmCoord gemm_size1, gemm_size2;
  curandState* randStates;
  bool refCheck;
  ElementComputeEpilogue alpha;
  ElementComputeEpilogue beta;

  AttentionParams(int problem[4], bool check) {
    gemm_size1 = cutlass::gemm::GemmCoord(problem[0], problem[1] * 3, problem[2]);
    gemm_size2 = cutlass::gemm::GemmCoord(problem[0], problem[3], problem[1]);
    alpha = ElementComputeEpilogue(1);
    beta = ElementComputeEpilogue(0);
  
    x    = HostTensor(gemm_size1.mk());
    qkv  = HostTensor(gemm_size1.kn());
    xqkv = HostTensor(gemm_size1.mn());
    xdot = HostTensor({gemm_size1.m(), gemm_size1.n()/3});
    w2   = HostTensor(gemm_size2.kn());
    xw12 = HostTensor(gemm_size2.mn());

    ref_xdot = HostTensor({gemm_size1.m(), gemm_size1.n()/3});
    ref_xqkv = HostTensor(gemm_size1.mn());
    ref_xw12 = HostTensor(gemm_size2.mn());

    size_t numRandStates = gemm_size1.m() * 1024;
    CUDA_CHECK(cudaMalloc(&randStates, sizeof(curandState)*(numRandStates)));
    init_curand_states<<<numRandStates/128, 128>>>(randStates, numRandStates);
    CUDA_CHECK(cudaDeviceSynchronize());
    refCheck = check;
  }

  void initIns() {
    if (refCheck) {
      memset_random2(x.host_data(), ElementOutput(0.02), 
                     ElementOutput(0.03), x.size());
      memset_random2(qkv.host_data(), ElementOutput(0.01), 
                     ElementOutput(0.035), qkv.size());
      memset_random2(w2.host_data(), ElementOutput(0.01),
                     ElementOutput(0.05), w2.size());
    } else {
      cutlass::reference::host::TensorFill(x.host_view(),
                                           ElementOutput(0.05));
      cutlass::reference::host::TensorFill(qkv.host_view(),
                                           ElementOutput(0.5));
      cutlass::reference::host::TensorFill(w2.host_view(),
                                           ElementOutput(0.01));
    }

    // Copy data from host to GPU
    x.sync_device();
    qkv.sync_device();
    w2.sync_device();
  }

  void initOuts() {
    //Zeros all output tensors
    cutlass::reference::host::TensorFill(xqkv.host_view());
    cutlass::reference::host::TensorFill(xw12.host_view());
    cutlass::reference::host::TensorFill(xdot.host_view());
  }

  void initRefs() {
    cutlass::reference::host::TensorFill(ref_xqkv.host_view());
    cutlass::reference::host::TensorFill(ref_xdot.host_view());
    cutlass::reference::host::TensorFill(ref_xw12.host_view());
  }
};

template<uint NTHREADS, typename T, typename AT, uint TileM, uint TileN, uint RowTile, bool enableOverlap>
__global__ void selfAttnDotProdSoftmaxDropout(uint32_t M, uint32_t N,
                                              T* XQKV, T* out, float p,
                                              curandState* randStates,
                                              MiddleCuStage cons1, MiddleCuStage prod2) {
  extern __shared__ half xqkRows[];

  __shared__ AT sum;
  if (enableOverlap)
    prod2.tile(nullptr);
  int linearThreadId = blockIdx.x * blockDim.x + threadIdx.x;
  curandState* localRandState = &randStates[linearThreadId];
  // __shared__ shRandStates[sizeof(curandState) * NTHREADS];
  uint ROW = blockIdx.x * RowTile;
  const uint tileRow = blockIdx.x;
  const uint tileM = ROW/TileM;
  if (enableOverlap) {
    // && tileM == 0) printf("TileM %d TileN %d ROW %d\n", TileM, TileN, ROW);
    // handle1.waitOnTilesWithSyncValue(tileM, 0, 0, 1);
    // if (tileM < M/TileM) {
    //   {tileM + 1, 0, 0};
    //   handle1.waitOnTile();
    // }
  }

  for (uint ti = 0; ti < RowTile && ROW < M; ti++) {
    if (threadIdx.x == 0) {
      sum = 0;
    }

    AT threadSum = (AT)0.0f;

    for (uint COL = threadIdx.x; COL < N; COL += blockDim.x) {
      if (enableOverlap) {
        if (ti == 0 && ROW % TileM == 0) {
          dim3 tile = {tileM, COL/TileN, 0};
          cons1.wait(tile, (COL/TileN)%NTHREADS);
        }
      }
      T xq = XQKV[ROW * 3 * N + COL];
      if (enableOverlap  && ti == 0 && ROW % TileM == 0) {
        dim3 tile = {tileM, N/TileN + COL/TileN, 0};
        #ifdef TILESYNC
        cons1.wait(tile, (COL/TileN)%NTHREADS);
        #endif
      }
      T xk = XQKV[ROW * 3 * N + (COL + N)];
      T xqk = xq * xk;
      threadSum += (AT)exp((AT)xqk);
      xqkRows[COL] = xqk;
    }
    __syncthreads();
    atomicAdd(&sum, (AT)threadSum);
    __syncthreads();
    for (uint COL = threadIdx.x; COL < N; COL += blockDim.x) {
      float r = curand_uniform(localRandState);
      // if (enableOverlap && ti == 0) {
      //   if (rowSyncOrTileSync) {

      //   } else {
      if (enableOverlap && ti == 0 && ROW % TileM == 0) {
        dim3 tile = {tileM, N/TileN*2 + COL/TileN, 0};
        #ifndef TILESYNC
        cons1.wait(tile, (COL/TileN)%NTHREADS);
        #endif
      }
      __half v = (r <= p) ? (__half)(((float)(exp((AT)xqkRows[COL]) * 
                                     (float)XQKV[ROW* 3 * N + (COL + 2 * N)]))/sum) : (__half)0.0f;
      out[ROW * N + COL] = v;
      if (enableOverlap && ti == SoftmaxRowTile - 1) {
        dim3 tile = {tileM, COL/TileN, 0};
        prod2.post(tile, ((COL/TileN)*TileN)%NTHREADS);
      }
    }
    __syncthreads();

    ROW++;
  }

  // if (enableOverlap) {
  //   if (rowSyncOrTileSync) {
  //     // tileM = ROW/TileM;
  //     handle2.setRowStatus(tileM, 0, 0, RowTile);
  //   } else {
      
  //   }
  // }
}

void attnRefMatmul(cutlass::gemm::GemmCoord size, ElementOutput* a, ElementOutput* b, ElementOutput* c) {
  ref_matmul<ElementOutput, ElementAccumulator>(size.m(), size.n(), 
                                                size.k(), a, b, c);
}

cudaError_t host_attention(AttentionParams& attnParams) {
  attnRefMatmul(attnParams.gemm_size1, attnParams.x.device_data(), 
                attnParams.qkv.device_data(), attnParams.ref_xqkv.host_data());
  
  size_t xq_size = attnParams.ref_xdot.size();
  assert(attnParams.ref_xdot.size() == attnParams.gemm_size1.m() * attnParams.gemm_size1.n()/3);
  size_t N = attnParams.gemm_size1.n()/3;
  ElementOutput* host_xqkv = attnParams.ref_xqkv.host_data();
  ElementOutput* host_xdot = attnParams.ref_xdot.host_data();

  for (size_t row = 0; row < attnParams.gemm_size1.m(); row++) {
    for (size_t col = 0; col < attnParams.gemm_size1.n()/3; col++) {
      ElementOutput xqk = host_xqkv[row * 3 * N + col] * host_xqkv[row * 3 * N + (col + N)];
      host_xdot[row * N + col] = xqk;
    }
  }

  for (size_t ROW = 0; ROW < attnParams.gemm_size1.m(); ROW++) {
    float sum = 0.0f;
    for (size_t COL = 0; COL < attnParams.gemm_size1.n()/3; COL++) {
      sum += exp((float)host_xdot[ROW*N + COL]);
    }
    
    for (size_t COL = 0; COL < attnParams.gemm_size1.n()/3; COL++) {
      //Assume dropout probability is 1.0
      host_xdot[ROW*N + COL] = (exp(host_xdot[ROW*N + COL]) * host_xqkv[ROW*3*N + COL+2*N])/sum;
    }
  }
  
  attnParams.ref_xdot.sync_device();

  attnRefMatmul(attnParams.gemm_size2, attnParams.ref_xdot.device_data(), 
                attnParams.w2.device_data(), attnParams.ref_xw12.host_data());
  
  return cudaSuccess;
}

cudaError_t check_results(AttentionParams& attnParams) {
  attnParams.xqkv.sync_host();
  printf("Checking First GeMM output\n");
  bool eq = equals(attnParams.ref_xqkv.size(), 
                   attnParams.ref_xqkv.host_data(), 
                   attnParams.xqkv.host_data(), 1e-1f);
  if (eq == false) {
    printf("First GeMM not correct\n");
    return cudaErrorUnknown;
  }
  attnParams.xdot.sync_host();
  printf("Checking Dot Dropout kernel\n");
  eq = equals(attnParams.ref_xdot.size(), attnParams.ref_xdot.host_data(),
              attnParams.xdot.host_data(), 1e-1f);
  if (eq == false) {
    printf("Dot not correct\n");
    return cudaErrorUnknown;
  }

  attnParams.xw12.sync_host();
  printf("Checking second GeMM\n");
  eq = equals(attnParams.ref_xw12.size(), attnParams.ref_xw12.host_data(), 
              attnParams.xw12.host_data(), 1e-1);
  if (eq == false) {
    printf("Second GeMM not correct\n");
    return cudaErrorUnknown;
  }
  printf("Self-Attention Passed\n");

  return cudaSuccess;
}

__global__ void print_kernel(ElementOutput* data) {
  if (threadIdx.x < 10) {
    printf("%p %f\n", data, (float)data[threadIdx.x]);
  }
}

//Run our baseline of Self-Attention
template<typename GemmTy1, typename GemmTy2>
cudaError_t runAttentionBaseline(int split_k1, int split_k2,
                                 AttentionParams& attnParams,
                                 cudaStream_t streams[],
                                 double& execTime,
                                 double& matmul1Time,
                                 double& softmaxTime,
                                 double& matmul2Time,
                                 int iters = 100) {  
  // ElementOutput* device_xqkv = tensor_xqkv.device_data();
  cutlass::Status status;

  //Setup First GeMM
  typename GemmTy1::Arguments args1{attnParams.gemm_size1,
                                    attnParams.x.device_ref(),
                                    attnParams.qkv.device_ref(),
                                    attnParams.xqkv.device_ref(),
                                    attnParams.xqkv.device_ref(),
                                    {attnParams.alpha, attnParams.beta},
                                    split_k1};
  size_t workspace_size = GemmTy1::get_workspace_size(args1);
  cutlass::device_memory::allocation<uint8_t> workspace1(workspace_size);
  GemmTy1 gemm_op1;
  status = gemm_op1.can_implement(args1);
  CUTLASS_CHECK(status);
  status = gemm_op1.initialize(args1, workspace1.get());
  CUTLASS_CHECK(status);

  //Setup Second GeMM
  typename GemmTy2::Arguments args2{attnParams.gemm_size2,
                                    attnParams.xdot.device_ref(),
                                    attnParams.w2.device_ref(),
                                    attnParams.xw12.device_ref(),
                                    attnParams.xw12.device_ref(),
                                    {attnParams.alpha, attnParams.beta},
                                    split_k2};
  workspace_size = GemmTy2::get_workspace_size(args2);
  cutlass::device_memory::allocation<uint8_t> workspace2(workspace_size);
  GemmTy2 gemm_op2;
  status = gemm_op2.can_implement(args2);
  CUTLASS_CHECK(status);
  status = gemm_op2.initialize(args2, workspace2.get());
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
    
    selfAttnDotProdSoftmaxDropout<SoftmaxThreads, half, float, 
                                  ShapeMMAThreadBlock::kM, ShapeMMAThreadBlock::kN,
                                  SoftmaxRowTile, false>
                                  <<<DIVUP(attnParams.gemm_size1.m(), SoftmaxRowTile), 
                                    SoftmaxThreads, 
                                    attnParams.gemm_size1.n()/3 * sizeof(half), 
                                    streams[0]>>>
                                    (attnParams.gemm_size1.m(), 
                                    attnParams.gemm_size1.n()/3, 
                                    (half*)attnParams.xqkv.device_data(),
                                    (half*)attnParams.xdot.device_data(), 
                                    1.0f, attnParams.randStates,
                                    MiddleCuStage(), MiddleCuStage());
    CUDA_CHECK(cudaStreamSynchronize(streams[0]));
    double middle2 = timeInMicroSeconds();
    double iterSoftmax = middle2-middle1;
    softmaxTime += iterSoftmax;
    status = gemm_op2(streams[0]);
    CUTLASS_CHECK(status);
    CUDA_CHECK(cudaDeviceSynchronize());
    double middle3 = timeInMicroSeconds();
    double iterMatmul2 = middle3-middle2;
    matmul2Time += iterMatmul2;
    double end = timeInMicroSeconds();
    if (iters > 10)
      printf("{\"Total\": %lf, \"matmul1Time\": %lf, \"softmaxTime\": %lf, \"matmul2Time\": %lf}\n",
             end-start, iterMatMul1, iterSoftmax, iterMatmul2);
    execTime += end-start;
  }

  return cudaSuccess;
}

template<typename GemmTy1, typename GemmTy2, typename GemmSplitKTy1, typename GemmSplitKTy2>
cudaError_t runAttentionBaseline(int split_k1, int split_k2,
                                 AttentionParams& attnParams, 
                                 cudaStream_t streams[],
                                 double& execTime,
                                 double& matmul1Time,
                                 double& softmaxTime,
                                 double& matmul2Time,
                                 int iters = 100) {
  cudaError_t result;
  if (split_k1 == 1 && split_k2 == 1) {
    result = runAttentionBaseline<GemmTy1, GemmTy2>(split_k1, split_k2, attnParams, streams, execTime, matmul1Time, softmaxTime, matmul2Time, iters);
  } else if (split_k1 > 1 && split_k2 == 1) {
    result = runAttentionBaseline<GemmSplitKTy1, GemmTy1>(split_k1, split_k2, attnParams, streams, execTime, matmul1Time, softmaxTime, matmul2Time, iters);
  } else if (split_k1 == 1 && split_k2 > 1) {
    result = runAttentionBaseline<GemmTy1, GemmSplitKTy2>(split_k1, split_k2, attnParams, streams, execTime, matmul1Time, softmaxTime, matmul2Time, iters);
  } else if (split_k1 > 1 && split_k2 > 1) {
    result = runAttentionBaseline<GemmSplitKTy1, GemmSplitKTy2>(split_k1, split_k2, attnParams, streams, execTime, matmul1Time, softmaxTime, matmul2Time, iters);
  }

  return result;
}

//Self-Attention using CuSync
template<typename GemmTy1, typename GemmTy2>
cudaError_t runAttentionCuSync(int split_k1, int split_k2, 
                               AttentionParams& attnParams, 
                               CuSyncImpl1& handle1,
                               CuSyncImpl2& handle2,
                               cudaStream_t streams[],
                               double& execTime,
                               int iters = 100) {  
  //Setup first GeMM
  typename GemmTy1::Arguments args1{handle1.prod(),
                                    attnParams.gemm_size1,
                                    attnParams.x   .device_ref(),
                                    attnParams.qkv .device_ref(),
                                    attnParams.xqkv.device_ref(),
                                    attnParams.xqkv.device_ref(),
                                    {attnParams.alpha, attnParams.beta},
                                    split_k1};
  size_t workspace_size = GemmTy1::get_workspace_size(args1);
  cutlass::device_memory::allocation<uint8_t> workspace1(workspace_size);
  GemmTy1 gemm_op1;
  cutlass::Status status;
  status = gemm_op1.can_implement(args1);
  CUTLASS_CHECK(status);
  status = gemm_op1.initialize(args1, workspace1.get());
  CUTLASS_CHECK(status);

  //Setup second GeMM
  typename GemmTy2::Arguments args2{handle2.cons(),
                                    attnParams.gemm_size2,
                                    attnParams.xdot.device_ref(),
                                    attnParams.w2  .device_ref(),
                                    attnParams.xw12.device_ref(),
                                    attnParams.xw12.device_ref(),
                                    {attnParams.alpha, attnParams.beta},
                                    split_k2};
  workspace_size = GemmTy2::get_workspace_size(args2);
  cutlass::device_memory::allocation<uint8_t> workspace2(workspace_size);
  GemmTy2 gemm_op2;
  status = gemm_op2.can_implement(args2);
  CUTLASS_CHECK(status);
  status = gemm_op2.initialize(args2, workspace2.get());
  CUTLASS_CHECK(status);
  
  execTime = 0;
  
  //Run Kernels in Self-Attention
  for (int r = 0; r < iters; r++) {
    handle1.prod().iter += 1;
    handle1.cons().iter += 1;
    handle2.prod().iter += 1;
    handle2.cons().iter += 1;
    gemm_op1.params_.custage.iter += 1;
    gemm_op2.params_.custage.iter += 1;

    double start = timeInMicroSeconds();
    status = gemm_op1.run(true, NULL, streams[0]);
    CUTLASS_CHECK(status);
    
#ifndef AVOID_WAIT_KERNEL
    handle1.invokeWaitKernel(streams[1]);
#endif
    selfAttnDotProdSoftmaxDropout<SoftmaxThreads, half, float, 
                                  ShapeMMAThreadBlock::kM, ShapeMMAThreadBlock::kN, 
                                  SoftmaxRowTile, true>
                                <<<DIVUP(attnParams.gemm_size1.m(), SoftmaxRowTile), 
                                   SoftmaxThreads, attnParams.gemm_size1.n()/3 * sizeof(half), 
                                   streams[1]>>>
                                (attnParams.gemm_size1.m(), 
                                 attnParams.gemm_size1.n()/3,
                                 (half*)attnParams.xqkv.device_data(),
                                 (half*)attnParams.xdot.device_data(),
                                 1.0f,
                                 attnParams.randStates,
                                 handle1.cons(), handle2.prod());

                                 // CUDA_CHECK(cudaDeviceSynchronize());
#ifndef AVOID_WAIT_KERNEL    
    handle2.invokeWaitKernel(streams[2]);
#endif
    
    status = gemm_op2.run(true, NULL, streams[2]);
    CUTLASS_CHECK(status);

    if (status != cutlass::Status::kSuccess) {
      return cudaErrorUnknown;
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    double end = timeInMicroSeconds();

    if (iters > 10)
      printf("{\"Total\": %lf}\n",end-start);
    execTime += end-start;
  }

  return cudaSuccess;
}

template<typename GemmTy1, typename GemmTy2, typename GemmSplitKTy1, typename GemmSplitKTy2>
cudaError_t runAttentionCuSync(int split_k1, int split_k2,
                               AttentionParams& attnParams,
                               CuSyncImpl1& handle1,
                               CuSyncImpl2& handle2,
                               cudaStream_t streams[],
                               double& execTime,
                               int iters = 100) {
  cudaError_t result;
  if (split_k1 == 1 && split_k2 == 1) {
    result = runAttentionCuSync<GemmTy1, GemmTy2>(split_k1, split_k2, attnParams, 
                                                  handle1, handle2, streams, execTime, 
                                                  iters);
  } else if (split_k1 == 1 && split_k2 > 1) {
    result = runAttentionCuSync<GemmTy1, GemmSplitKTy2>(split_k1, split_k2, attnParams, 
                                                        handle1, handle2, streams, execTime, 
                                                        iters);
  } else if (split_k1 > 1 && split_k2 == 1) {
    result = runAttentionCuSync<GemmSplitKTy1, GemmTy2>(split_k1, split_k2, attnParams, 
                                                        handle1, handle2, streams, execTime, 
                                                        iters);
  } else if (split_k1 > 1 && split_k2 > 1) {
    result = runAttentionCuSync<GemmSplitKTy1, GemmSplitKTy2>(split_k1, split_k2, attnParams, 
                                                              handle1, handle2, streams, 
                                                              execTime, iters);
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
  const uint NUM_ARGS = 5;
  std::string argNames[NUM_ARGS] = {"--model", "--batch", "--check", "--split-k1", "--split-k2"};
  std::string argHelp[NUM_ARGS] = {"GPT-3 or LLaMa", "Batch size", "Check results", 
                                   "Split K for first GeMM", "Split K for second GeMM"};
  
  if (argc < NUM_ARGS+1) {
    std::cout << "usage: " << std::endl
              << argNames[0] << " gpt3|llama " << argHelp[0] << std::endl 
              << argNames[1] << " <int>" << argHelp[1] << std::endl
              << argNames[2] << " true|false" << argHelp[2] << std::endl
              << argNames[3] << " <int> " << argHelp[3] << std::endl
              << argNames[4] << " <int> " << argHelp[4] << std::endl;
    return 0;
  }

  std::string model = "";
  uint batch = 0;
  bool doChecking = false;
  uint split_k1 = 1;
  uint split_k2 = 1;

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
      split_k1 = atoi(argv[i+1]);
      i=i+1;
    } else if (arg.find(argNames[4]) == 0) {
      split_k2 = atoi(argv[i+1]);
      i=i+1;
    }
  }

  if (model == "" || batch == 0) {
    std::cout<<"invalid model or batch" <<std::endl;
    return 0;
  }
    
  std::cout << "model=" << model << " batch=" << batch << "check="<<doChecking <<std::endl;
  int problem[4] = {0,0,0,0};
  problem[0] = batch;
  
  if (model=="gpt3") {
    problem[1] = 12288/8;
    problem[2] = 12288;
    problem[3] = 12288;
  } else if (model=="llama") {
    problem[1] = 8192/8;
    problem[2] = 8192;
    problem[3] = 8192;
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
  AttentionParams attnParams(problem, doChecking);
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
  double softmaxTime = 0;
  double matmul2Time = 0;
  #define ENABLE_NORMAL_GEMM

  if (true) {
    result = runAttentionBaseline<Gemm1, Gemm2, GemmSplitK1, GemmSplitK2>(split_k1, split_k2, attnParams, streams, baselineTime, matmul1Time, softmaxTime, matmul2Time, 1);

    CUDA_CHECK(cudaDeviceSynchronize());
    if (doChecking) {
      result = check_results(attnParams);
      CUDA_CHECK(result);
    }

    result = runAttentionBaseline<Gemm1, Gemm2, GemmSplitK1, GemmSplitK2>(split_k1, split_k2, attnParams, streams, baselineTime, matmul1Time, softmaxTime, matmul2Time, warmup);

    CUDA_CHECK(cudaDeviceSynchronize());
    matmul1Time = 0;
    softmaxTime = 0;
    matmul2Time = 0;
    printf("START-BASELINE:\n");
    result = runAttentionBaseline<Gemm1, Gemm2, GemmSplitK1, GemmSplitK2>(split_k1, split_k2, attnParams, streams, baselineTime, matmul1Time, softmaxTime, matmul2Time, epochs);

    CUDA_CHECK(result);
  
    printf("END-BASELINE: {\"Total\": %lf, \"matmul1Time\": %lf, \"softmaxTime\": %lf, \"matmul2Time\": %lf} microseconds\n", baselineTime/(float)epochs, matmul1Time/(float)epochs, softmaxTime/(float)epochs, matmul2Time/(float)epochs);
  }
  
  attnParams.initOuts();

  dim3 gridDim1 = {(uint)DIVUP(attnParams.gemm_size1.m(), ShapeMMAThreadBlock::kM),
                   (uint)DIVUP(attnParams.gemm_size1.n(), ShapeMMAThreadBlock::kN),
                   split_k1};
  dim3 gridDim2 = {(uint)DIVUP(attnParams.gemm_size1.m(), SoftmaxRowTile), 1, 1};
  dim3 gridDim3 = {(uint)DIVUP(attnParams.gemm_size2.m(), ShapeMMAThreadBlock::kM), 
                   (uint)DIVUP(attnParams.gemm_size2.n(), ShapeMMAThreadBlock::kN),
                   split_k2};
  dim3 tileSize = {ShapeMMAThreadBlock::kM, ShapeMMAThreadBlock::kN, 1};
  
#ifdef ROWSYNC
  using Sync1 = RowSync;
  RowSync sync1(gridDim1.y);
  using Sync2 = RowSync;
  Sync2 sync2(min(ShapeMMAThreadBlock::kM, attnParams.gemm_size1.m()), SoftmaxRowTile); 
#elif defined(TILESYNC)
  using Sync1 = TileSync<1>;
  using Sync2 = Sync1;
  TileSync<1> sync1;
  uint waitValue = DIVUP(min(attnParams.gemm_size1.m(), ShapeMMAThreadBlock::kM), SoftmaxRowTile);
  TileSync<1> sync2(waitValue, 1);
#elif defined(STRIDEDSYNC)
    StridedSyncImpl sync1;
    uint waitValue = DIVUP(min(attnParams.gemm_size1.m(), ShapeMMAThreadBlock::kM), SoftmaxRowTile);
    TileSync<1> sync2(waitValue, 1);
#else
  #error "Unknown Policy"
#endif

  ProdCuStage prod1(gridDim1, tileSize, sync1);
  MiddleCuStage cons1(gridDim2, {SoftmaxRowTile, 1, 1}, sync1);
  ConsCuStage cons2(gridDim3, tileSize, sync2);

  CuSyncImpl1 handle1(prod1, cons1);
  CuSyncImpl2 handle2(cons1, cons2);
  handle2.iter = 0;
  handle1.iter = 0;
  handle1.prod().iter = handle1.cons().iter = 0;
  handle2.prod().iter = handle2.cons().iter = 0;
  
  double overlapTime = 0;
  matmul1Time = 0;
  softmaxTime = 0;
  matmul2Time = 0;
  if (true) {
    result = runAttentionCuSync<CuSyncGemm1, CuSyncGemm2, CuSyncGemmSplitK1, CuSyncGemmSplitK2>(split_k1, split_k2, attnParams, handle1, handle2, streams, overlapTime, 1);

    CUDA_CHECK(cudaDeviceSynchronize());
    if (doChecking) {
      result = check_results(attnParams);
      if (result != cudaSuccess) {
        return 1;
      }
    }
    //warmup
    result = runAttentionCuSync<CuSyncGemm1, CuSyncGemm2, CuSyncGemmSplitK1, CuSyncGemmSplitK2>(split_k1, split_k2, attnParams, handle1, handle2, streams, overlapTime, warmup);

    CUDA_CHECK(cudaDeviceSynchronize());
    printf("START-OVERLAPPED\n");
    result = runAttentionCuSync<CuSyncGemm1, CuSyncGemm2, CuSyncGemmSplitK1, CuSyncGemmSplitK2>(split_k1, split_k2, attnParams, handle1, handle2, streams, overlapTime, epochs);
    
    printf("END-OVERLAPPED: {\"Total\": %lf} microseconds\n", overlapTime/(float)epochs);
  }

  return 0;
}
