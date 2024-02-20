// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <iostream>
#include <sstream>
#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/cusyncgemm.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"
#include "helper.h"
#include <curand_kernel.h>
#include "cutlass/tensor_ref.h"
#include "cutlass/tensor_coord.h"
#include "cutlass/epilogue/thread/linear_combination_silu.h"

#include <time.h>
#include <sys/time.h>

#include<cuSync.h>
#define DIVUP(x, y) (((x) + (y) - 1)/(y))

#if defined(TILESYNC)
#define NO_ATOMIC_ADD
#define REORDER_TILE_LOADS
#endif

#define AVOID_WAIT_KERNEL
#define AVOID_CUSTOM_ORDER
#define REORDER_TILE_LOADS

/*Notes:
  TileSync with MiddleCuStage being a producer works best because
  it only needs to synchronize on XW 
*/

#ifdef ROWSYNC
  using ProdCuStage = CuStage<CuStageType::Producer, RowMajor, RowSync>;
  using MiddleCuStage = CuStage<CuStageType::Producer | CuStageType::LLaMAMiddle, RowMajor, RowSync>;
  using ConsCuStage = CuStage<CuStageType::Consumer, RowMajor, RowSync>;
  using Sync = RowSync;
#elif defined(TILEBATCH)
  using ProdCuStage = CuStage<CuStageType::Producer, RowMajor, TileSync<2>>;
  using MiddleCuStage = CuStage<CuStageType::Producer | CuStageType::Consumer, RowMajor, TileSync<2>>;
  using ConsCuStage = CuStage<CuStageType::Consumer, RowMajor, TileSync<2>>;
  using Sync = TileSync<2>;
#elif defined(TILESYNC)
  using ProdCuStage = CuStage<CuStageType::Producer, RowMajor, TileSync<1>>;
  using MiddleCuStage = CuStage<CuStageType::Producer | CuStageType::LLaMAMiddle, RowMajor, TileSync<1>>;
  using ConsCuStage = CuStage<CuStageType::Consumer, RowMajor, TileSync<1>>;
  using Sync = TileSync<1>;
#elif defined(BATCHEDROW)
  using ProdCuStage = CuStage<CuStageType::Producer, RowMajor, BatchedRowSync>;
  using ConsCuStage = CuStage<CuStageType::Consumer, RowMajor, BatchedRowSync>;
  using Sync = BatchedRowSync;
#else
  #error "Unknown Synchronization"
#endif

static double convertTimeValToDouble(struct timeval _time) {
  return ((double)_time.tv_sec)*1e6 + ((double)_time.tv_usec);
}

static struct timeval getTimeOfDay () {
  struct timeval _time;

  if (gettimeofday (&_time, NULL) == -1) {
    fprintf (stderr, "gettimeofday returned -1\n");
    perror ("");
    abort ();
  }

  return _time;
}

static double timeInMicroSeconds() {
  return convertTimeValToDouble(getTimeOfDay());
}

static double getCurrentTime() {
  return timeInMicroSeconds();
}

// #include "common.h"

#ifndef EVAL_TILE_SIZES
//Tile sizes of all GeMMs
using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<32, 256, 32>;
using ShapeMMAWarp = cutlass::gemm::GemmShape<32, 64, 32>;
#else
//<eval tiles>
using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 32>;  
using ShapeMMAWarp = cutlass::gemm::GemmShape<128, 64, 32>;
//</eval tiles>
#endif

using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;

//Element types of A, B, and C
using ElementAccumulator = float;
using ElementInputA = cutlass::half_t;
using ElementInputB = cutlass::half_t;
using ElementOutput = cutlass::half_t;
using ElementComputeEpilogue = cutlass::half_t;

//All matrices are in RowMajor
using LayoutInputA = cutlass::layout::RowMajor;
// using LayoutInputB = cutlass::layout::RowMajor;
// using LayoutOutput = cutlass::layout::RowMajor;

//Use FP-16 Tensor Cores
using MMAOp = cutlass::arch::OpClassTensorOp;

using SmArch = cutlass::arch::Sm70;

//First GeMM in MLP is fused with GELU
using EpilogueOp1 = cutlass::epilogue::thread::LinearCombinationSilu<
    ElementOutput,                                        
    128 / cutlass::sizeof_bits<ElementOutput>::value,
    ElementAccumulator,
    ElementComputeEpilogue>;

using EpilogueOp2 = cutlass::epilogue::thread::LinearCombinationSwiGLU<
    ElementOutput,                                        
    128 / cutlass::sizeof_bits<ElementOutput>::value,     
    ElementAccumulator,
    ElementComputeEpilogue,
    cutlass::epilogue::thread::ScaleType::SwishScaling>;

//Third GeMM in MLP performs no extra fused computations 
using EpilogueOp3 = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,                                        
    128 / cutlass::sizeof_bits<ElementOutput>::value,     
    ElementAccumulator,
    ElementComputeEpilogue>;

cudaStream_t CudaStreams[128] = {NULL};

template<typename EpilogueOp, int AlignmentB, bool splitK>
class BaseMLPGemm : public cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, 
                                                       ElementInputA, LayoutInputA,
                                                       ElementOutput, LayoutInputA,
                                                        ElementAccumulator, MMAOp,
                                                        SmArch, ShapeMMAThreadBlock,
                                                        ShapeMMAWarp, ShapeMMAOp,
                                                        EpilogueOp, 
                                                        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
                                                        2, 8, AlignmentB, splitK> {};
// Baseline GeMMs
using Gemm1 = BaseMLPGemm<EpilogueOp1, 8, false>;
using Gemm2 = BaseMLPGemm<EpilogueOp2, 8, false>;
using Gemm3 = BaseMLPGemm<EpilogueOp3, 8, false>;

//Baseline GeMMs with SplitK enabled
using GemmSplitK1 = BaseMLPGemm<EpilogueOp1, 8, true>;
using GemmSplitK2 = BaseMLPGemm<EpilogueOp2, 8, true>;
using GemmSplitK3 = BaseMLPGemm<EpilogueOp3, 8, true>;

//CuSync GeMMs
using CuSyncImpl1 = CuSync<ProdCuStage, MiddleCuStage>;
using CuSyncImpl2 = CuSync<MiddleCuStage, ConsCuStage>;

template<typename CuStage, typename EpilogueOp, int AlignmentB, bool splitK>
class CuSyncMLPGemm : public cutlass::gemm::device::CuSyncGemm<CuStage, ElementInputA, LayoutInputA, 
                                                       ElementInputA, LayoutInputA,
                                                       ElementOutput, LayoutInputA,
                                                        ElementAccumulator, MMAOp,
                                                        SmArch, ShapeMMAThreadBlock,
                                                        ShapeMMAWarp, ShapeMMAOp,
                                                        EpilogueOp, 
                                                        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 
                                                        2, 8, AlignmentB, splitK> {};

using CuSyncGemm1 = CuSyncMLPGemm<ProdCuStage, EpilogueOp1, 8, false>;
using CuSyncGemm2 = CuSyncMLPGemm<MiddleCuStage, EpilogueOp2, 8, false>;
using CuSyncGemm3 = CuSyncMLPGemm<ConsCuStage, EpilogueOp3, 8, false>;

using CuSyncGemmSplitK1 = CuSyncMLPGemm<ProdCuStage, EpilogueOp1, 8, true>;
using CuSyncGemmSplitK2 = CuSyncMLPGemm<MiddleCuStage, EpilogueOp2, 8, true>;
using CuSyncGemmSplitK3 = CuSyncMLPGemm<ConsCuStage, EpilogueOp3, 8, true>;

using HostTensor = cutlass::HostTensor<ElementInputA, LayoutInputA>;
using TensorRef = cutlass::TensorRef<ElementInputA, LayoutInputA>;
using TensorRefC = cutlass::TensorRef<ElementOutput, LayoutInputA>;

enum MLPType {
  GPT3,
  LLaMa    
};

const int H = 8192;
const int multiple_of = 128;
const int d = ((H/3 + multiple_of-1)/multiple_of)*multiple_of;

template<typename GemmTy1, typename GemmTy2, typename GemmTy3>
struct MLPParameters {
  TensorRef x; //[B, H]
  TensorRef w1; //[H, 4H/8] in GPT-3 and [H, H/3] in LLaMa
  //xw1 = GeLU(x * w1)
  TensorRef xw1; //[B, 4 H / 8]
  TensorRef w2; //[4H/8, H] in GPT-3 and [H/3, H] in LLaMa
  //xw12 = xw1 * w2
  TensorRef xw12; //[B, H]

  //For LLaMa only
  TensorRef v; //[H, H/3] in LLaMa
  TensorRef xv; //[B, H/3] in LLaMa
  
  TensorRef ref_xw1;
  TensorRef ref_xw12;

  //For LLaMa only
  TensorRef ref_xv;

  bool checkResults;

  cutlass::gemm::GemmCoord gemm_size1;
  cutlass::gemm::GemmCoord gemm_size2;
  ElementComputeEpilogue alpha;
  ElementComputeEpilogue beta;

  std::string model;

  GemmTy1 gemm1;
  GemmTy2 gemm2;
  GemmTy3 gemm3;

  cutlass::device_memory::allocation<uint8_t> workspace1;
  cutlass::device_memory::allocation<uint8_t> workspace2;
  cutlass::device_memory::allocation<uint8_t> workspace3;

  MLPParameters() {
    
  }

  MLPParameters(std::string model_, uint batch,  
                const ElementInputA* w1Ptr, const ElementInputA* vPtr, 
                const ElementInputA* w2Ptr) {
    alpha = ElementComputeEpilogue(1.0);
    beta = ElementComputeEpilogue(0.0);
    model = model_;
    gemm_size1 = cutlass::gemm::GemmCoord(batch, d, H);
    gemm_size2 = cutlass::gemm::GemmCoord(batch, H, d);
  //  std::cout << "GeMM 1 Size: " << gemm_size1.m() << ", " << 
  //   gemm_size1.n() << ", " << gemm_size1.k() << std::endl;
  //  std::cout << "GeMM 2 Size: " << gemm_size2.m() << ", " << 
  //    gemm_size2.n() << ", " << gemm_size2.k() << std::endl;
    // auto extent_ = LayoutInputA::TensorCoord();
    x = TensorRef();
    w1 = TensorRef((ElementInputA*)w1Ptr, LayoutInputA::packed(gemm_size1.kn()));
    w2 = TensorRef((ElementInputA*)w2Ptr, LayoutInputA::packed(gemm_size2.kn()));
    v = TensorRef((ElementInputA*)vPtr, LayoutInputA::packed(gemm_size1.kn()));
    checkResults = false;
  }

  void initGeMMs(typename GemmTy1::Arguments& argsXW1,
                 typename GemmTy2::Arguments& argsXV,
                 typename GemmTy3::Arguments& argsXW12)
  {
    size_t workspace_size = GemmTy1::get_workspace_size(argsXW1);
    workspace1 = cutlass::device_memory::allocation<uint8_t>(workspace_size);
    cutlass::Status status = gemm1.can_implement(argsXW1);
    CUTLASS_CHECK(status);
    status = gemm1.initialize(argsXW1, workspace1.get());
    CUTLASS_CHECK(status);

    workspace_size = GemmTy2::get_workspace_size(argsXV);
    workspace2 = cutlass::device_memory::allocation<uint8_t>(workspace_size);
    status = gemm2.can_implement(argsXV);
    CUTLASS_CHECK(status);
    status = gemm2.initialize(argsXV, workspace2.get());
    CUTLASS_CHECK(status);
    
    workspace_size = GemmTy3::get_workspace_size(argsXW12);
    workspace3 = cutlass::device_memory::allocation<uint8_t>(workspace_size);
    status = gemm3.can_implement(argsXW12);
    CUTLASS_CHECK(status);
    status = gemm3.initialize(argsXW12, workspace3.get());
    CUTLASS_CHECK(status);
  }

  void setInput(ElementInputA* xPtr) {
    this->x = TensorRef(xPtr, LayoutInputA::packed(gemm_size1.mk()));
    gemm1.updateA(this->x);
  }

  void setIntermediate(ElementInputA* silu, ElementInputA* xv) {
    this->xv = TensorRef(xv, LayoutInputA::packed(gemm_size1.mn()));
    this->xw1 = TensorRef(silu, LayoutInputA::packed(gemm_size1.mn()));

    gemm1.updateC(this->xw1);
    gemm1.updateD(this->xw1);

    gemm2.updateA(this->x);
    gemm2.updateC(this->xw1);
    gemm2.updateD(this->xv);

    gemm3.updateA(this->xv);
  }

  void setOutput(ElementInputA* out) {
    this->xw12 = TensorRef(out, LayoutInputA::packed(gemm_size2.mn()));

    gemm3.updateC(this->xw12);
    gemm3.updateD(this->xw12);
  }

  bool isGPT3() {return model == "gpt3";}
  bool isLLaMa() {return model == "llama";}
};


template<typename GemmTy1, typename GemmTy2, typename GemmTy3>
struct BaselineMLPParams : public MLPParameters<GemmTy1, GemmTy2, GemmTy3> {
  typename GemmTy1::Arguments argsXW1;
  typename GemmTy2::Arguments argsXV;
  typename GemmTy3::Arguments argsXW12;

  using Parent = MLPParameters<GemmTy1, GemmTy2, GemmTy3>;
  BaselineMLPParams(std::string model_, int split_k1, int split_k2, uint batch, const ElementInputA* w1Ptr, const ElementInputA* vPtr, const ElementInputA* w2Ptr) : 
    MLPParameters<GemmTy1, GemmTy2, GemmTy3>(model_, batch, w1Ptr, vPtr, w2Ptr)
  {
    //Setup XW1 GeMM
    argsXW1 = typename GemmTy1::Arguments {Parent::gemm_size1,
                                           Parent::x, Parent::w1,
                                           Parent::xw1, Parent::xw1,
                                           {Parent::alpha, Parent::beta},
                                           split_k1};

    //Setup XV GeMM
    argsXV = typename GemmTy2::Arguments {Parent::gemm_size1,
                                          Parent::x, Parent::v,
                                          Parent::xw1, Parent::xv,
                                          {Parent::alpha, ElementComputeEpilogue(1.0f)},
                                          split_k1};
    //Setup XW12 GeMM
    argsXW12 = typename GemmTy3::Arguments {Parent::gemm_size2, 
                                            Parent::xv, Parent::w2, 
                                            Parent::xw12, Parent::xw12, 
                                            {Parent::alpha, Parent::beta},         
                                            split_k2};
    
    Parent::initGeMMs(argsXW1, argsXV, argsXW12);
    
    // CUDA_CHECK(cudaMemset(Parent::gemm1.params_.semaphore, 0, 4));
  }
};

template<typename GemmTy1, typename GemmTy2, typename GemmTy3>
struct CuSyncMLPParameters  : public MLPParameters<GemmTy1, GemmTy2, GemmTy3> {
  using Parent = MLPParameters<GemmTy1, GemmTy2, GemmTy3>;

  typename GemmTy1::Arguments argsXW1;
  typename GemmTy2::Arguments argsXV;
  typename GemmTy3::Arguments argsXW12;

  ProdCuStage prodStage;
  MiddleCuStage middleStage;
  ConsCuStage consStage;
  CuSyncImpl1 cuSyncHandle1;
  CuSyncImpl2 cuSyncHandle2;
  CuSyncMLPParameters(std::string model_, int split_k1, int split_k2, uint batch, const ElementInputA* w1Ptr, const ElementInputA* vPtr, const ElementInputA* w2Ptr) : 
  MLPParameters<GemmTy1, GemmTy2, GemmTy3>(model_, batch, w1Ptr, vPtr, w2Ptr) {
    //Setup CuSync stages
    auto& gemm_size1 = Parent::gemm_size1;
    auto& gemm_size2 = Parent::gemm_size2;
    dim3 gridDim1 = {(uint)DIVUP(gemm_size1.m(), ShapeMMAThreadBlock::kM), 
                     (uint)DIVUP(gemm_size1.n(), ShapeMMAThreadBlock::kN), 
                     split_k1};
    dim3 gridDim2 = {(uint)DIVUP(gemm_size2.m(), ShapeMMAThreadBlock::kM), 
                     (uint)DIVUP(gemm_size2.n(), ShapeMMAThreadBlock::kN), 
                     split_k2};
    dim3 tileSize = {ShapeMMAThreadBlock::kM, ShapeMMAThreadBlock::kN, 1};

    #if defined(ROWSYNC)
      using Sync = RowSync;
      RowSync sync((uint)DIVUP(::d, ShapeMMAThreadBlock::kN));
    #elif defined(TILEBATCH)
      using Sync = TileSync<2>;
      Sync sync;
    #elif defined(TILESYNC)
      using Sync = TileSync<1>;
      Sync sync;
    #elif defined(BATCHEDROW)
      using Sync = BatchedRowSync;
      BatchedRowSync sync(gridDim1.y, 1);
    #else
      #error "Unknown Policy"
    #endif

    prodStage = ProdCuStage(gridDim1, tileSize, sync);
    middleStage = MiddleCuStage(gridDim1, tileSize, sync);
    consStage = ConsCuStage(gridDim2, tileSize, sync);

    prodStage.iter = consStage.iter = middleStage.iter = 0;

    cuSyncHandle1 = CuSyncImpl1(prodStage, middleStage);
    cuSyncHandle2 = CuSyncImpl2(middleStage, consStage);

    cuSyncHandle1.iter = 0;
    cuSyncHandle1.prod().iter = cuSyncHandle1.cons().iter = 0;
    cuSyncHandle2.iter = 0;
    cuSyncHandle2.prod().iter = cuSyncHandle2.cons().iter = 0;
    cuSyncHandle1.cons().kernelExecuted_ = cuSyncHandle2.prod().kernelExecuted_;
    
    //Setup XW1 GeMM
    argsXW1 = typename GemmTy1::Arguments {cuSyncHandle1.prod(),
                                          gemm_size1,
                                          Parent::x, Parent::w1, 
                                          Parent::xw1, Parent::xw1,
                                          {Parent::alpha, Parent::beta},
                                          split_k1};
    //Setup XV GeMM
    argsXV = typename GemmTy2::Arguments {cuSyncHandle1.cons(),
                                          gemm_size1,
                                          Parent::x, Parent::v,
                                          Parent::xw1, Parent::xv,
                                          {Parent::alpha, ElementComputeEpilogue(1.0f)},
                                          split_k1};
    //Setup XW12 GeMM
    argsXW12 = typename GemmTy3::Arguments {cuSyncHandle2.cons(),
                                            gemm_size2, 
                                            Parent::xv, Parent::w2,
                                            Parent::xw12, Parent::xw12,
                                            {Parent::alpha, Parent::beta},
                                            split_k2};
    
    Parent::initGeMMs(argsXW1, argsXV, argsXW12);
  }
};


/*LLaMA Baseline MLP*/
template<typename GemmTy1, typename GemmTy2, typename GemmTy3>
cudaError_t runBaselineLLaMA(BaselineMLPParams<GemmTy1, GemmTy2, GemmTy3>& mlpParams,
                             cudaStream_t stream1,
                             int iters = 100) {
  //Run kernels
  for (int r = 0; r < iters; r++) {    
    auto status = mlpParams.gemm1(stream1);
    CUTLASS_CHECK(status);
    // CUDA_CHECK(cudaStreamSynchronize(stream1));
    // CUDA_CHECK(cudaDeviceSynchronize());
    // double middle1 = timeInMicroSeconds();
    // double iterMatMul1 = middle1-start;
    // matmul1Time += iterMatMul1;
    
    // return cudaSuccess;

    status = mlpParams.gemm2(stream1);
    CUTLASS_CHECK(status);
    // CUDA_CHECK(cudaStreamSynchronize(stream1));
    // CUDA_CHECK(cudaDeviceSynchronize());
    // double middle2 = timeInMicroSeconds();
    // double iterMatMul2 = middle2-middle1;
    // matmul2Time += iterMatMul2;

    status = mlpParams.gemm3(stream1);
    CUTLASS_CHECK(status);
    // CUDA_CHECK(cudaStreamSynchronize(stream1));
    CUDA_CHECK(cudaDeviceSynchronize());
    // double middle3 = timeInMicroSeconds();
    // double iterMatmul3 = middle3-middle2;
    // matmul3Time += iterMatmul3;
    // double end = timeInMicroSeconds();
    // if (iters > 10)
    //   printf("{\"Total\": %lf, \"matmul1Time\": %lf, \"matmul2Time\": %lf, \"matmul3Time\": %lf}\n",end-start, iterMatMul1, iterMatMul2, iterMatmul3);
    // execTime += end-start;
    // CUDA_CHECK(cudaDeviceSynchronize());
  }

  return cudaSuccess;
}

template<typename GemmTy1, typename GemmTy2, typename GemmTy3>
cudaError_t runCuSyncMLP(CuSyncMLPParameters<GemmTy1, GemmTy2, GemmTy3>& mlpParams,
                         double& execTime,
                         int iters = 100) {
  execTime = 0;
  // mlpParams.cuSyncHandle1.prod().iter += 1;
  // mlpParams.cuSyncHandle1.cons().iter += 1;
  // mlpParams.cuSyncHandle2.prod().iter += 1;
  // mlpParams.cuSyncHandle2.cons().iter += 1;

  for (int r = 0; r < iters; r++) {
    mlpParams.gemm1.params_.custage.iter += 1;
    mlpParams.gemm2.params_.custage.iter += 1;
    mlpParams.gemm3.params_.custage.iter += 1;

    auto status = mlpParams.gemm1.run(true, NULL, CudaStreams[0]);
    CUTLASS_CHECK(status);
    
  #ifndef AVOID_WAIT_KERNEL
    mlpParams.cuSyncHandle1.invokeWaitKernel(CudaStreams[1]);
  #endif
    status = mlpParams.gemm2.run(true, NULL, CudaStreams[1]);
    CUTLASS_CHECK(status);
    
  #ifndef AVOID_WAIT_KERNEL
    mlpParams.cuSyncHandle2.invokeWaitKernel(CudaStreams[2]);
  #endif
    status = mlpParams.gemm3.run(true, NULL, CudaStreams[2]);
    CUTLASS_CHECK(status);
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  return cudaSuccess;
}

// cudaError_t runBaselineLLaMA(int split_k1, int split_k2, 
//                         MLPParameters& mlpParams,
//                         cudaStream_t stream1,
//                         cudaStream_t stream2,
//                         double& execTime,
//                         double& matmul1Time,
//                         double& matmul2Time,
//                         double& matmul3Time,
//                         int iters = 100) {
//   cudaError_t result;
//   execTime = 0;
//   matmul1Time = 0;
//   matmul2Time = 0;
//   matmul3Time = 0;
//   if (split_k1 == 1 && split_k2 == 1) {
//     result = runBaselineLLaMA<Gemm1, Gemm2, Gemm3>(split_k1, split_k2, mlpParams, stream1, stream2, execTime, matmul1Time, matmul2Time, matmul3Time, iters);
//   } else if (split_k1 > 1 && split_k2 == 1) {
//     result = runBaselineLLaMA<GemmSplitK1, GemmSplitK2, Gemm3>(split_k1, split_k2, mlpParams, stream1, stream2, execTime, matmul1Time, matmul2Time, matmul3Time, iters);
//   } else if (split_k1 == 1 && split_k2 > 1) {
//     result = runBaselineLLaMA<Gemm1, GemmSplitK2, Gemm3>(split_k1, split_k2, mlpParams, stream1, stream2, execTime, matmul1Time, matmul2Time, matmul3Time, iters);
//   } else {
//     result = runBaselineLLaMA<GemmSplitK1, GemmSplitK2, GemmSplitK3>(split_k1, split_k2, mlpParams, stream1, stream2, execTime, matmul1Time, matmul2Time, matmul3Time, iters);
//   }

//   return result;
// }

#define SPLIT_K

#ifdef SPLIT_K
int baseline_split_k1 = 8;
int baseline_split_k2 = 4;
using Baseline = BaselineMLPParams<GemmSplitK1, GemmSplitK2, GemmSplitK3>;
#else
int baseline_split_k1 = 1;
int baseline_split_k2 = 1;
using Baseline = BaselineMLPParams<Gemm1, Gemm2, Gemm3>;
#endif

extern "C" 
Baseline* initMLPParams(const void* w1, const void* v, const void* w2, const uint batch) {
//printf("w1 %p v %p w2 %p\n", w1, v, w2);

Baseline* llamaMLPParams = new Baseline(std::string("llama"), 
                                 baseline_split_k1, baseline_split_k2, batch, 
                                 (const ElementInputA*)w1, 
                                 (const ElementInputA*)v,
                                 (const ElementInputA*)w2);
  return llamaMLPParams;
}

extern "C"
void runLLAMA(Baseline* llamaMLPParams, const void* x, const void* silu, const void* xv, const void* out, int runs) {
  double times = 0;
  llamaMLPParams->setInput((ElementInputA*)x);
  llamaMLPParams->setIntermediate((ElementInputA*)silu, (ElementInputA*)xv);
  llamaMLPParams->setOutput((ElementInputA*)out);
  runBaselineLLaMA(*llamaMLPParams, CudaStreams[0], runs);
}

void initStreams() {
  if (CudaStreams[0] != NULL) return;
  int highestPriority;
  int lowestPriority;
  CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&lowestPriority, &highestPriority));
  for (int i = highestPriority; i <= lowestPriority; i++) {
    CUDA_CHECK(cudaStreamCreateWithPriority(&CudaStreams[i - highestPriority], 0, i));
  }
}

#ifdef SPLIT_K
int cusync_split_k1 = 8;
int cusync_split_k2 = 4;
using CuSyncMLP = CuSyncMLPParameters<CuSyncGemmSplitK1, CuSyncGemmSplitK2, CuSyncGemmSplitK3>;
#else
using CuSyncMLP = CuSyncMLPParameters<CuSyncGemm1, CuSyncGemm2, CuSyncGemm3>;
int cusync_split_k1 = 1;
int cusync_split_k2 = 1;
#endif

extern "C"
CuSyncMLP* initCuSyncMLPParams(const void* w1, const void* v, const void* w2, const uint batch) {
//printf("w1 %p v %p w2 %p\n", w1, v, w2);

  initStreams();

  CuSyncMLP* llamaMLPParams = 
  new CuSyncMLP(std::string("llama"), 
                cusync_split_k1, cusync_split_k2, batch, 
                                 (const ElementInputA*)w1, 
                                 (const ElementInputA*)v,
                                 (const ElementInputA*)w2);
  return llamaMLPParams;
}

extern "C"
void runCuSyncLLAMA(CuSyncMLP* llamaMLPParams, const void* x, const void* silu, const void* xv, const void* out, int runs) {
  double times = 0;
  llamaMLPParams->setInput((ElementInputA*)x);
  llamaMLPParams->setIntermediate((ElementInputA*)silu, (ElementInputA*)xv);
  llamaMLPParams->setOutput((ElementInputA*)out);
  runCuSyncMLP(*llamaMLPParams, times, runs);
}
