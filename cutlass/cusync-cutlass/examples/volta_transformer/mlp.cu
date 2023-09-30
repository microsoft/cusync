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

// #define LLAMA

#if defined(TILESYNC)
#if !defined(LLAMA)
  #define NO_ATOMIC_ADD
#endif
#define REORDER_TILE_LOADS
#endif

// #define AVOID_CUSTOM_ORDER
// #define AVOID_WAIT_KERNEL

// #if defined(TILESYNC) || defined(TILEBATCH)
// #define AVOID_CUSTOM_ORDER
// #define AVOID_WAIT_KERNEL
// #endif 

#include<cuSync.h>

#ifdef ROWSYNC
  using ProdCuStage = CuStage<CuStageType::Producer, RowMajor, RowSync>;
  using MiddleCuStage = CuStage<CuStageType::Producer | CuStageType::Consumer, RowMajor, RowSync>;
  using ConsCuStage = CuStage<CuStageType::Consumer, RowMajor, RowSync>;
  using Sync = RowSync;
#elif defined(TILEBATCH)
  using ProdCuStage = CuStage<CuStageType::Producer, RowMajor, TileSync<2>>;
  using MiddleCuStage = CuStage<CuStageType::Producer | CuStageType::Consumer, RowMajor, TileSync<2>>;
  using ConsCuStage = CuStage<CuStageType::Consumer, RowMajor, TileSync<2>>;
  using Sync = TileSync<2>;
#elif defined(TILESYNC)
  using ProdCuStage = CuStage<CuStageType::Producer, RowMajor, TileSync<1>>;
  using MiddleCuStage = CuStage<CuStageType::Producer | CuStageType::Consumer, RowMajor, TileSync<1>>;
  using ConsCuStage = CuStage<CuStageType::Consumer, RowMajor, TileSync<1>>;
  using Sync = TileSync<1>;
#elif defined(BATCHEDROW)
  using ProdCuStage = CuStage<CuStageType::Producer, RowMajor, BatchedRowSync>;
  using ConsCuStage = CuStage<CuStageType::Consumer, RowMajor, BatchedRowSync>;
  using Sync = BatchedRowSync;
#else
  #error "Unknown Synchronization"
#endif

#include "common.h"
const uint GLURowTile = 1;

#ifndef EVAL_TILE_SIZES
//Tile sizes of all GeMMs
using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<256, 128, 32>;
using ShapeMMAWarp = cutlass::gemm::GemmShape<128, 64, 32>;
#else
//<eval tiles>
using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<32, 256, 32>;  
using ShapeMMAWarp = cutlass::gemm::GemmShape<32, 64, 32>;
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
using LayoutInputB = cutlass::layout::RowMajor;
using LayoutOutput = cutlass::layout::RowMajor;

//Use FP-16 Tensor Cores
using MMAOp = cutlass::arch::OpClassTensorOp;

using SmArch = cutlass::arch::Sm70;

//First GeMM in MLP is fused with GELU
#ifdef LLAMA
using EpilogueOp1 = cutlass::epilogue::thread::LinearCombination<
#elif defined(GPT3)
using EpilogueOp1 = cutlass::epilogue::thread::LinearCombinationGELU<
#endif
    ElementOutput,                                        
    128 / cutlass::sizeof_bits<ElementOutput>::value,
    ElementAccumulator, 
    ElementComputeEpilogue,                              
    cutlass::epilogue::thread::ScaleType::NoBetaScaling>;

//Second GeMM in MLP performs no extra fused computations 
using EpilogueOp2 = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,                                        
    128 / cutlass::sizeof_bits<ElementOutput>::value,     
    ElementAccumulator,
    ElementComputeEpilogue>;

template<typename EpilogueOp, bool splitK>
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
using Gemm1 = BaseMLPGemm<EpilogueOp1, false>;
using Gemm2 = BaseMLPGemm<EpilogueOp2, false>;

//Baseline GeMMs with SplitK enabled
using GemmSplitK1 = BaseMLPGemm<EpilogueOp1, true>;
using GemmSplitK2 = BaseMLPGemm<EpilogueOp2, true>;

//CuSync GeMMs
using CuSyncImpl = CuSync<ProdCuStage, ConsCuStage>;

using CuSyncImpl1 = CuSync<ProdCuStage, MiddleCuStage>;
using CuSyncImpl2 = CuSync<MiddleCuStage, ConsCuStage>;

template<typename CuStage, typename EpilogueOp, bool splitK>
class CuSyncMLPGemm : public cutlass::gemm::device::CuSyncGemm<CuStage, ElementInputA, LayoutInputA, 
                                                       ElementInputB, LayoutInputB,
                                                       ElementOutput, LayoutOutput,
                                                        ElementAccumulator, MMAOp,
                                                        SmArch, ShapeMMAThreadBlock,
                                                        ShapeMMAWarp, ShapeMMAOp,
                                                        EpilogueOp, 
                                                        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 
                                                        2, 8, 8, splitK> {};

using CuSyncGemm1 = CuSyncMLPGemm<ProdCuStage, EpilogueOp1, false>;
using CuSyncGemm2 = CuSyncMLPGemm<ConsCuStage, EpilogueOp2, false>;

using CuSyncGemmSplitK1 = CuSyncMLPGemm<ProdCuStage, EpilogueOp1, true>;
using CuSyncGemmSplitK2 = CuSyncMLPGemm<ConsCuStage, EpilogueOp2, true>;

using HostTensor = cutlass::HostTensor<ElementInputA, LayoutInputA>;

enum MLPType {
  GPT3,
  LLaMa    
};

struct MLPParameters {
  HostTensor x; //[B, H]
  HostTensor w1; //[H, 4H/8] in GPT-3
  //xw1 = GeLU(x * w1)
  HostTensor xw1; //[B, 4 H / 8]
  HostTensor w2; //[4H/8, H] in GPT-3 and [H/3, H] in LLaMa
  //xw12 = xw1 * w2
  HostTensor xw12; //[B, H]

  //For LLaMa only
  HostTensor vw1; //[B, 2*H/3] in LLAMA
  HostTensor xvw1; //[B, 2*H/3] in LLaMa
  HostTensor glu; //[B, H/3] in LLaMa

  HostTensor ref_xw1;
  HostTensor ref_xw12;

  //For LLaMa only
  HostTensor ref_xv;

  bool checkResults;

  cutlass::gemm::GemmCoord gemm_size1;
  cutlass::gemm::GemmCoord gemm_size2;
  ElementComputeEpilogue alpha;
  ElementComputeEpilogue beta;

  std::string model;

  MLPParameters(std::string model_, uint batch, bool check) {
    alpha = ElementComputeEpilogue(1.0);
    beta = ElementComputeEpilogue(0.0);
    model = model_;

    if (model == "gpt3") {
      gemm_size1 = cutlass::gemm::GemmCoord(batch, 4*12288/8, 12288);
      gemm_size2 = cutlass::gemm::GemmCoord(batch, 12288, 4*12288/8);
    } else if (model=="llama") {
      int d = ((8192/3 + 127)/128)*128;
      gemm_size1 = cutlass::gemm::GemmCoord(batch, 2*d, 8192);
      gemm_size2 = cutlass::gemm::GemmCoord(batch, 8192, d);
    }
    std::cout << "GeMM 1 Size: " << gemm_size1.m() << ", " << 
      gemm_size1.n() << ", " << gemm_size1.k() << std::endl;
    std::cout << "GeMM 2 Size: " << gemm_size2.m() << ", " << 
      gemm_size2.n() << ", " << gemm_size2.k() << std::endl;
    
    x = HostTensor(gemm_size1.mk());
    w1 = HostTensor(gemm_size1.kn());
    xw1 = HostTensor(gemm_size1.mn());
    w2 = HostTensor(gemm_size2.kn());
    xw12 = HostTensor(gemm_size2.mn());
    ref_xw1 = HostTensor(gemm_size1.mn());
    ref_xw12 = HostTensor(gemm_size2.mn());

    if (model == "llama") {
      xvw1 = HostTensor(gemm_size1.mn());
      vw1 = HostTensor(gemm_size1.kn());
      glu = HostTensor(gemm_size2.mk());
      ref_xv = HostTensor(gemm_size1.mn());
    }
    checkResults = check;
  }

  void initIns() {
    if (checkResults) {
      ElementOutput values[5] = {ElementOutput(0.05), ElementOutput(0.2), ElementOutput(0.01), ElementOutput(3), ElementOutput(0.4)};
      memset_random(x.host_data(), 5, values, x.size());
      memset_random(w1.host_data(), 5, values, w1.size());
      memset_random2(w2.host_data(), ElementOutput(0.01), ElementOutput(0.05), w2.size());
      if (model == "llama") {
        memset_random2(vw1.host_data(), ElementOutput(0.01), ElementOutput(0.2), vw1.size());
      }
    } else {
      cutlass::reference::host::TensorFill(x.host_view(), ElementOutput(0.05));
      cutlass::reference::host::TensorFill(w1.host_view(), ElementOutput(0.5));
      cutlass::reference::host::TensorFill(w2.host_view(), ElementOutput(0.01));
      if (model == "llama") {
        cutlass::reference::host::TensorFill(vw1.host_view(), ElementOutput(0.5));
      }
    }
    // Copy data from host to GPU
    x.sync_device();
    w1.sync_device();
    w2.sync_device();
    if (model == "llama") {
      vw1.sync_device();
    }
  }
  
  void initOuts() {
    cutlass::reference::host::TensorFill(xw1.host_view());
    cutlass::reference::host::TensorFill(xw12.host_view());
      
    xw1.sync_device();
    xw12.sync_device();
    if (model == "llama") {
      cutlass::reference::host::TensorFill(xvw1.host_view());
      xvw1.sync_device();
      cutlass::reference::host::TensorFill(glu.host_view());
      glu.sync_device();
    }
  }

  void initRefs() {
    cutlass::reference::host::TensorFill(ref_xw12.host_view());
    cutlass::reference::host::TensorFill(ref_xw1.host_view());

    ref_xw12.sync_device();
    ref_xw1.sync_device();
    if (model == "llama") {
      cutlass::reference::host::TensorFill(ref_xv.host_view());
      ref_xv.sync_device(); 
    }
  }

  bool isGPT3() {return model == "gpt3";}
  bool isLLaMa() {return model == "llama";}
};

/** Reference MLP for correctness check **/
cudaError_t referenceMLP(MLPParameters& mlpParams) {
  ref_matmul<ElementOutput, ElementAccumulator>(mlpParams.gemm_size1.m(), 
                                                mlpParams.gemm_size1.n(), 
                                                mlpParams.gemm_size1.k(),
                                                mlpParams.x.device_data(), 
                                                mlpParams.w1.device_data(), 
                                                mlpParams.ref_xw1.host_data());
  CUDA_CHECK(cudaMemcpy(mlpParams.ref_xw1.device_data(), mlpParams.ref_xw1.host_data(), 
             sizeof(ElementOutput) * mlpParams.ref_xw1.size(), cudaMemcpyHostToDevice));
  
  if (mlpParams.isLLaMa()) {
    printf("check not supported in llama\n");
    ref_matmul<ElementOutput, ElementAccumulator>(mlpParams.gemm_size1.m(), 
                                                  mlpParams.gemm_size1.n(), 
                                                  mlpParams.gemm_size1.k(),
                                                  mlpParams.x.device_data(), 
                                                  mlpParams.vw1.device_data(), 
                                                  mlpParams.ref_xv.host_data());
    //Compute XW1 (dot) XV
    for (int b = 0; b < mlpParams.gemm_size1.m(); b++) {
      for (int n = 0; n < mlpParams.gemm_size1.n(); n++) {
        uint index = b * mlpParams.gemm_size1.n() + n;
        mlpParams.ref_xv.host_data()[index] = mlpParams.ref_xw1.host_data()[index] * 
                                              mlpParams.ref_xv.host_data()[index];
      }
    }

    mlpParams.ref_xv.sync_device();

    ref_matmul<ElementOutput, ElementAccumulator>(mlpParams.gemm_size2.m(),
                                                  mlpParams.gemm_size2.n(),
                                                  mlpParams.gemm_size2.k(), 
                                                  mlpParams.ref_xv.device_data(),
                                                  mlpParams.w2.device_data(), 
                                                  mlpParams.ref_xw12.host_data());
  } else {
    ref_matmul<ElementOutput, ElementAccumulator>(mlpParams.gemm_size2.m(),
                                                  mlpParams.gemm_size2.n(),
                                                  mlpParams.gemm_size2.k(), 
                                                  mlpParams.ref_xw1.device_data(),
                                                  mlpParams.w2.device_data(), 
                                                  mlpParams.ref_xw12.host_data());
  }

  return cudaSuccess;
}

cudaError_t checkMLPResults(MLPParameters& mlpParams) {
  ElementOutput* hostC = new ElementOutput[mlpParams.ref_xw1.size()];
  CUDA_CHECK(cudaMemcpy(hostC, mlpParams.xw1.device_data(), 
                        mlpParams.xw1.size() * sizeof(ElementOutput), 
                        cudaMemcpyDeviceToHost));
  printf("Checking first GeMM\n");
  bool eq = equals(mlpParams.ref_xw1.size(), mlpParams.ref_xw1.host_data(), hostC, 1e-5);
  if (eq == false) {
    printf("First GeMM not correct\n");
    return cudaErrorUnknown;
  }
  printf("First GeMM passed\n");
  ElementOutput* hostE = new ElementOutput[mlpParams.ref_xw12.size()];
  CUDA_CHECK(cudaMemcpy(hostE, mlpParams.xw12.device_data(), 
                        mlpParams.xw12.size() * sizeof(ElementOutput), 
                        cudaMemcpyDeviceToHost));
  //For LLaMa not checking XV
  printf("Checking second GeMM\n");
  eq = equals(mlpParams.ref_xw12.size(), mlpParams.ref_xw12.host_data(), hostE, 1e-1f);
  if (eq == false) {
    printf("Second GeMM not correct \n");
    return cudaErrorUnknown;
  }

  printf("Second GeMM passed\n");

  return cudaSuccess;
}

/*GPT3 Baseline MLP*/
template<typename GemmTy1, typename GemmTy2>
cudaError_t runBaselineGPT3(int split_k1, int split_k2, 
                            MLPParameters& mlpParams,
                            cudaStream_t stream,
                            double& execTime, double& matmul1Time, double& softmaxTime, double& matmul2Time,
                            int iters = 100) {
  //Setup first GeMM
  typename GemmTy1::Arguments args1 {
    mlpParams.gemm_size1,
    mlpParams.x.device_ref(), 
    mlpParams.w1.device_ref(),
    mlpParams.xw1.device_ref(),
    mlpParams.xw1.device_ref(),
    {mlpParams.alpha, mlpParams.beta},
    split_k1};

  size_t workspace_size = GemmTy1::get_workspace_size(args1);
  cutlass::device_memory::allocation<uint8_t> workspace1(workspace_size);
  GemmTy1 gemm_op1;
  cutlass::Status status = gemm_op1.can_implement(args1);
  CUTLASS_CHECK(status);
  status = gemm_op1.initialize(args1, workspace1.get());
  CUTLASS_CHECK(status);

  //Setup Second GeMM
  typename GemmTy2::Arguments args2{ 
    mlpParams.gemm_size2, 
    mlpParams.xw1.device_ref(), 
    mlpParams.w2.device_ref(), 
    mlpParams.xw12.device_ref(), 
    mlpParams.xw12.device_ref(), 
    {mlpParams.alpha, mlpParams.beta},         
    split_k2};
  
  GemmTy2 gemm_op2;
  workspace_size = GemmTy2::get_workspace_size(args2);
  cutlass::device_memory::allocation<uint8_t> workspace2(workspace_size);
  status = gemm_op2.can_implement(args2);
  CUTLASS_CHECK(status);
  status = gemm_op2.initialize(args2, workspace2.get());
  CUTLASS_CHECK(status);
  
  execTime = 0;
  
  //Run kernels
  for (int r = 0; r < iters; r++) {    
    double start = timeInMicroSeconds();
    status = gemm_op1(args1, workspace1.get(), stream);
    CUTLASS_CHECK(status);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    double middle1 = timeInMicroSeconds();
    double iterMatMul1 = middle1-start;
    matmul1Time += iterMatMul1;
    status = gemm_op2(args2, workspace2.get(), stream);
    CUTLASS_CHECK(status);
    CUDA_CHECK(cudaDeviceSynchronize());
    double middle3 = timeInMicroSeconds();
    double iterMatmul2 = middle3-middle1;
    matmul2Time += iterMatmul2;
    double end = timeInMicroSeconds();
    if (iters > 10)
      printf("{\"Total\": %lf, \"matmul1Time\": %lf, \"matmul2Time\": %lf}\n",end-start,iterMatMul1, iterMatmul2);
    execTime += end-start;
  }

  return cudaSuccess;
}

cudaError_t runBaselineGPT3(int split_k1, int split_k2, 
                        MLPParameters& mlpParams,
                        cudaStream_t stream,
                        double& execTime,
                        double& matmul1Time,
                        double& softmaxTime,
                        double& matmul2Time,
                        int iters = 100) {
  cudaError_t result;
  execTime = 0;
  matmul1Time = 0;
  softmaxTime = 0;
  matmul2Time = 0;
  if (split_k1 == 1 && split_k2 == 1) {
    result = runBaselineGPT3<Gemm1, Gemm2>(split_k1, split_k2, mlpParams, stream, execTime, matmul1Time, softmaxTime, matmul2Time, iters);
  } else if (split_k1 > 1 && split_k2 == 1) {
    result = runBaselineGPT3<GemmSplitK1, Gemm2>(split_k1, split_k2, mlpParams, stream, execTime, matmul1Time, softmaxTime, matmul2Time, iters);
  } else if (split_k1 == 1 && split_k2 > 1) {
    result = runBaselineGPT3<Gemm1, GemmSplitK2>(split_k1, split_k2, mlpParams, stream, execTime, matmul1Time, softmaxTime, matmul2Time, iters);
  } else {
    result = runBaselineGPT3<GemmSplitK1, GemmSplitK2>(split_k1, split_k2, mlpParams, stream, execTime, matmul1Time, softmaxTime, matmul2Time, iters);
  }

  return result;
}

/*LLaMA Baseline MLP*/
template<typename T, uint H3>
__global__ void gluKernel(T* xvw1, T* glu) {
  int ROW = blockIdx.x;

  for (int i = threadIdx.x; i < H3; i += blockDim.x) {
    float xw1 = xvw1[ROW * (2 * H3) + i];
    float xv =  xvw1[ROW * (2 * H3) + i + H3];
    glu[ROW * H3 + i] = xw1 * xv;
  }
}

template<typename GemmTy1, typename GemmTy2>
cudaError_t runBaselineLLaMA(int split_k1, int split_k2, 
                             MLPParameters& mlpParams,
                             cudaStream_t stream1,
                             cudaStream_t stream2,
                             double& execTime, double& matmul1Time, 
                             double& matmul2Time, double& matmul3Time,
                             int iters = 100) {
  //Setup XW1 GeMM
  typename GemmTy1::Arguments argsXW1{
    mlpParams.gemm_size1,
    mlpParams.x.device_ref(), 
    mlpParams.w1.device_ref(),
    mlpParams.xvw1.device_ref(),
    mlpParams.xvw1.device_ref(),
    {mlpParams.alpha, mlpParams.beta},
    split_k1};

  size_t workspace_size = GemmTy1::get_workspace_size(argsXW1);
  cutlass::device_memory::allocation<uint8_t> workspace1(workspace_size);
  GemmTy1 gemm_opXVW1;
  cutlass::Status status = gemm_opXVW1.can_implement(argsXW1);
  CUTLASS_CHECK(status);
  status = gemm_opXVW1.initialize(argsXW1, workspace1.get());
  CUTLASS_CHECK(status);

  //Setup XW12 GeMM
  typename GemmTy2::Arguments argsXW12{
    mlpParams.gemm_size2, 
    mlpParams.glu.device_ref(), 
    mlpParams.w2.device_ref(), 
    mlpParams.xw12.device_ref(), 
    mlpParams.xw12.device_ref(), 
    {mlpParams.alpha, mlpParams.beta},         
    split_k2};
  
  GemmTy2 gemm_opXW12;
  workspace_size = GemmTy2::get_workspace_size(argsXW12);
  cutlass::device_memory::allocation<uint8_t> workspace3(workspace_size);
  status = gemm_opXW12.can_implement(argsXW12);
  CUTLASS_CHECK(status);
  status = gemm_opXW12.initialize(argsXW12, workspace3.get());
  CUTLASS_CHECK(status);
  
  execTime = 0; 

  //Run kernels
  for (int r = 0; r < iters; r++) {    
    double start = timeInMicroSeconds();
    status = gemm_opXVW1(stream1);
    CUTLASS_CHECK(status);
    CUDA_CHECK(cudaDeviceSynchronize());
    double middle1 = timeInMicroSeconds();
    double iterMatMul1 = middle1-start;
    matmul1Time += iterMatMul1;

    //glu
    gluKernel<half, ((8192/3+127)/128)*128><<<mlpParams.gemm_size1.m(), 
                                              ShapeMMAThreadBlock::kN, 0, stream1>>>
      ((half*)mlpParams.xvw1.device_data(), (half*)mlpParams.glu.device_data());
    CUDA_CHECK(cudaDeviceSynchronize());
    double middle2 = timeInMicroSeconds();
    double iterMatMul2 = middle2-middle1;
    matmul2Time += iterMatMul2;

    status = gemm_opXW12(stream1);
    CUTLASS_CHECK(status);
    CUDA_CHECK(cudaDeviceSynchronize());
    double middle3 = timeInMicroSeconds();
    double iterMatmul3 = middle3-middle2;
    matmul3Time += iterMatmul3;
    double end = timeInMicroSeconds();
    if (iters > 10)
      printf("{\"Total\": %lf, \"matmul1Time\": %lf, \"matmul2Time\": %lf, \"matmul3Time\": %lf}\n",end-start, iterMatMul1, iterMatMul2, iterMatmul3);
    execTime += end-start;
  }

  return cudaSuccess;
}

cudaError_t runBaselineLLaMA(int split_k1, int split_k2, 
                        MLPParameters& mlpParams,
                        cudaStream_t stream1,
                        cudaStream_t stream2,
                        double& execTime,
                        double& matmul1Time,
                        double& matmul2Time,
                        double& matmul3Time,
                        int iters = 100) {
  cudaError_t result;
  execTime = 0;
  matmul1Time = 0;
  matmul2Time = 0;
  matmul3Time = 0;
  if (split_k1 == 1 && split_k2 == 1) {
    result = runBaselineLLaMA<Gemm1, Gemm2>(split_k1, split_k2, mlpParams, stream1, stream2, execTime, matmul1Time, matmul2Time, matmul3Time, iters);
  } else if (split_k1 > 1 && split_k2 == 1) {
    result = runBaselineLLaMA<GemmSplitK1, Gemm2>(split_k1, split_k2, mlpParams, stream1, stream2, execTime, matmul1Time, matmul2Time, matmul3Time, iters);
  } else if (split_k1 == 1 && split_k2 > 1) {
    result = runBaselineLLaMA<Gemm1, GemmSplitK2>(split_k1, split_k2, mlpParams, stream1, stream2, execTime, matmul1Time, matmul2Time, matmul3Time, iters);
  } else {
    result = runBaselineLLaMA<GemmSplitK1, GemmSplitK2>(split_k1, split_k2, mlpParams, stream1, stream2, execTime, matmul1Time, matmul2Time, matmul3Time, iters);
  }

  return result;
}


/*CuSync GPT-3 MLP*/
template<typename GemmTy1, typename GemmTy2>
cudaError_t runCuSyncGPT3(int split_k1, int split_k2,
                      MLPParameters& mlpParams,
                      CuSyncImpl& handle,
                      cudaStream_t producer_stream, 
                      cudaStream_t consumer_stream,
                      double& execTime,
                      int iters = 100) {
  typename GemmTy1::Arguments args1{handle.prod(),
                                     mlpParams.gemm_size1,
                                     mlpParams.x.device_ref(),
                                     mlpParams.w1.device_ref(),
                                     mlpParams.xw1.device_ref(),
                                     mlpParams.xw1.device_ref(),
                                     {mlpParams.alpha, mlpParams.beta},         
                                     split_k1};
  GemmTy1 gemm_op1;
  size_t workspace_size = GemmTy1::get_workspace_size(args1);
  cutlass::device_memory::allocation<uint8_t> workspace1(workspace_size);
  cutlass::Status status = gemm_op1.can_implement(args1);
  CUTLASS_CHECK(status);
  status = gemm_op1.initialize(args1, workspace1.get());
  CUTLASS_CHECK(status);

  typename GemmTy2::Arguments args2{handle.cons(),
                                    mlpParams.gemm_size2,  
                                    mlpParams.xw1.device_ref(),
                                    mlpParams.w2.device_ref(),
                                    mlpParams.xw12.device_ref(),
                                    mlpParams.xw12.device_ref(),
                                    {mlpParams.alpha, mlpParams.beta},
                                    split_k2};

  GemmTy2 gemm_op2;
  workspace_size = GemmTy2::get_workspace_size(args2);
  cutlass::device_memory::allocation<uint8_t> workspace2(workspace_size);
  status = gemm_op2.can_implement(args2);
  CUTLASS_CHECK(status);
  status = gemm_op2.initialize(args2, workspace2.get());
  CUTLASS_CHECK(status);

  execTime = 0;
  
  for (int r = 0; r < iters; r++) {
    handle.prod().iter += 1;
    handle.cons().iter += 1;
    gemm_op2.params_.custage.iter += 1;
    gemm_op1.params_.custage.iter += 1;
    
    double start = timeInMicroSeconds();
    status = gemm_op1.run(true, NULL, producer_stream);
    CUTLASS_CHECK(status);

    // CUDA_CHECK(cudaDeviceSynchronize());
  #ifndef AVOID_WAIT_KERNEL
    handle.invokeWaitKernel(consumer_stream);
  #endif  
    status = gemm_op2.run(true, NULL, consumer_stream);
    CUTLASS_CHECK(status);
    CUDA_CHECK(cudaDeviceSynchronize());
    double end = timeInMicroSeconds();
    if (iters > 10)
      printf("{\"Total\": %lf}\n",end-start);
    execTime += end-start;
  }

  return cudaSuccess;
}

cudaError_t runCuSyncGPT3(int split_k1, int split_k2, MLPParameters& mlpParams,
                      CuSyncImpl& handle,
                      cudaStream_t producer_stream, cudaStream_t consumer_stream,
                      double& execTime, int iters = 100) {
  cudaError_t result;
  execTime = 0;

  if (split_k1 == 1 && split_k2 == 1) {
    result = runCuSyncGPT3<CuSyncGemm1, CuSyncGemm2>(split_k1, split_k2, mlpParams, handle, producer_stream, consumer_stream, execTime, iters);
  } else if (split_k1 > 1 && split_k2 == 1) {
    result = runCuSyncGPT3<CuSyncGemmSplitK1, CuSyncGemm2>(split_k1, split_k2, mlpParams, handle, producer_stream, consumer_stream, execTime, iters);
  } else if (split_k1 == 1 && split_k2 > 1) {
    result = runCuSyncGPT3<CuSyncGemm1, CuSyncGemmSplitK2>(split_k1, split_k2, mlpParams, handle, producer_stream, consumer_stream, execTime, iters);
  } else {
    result = runCuSyncGPT3<CuSyncGemmSplitK1, CuSyncGemmSplitK2>(split_k1, split_k2, mlpParams, handle, producer_stream, consumer_stream, execTime, iters);
  }

  return result;
}

/**CuSync LLaMa in MLP*/
template<typename T, uint RowTile, uint H3>
__global__ void cusyncgluKernel(uint M, T* xvw1, T* glu, MiddleCuStage cons1, MiddleCuStage prod2) {
  uint ROW = blockIdx.x * RowTile;
  prod2.tile(nullptr);
  for (uint ti = 0; ti < RowTile && ROW < M; ti++) {
    for (uint i = threadIdx.x; i < H3; i += blockDim.x) {
      if (ti == 0) {
        dim3 tile = {ROW/ShapeMMAThreadBlock::kM, i/ShapeMMAThreadBlock::kN, 0};
        cons1.wait(tile);
      }
      float xw1 = xvw1[ROW * (2 * H3) + i];
      float xv =  xvw1[ROW * (2 * H3) + i + H3];
      glu[ROW * H3 + i] = xw1 * xv;
      if (ti == RowTile - 1) {
        dim3 tile = {ROW/ShapeMMAThreadBlock::kM, i/ShapeMMAThreadBlock::kN, 0};
        prod2.post(tile);
      }
    }
    ROW++;
  }
}

template<typename GemmTy1, typename GemmTy2>
cudaError_t runCuSyncLLaMA(int split_k1, int split_k2,
                           MLPParameters& mlpParams,
                           CuSyncImpl1& handle1,
                           CuSyncImpl2& handle2,
                           cudaStream_t* streams,
                           double& execTime,
                           int iters = 100) {
  typename GemmTy1::Arguments argsXW1{handle1.prod(),
                                      mlpParams.gemm_size1,
                                      mlpParams.x.device_ref(),
                                      mlpParams.w1.device_ref(),
                                      mlpParams.xvw1.device_ref(),
                                      mlpParams.xvw1.device_ref(),
                                      {mlpParams.alpha, mlpParams.beta},         
                                      split_k1};
  GemmTy1 gemm_opXVW1;
  size_t workspace_size = GemmTy1::get_workspace_size(argsXW1);
  cutlass::device_memory::allocation<uint8_t> workspace1(workspace_size);
  cutlass::Status status = gemm_opXVW1.can_implement(argsXW1);
  CUTLASS_CHECK(status);
  status = gemm_opXVW1.initialize(argsXW1, workspace1.get());
  CUTLASS_CHECK(status);

  typename GemmTy2::Arguments argsXW12{handle2.cons(),
                                       mlpParams.gemm_size2,  
                                       mlpParams.xvw1.device_ref(),
                                       mlpParams.w2.device_ref(),
                                       mlpParams.xw12.device_ref(),
                                       mlpParams.xw12.device_ref(),
                                       {mlpParams.alpha, mlpParams.beta},
                                       split_k2};

  GemmTy2 gemm_opXW12;
  workspace_size = GemmTy2::get_workspace_size(argsXW12);
  cutlass::device_memory::allocation<uint8_t> workspace3(workspace_size);
  status = gemm_opXW12.can_implement(argsXW12);
  CUTLASS_CHECK(status);
  status = gemm_opXW12.initialize(argsXW12, workspace3.get());
  CUTLASS_CHECK(status);

  execTime = 0;
  
  for (int r = 0; r < iters; r++) {
    handle1.prod().iter += 1;
    handle1.cons().iter += 1;
    handle2.prod().iter += 1;
    handle2.cons().iter += 1;

    gemm_opXW12.params_.custage.iter += 1;
    gemm_opXVW1.params_.custage.iter += 1;

    double start = timeInMicroSeconds();
    status = gemm_opXVW1.run(true, NULL, streams[0]);
    CUTLASS_CHECK(status);

  #ifndef AVOID_WAIT_KERNEL
    handle1.invokeWaitKernel(streams[1]);
  #endif
    //glu
    cusyncgluKernel<half, GLURowTile, ((8192/3+127)/128)*128>
      <<<mlpParams.gemm_size1.m(), ShapeMMAThreadBlock::kN, 0, streams[1]>>>
      (mlpParams.gemm_size1.m(), (half*)mlpParams.xvw1.device_data(), 
       (half*)mlpParams.glu.device_data(), handle1.cons(), handle2.prod());

  #ifndef AVOID_WAIT_KERNEL
    handle2.invokeWaitKernel(streams[2]);
  #endif  
    status = gemm_opXW12.run(true, NULL, streams[2]);
    CUTLASS_CHECK(status);
    CUDA_CHECK(cudaDeviceSynchronize());

    double end = timeInMicroSeconds();
    if (iters > 10)
      printf("{\"Total\": %lf}\n",end-start);
    execTime += end-start;
  }

  return cudaSuccess;
}

cudaError_t runCuSyncLLaMA(int split_k1, int split_k2, 
                          MLPParameters& mlpParams,
                          CuSyncImpl1& handle1,
                          CuSyncImpl2& handle2,
                          cudaStream_t* streams,
                          double& execTime, int iters = 100) {
  cudaError_t result;
  execTime = 0;

  if (split_k1 == 1 && split_k2 == 1) {
    result = runCuSyncLLaMA<CuSyncGemm1, CuSyncGemm2>(split_k1, split_k2, mlpParams, handle1, handle2, streams, execTime, iters);
  } else if (split_k1 > 1 && split_k2 == 1) {
    result = runCuSyncLLaMA<CuSyncGemmSplitK1, CuSyncGemm2>(split_k1, split_k2, mlpParams, handle1, handle2, streams, execTime, iters);
  } else if (split_k1 == 1 && split_k2 > 1) {
    result = runCuSyncLLaMA<CuSyncGemm1, CuSyncGemmSplitK2>(split_k1, split_k2, mlpParams, handle1, handle2, streams, execTime, iters);
  } else {
    result = runCuSyncLLaMA<CuSyncGemmSplitK1, CuSyncGemmSplitK2>(split_k1, split_k2, mlpParams, handle1, handle2, streams, execTime, iters);
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
    std::cerr << "Volta Tensor Ops must be run on a machine"
              << "with compute capability of 70, 72, or 75."
              << std::endl;
    return 0;
  }
  
  const uint NUM_ARGS = 5;
  std::string argNames[NUM_ARGS] = {"--model", "--batch", "--check", "--split-k1", "--split-k2"};
  std::string argHelp[NUM_ARGS] = {"GPT3 or LLaMa", "Batch size", "Check results", 
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
    
  std::cout << "model=" << model << " batch=" << batch << " check="<<doChecking <<std::endl;

  cudaStream_t producer_stream;
  cudaStream_t producer_stream2;
  cudaStream_t consumer_stream;
  CUDA_CHECK(cudaStreamCreate(&producer_stream));
  CUDA_CHECK(cudaStreamCreate(&producer_stream2));
  CUDA_CHECK(cudaStreamCreate(&consumer_stream));

  MLPParameters mlpParams(model, batch, doChecking);
  mlpParams.initIns();
  mlpParams.initOuts();
  mlpParams.initRefs();
  
  cudaError_t result;
  int epochs = 20;
  int warmup = 10;

  if (doChecking) {
    //Run our reference MLP
    result = referenceMLP(mlpParams);
    if (result != cudaSuccess) {
      return 1;
    }
  }

  //Run baseline MLP
  double baselineTime = 0;
  double matmul1Time = 0;
  double softmaxTime = 0;
  double matmul2Time = 0;

  if (mlpParams.isGPT3()) {
    result = runBaselineGPT3(split_k1, split_k2, mlpParams, producer_stream, 
                             baselineTime, matmul1Time, softmaxTime, matmul2Time, 1);

    CUDA_CHECK(cudaDeviceSynchronize());

    if (doChecking) {
      result = checkMLPResults(mlpParams);
      if (result != cudaSuccess) {
        return 1;
      }
    }

    result = runBaselineGPT3(split_k1, split_k2, mlpParams, producer_stream, 
                             baselineTime, matmul1Time, softmaxTime, matmul2Time, warmup);

    CUDA_CHECK(cudaDeviceSynchronize());
    printf("START-BASELINE:\n");
    result = runBaselineGPT3(split_k1, split_k2, mlpParams, producer_stream, 
                         baselineTime, matmul1Time, softmaxTime, matmul2Time, epochs);
    CUDA_CHECK(result);
    printf("END-BASELINE:\n");
    printf("Average time %lf microseconds\n", baselineTime/(float)epochs);
  } else if (mlpParams.isLLaMa()) {
    result = runBaselineLLaMA(split_k1, split_k2, mlpParams, producer_stream, 
                              producer_stream2, baselineTime, matmul1Time, softmaxTime, matmul2Time, 1);

    CUDA_CHECK(cudaDeviceSynchronize());

    if (doChecking) {
      result = checkMLPResults(mlpParams);
      if (result != cudaSuccess) {
        return 1;
      }
    }

    result = runBaselineLLaMA(split_k1, split_k2, mlpParams, producer_stream, 
                              producer_stream2, baselineTime, matmul1Time, softmaxTime, matmul2Time, warmup);

    CUDA_CHECK(cudaDeviceSynchronize());
    printf("START-BASELINE:\n");
    result = runBaselineLLaMA(split_k1, split_k2, mlpParams, producer_stream, 
                              producer_stream2, baselineTime, matmul1Time, softmaxTime, matmul2Time, epochs);
    CUDA_CHECK(result);
    printf("END-BASELINE:\n");
    printf("Average time %lf microseconds\n", baselineTime/(float)epochs);
  }

  
  if (doChecking) {
    mlpParams.initOuts();
  }
  printf("mlpParams.gemm_size1.n() %d ShapeMMAThreadBlock::kN %d\n", mlpParams.gemm_size1.n(), ShapeMMAThreadBlock::kN);
  //Setup cusync gemm
  dim3 gridDim1 = {(uint)DIVUP(mlpParams.gemm_size1.m(), ShapeMMAThreadBlock::kM), 
                  (uint)DIVUP(mlpParams.gemm_size1.n(), ShapeMMAThreadBlock::kN), 
                  split_k1};
  dim3 gridDim2 = {(uint)DIVUP(mlpParams.gemm_size2.m(), ShapeMMAThreadBlock::kM), 
                   (uint)DIVUP(mlpParams.gemm_size2.n(), ShapeMMAThreadBlock::kN), 
                   split_k2};
  dim3 tileSize = {ShapeMMAThreadBlock::kM, ShapeMMAThreadBlock::kN, 1};

#if defined(ROWSYNC)
  using Sync = RowSync;
  uint waitValue = gridDim1.y;
  RowSync sync(gridDim1.y);
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

  int highestPriority;
  int lowestPriority;
  CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&lowestPriority, &highestPriority));
  CUDA_CHECK(cudaStreamCreateWithPriority(&consumer_stream, 0, lowestPriority));
  cudaStream_t streams[(lowestPriority - highestPriority + 1)];
  for (int i = highestPriority; i <= lowestPriority; i++) {
    CUDA_CHECK(cudaStreamCreateWithPriority(&streams[i - highestPriority], 0, i));
  }
  
  //Run cusync mlp
  if (mlpParams.isGPT3()) {
    ProdCuStage prod(gridDim1, tileSize, sync);
    ConsCuStage cons(gridDim2, tileSize, sync);

    prod.iter = cons.iter = 0;

    CuSyncImpl cuSyncHandle(prod, cons);
    
    double overlapTime = 0;
    cuSyncHandle.iter = 0;
    cuSyncHandle.prod().iter = cuSyncHandle.cons().iter = 0;

    result = runCuSyncGPT3(split_k1, split_k2, mlpParams, cuSyncHandle, producer_stream, consumer_stream, overlapTime, 1);

    CUDA_CHECK(cudaDeviceSynchronize());
    if (doChecking) {
      result = checkMLPResults(mlpParams);
      if (result != cudaSuccess) {
        return 1;
      }
    }

    result = runCuSyncGPT3(split_k1, split_k2, mlpParams, cuSyncHandle, producer_stream, consumer_stream, overlapTime, warmup);
    
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("START-OVERLAPPED:\n");
    
    result = runCuSyncGPT3(split_k1, split_k2, mlpParams, cuSyncHandle, producer_stream, consumer_stream, overlapTime, epochs);
    
    CUDA_CHECK(result);
    printf("END-OVERLAPPED:\n");
    
    printf("Average time %lf microseconds\n", overlapTime/(float)epochs);
  } else if (mlpParams.isLLaMa()) {
#if defined(ROWSYNC)
  RowSync sync2(min(ShapeMMAThreadBlock::kM, mlpParams.gemm_size1.m()), GLURowTile);
#elif defined(TILESYNC)
  using Sync = TileSync<1>;
  uint waitValue = DIVUP(min(mlpParams.gemm_size1.m(), ShapeMMAThreadBlock::kM), GLURowTile);
  Sync sync2(waitValue, 1);
#else
  #error "Unknown Policy"
#endif
    ProdCuStage prod(gridDim1, tileSize, sync);
    dim3 gridMiddle = {(uint)DIVUP(mlpParams.gemm_size1.m(), GLURowTile), 1, 1};
    MiddleCuStage middle(gridMiddle, {GLURowTile, 1, 1}, sync);
    ConsCuStage cons(gridDim2, tileSize, sync2);

    prod.iter = cons.iter = middle.iter = 0;

    CuSyncImpl1 cuSyncHandle1(prod, middle);
    CuSyncImpl2 cuSyncHandle2(middle, cons);
    
    double overlapTime = 0;
    cuSyncHandle1.iter = 0;
    cuSyncHandle1.prod().iter = cuSyncHandle1.cons().iter = 0;
    cuSyncHandle2.iter = 0;
    cuSyncHandle2.prod().iter = cuSyncHandle2.cons().iter = 0;

    // cuSyncHandle2.prod().setTileStatusToWait(cuSyncHandle1.cons().getTileStatusToWait());
    cuSyncHandle1.cons().kernelExecuted_ = cuSyncHandle2.prod().kernelExecuted_;

    result = runCuSyncLLaMA(split_k1, split_k2, mlpParams, cuSyncHandle1, cuSyncHandle2, streams, overlapTime, 1);

    CUDA_CHECK(cudaDeviceSynchronize());
    if (doChecking) {
      result = checkMLPResults(mlpParams);
      if (result != cudaSuccess) {
        return 1;
      }
    }

    result = runCuSyncLLaMA(split_k1, split_k2, mlpParams, cuSyncHandle1, cuSyncHandle2, streams, overlapTime, warmup);
    
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("START-OVERLAPPED:\n");
    
    result = runCuSyncLLaMA(split_k1, split_k2, mlpParams, cuSyncHandle1, cuSyncHandle2, streams, overlapTime, epochs);
    
    CUDA_CHECK(result);
    printf("END-OVERLAPPED:\n");
    
    printf("Average time %lf microseconds\n", overlapTime/(float)epochs);
  }

  return 0;
}
