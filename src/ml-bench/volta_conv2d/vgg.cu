// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <iostream>
#include <fstream>
#include <sstream>

//<OPTIMIZATIONS>
//</OPTIMIZATIONS>
// #define AVOID_CUSTOM_ORDER
// #define AVOID_WAIT_KERNEL


#ifdef TILESYNC
#define NO_ATOMIC_ADD
  // #define REORDER_TILE_LOADS
#endif

#include <cusync/cusync.h>
using namespace cusync;

const uint Opts = 
#ifdef AVOID_CUSTOM_ORDER
  Optimizations::AvoidCustomOrder |
#endif
#ifdef AVOID_WAIT_KERNEL
  Optimizations::AvoidWaitKernel  |
#endif
// #ifdef NO_ATOMIC_ADD
//   Optimizations::NoAtomicAdd      |
// #endif
  Optimizations::NoOptimization;


#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"
#include "cutlass/conv/kernel/cusyncdefault_conv2d_fprop.h"
#include "cutlass/conv/device/cusyncimplicit_gemm_convolution.h"
#include "cutlass/gemm/threadblock/cusync_threadblock_swizzle.h"
#include "cutlass/conv/conv2d_problem_size.h"
#include "cutlass/conv/convolution.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/convolution.h"
#include "cutlass/util/tensor_view_io.h"

#include "helper.h"
#include <time.h>
#include <sys/time.h>

using namespace cutlass::conv;

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


using ElementAccumulator = float;
using ElementComputeEpilogue = cutlass::half_t;
using ElementInputA = cutlass::half_t;
using ElementInputB = ElementInputA;
using ElementOutput = ElementInputB;

using LayoutInputA = cutlass::layout::TensorNHWC;
using LayoutInputB = cutlass::layout::TensorNHWC;
using LayoutOutput = cutlass::layout::TensorNHWC;

using MMAOp = cutlass::arch::OpClassTensorOp;
using SmArch = cutlass::arch::Sm70;

//<eval tiles>
using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 32>;
using WarpShape = cutlass::gemm::GemmShape<32, 32, 32>;
//</eval tiles>

#ifdef ROWSYNC 
  using Sync = RowSync<ThreadblockShape::kM>;
  using FirstStage = CuStage<TransposeXYOrder, NoSync,  Sync, Opts>;
  using Mid1Stage  = CuStage<TransposeXYOrder, Sync, Sync, Opts>;
  using Mid2Stage  = CuStage<TransposeXYOrder, Sync, Sync, Opts>;
  using FinalStage = CuStage<TransposeXYOrder, Sync, NoSync,  Opts>;
#elif defined(TILESYNC)
  using Conv2DSync = Conv2DTileSync<TransposeXYOrder, 3, 3, ThreadblockShape::kM, ThreadblockShape::kN>;
  using Sync = TileSync<TransposeXYOrder, ThreadblockShape::kM, ThreadblockShape::kN>;
  using FirstStage = CuStage<TransposeXYOrder, NoSync,     Sync, Opts>;
  using Mid1Stage  = CuStage<TransposeXYOrder, Conv2DSync, Sync, Opts>;
  using Mid2Stage  = CuStage<TransposeXYOrder, Conv2DSync, Sync, Opts>;
  using FinalStage = CuStage<TransposeXYOrder, Conv2DSync, NoSync, Opts>;
#else
  #error "Unknown Synchronization"
#endif 

using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;

constexpr int NumStages = 2;

using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,
    8,
    ElementAccumulator,
    ElementComputeEpilogue>;

#ifdef STREAM_K
using SwizzleThreadBlock = cutlass::gemm::threadblock::ThreadblockSwizzleStreamK;
#else
using SwizzleThreadBlock = cutlass::gemm::threadblock::CuSyncGemmHorizontalThreadblockSwizzle;
#endif

using Conv2dFpropKernel = 
  typename kernel::DefaultConv2dFprop<ElementInputA, LayoutInputA,
                                              ElementInputB, LayoutInputB,
                                              ElementOutput, LayoutOutput,
                                              ElementAccumulator,
                                              MMAOp,
                                              SmArch,
                                              ThreadblockShape,
                                              WarpShape,
                                              InstructionShape,
                                              EpilogueOp,
                                              SwizzleThreadBlock,
                                              NumStages,
                                              cutlass::arch::OpMultiplyAdd,
                                              cutlass::conv::IteratorAlgorithm::kAnalytic>::Kernel;

using BaselineImplicitGemm = device::ImplicitGemmConvolution<Conv2dFpropKernel>;

using CuSyncSwizzleThreadBlock = cutlass::gemm::threadblock::CuSyncGemmHorizontalThreadblockSwizzle;

using FirstConv2dKernel =
  typename kernel::CuSyncDefaultConv2dFprop<FirstStage,
                                            ElementInputA, LayoutInputA,
                                            ElementInputB, LayoutInputB,
                                            ElementOutput, LayoutOutput,
                                            ElementAccumulator,
                                            MMAOp,
                                            SmArch,
                                            ThreadblockShape,
                                            WarpShape,
                                            InstructionShape,
                                            EpilogueOp,
                                            CuSyncSwizzleThreadBlock,
                                            NumStages,
                                            cutlass::arch::OpMultiplyAdd,
                                            cutlass::conv::IteratorAlgorithm::kAnalytic
                                          >::Kernel;

using Mid1Conv2dKernel =
  typename kernel::CuSyncDefaultConv2dFprop<Mid1Stage,
                                            ElementInputA, LayoutInputA,
                                            ElementInputB, LayoutInputB,
                                            ElementOutput, LayoutOutput,
                                            ElementAccumulator,
                                            MMAOp,
                                            SmArch,
                                            ThreadblockShape,
                                            WarpShape,
                                            InstructionShape,
                                            EpilogueOp,
                                            CuSyncSwizzleThreadBlock,
                                            NumStages,
                                            cutlass::arch::OpMultiplyAdd,
                                            cutlass::conv::IteratorAlgorithm::kAnalytic
                                          >::Kernel;

using Mid2Conv2dKernel =
  typename kernel::CuSyncDefaultConv2dFprop<Mid2Stage,
                                            ElementInputA, LayoutInputA,
                                            ElementInputB, LayoutInputB,
                                            ElementOutput, LayoutOutput,
                                            ElementAccumulator,
                                            MMAOp,
                                            SmArch,
                                            ThreadblockShape,
                                            WarpShape,
                                            InstructionShape,
                                            EpilogueOp,
                                            CuSyncSwizzleThreadBlock,
                                            NumStages,
                                            cutlass::arch::OpMultiplyAdd,
                                            cutlass::conv::IteratorAlgorithm::kAnalytic
                                          >::Kernel;
using FinalConv2dKernel =
  typename kernel::CuSyncDefaultConv2dFprop<FinalStage,
                                            ElementInputA, LayoutInputA,
                                            ElementInputB, LayoutInputB,
                                            ElementOutput, LayoutOutput,
                                            ElementAccumulator,
                                            MMAOp,
                                            SmArch,
                                            ThreadblockShape,
                                            WarpShape,
                                            InstructionShape,
                                            EpilogueOp,
                                            CuSyncSwizzleThreadBlock,
                                            NumStages,
                                            cutlass::arch::OpMultiplyAdd,
                                            cutlass::conv::IteratorAlgorithm::kAnalytic
                                          >::Kernel;
using CuSyncImplicitGemm1 = device::CuSyncImplicitGemmConvolution<FirstStage, FirstConv2dKernel>;
using CuSyncImplicitGemm2 = device::CuSyncImplicitGemmConvolution<Mid1Stage, Mid1Conv2dKernel>;
using CuSyncImplicitGemm3 = device::CuSyncImplicitGemmConvolution<Mid2Stage, Mid2Conv2dKernel>;
using CuSyncImplicitGemm4 = device::CuSyncImplicitGemmConvolution<FinalStage, FinalConv2dKernel>;

//Check for tensor equality
template<typename T>
bool equals(size_t size, T* mat1, T* mat2, float err) {
  for (size_t i = 0; i < size; i++) {
    float e1 = (float)mat1[i];
    float e2 = (float)mat2[i];
    
    float v = err;
    bool ret = true;
    if (abs(e1) < v && abs(e2) < v) {
      // printf("%f , %f at %lu\n", e1, e2, i);
      ret = true;
    } else if (abs(e1) < v) {
      ret = false;
    } else if (abs(e2) < v) {
      ret = false;
    } else {
      float err = abs(e1 - e2)/abs(e1);
      if (err <= v) {
        ret = true;
      } else {
        ret = false;
      }
    }

    if (ret == false) {
      printf("%f != %f at %lu\n", e1, e2, i);
      return false;
    }
  }

  return true;
}

// Command line options parsing
struct Options {
  bool help;
  cutlass::Tensor4DCoord input_size;
  cutlass::Tensor4DCoord filter_size;
  cutlass::Tensor4DCoord padding;
  cutlass::MatrixCoord conv_stride;
  cutlass::MatrixCoord dilation;
  bool reference_check;
  bool measure_performance;
  int iterations;
  bool save_workspace;
  ElementComputeEpilogue alpha;
  ElementComputeEpilogue beta;
  bool benchmark;
  std::string tag;
  int split_k_slices;
  bool rowSyncOrTileSync;

  Options():
    help(false),
    input_size(1, 32, 32, 32),
    filter_size(32, 3, 3, 32),
    padding(1, 1, 1, 1),
    conv_stride(1, 1),
    dilation(1, 1),
    reference_check(false),
    measure_performance(true),
    iterations(20),
    save_workspace(false),
    alpha(1),
    beta(0),
    split_k_slices(0),
    rowSyncOrTileSync(false),
    benchmark(false) { }

  // Verify the problem size is compatible with the CUTLASS Convolution implementation.
  bool valid() {

    //
    // CUTLASS attempts to load 128b vectors of int4b_t elements. Consequently,
    // all pointers, strides, and tensor extents must be divisible by 32 elements.
    //
    int const kAlignment = 32;

    if ((input_size.c() % kAlignment) ||
      (filter_size.n() % kAlignment)) {

      // misaligned tensors
      return false;
    }

    // Invalid padding
    if ((padding.h() != filter_size.h() / 2) ||
      (padding.w() != filter_size.w() / 2)) {

      return false;
    }

    return true;
  }

  /// Updates input and filter sizes
  void update(cutlass::Tensor4DCoord input_size,
              cutlass::Tensor4DCoord filter_size) {

    this->input_size = input_size;
    this->filter_size = filter_size;

    padding.n() = filter_size.h() / 2;
    padding.h() = filter_size.h() / 2;
    padding.w() = filter_size.w() / 2;
    padding.c() = filter_size.w() / 2;
  }

  // Parses the command line
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
    }

    if (cmd.check_cmd_line_flag("ref-check")) {
      reference_check = true;
    }

    if (cmd.check_cmd_line_flag("perf-check")) {
      measure_performance = true;
    }

    if (cmd.check_cmd_line_flag("save-workspace")) {
      save_workspace = true;
    }

    if (cmd.check_cmd_line_flag("benchmark")) {
      benchmark = true;
    }

    cmd.get_cmd_line_argument("split_k_slices", split_k_slices);
  
    cmd.get_cmd_line_argument("n", input_size.n());
    cmd.get_cmd_line_argument("h", input_size.h());
    cmd.get_cmd_line_argument("w", input_size.w());
    cmd.get_cmd_line_argument("c", input_size.c());

    cmd.get_cmd_line_argument("k", filter_size.n());
    cmd.get_cmd_line_argument("r", filter_size.h());
    cmd.get_cmd_line_argument("s", filter_size.w());
    filter_size.c() = input_size.c(); 

    cmd.get_cmd_line_argument("alpha", alpha);
    cmd.get_cmd_line_argument("beta", beta);
    
    cmd.get_cmd_line_argument("iterations", iterations);
    cmd.get_cmd_line_argument("tag", tag);

    if (filter_size.h() == 3 && filter_size.w() == 3) {
      padding = {1, 1, 1, 1};
    }
    else {
      filter_size.h() = 1;
      filter_size.w() = 1;
      padding = {0, 0, 0, 0};
    }
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "09_turing_tensorop_conv2dfprop example\n\n"
      << "  This example uses Turing's Tensor Core operators on int4 data types to compute\n"
      << "  forward convolution on tensors of layout NHWC.\n\n"
      << "Options:\n\n"
      << "  --help               If specified, displays this usage statement.\n\n"
      << "  --n=<int>            Input tensor extent N\n"
      << "  --h=<int>            Input tensor extent H\n"
      << "  --w=<int>            Input tensor extent W\n"
      << "  --c=<int>            Input tensor extent C\n"
      << "  --k=<int>            Filter extent K\n"
      << "  --r=<int>            Filter extent R\n"
      << "  --s=<int>            Filter extent S\n\n"
      << "  --alpha=<float>      Epilogue scalar alpha\n"
      << "  --beta=<float>       Epilogue scalar beta\n\n"
      << "  --ref-check          If set (true), reference check on the host is computed\n"
      << "  --perf-check         If set (true), performance is measured.\n"
      << "  --benchmark          If set (true), performance benchmarking on several layers and batch-size.\n"
      << "  --iterations=<int>   Number of profiling iterations to perform.\n"
      << "  --save-workspace     If set, workspace is written to a text file.\n"
      << "  --tag=<string>       String to replicate across the first column in the results table\n"
      << "  --split_k_slices=<int> ";

    out << "\n\nExamples:\n\n"
      << "$ ./examples/09_turing_tensorop_conv2dfprop/09_turing_tensorop_conv2dfprop  --n=32 --h=224 --w=224 --c=128 --k=256 --r=1 --s=1\n\n"
      << "$ ./examples/09_turing_tensorop_conv2dfprop/09_turing_tensorop_conv2dfprop  --n=1 --h=224 --w=224 --c=32 --k=32 --r=3 --s=3 --ref-check\n\n";

    return out;
  }
  
  /// Computes the output tensor size (NPQK)
  cutlass::Tensor4DCoord output_size() const {
    return cutlass::Tensor4DCoord(
      input_size.n(),
      (input_size.h() + padding.n() + padding.h() - filter_size.h()) / conv_stride.row() + 1,
      (input_size.w() + padding.w() + padding.c() - filter_size.w()) / conv_stride.column() + 1,
      filter_size.n());
  }

  /// Compute performance in GFLOP/s
  double gflops(double runtime_s) const {

    // Number of multiply-adds = NPQK * CRS
    int64_t fmas = output_size().product() * int64_t(filter_size.h() * filter_size.w() * filter_size.c());
    
    // Two flops per multiply-add
    return 2.0 * double(fmas) / double(1.0e9) / runtime_s;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

struct Result {
  double runtime_ms;
  double gflops;
  cutlass::Status status;
  cutlass::Status reference_check;
  cudaError_t error;

  Result(): 
    runtime_ms(0), 
    gflops(0),
    status(cutlass::Status::kSuccess),
    reference_check(cutlass::Status::kInvalid),
    error(cudaSuccess) { }

  static std::ostream & print_header(std::ostream &out, Options const &options) {

    if (!options.tag.empty()) {
      out << "Name,";
    }

    out << "Layer,N,H,W,C,K,R,S,Runtime,GFLOPs";

    return out;
  }

  std::ostream & print(std::ostream &out, int idx, Options const &options) {

    if (!options.tag.empty()) {
      out << options.tag << ",";
    }

    out 
      << "conv_" << idx << ","
      << options.input_size.n() << ","
      << options.input_size.h() << ","
      << options.input_size.w() << ","
      << options.input_size.c() << ","
      << options.filter_size.n() << ","
      << options.filter_size.h() << ","
      << options.filter_size.w() << ","
      << runtime_ms << ","
      << gflops;

    return out;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template<typename TensorA, typename TensorB, typename TensorC>
void runBaseline(cutlass::conv::Conv2dProblemSize problem_size, 
                 const Options& options, cudaStream_t* streams, 
                 TensorA& tensor_x, TensorB& tensor_w1, TensorB& tensor_w2, TensorB& tensor_w3, TensorB& tensor_w4, 
                 TensorC& tensor_y1, TensorC& tensor_y2, TensorC& tensor_y3, TensorC& tensor_y4,
                 double& elapsedTime, double& conv1Time, double& conv2Time, 
                 double& conv3Time, double& conv4Time, int runs) {
  // Construct ImplicitGemm::Argument structure with conv2d 
  // problem size, data pointers, and epilogue values
  typename BaselineImplicitGemm::Arguments args1{
    problem_size,
    tensor_x.device_ref(),
    tensor_w1.device_ref(),
    tensor_y1.device_ref(),
    tensor_y1.device_ref(),
    {options.alpha, options.beta},
  };

  BaselineImplicitGemm implicit_gemm_op1;
  size_t workspace_size1 = implicit_gemm_op1.get_workspace_size(args1);
  cutlass::device_memory::allocation<uint8_t> workspace1(workspace_size1);
  auto status = implicit_gemm_op1.can_implement(args1);
  CUTLASS_CHECK(status);
  status = implicit_gemm_op1.initialize(args1, workspace1.get());
  CUTLASS_CHECK(status);

  typename BaselineImplicitGemm::Arguments args2{
    problem_size,
    tensor_y1.device_ref(),
    tensor_w2.device_ref(),
    tensor_y2.device_ref(),
    tensor_y2.device_ref(),
    {options.alpha, options.beta},
  };

  BaselineImplicitGemm implicit_gemm_op2;
  size_t workspace_size2 = implicit_gemm_op2.get_workspace_size(args2);
  cutlass::device_memory::allocation<uint8_t> workspace2(workspace_size2);
  status = implicit_gemm_op2.can_implement(args2);
  CUTLASS_CHECK(status);
  status = implicit_gemm_op2.initialize(args2, workspace2.get());
  CUTLASS_CHECK(status);

  typename BaselineImplicitGemm::Arguments args3{
    problem_size,
    tensor_y2.device_ref(),
    tensor_w3.device_ref(),
    tensor_y3.device_ref(),
    tensor_y3.device_ref(),
    {options.alpha, options.beta},
  };

  BaselineImplicitGemm implicit_gemm_op3;
  size_t workspace_size3 = implicit_gemm_op2.get_workspace_size(args3);
  cutlass::device_memory::allocation<uint8_t> workspace3(workspace_size3);
  status = implicit_gemm_op3.can_implement(args3);
  CUTLASS_CHECK(status);
  status = implicit_gemm_op3.initialize(args3, workspace3.get());
  CUTLASS_CHECK(status);

  typename BaselineImplicitGemm::Arguments args4{
    problem_size,
    tensor_y3.device_ref(),
    tensor_w4.device_ref(),
    tensor_y4.device_ref(),
    tensor_y4.device_ref(),
    {options.alpha, options.beta},
  };

  BaselineImplicitGemm implicit_gemm_op4;
  size_t workspace_size4 = implicit_gemm_op4.get_workspace_size(args4);
  cutlass::device_memory::allocation<uint8_t> workspace4(workspace_size4);
  status = implicit_gemm_op4.can_implement(args4);
  CUTLASS_CHECK(status);
  status = implicit_gemm_op4.initialize(args4, workspace4.get());
  CUTLASS_CHECK(status);

  for (int i = 0; i < runs; i++) {
    double start = getCurrentTime();
    auto status = implicit_gemm_op1(args1, workspace1.get(), streams[0]);

    CUTLASS_CHECK(status);
    cudaDeviceSynchronize();
    double middle1 = getCurrentTime();
    conv1Time += middle1 - start;
    status = implicit_gemm_op2(args2, workspace2.get(), streams[0]);

    CUTLASS_CHECK(status);
    cudaDeviceSynchronize();
    double middle2 = getCurrentTime();
    conv2Time += middle2 - middle1;

    status = implicit_gemm_op3(args3, workspace3.get(), streams[0]);

    CUTLASS_CHECK(status);
    cudaDeviceSynchronize();
    double middle3 = getCurrentTime();
    conv3Time += middle3 - middle2;

    status = implicit_gemm_op2(args3, workspace3.get(), streams[0]);

    CUTLASS_CHECK(status);
    cudaDeviceSynchronize();
    double end = getCurrentTime();
    conv4Time += end - middle3;

    elapsedTime += end - start;
    printf("{\"Total\": %lf, \"conv1\": %lf, \"conv2\": %lf, \"conv3\": %lf, \"conv4\": %lf}\n",
           end-start, middle1-start, middle2-middle1, middle3-middle2, end-middle3);
  }
}

template<typename TensorA, typename TensorB, typename TensorC>
void runConvolution(cutlass::conv::Conv2dProblemSize problem_size, 
                    const Options& options, cudaStream_t* streams, 
                    FirstStage& stage1, Mid1Stage& stage2, Mid2Stage& stage3, FinalStage& stage4, 
                    TensorA& tensor_x, TensorB& tensor_w1, TensorB& tensor_w2, TensorB& tensor_w3, TensorB& tensor_w4, 
                    TensorC& tensor_y1, TensorC& tensor_y2, TensorC& tensor_y3, TensorC& tensor_y4,
                    double& elapsedTime, double& conv1Time, double& conv2Time, int runs) {
  // Construct ImplicitGemm::Argument structure with conv2d
  // problem size, data pointers, and epilogue values
  typename CuSyncImplicitGemm1::Arguments args1{
    stage1,
    problem_size,
    tensor_x.device_ref(),
    tensor_w1.device_ref(),
    tensor_y1.device_ref(),
    tensor_y1.device_ref(),
    {options.alpha, options.beta},
  };
  
  CuSyncImplicitGemm1 implicit_gemm_op1;
  size_t workspace_size1 = implicit_gemm_op1.get_workspace_size(args1);
  cutlass::device_memory::allocation<uint8_t> workspace1(workspace_size1);
  auto status = implicit_gemm_op1.can_implement(args1);
  CUTLASS_CHECK(status);
  status = implicit_gemm_op1.initialize(args1, workspace1.get());
  CUTLASS_CHECK(status);

  typename CuSyncImplicitGemm2::Arguments args2{
    stage2,
    problem_size,
    tensor_y1.device_ref(),
    tensor_w2.device_ref(),
    tensor_y2.device_ref(),
    tensor_y2.device_ref(),
    {options.alpha, options.beta},
  };

  CuSyncImplicitGemm2 implicit_gemm_op2;
  size_t workspace_size2 = implicit_gemm_op2.get_workspace_size(args2);
  cutlass::device_memory::allocation<uint8_t> workspace2(workspace_size2);
  status = implicit_gemm_op2.can_implement(args2);
  CUTLASS_CHECK(status);
  status = implicit_gemm_op2.initialize(args2, workspace2.get());
  CUTLASS_CHECK(status);

  typename CuSyncImplicitGemm3::Arguments args3{
    stage3,
    problem_size,
    tensor_y2.device_ref(),
    tensor_w3.device_ref(),
    tensor_y3.device_ref(),
    tensor_y3.device_ref(),
    {options.alpha, options.beta},
  };

  CuSyncImplicitGemm3 implicit_gemm_op3;
  size_t workspace_size3 = implicit_gemm_op3.get_workspace_size(args3);
  cutlass::device_memory::allocation<uint8_t> workspace3(workspace_size3);
  status = implicit_gemm_op3.can_implement(args3);
  CUTLASS_CHECK(status);
  status = implicit_gemm_op3.initialize(args3, workspace3.get());
  CUTLASS_CHECK(status);

  typename CuSyncImplicitGemm4::Arguments args4{
    stage4,
    problem_size,
    tensor_y3.device_ref(),
    tensor_w4.device_ref(),
    tensor_y4.device_ref(),
    tensor_y4.device_ref(),
    {options.alpha, options.beta},
  };

  CuSyncImplicitGemm4 implicit_gemm_op4;
  size_t workspace_size4 = implicit_gemm_op4.get_workspace_size(args4);
  cutlass::device_memory::allocation<uint8_t> workspace4(workspace_size4);
  status = implicit_gemm_op4.can_implement(args4);
  CUTLASS_CHECK(status);
  status = implicit_gemm_op4.initialize(args4, workspace4.get());
  CUTLASS_CHECK(status);

  for (int i = 0; i < runs; i++) {
    double start = getCurrentTime();
    auto status = implicit_gemm_op1(streams[0]);
  //  waitKernel<<<1,1,0,streams[1]>>>((uint*)&kernelExecuted[0], args1.overlap_handle.iter);
    // CUDA_CHECK(cudaDeviceSynchronize());
    // CUTLASS_CHECK(status);
#ifndef AVOID_WAIT_KERNEL
    stage1.invokeWaitKernel(streams[1]);
#endif
    // double middle1 = getCurrentTime();
    // conv1Time += middle1 - start;
    status = implicit_gemm_op2(streams[1]);
    CUTLASS_CHECK(status);

#ifndef AVOID_WAIT_KERNEL
    stage2.invokeWaitKernel(streams[2]);
#endif
    status = implicit_gemm_op3(streams[2]);

#ifndef AVOID_WAIT_KERNEL
    stage3.invokeWaitKernel(streams[3]);
#endif
    status = implicit_gemm_op4(streams[3]);

    cudaDeviceSynchronize();
    
    double end = getCurrentTime();
    // conv2Time += end - middle1;
    elapsedTime += end - start;
    printf("{\"Total\": %lf, \"conv1\": %lf, \"conv2\": %lf}\n",end-start,conv1Time,conv2Time);
    stage1.incrementIter();
    stage2.incrementIter();
    stage3.incrementIter();
    stage4.incrementIter();
    
    implicit_gemm_op1.params_.custage.incrementIter();
    implicit_gemm_op2.params_.custage.incrementIter();
    implicit_gemm_op3.params_.custage.incrementIter();
    implicit_gemm_op4.params_.custage.incrementIter();
  }
}

/// Runs one benchmark
Result profile_convolution(Options const &options) {
  // Check the problem size is supported or not 

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

  Result result;

  //
  // Allocate host-device tensors using the CUTLASS Utilities.
  //

  cutlass::HostTensor<ElementInputA, LayoutInputA> tensor_x(options.input_size);
  
  cutlass::HostTensor<ElementInputB, LayoutInputB> tensor_w1(options.filter_size);
  cutlass::HostTensor<ElementInputB, LayoutInputB> tensor_w2(options.filter_size);
  cutlass::HostTensor<ElementInputB, LayoutInputB> tensor_w3(options.filter_size);
  cutlass::HostTensor<ElementInputB, LayoutInputB> tensor_w4(options.filter_size);

  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_y1(options.output_size());
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_y2(options.output_size());
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_y3(options.output_size());
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_y4(options.output_size());

  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_ref_y1(options.output_size());
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_ref_y2(options.output_size());

  //
  // Initialize tensors
  //

  cutlass::reference::host::TensorFillRandomUniform(
      tensor_x.host_view(), ElementInputA(1.0f));
  cutlass::reference::host::TensorFillRandomUniform(
      tensor_w1.host_view(), ElementInputB(1.0f));
  cutlass::reference::host::TensorFillRandomUniform(
      tensor_y1.host_view(), ElementOutput(1.0f));
  // Fill tensor A on host with uniform-distribution random data
  // cutlass::reference::host::TensorFillRandomUniform(
  //     tensor_x.host_view(),
  //     1,
  //     ElementInputA(1),
  //     ElementInputA(-1),
  //     0);

  // // Fill tensor B on host with uniform-distribution random data
  // cutlass::reference::host::TensorFillRandomUniform(
  //     tensor_w1.host_view(),
  //     1,
  //     ElementInputB(1),
  //     ElementInputB(-1),
  //     0);
  
  // // Fill tensor B on host with uniform-distribution random data
  // cutlass::reference::host::TensorFillRandomUniform(
  //   tensor_w2.host_view(),
  //   1,
  //   ElementInputB(1),
  //   ElementInputB(-1),
  //   0);

  // Fill tensor C on host with zeros
  cutlass::reference::host::TensorFill(
    tensor_y1.host_view());
  cutlass::reference::host::TensorFill(
    tensor_y2.host_view());

  // Fill tensor C for reference on host with zeros
  cutlass::reference::host::TensorFill(
    tensor_ref_y1.host_view());
  cutlass::reference::host::TensorFill(
    tensor_ref_y2.host_view());
  
  // Copy data from host to GPU
  tensor_x.sync_device();
  tensor_w1.sync_device();
  tensor_w2.sync_device();
  tensor_y1.sync_device();
  tensor_y2.sync_device();
  tensor_ref_y1.sync_device();
  tensor_ref_y2.sync_device();

  // mode (kCrossCorrelation or kConvolution)
  cutlass::conv::Mode mode = cutlass::conv::Mode::kCrossCorrelation;

  // Construct Conv2dProblemSize with user defined output size
  cutlass::conv::Conv2dProblemSize problem_size(
      options.input_size,
      options.filter_size,
      options.padding,
      options.conv_stride,
      options.dilation,
      options.output_size(),
      mode,
      options.split_k_slices);
  //
  // Optional reference check
  //

  int warmup = 5;
  int epochs = 20;
  double elapsedTime = 0;
  double conv1Time = 0, conv2Time = 0, conv3Time = 0, conv4Time = 0;
  
  runBaseline(problem_size, options, &streams[0], 
              tensor_x, tensor_w1, tensor_w2, tensor_w3, tensor_w4, 
              tensor_y1, tensor_y2, tensor_y3, tensor_y4,
              elapsedTime, conv1Time, conv2Time, conv3Time, conv4Time, 1);

  if (options.reference_check) {
    std::cout << "Verification on host...\n";
    
    using NumericConverter = cutlass::NumericConverter<ElementOutput, ElementComputeEpilogue>;
    auto HostConv2d = cutlass::reference::host::Conv2dFprop<ElementInputA, LayoutInputA,
                                                       ElementInputB, LayoutInputB,
                                                       ElementOutput, LayoutOutput,
                                                       ElementComputeEpilogue,
                                                       ElementAccumulator,
                                                       ElementOutput,
                                                       NumericConverter>;
    HostConv2d(problem_size, tensor_x.host_ref(), tensor_w1.host_ref(),
               tensor_y1.host_ref(), tensor_ref_y1.host_ref(),
               options.alpha, options.beta);
    HostConv2d(problem_size, tensor_y1.host_ref(), tensor_w2.host_ref(),
               tensor_y2.host_ref(), tensor_ref_y2.host_ref(),
               options.alpha, options.beta);

    // Check if output from CUTLASS kernel and reference kernel are equal or not
    tensor_y1.sync_host(); tensor_y2.sync_host();
    bool passed = equals(tensor_y1.size(), tensor_ref_y1.host_data(), tensor_y1.host_data(), 1e-2);
    
    if (!passed) {
      result.reference_check = cutlass::Status::kErrorInternal;
      std::cout << "ERROR - First conv incorrect.\n";
      return result;
    } else {
      result.reference_check = cutlass::Status::kSuccess;
      std::cout << "First Passed.\n";
    }

    passed = equals(tensor_y2.size(), tensor_ref_y2.host_data(), tensor_y2.host_data(), 1e-2);
    
    if (!passed) {
      result.reference_check = cutlass::Status::kErrorInternal;
      std::cout << "ERROR - second conv incorrect.\n";
      return result;
    } else {
      result.reference_check = cutlass::Status::kSuccess;
      std::cout << "Second Passed.\n";
    }
  }

  runBaseline(problem_size, options, &streams[0], 
              tensor_x, tensor_w1, tensor_w2, tensor_w3, tensor_w4, 
              tensor_y1, tensor_y2, tensor_y3, tensor_y4,
              elapsedTime, conv1Time, conv2Time, conv3Time, conv4Time, warmup);

  elapsedTime = 0;
  conv1Time = 0;
  conv2Time = 0;
  printf("START-BASELINE:\n");
  runBaseline(problem_size, options, &streams[0], 
              tensor_x, tensor_w1, tensor_w2, tensor_w3, tensor_w4, 
              tensor_y1, tensor_y2, tensor_y3, tensor_y4,
              elapsedTime, conv1Time, conv2Time, conv3Time, conv4Time, epochs);
  
  printf("END-BASELINE: {Total: %lf, Conv1: %lf, Conv2: %lf} micro seconds\n", elapsedTime/epochs, conv1Time/epochs, conv2Time/epochs);


  auto gemm_problem_size = cutlass::conv::implicit_gemm_problem_size(cutlass::conv::Operator::kFprop, problem_size);
  printf("gemm problem size: {%d, %d, %d}\n", gemm_problem_size.m(), gemm_problem_size.n(), gemm_problem_size.k());
  cutlass::gemm::GemmCoord tileSize = {ThreadblockShape::kM, ThreadblockShape::kN, 1};
  cutlass::gemm::GemmCoord gridDim = CuSyncSwizzleThreadBlock().get_tiled_shape(gemm_problem_size, tileSize, options.split_k_slices);
  printf("gridDim: {%d, %d, %d}\n", gridDim.m(), gridDim.n(), gridDim.k());

  
#if defined(ROWSYNC)
  Sync sync1(gridDim.n());
  Sync sync2(gridDim.n());
#elif defined(TILESYNC)
  Sync sync1;
  Conv2DSync sync2;
#else
  #error "Unkown Policy"
#endif

  auto grid_shape = CuSyncSwizzleThreadBlock().get_grid_shape(gridDim);
  printf("gridShape: {%d, %d, %d}\n", grid_shape.x, grid_shape.y, grid_shape.z);
  FirstStage prod(grid_shape, {1,1,1}, NoSync(), sync1);
  Mid1Stage  mid1(grid_shape, {1,1,1}, sync2, sync1);
  Mid2Stage  mid2(grid_shape, {1,1,1}, sync2, sync1);
  FinalStage cons(grid_shape, {1,1,1}, sync2, NoSync());

  CuSync::setProducerConsumerPair(prod, mid1);
  CuSync::setProducerConsumerPair(mid1, mid2);
  CuSync::setProducerConsumerPair(mid2, cons);

  cutlass::reference::host::TensorFill(
    tensor_y1.host_view());
  cutlass::reference::host::TensorFill(
    tensor_y2.host_view());
  cutlass::reference::host::TensorFill(
    tensor_y3.host_view());
  cutlass::reference::host::TensorFill(
    tensor_y4.host_view());

  tensor_y1.sync_device();
  tensor_y2.sync_device();
  tensor_y3.sync_device();
  tensor_y4.sync_device();
  
  
#ifndef STREAM_K
  runConvolution(problem_size, options, &streams[0], 
                 prod, mid1, mid2, cons, 
                 tensor_x, tensor_w1, tensor_w2, tensor_w3, tensor_w4, 
                 tensor_y1, tensor_y2, tensor_y3, tensor_y4,
                 elapsedTime, conv1Time, conv2Time, 1);

  if (options.reference_check) {
    // Check if output from CUTLASS kernel and reference kernel are equal or not
    tensor_y1.sync_host(); tensor_y2.sync_host();
    bool passed = equals(tensor_y1.size(), tensor_ref_y1.host_data(), tensor_y1.host_data(), 1e-2);

    if (!passed) {
      result.reference_check = cutlass::Status::kErrorInternal;
      std::cout << "ERROR - First conv incorrect.\n";
      return result;
    } else {
      result.reference_check = cutlass::Status::kSuccess;
      std::cout << "First Passed.\n";
    }

    passed = equals(tensor_y1.size(), tensor_ref_y2.host_data(), tensor_y2.host_data(), 1e-2);

    if (!passed) {
      result.reference_check = cutlass::Status::kErrorInternal;
      std::cout << "ERROR - second conv incorrect.\n";
      return result;
    } else {
      result.reference_check = cutlass::Status::kSuccess;
      std::cout << "Second Passed.\n";
    }
  }
  runConvolution(problem_size, options, &streams[0], 
                 prod, mid1, mid2, cons, 
                 tensor_x, tensor_w1, tensor_w2, tensor_w3, tensor_w4, 
                 tensor_y1, tensor_y2, tensor_y3, tensor_y4,
                 elapsedTime, conv1Time, conv2Time, warmup);
  elapsedTime = 0;
  conv1Time = 0;
  conv2Time = 0;
  printf("START-OVERLAP:\n");
  runConvolution(problem_size, options, &streams[0], 
                 prod, mid1, mid2, cons, 
                 tensor_x, tensor_w1, tensor_w2, tensor_w3, tensor_w4, 
                 tensor_y1, tensor_y2, tensor_y3, tensor_y4,
                 elapsedTime, conv1Time, conv2Time, epochs);
  printf("END-OVERLAP {Total: %lf, Conv1: %lf, Conv2: %lf} micro seconds\n", elapsedTime/epochs, conv1Time/epochs, conv2Time/epochs);
#endif

  return result;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char const **args) {

  // Turing Tensor Core operations exposed with mma.sync are first available in CUDA 10.2.
  //
  // CUTLASS must be compiled with CUDA 10.2 Toolkit to run these examples.
  if (!(__CUDACC_VER_MAJOR__ > 10 || (__CUDACC_VER_MAJOR__ == 10 && __CUDACC_VER_MINOR__ >= 2))) {
    std::cerr << "Turing Tensor Core operations must be compiled with CUDA 10.2 Toolkit or later." << std::endl;
    return 0;
  }

  cudaDeviceProp props;
  CUDA_CHECK(cudaGetDeviceProperties(&props, 0));

  // if (!(props.major > 7 || (props.major == 7 && props.minor >= 5))) {
  //   std::cerr << "Turing Tensor Ops must be run on a machine with compute capability at least 75."
  //             << std::endl;
  //   return 0;
  // }

  Options options;
  
  options.parse(argc, args);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }
  // Execute one problem size
  if (!options.valid()) {
    std::cerr << "Invalid problem." << std::endl;
    return -1;
  }

  Result result = profile_convolution(options);

  Result::print_header(std::cout, options) << std::endl;
  result.print(std::cout, 1, options) << std::endl;

  return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////



