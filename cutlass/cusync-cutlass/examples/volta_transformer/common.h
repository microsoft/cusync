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

/**
This example shows how to run matrix multiplication kernels using functions and data structures
provided by CUTLASS using tensor cores; which we run on a NVIDIA Volta GPU.

Writing a single high performance matrix multiplication kernel is hard but do-able. Whereas writing
high performance kernels at scale which works for multiple problem sizes with good abstractions is
really hard. CUTLASS solves this problem by providing simplified abstractions to compose
multiple sections of gemm kernel. When used properly, the kernels can hit peak performance of GPU
easily.

CUTLASS divides a kernel into hierarchical composable sections. Which means, at each thread, warp
and thread-block level, they compute on their own tile-size with higher level of tile sizes being
composed from lower level ones. Multiple thread-tiles (tile size each thread computes) can be used
to form warp-tiles (tile size each warp computes) and multiple warp tiles can be used to compute
threadblock-tile (tile size computed by a threadblock).

In thie example, we split variable initialization into
1. Setting up data properties : describes how matrices are laid out in the memory and how the kernel
can view them (logical to physical mapping)
2. Setting up computation properties : describes how the above set matrices will be used to compute
output of matrix multiplication.

First, we setup the data types of matrices A, B, C and D along with alpha, beta as the equation for
GEMM is D = alpha * A * B + beta * C. In CUTLASS, the kernels first compute A * B and leaves the
rest of the computation to end of the kernel as alpha * X + beta * C is a simple element-wise
operation on X (A * B) and C. We call this as epilogue of kernel. Hence, we setup data types for
alpha and beta to be equal to ElementComputeEpilogue = float. As we want to MMA instructions on
Volta and they support only half-precision floating point (fp16 or half), we use data type for
elements in input matrix A and B as cutlass::half_t. Volta also supports accumulation of partial dot
product to fp32, which can store wider range of numbers, we use it as data type of output matrix
elements and accumulation. We convey this to CUTLASS kernel by initializing template variables
ElementAccumulator (float), ElementComputeEpilogue (float), ElementInputA (cutlass::half_t),
ElementInputB (cutlass::half_t), ElementOutput (float). Communicating just the data type is not
enough. As the data is laid out linearly in memory, we have to convey the layout of matrices. We do
that by initializing template variable LayoutInputA to column major cutlass variable, LayoutInputB
to row major and LayoutOutput to row major. Next, we setup rules to comptue alpha * X + beta * C
which is called epilogue of the kernel. We initialize template variable EpilogueOp, which takes the
data type of output ElementOutput (int32_t), the number of elements per vector memory access (16),
data type of accumulator (int32_t) and data type of computation of linear combination (alpha * X +
beta * C).

Now that we setup the properties of data, we have to setup properties of computation.

Second, we create template variables of tile sizes for thread-block, warp and mma-op to 128x128x32,
64x64x32, 8x8x4 (MxNxK) respectively. When passed to instantiate CUTLASS GEMM kernel, it internally
deduce the amount of threads needed per thread-block, amount of shared memory, storing data in
bank-conflict free manner, and ton of other variables required to compose, intialize and launch a
high performance GEMM kernel. This is the beauty of CUTLASS, it relieves developer from
understanding and coding complicated hardware optimizations which can easily go wrong.

CUTLASS also supports multiple MMA pipelines in a CTA. What are MMA pipelines? MMA pipelines
constitute the whole process of loading input data from global memory to shared memory, loading data
from shared memory to registers, doing matrix multiplication, store to global memory. The below flow
sequence shows a typical mma pipeline.

matrix in global memory -> registers -> tile in shared memory -> registers -> mma -> registers ->
output to global memory

The problem with single pipeline is, each stage is synchronous which means, each stage has to wait
until the previous finished executing. There are stages in the pipeline which do not have fixed
latency, for example, the loads from global memory and shared memory. Therefore, we can add one more
pipeline with a phase shift in mma kernel to hide latency from global and shared memory loads.
Finally, the pipeline in a kernel looks like

(1) matrix in global memory -> (2) registers -> (3) tile in shared memory -> (4) registers -> (5)
mma -> (6) registers -> (7) output to global memory (1) <null> -> (2) <null> -> (3) matrix in global
memory -> (4) registers -> (5) tile in shared memory -> (6) registers -> (7) mma -> (8) registers ->
(9) output to global memory

This way, you can hide the second global memoroy load latency by doing computation on already loaded
input data.

There are few more template variables initialized such as, which threadblock tile of output matrix
is done which threadblock launched on an SM, CUDA SM architecture of GPU you want to run on.

These are all put together to create a template variable which describes CUTLASS GEMM kernel using
cutlass::gemm::device::Gemm template.

The next step is to intialize physical data, instantiate and initialize CUTLASS kernel and run it.
We use CUTLASS utilities to initialize, fill, compare matrices as they are simple and doesn't come
in the way of learning CUTLASS.

Once all the matrices are initialized and filled with data, create arguments tuple to launch CUTLASS
kernel which takes problem size (M = 5120, N = 4096 and K = 4096), matrices, alpha, beta and the
important one, split k-dimension factor. Along with that, we query CUTLASS if any scratch-space
memory required by the kernel we instantiated. If yes, we create it and pass it along with other
arguments created to intialize CUTLASS kernel then, the kernel is launched.

In this example, we later on launch a reference gemm kernel (from CUTLASS utilities) to compare if
the output from CUTLASS kernel is same as reference GEMM kernel.
*/

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

#include <time.h>
#include <sys/time.h>
#include <cublas_v2.h>

#define DIVUP(x, y) (((x) + (y) - 1)/(y))

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

#define CUBLASCHECK(cmd) do {                       \
  cublasStatus_t e = cmd;                           \
  if (e != CUBLAS_STATUS_SUCCESS) {                 \
    printf("Failed: CUBLAS error %s: %d '%d'\n",    \
           __FILE__, __LINE__, cmd);                \
    assert(false);                                  \
  }                                                 \
} while(0)                                        

template<typename T, typename AT>
__global__ void ref_cudamatmul(uint32_t M, uint32_t N, uint32_t K,
                                           T* A, T* B, T* C) {
    int ROW = blockIdx.y*blockDim.y+threadIdx.y;
    int COL = blockIdx.x*blockDim.x+threadIdx.x;

    AT tmpSum = 0;

    if (ROW < M && COL < N) {
        // each thread computes one element of the block sub-matrix
        for (uint32_t i = 0; i < K; i++) {
            tmpSum += ((AT)A[ROW * K + i]) * ((AT)B[i * N + COL]);
        }

        C[ROW * N + COL] = (T)tmpSum;
    }
}

template<typename T, typename AT>
void ref_matmul(uint32_t M, uint32_t N, uint32_t K, T* mat1, T* mat2, T* host_res) {
  T* dev_refC = NULL;
  CUDA_CHECK(cudaMalloc(&dev_refC, sizeof(T)*M*N));
  dim3 block = {32, 32, 1};
  dim3 grid = {N/block.y + 1, M/block.x + 1, 1};
  ref_cudamatmul<T,AT><<<grid, block>>>(M, N, K, mat1, mat2, dev_refC);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(host_res, dev_refC, sizeof(T)*M*N, cudaMemcpyDeviceToHost));
}

template<typename T, typename AT>
void ref_cpumatmul(uint32_t M, uint32_t N, uint32_t K, T* mat1, T* mat2, T* res)
{
  uint32_t i, j, k;
    for (i = 0; i < M; i++) {
      #pragma omp parallel for
      for (j = 0; j < N; j++) {
          AT accum = 0;
          for (k = 0; k < K; k++)
              accum += ((float)mat1[i*K + k]) * ((float)mat2[k*N + j]);
          res[i*N + j] = T(accum);
      }
    }
}

template<typename T>
bool equals(size_t size, T* mat1, T* mat2, float err) {
  bool eq = true;
  for (size_t i = 0; i < size; i++) {
    float e1 = (float)mat1[i];
    float e2 = (float)mat2[i];
    
    float v = err;
    bool ret = true;
    printf("%f , %f at %lu\n", e1, e2, i);
    if (abs(e1) < v && abs(e2) < v) {
      
      ret = true;
    } else if (abs(e1) < v) {
      ret = false;
    } else if (abs(e2) < v) {
      ret = false;
    } else {
      float err = abs(e1 - e2)/abs(e1);
      if (err <= v) {
        ret = true;
      // printf("243: %f , %f at %lu\n", e1, e2, i);
      } else {
        ret = false;
      }
    }

    if (ret == false) {
      // printf("%f != %f at %lu\n", e1, e2, i);
      eq = false;
    }
  }
  return eq;
  return true;
}

template<typename T>
__global__ void printKernel(size_t sz, T* data) {
  if (threadIdx.x == 0) {
    for (size_t i = 65536; i < sz; i++) {
      printf("%f at %lu \n", (float)data[i], i);
    }
  }
}

template<class T>
void memset_value(T*f, T v, size_t nelems) 
{
  T* h_buff = (T*)malloc(sizeof(T)*nelems);
  assert(h_buff != nullptr);
  for (uint64_t i = 0; i < nelems; i++) {
    h_buff[i] = v;
  }

  CUDA_CHECK(cudaMemcpy(f, h_buff, sizeof(T)*nelems, cudaMemcpyHostToDevice));
  free(h_buff);
}

template<class T>
void memset_random2(T*f, T v1, T v2, size_t nelems)
{
  // T* h_buff = (T*)malloc(sizeof(T)*nelems);
  assert(f != nullptr);
  for (uint64_t i = 0; i < nelems; i++) {
    if (rand()%2 == 0)
      f[i] = v1;
    else
      f[i] = v2;
    // printf("%f\n", (float)f[i]);
  }

  // CUDA_CHECK(cudaMemcpy(f, h_buff, sizeof(T)*nelems, cudaMemcpyHostToDevice));
  // free(h_buff);
}

template<class T>
void memset_random(T*f, int numVals, T* values, size_t nelems)
{
  // T* h_buff = (T*)malloc(sizeof(T)*nelems);
  assert(f != nullptr);
  for (uint64_t i = 0; i < nelems; i++) {
    f[i] =  values[rand()%numVals];
  }

  // CUDA_CHECK(cudaMemcpy(f, h_buff, sizeof(T)*nelems, cudaMemcpyHostToDevice));
  // free(h_buff);
}

__global__ void init_curand_states(curandState* states, size_t num_states)
{
  int thread_id = blockIdx.x*blockDim.x + threadIdx.x;
  if (thread_id < num_states)
    curand_init(thread_id, threadIdx.x, 0, &states[thread_id]);
}

int run(int argc, char* arg[]);
int main(int argc, char* argv[]) {

  // Volta Tensor Core operations exposed with mma.sync are first available in CUDA 10.1.
  //
  // CUTLASS must be compiled with CUDA 10.1 Toolkit to run these examples.
  if (!(__CUDACC_VER_MAJOR__ > 10 || (__CUDACC_VER_MAJOR__ == 10 && __CUDACC_VER_MINOR__ >= 1))) {
    std::cerr << "Volta Tensor Core operations must be compiled with CUDA 10.1 Toolkit or later." << std::endl;

    // Returning zero when built on older Toolkits so tests pass. The actions of this SDK example are no-op.
    return 0;
  }
  else {
    return run(argc, argv);
  }
}

