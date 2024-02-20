// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
#include <iostream>
#include <sstream>
#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/cusyncgemm.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/threadblock/cusync_threadblock_swizzle.h"

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

  if (ROW < M && COL < N) {
    AT tmpSum = (AT)0.0f;
    // each thread computes one element of the block sub-matrix
    for (uint32_t i = 0; i < K; i++) {
        tmpSum += (AT)(A[ROW * K + i]) * (AT)(B[i * N + COL]);
    }

    C[ROW * N + COL] = (T)tmpSum;
  }
}

template<typename T, typename AT>
void ref_matmul(uint32_t M, uint32_t N, uint32_t K, T* mat1, T* mat2, T* host_res) {
  T* dev_refC = NULL;
  CUDA_CHECK(cudaMalloc(&dev_refC, sizeof(T)*M*N));
  dim3 block = {32, 32, 1};
  dim3 grid = {N/block.x + 1, M/block.y + 1, 1};
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
    if (abs(e1) < v && abs(e2) < v) {
      
      ret = true;
    } else if (abs(e1) < v) {
      ret = false;
    } else if (abs(e2) < v) {
      ret = false;
    } else {
      float err = abs(abs(e1) - abs(e2))/max(abs(e1), abs(e2));
      if (err <= v) {
        ret = true;
      } else {
        printf("243: %f , %f at %lu, %f\n", e1, e2, i, err);
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

