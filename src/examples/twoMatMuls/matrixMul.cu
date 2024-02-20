// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

/**
 * Matrix multiplication: C = A * B.
 * Host code.
 *
 * This sample implements matrix multiplication which makes use of shared memory
 * to ensure data reuse, the matrix multiplication is done using tiling
 * approach. It has been written for clarity of exposition to illustrate various
 * CUDA programming principles, not with the goal of providing the most
 * performant generic kernel for matrix multiplication. See also: V. Volkov and
 * J. Demmel, "Benchmarking GPUs to tune dense linear algebra," in Proc. 2008
 * ACM/IEEE Conf. on Supercomputing (SC '08), Piscataway, NJ: IEEE Press, 2008,
 * pp. Art. 31:1-11.
 */
 
// System includes
#include <assert.h>
#include <stdio.h>

// CUDA runtime
#include <cuda_runtime.h>

// CuSync include
#include <cusync/cusync.h>

/**
 * Matrix multiplication (CUDA Kernel) on the device: C = A * B
 * wA is A's width and wB is B's width
 */

using namespace cusync;

//Define Producer and Consumer CuStage
const int BLOCK_SIZE = 32;
using Sync = TileSync<IdentityOrder, BLOCK_SIZE, BLOCK_SIZE>;
using ProdCuStage = CuStage<IdentityOrder, NoSync, Sync>;
using ConsCuStage = CuStage<IdentityOrder, Sync, NoSync>;

template <typename CuStageTy>
__global__ void MatrixMulCUDA(CuStageTy custage, float *C, float *A,
                              float *B, int wA, int wB) {
  __shared__ int tileSh[3];
  // Get tile to compute by this thread block
  dim3 tile = custage.tile((dim3*)&tileSh[0]);

  // Block index
  int bx = tile.x;
  int by = tile.y;

  // Thread index
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // Index of the first sub-matrix of A processed by the block
  int aBegin = wA * BLOCK_SIZE * by;

  // Index of the last sub-matrix of A processed by the block
  int aEnd = aBegin + wA - 1;

  // Step size used to iterate through the sub-matrices of A
  int aStep = BLOCK_SIZE;

  // Index of the first sub-matrix of B processed by the block
  int bBegin = BLOCK_SIZE * bx;

  // Step size used to iterate through the sub-matrices of B
  int bStep = BLOCK_SIZE * wB;

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  float Csub = 0;

  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
    // Declaration of the shared memory array As used to
    // store the sub-matrix of A
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

    // Declaration of the shared memory array Bs used to
    // store the sub-matrix of B
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    // Load the matrices from device memory
    // to shared memory; each thread loads
    // one element of each matrix
    // Wait for tile of A to be computed by producer kernel
    
    dim3 tile = {(uint32_t)(a - aBegin), (uint32_t)by * BLOCK_SIZE, 1};
    custage.wait(tile);

    As[ty][tx] = A[a + wA * ty + tx];
    Bs[ty][tx] = B[b + wB * ty + tx];

    // Synchronize to make sure the matrices are loaded
    __syncthreads();

    // Multiply the two matrices together;
    // each thread computes one element
    // of the block sub-matrix
#pragma unroll
    for (int k = 0; k < BLOCK_SIZE; ++k) {
      Csub += As[ty][k] * Bs[k][tx];
    }

    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    __syncthreads();
  }

  // Write the block sub-matrix to device memory;
  // each thread writes one element
  int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
  C[c + wB * ty + tx] = Csub;

  // Post the status of tile when computed
  custage.post({(uint32_t)bx * BLOCK_SIZE, (uint32_t)by * BLOCK_SIZE, 0});
}

void ConstantInit(float *data, int size, float val) {
  for (int i = 0; i < size; ++i) {
    data[i] = val;
  }
}

/**
 * Run a simple test of matrix multiplication using CUDA
 */
int MatrixMultiply(int argc, char **argv, int block_size, const dim3 &dimsA,
                   const dim3 &dimsB, const dim3 &dimsD) {
  // Allocate host memory for matrices A and B
  unsigned int size_A = dimsA.x * dimsA.y;
  unsigned int mem_size_A = sizeof(float) * size_A;
  float *h_A;
  CUDA_CHECK(cudaMallocHost(&h_A, mem_size_A));
  unsigned int size_B = dimsB.x * dimsB.y;
  unsigned int mem_size_B = sizeof(float) * size_B;
  float *h_B;
  CUDA_CHECK(cudaMallocHost(&h_B, mem_size_B));
  float *h_D;
  CUDA_CHECK(cudaMallocHost(&h_D, mem_size_A));
  
  cudaStream_t prod_stream, cons_stream;

  // Initialize host memory
  const float valB = 0.01f;
  ConstantInit(h_A, size_A, 1.0f);
  ConstantInit(h_B, size_B, valB);
  ConstantInit(h_D, size_B, valB);

  // Allocate device memory
  float *d_A, *d_B, *d_C, *d_D, *d_E;

  // Allocate host matrix C and E
  dim3 dimsC(dimsB.x, dimsA.y, 1);
  unsigned int mem_size_C = dimsC.x * dimsC.y * sizeof(float);
  float *h_C;
  CUDA_CHECK(cudaMallocHost(&h_C, mem_size_C));

  dim3 dimsE(dimsB.x, dimsA.y, 1);
  unsigned int mem_size_E = dimsC.x * dimsC.y * sizeof(float);
  float *h_E;
  CUDA_CHECK(cudaMallocHost(&h_E, mem_size_E));

  if (h_C == NULL) {
    fprintf(stderr, "Failed to allocate host matrix C!\n");
    exit(EXIT_FAILURE);
  }

  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), mem_size_A));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_B), mem_size_B));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_C), mem_size_C));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_D), mem_size_B));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_E), mem_size_E));

  // Allocate CUDA events that we'll use for timing
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaStreamCreateWithFlags(&cons_stream, cudaStreamNonBlocking));
  CUDA_CHECK(cudaStreamCreateWithFlags(&prod_stream, cudaStreamNonBlocking));

  // copy host memory to device
  CUDA_CHECK(
      cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice));
  CUDA_CHECK(
      cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice));
  CUDA_CHECK(
      cudaMemcpy(d_D, h_D, mem_size_B, cudaMemcpyHostToDevice));
  
  // Setup execution parameters
  dim3 threads(block_size, block_size, 1);
  dim3 grid(dimsB.x / threads.x, dimsA.y / threads.y, 1);
  
  // Create CuSync and CuStage
  Sync sync;
  dim3 tilesize = threads;
  ProdCuStage prod(grid, tilesize, NoSync(), sync);
  ConsCuStage cons(grid, tilesize, sync, NoSync());
  CuSync::setProducerConsumerPair(prod, cons);

  // Create and start timer
  printf("Computing result using CUDA Kernel...\n");

  assert (block_size == 32);
  // Invoke producer kernel (C = A * B)
  MatrixMulCUDA<ProdCuStage>
        <<<grid, threads, 0, prod_stream>>>(prod, d_C, d_A, d_B, dimsA.x, dimsB.x);

  //Invoke wait kernel
  prod.invokeWaitKernel(cons_stream);
  
  //Invoke consumer kernel (E = C * D)
  MatrixMulCUDA<ConsCuStage>
        <<<grid, threads, 0, cons_stream>>>(cons, d_E, d_C, d_D, dimsA.x, dimsB.x);
  
  CUDA_CHECK(cudaDeviceSynchronize());

  //for next run increment the iteration counter
  prod.incrementIter();
  cons.incrementIter();

  printf("Execution done\n");
  
  // Copy result from device to host
  CUDA_CHECK(
      cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost));
  CUDA_CHECK(
      cudaMemcpy(h_E, d_E, mem_size_C, cudaMemcpyDeviceToHost));

  printf("Checking computed result for correctness: \n");
  bool correct = true;

  // test relative error by the formula
  //     |<x, y>_cpu - <x,y>_gpu|/<|x|, |y|>  < eps
  double eps = 1.e-5;  // machine zero
  // Check C
  for (int i = 0; i < static_cast<int>(dimsC.x * dimsC.y); i++) {
    double abs_err = fabs(h_C[i] - (dimsA.x * valB));
    double dot_length = dimsA.x;
    double abs_val = fabs(h_C[i]);
    double rel_err = abs_err / abs_val / dot_length;

    if (rel_err > eps) {
      printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n", i,
             h_C[i], dimsA.x * valB, eps);
      correct = false;
      break;
    }
  }

  printf("C results: %s\n", correct ? "PASS" : "FAIL");

  for (int i = 0; i < static_cast<int>(dimsC.x * dimsC.y); i++) {
    double abs_err = fabs(h_E[i] - (dimsA.x * valB * dimsA.x * valB));
    double dot_length = dimsA.x;
    double abs_val = fabs(h_E[i]);
    double rel_err = abs_err / abs_val / dot_length;

    if (rel_err > eps) {
      printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n", i,
             h_E[i], dimsA.x * valB  * dimsA.x, eps);
      correct = false;
      break;
    }
  }

  printf("E results: %s\n", correct ? "PASS" : "FAIL");

  // Clean up memory
  CUDA_CHECK(cudaFreeHost(h_A));
  CUDA_CHECK(cudaFreeHost(h_B));
  CUDA_CHECK(cudaFreeHost(h_C));
  CUDA_CHECK(cudaFreeHost(h_D));
  CUDA_CHECK(cudaFreeHost(h_E));
  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));
  CUDA_CHECK(cudaFree(d_D));
  CUDA_CHECK(cudaFree(d_E));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  if (correct) {
    return EXIT_SUCCESS;
  } else {
    return EXIT_FAILURE;
  }
}

/**
 * Program main
 */
int main(int argc, char **argv) {
  printf("[Matrix Multiply Using CUDA] - Starting...\n");

  // This will pick the best possible CUDA capable device, otherwise
  // override the device ID based on input provided at the command line

  int block_size = BLOCK_SIZE;

  dim3 dimsA(4 * 2 * block_size, 4 * 2 * block_size, 1);
  dim3 dimsB = dimsA;
  dim3 dimsD = dimsA;

  printf("MatrixA(%d,%d), MatrixB(%d,%d), MatrixD(%d,%d)\n", dimsA.x, dimsA.y, dimsB.x,
         dimsB.y, dimsD.x, dimsD.y);

  int matrix_result = MatrixMultiply(argc, argv, block_size, dimsA, dimsB, dimsD);

  exit(matrix_result);
}
