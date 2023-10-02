#include "cusync-test.h"

#include "gtest/gtest.h"

using ProdCuStage = CuStage<CuStageType::Producer, RowMajor, TileSync<1>>;
using ConsCuStage = CuStage<CuStageType::Consumer, RowMajor, TileSync<1>>;

typedef uint ElemType;

/*
 * This kernel copies elements of in array to out array.
 * For each thread block, the kernel post the status of tile (thread block) 
 * and waits until the status of tile has reached expected value 
 */
template<typename CuStage>
__global__
void kernel(CuSyncTest cutest, CuStage custage, int idx, ElemType* in, ElemType* out) {
  dim3 tile = blockIdx;
  __shared__ int tileSh[3];
  tile = custage.tile((dim3*)&tileSh[0]);
  custage.wait(tile);
	
  uint linearid = threadIdx.x + blockIdx.x * blockDim.x;
	out[linearid] = in[linearid];
  
  custage.post(tile);
  cutest.setSemValue(idx, tile, custage);
}

/*
 * The test runs two kernels to copy from the source array to two output arrays.
 * The kernels are synchronized using the given synchronization. Finally,
 * checks the output of both copies and the value of semaphores are equal to the expected value.
 */
bool run() {
  ElemType* array1, *array2, *array3;
  size_t size = 1 << 20;

  //Allocate three arrays
  CUDA_CHECK(cudaMalloc(&array1, size * sizeof(ElemType)));
  CUDA_CHECK(cudaMalloc(&array2, size * sizeof(ElemType)));
  CUDA_CHECK(cudaMalloc(&array3, size * sizeof(ElemType)));

  ElemType* hostarray = new ElemType[size];
  //Initialize input array
  for (uint i = 0; i < size; i++) {
    hostarray[i] = i;
  }

  CUDA_CHECK(cudaMemcpy(array1, hostarray, size * sizeof(ElemType), cudaMemcpyHostToDevice));

  cudaStream_t prod_stream, cons_stream;
  CUDA_CHECK(cudaStreamCreateWithFlags(&cons_stream, cudaStreamNonBlocking));
  CUDA_CHECK(cudaStreamCreateWithFlags(&prod_stream, cudaStreamNonBlocking));

  dim3 threads(128, 1, 1);
  dim3 grid(size/threads.x, 1, 1);

  //Expected value of each semaphore is 1
  TileSync<1> sync;
  ProdCuStage prod(grid, threads, sync);
  ConsCuStage cons(grid, threads, sync);
  prod.iter = cons.iter = 1;
  initProducerConsumer(prod, cons);
  CuSyncTest cutest(1);
  
  //Invoke both kernels
  kernel<ProdCuStage><<<grid, threads, 0, prod_stream>>>(cutest, prod, -1, array1, array2);
  prod.invokeWaitKernel(cons_stream);
  kernel<ConsCuStage><<<grid, threads, 0, cons_stream>>>(cutest, cons, 0, array2, array3);

  CUDA_CHECK(cudaDeviceSynchronize());
  
  //Check that copies to array2 and array3 are correct 
  CUDA_CHECK(cudaMemcpy(hostarray, array2, size * sizeof(ElemType), cudaMemcpyDeviceToHost));
  bool eq = true;
  for (uint i = 0; i < size; i++) {
    eq = eq && (hostarray[i] == i);
  }

  CUDA_CHECK(cudaMemcpy(hostarray, array3, size * sizeof(ElemType), cudaMemcpyDeviceToHost));
  for (uint i = 0; i < size; i++) {
    eq = eq && (hostarray[i] == i);
  }

  delete hostarray;

  //Check that value of each semaphore is equal to the expected value
  eq = eq && cutest.allSemsCorrect();
  CUDA_CHECK(cudaFree(array1));
  CUDA_CHECK(cudaFree(array2));
  CUDA_CHECK(cudaFree(array3));

  return eq;
}

TEST(SimpleTest, SimpleTest) {
  EXPECT_TRUE(run());
}