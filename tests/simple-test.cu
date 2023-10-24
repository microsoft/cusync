#include "cusync-test.h"

#include "gtest/gtest.h"

typedef uint ElemType;

using namespace cusync;

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
template<typename SyncPolicy, typename ProdCuStage, typename ConsCuStage>
bool run(int iters) {
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

  CUDA_CHECK(cudaMemset(array2, 0, size * sizeof(ElemType)));
  CUDA_CHECK(cudaMemset(array3, 0, size * sizeof(ElemType)));

  cudaStream_t prod_stream, cons_stream;
  CUDA_CHECK(cudaStreamCreateWithFlags(&cons_stream, cudaStreamNonBlocking));
  CUDA_CHECK(cudaStreamCreateWithFlags(&prod_stream, cudaStreamNonBlocking));

  dim3 threads(128, 1, 1);
  dim3 grid(size/threads.x, 1, 1);

  //Expected value of each semaphore is 1
  SyncPolicy sync;
  ProdCuStage prod(grid, threads, NoSync(), sync);
  ConsCuStage cons(grid, threads, sync, NoSync());
  CuSync::setProducerConsumerPair(prod, cons);

  CuSyncTest cutest(1);
  
  //Invoke both kernels
  int i = 0;
  while (i < iters) {
    kernel<ProdCuStage><<<grid, threads, 0, prod_stream>>>(cutest, prod, -1, array1, array2);
    prod.invokeWaitKernel(cons_stream);
    kernel<ConsCuStage><<<grid, threads, 0, cons_stream>>>(cutest, cons, 0, array2, array3);

    CUDA_CHECK(cudaDeviceSynchronize());
    prod.incrementIter();
    cons.incrementIter();
    i++;
  }
  
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

  //Check that value of each semaphore is equal to the expected value
  eq = eq && cutest.allSemsCorrect();
  
  //Cleanup
  delete hostarray;
  CUDA_CHECK(cudaFree(array1));
  CUDA_CHECK(cudaFree(array2));
  CUDA_CHECK(cudaFree(array3));

  return eq;
}

TEST(SimpleTest_TileSync, NoOpts) {
  using Sync = TileSync<IdentityOrder, 1, 1>;
  using ProdCuStage = CuStage<IdentityOrder, NoSync, Sync>;
  using ConsCuStage = CuStage<IdentityOrder, Sync, NoSync>;
  bool result = run<Sync, ProdCuStage, ConsCuStage>(1);
  EXPECT_TRUE(result);
}

TEST(SimpleTest_TileSync_MultiIters, NoOpts) {
  using Sync = TileSync<IdentityOrder, 1, 1>;
  using ProdCuStage = CuStage<IdentityOrder, NoSync, Sync>;
  using ConsCuStage = CuStage<IdentityOrder, Sync, NoSync>;
  bool result = run<Sync, ProdCuStage, ConsCuStage>(2);
  EXPECT_TRUE(result);
}

TEST(SimpleTest_TileSync, NoAtomicAdd) {
  using Sync = TileSync<IdentityOrder, 1, 1>;
  using ProdCuStage = CuStage<IdentityOrder, NoSync, Sync, Optimizations::NoAtomicAdd>;
  using ConsCuStage = CuStage<IdentityOrder, Sync, NoSync>;
  
  bool result = run<Sync, ProdCuStage, ConsCuStage>(1);
  EXPECT_TRUE(result);
}

TEST(SimpleTest_TileSync, AvoidCustomOrder) {
  using Sync = TileSync<IdentityOrder, 1, 1>;
  using ProdCuStage = CuStage<IdentityOrder, NoSync, Sync, Optimizations::AvoidCustomOrder>;
  using ConsCuStage = CuStage<IdentityOrder, Sync, NoSync>;
  
  bool result = run<Sync, ProdCuStage, ConsCuStage>(1);
  EXPECT_TRUE(result);
}