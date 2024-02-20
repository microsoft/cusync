// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "cusync/wait-kernel.h"

/*
 * The wait kernel waits until the value of semaphore has reached the given value.
 * @semaphore: Address to the unsigned integer semaphore 
 * @givenValue: Given value of the semaphore 
*/
__global__
void waitKernel(volatile uint32_t* semaphore, uint32_t givenValue) {
  if (threadIdx.x == 0) {
    uint32_t currVal = globalLoad(semaphore);
    while(currVal < givenValue) {
      currVal = globalVolatileLoad(semaphore);
    }
  }
}

namespace cusync {
  void invokeWaitKernel(uint32_t* semaphore, uint32_t givenValue, cudaStream_t stream) {
      waitKernel<<<1,1,0,stream>>>((uint32_t*)semaphore, givenValue);   
  }
}
