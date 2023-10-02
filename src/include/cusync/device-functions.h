#pragma once


// __device__ __forceinline__
// uint semaphoreLoad(volatile uint* semaphore) {
//   uint state;
//   asm volatile ("ld.global.acquire.gpu.u32 %0, [%1];" : "=r"(state) : "l"(semaphore));
//   return state;
// }

__device__ __forceinline__
uint globalVolatileLoad(volatile uint* addr) {
  uint val;
  asm volatile ("ld.global.volatile.u32 {%0}, [%1];" : "=r"(val) : "l"(addr));
  return val;
}

__device__ __forceinline__ 
uint globalLoad(volatile uint* addr) {
  uint val;
  asm volatile ("ld.global.cg.u32 {%0}, [%1];" : "=r"(val) : "l"(addr));
  return val;
}

// __device__ __forceinline__
// uint bringToCache(volatile uint* addr) {
//   uint val;
//   asm ("ld.global.cg.u32 {%0}, [%1];" : "=r"(val) : "l"(addr));
//   return val;
// }