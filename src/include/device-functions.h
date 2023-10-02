#pragma once

__forceinline__ __device__ 
uint semaphoreLoad(volatile uint* semaphore) {
  uint state;
  asm volatile ("ld.global.acquire.gpu.u32 %0, [%1];" : "=r"(state) : "l"(semaphore));
  return state;
}

__device__ uint volatileLoad(volatile uint* addr) {
  uint val;
  asm volatile ("ld.global.volatile.u32 {%0}, [%1];" : "=r"(val) : "l"(addr));
  return val;
}

__device__ __forceinline__ uint glLoad(volatile uint* addr) {
  uint val;
  asm volatile ("ld.global.cg.u32 {%0}, [%1];" : "=r"(val) : "l"(addr));
  return val;
}

__device__ inline uint bringToCache(volatile uint* addr) {
  uint val;
  asm ("ld.global.cg.u32 {%0}, [%1];" : "=r"(val) : "l"(addr));
  return val;
}