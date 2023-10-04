#pragma once

/*
 * Volatile load of an unsigned integer from global memory
 *  @addr: global memory address of an unsigned integer  
 *  
 *  Returns the loaded unsigned integer 
 */
__device__ __forceinline__
uint globalVolatileLoad(volatile uint* addr) {
  uint val;
  asm volatile ("ld.global.volatile.u32 {%0}, [%1];" : "=r"(val) : "l"(addr));
  return val;
}

/*
 * Load of an unsigned integer from global memory and caching in L2 cache
 *  @addr: global memory address of an unsigned integer
 *  
 *  Returns the loaded unsigned integer 
 */
__device__ __forceinline__ 
uint globalLoad(volatile uint* addr) {
  uint val;
  asm volatile ("ld.global.cg.u32 {%0}, [%1];" : "=r"(val) : "l"(addr));
  return val;
}