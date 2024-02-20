// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "cusync_defines.h"

#pragma once

/*
 * Volatile load of an unsigned integer from global memory
 *  @addr: global memory address of an unsigned integer  
 *  
 *  Returns the loaded unsigned integer 
 */
CUSYNC_DEVICE
static uint32_t globalVolatileLoad(volatile uint32_t* addr) {
  uint32_t val;
  asm volatile ("ld.global.acquire.gpu.u32 {%0}, [%1];" : "=r"(val) : "l"(addr));
  return val;
}

/*
 * Load of an unsigned integer from global memory and caching in L2 cache
 *  @addr: global memory address of an unsigned integer
 *  
 *  Returns the loaded unsigned integer 
 */
CUSYNC_DEVICE
static uint32_t globalLoad(volatile uint32_t* addr) {
  uint32_t val;
  asm volatile ("ld.global.cg.u32 {%0}, [%1];" : "=r"(val) : "l"(addr));
  return val;
}
