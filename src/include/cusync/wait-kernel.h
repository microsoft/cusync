// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <stdint.h>
#include "device-functions.h"

#pragma once

namespace cusync {
  void invokeWaitKernel(uint32_t* semaphore, uint32_t givenValue, cudaStream_t stream);
}
