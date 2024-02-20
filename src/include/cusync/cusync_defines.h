// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
#if (defined(__CUDACC__) || defined(__NVCC__))
  #define CUSYNC_DEVICE __device__ __forceinline__
#else
  #define CUSYNC_DEVICE
#endif

#if (defined(__CUDACC__) || defined(__NVCC__))
  #define CUSYNC_HOST __host__ __forceinline__
#else
  #define CUSYNC_HOST
#endif

#if (defined(__CUDACC__) || defined(__NVCC__))
  #define CUSYNC_DEVICE_HOST __device__ __host__ __forceinline__
#else
  #define CUSYNC_DEVICE_HOST
#endif

#if (defined(__CUDACC__) || defined(__NVCC__))
  #define CUSYNC_GLOBAL __global__
#else
  #define CUSYNC_GLOBAL
#endif
