NVCC=/usr/local/cuda-12.2/bin/nvcc -std=c++17
ROOT=../../../
CUSYNC=$(ROOT)/src/include
NV_CUTLASS=$(ROOT)/src/include/cutlass/nvidia-cutlass
CUSYNC_CUTLASS=$(ROOT)/src/include/cutlass/cusync-cutlass
BUILD=build