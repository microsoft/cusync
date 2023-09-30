CUTLASS=../../
CUSYNC=../../../cusync/src/include/
NVCC=/usr/local/cuda-12.2/bin/nvcc

INCLUDES=-I$(CUTLASS)/include -I$(CUTLASS)/examples/common -I$(CUTLASS)/tools/util/include -I$(CUSYNC)

all: conv-rowsync

conv-rowsync: turing_tensorop_conv2dfprop.cu
	$(NVCC) -DCUTLASS_ENABLE_CUBLAS=1 -DCUTLASS_NAMESPACE=cutlass $(INCLUDES)  -O3 -DNDEBUG -Xcompiler=-fPIE   -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 -DCUTLASS_TEST_LEVEL=0 -DCUTLASS_TEST_ENABLE_CACHED_RESULTS=1 -DCUTLASS_CONV_UNIT_TEST_RIGOROUS_SIZE_ENABLED=1 -DCUTLASS_DEBUG_TRACE_LEVEL=0 -Xcompiler=-Wconversion -Xcompiler=-fno-strict-aliasing -gencode=arch=compute_70,code=[sm_70,compute_70] -std=c++17 -x cu ./turing_tensorop_conv2dfprop.cu -DROWSYNC -o $@

conv-tilesync: turing_tensorop_conv2dfprop.cu
	$(NVCC) -DCUTLASS_ENABLE_CUBLAS=1 -DCUTLASS_NAMESPACE=cutlass $(INCLUDES)  -O3 -DNDEBUG -Xcompiler=-fPIE   -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 -DCUTLASS_TEST_LEVEL=0 -DCUTLASS_TEST_ENABLE_CACHED_RESULTS=1 -DCUTLASS_CONV_UNIT_TEST_RIGOROUS_SIZE_ENABLED=1 -DCUTLASS_DEBUG_TRACE_LEVEL=0 -Xcompiler=-Wconversion -Xcompiler=-fno-strict-aliasing -gencode=arch=compute_70,code=[sm_70,compute_70] -std=c++17 -x cu ./turing_tensorop_conv2dfprop.cu -DTILESYNC -o $@

conv-eval-streamk: conv-eval-baseline.cu
	$(NVCC) -DCUTLASS_ENABLE_CUBLAS=1 -DCUTLASS_NAMESPACE=cutlass $(INCLUDES)  -O3 -DNDEBUG -Xcompiler=-fPIE   -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 -DCUTLASS_TEST_LEVEL=0 -DCUTLASS_TEST_ENABLE_CACHED_RESULTS=1 -DCUTLASS_CONV_UNIT_TEST_RIGOROUS_SIZE_ENABLED=1 -DCUTLASS_DEBUG_TRACE_LEVEL=0 -Xcompiler=-Wconversion -Xcompiler=-fno-strict-aliasing -gencode=arch=compute_70,code=[sm_70,compute_70] -std=c++17 -DROWSYNC -DSTREAM_K -x cu $< -o $@

conv-eval-baseline: conv-eval-baseline.cu
	$(NVCC) -DCUTLASS_ENABLE_CUBLAS=1 -DCUTLASS_NAMESPACE=cutlass $(INCLUDES)  -O3 -DNDEBUG -Xcompiler=-fPIE   -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 -DCUTLASS_TEST_LEVEL=0 -DCUTLASS_TEST_ENABLE_CACHED_RESULTS=1 -DCUTLASS_CONV_UNIT_TEST_RIGOROUS_SIZE_ENABLED=1 -DCUTLASS_DEBUG_TRACE_LEVEL=0 -Xcompiler=-Wconversion -Xcompiler=-fno-strict-aliasing -gencode=arch=compute_70,code=[sm_70,compute_70] -std=c++17 -DROWSYNC -x cu $< -o $@

conv-eval-rowsync: conv-eval-rowsync.cu
	$(NVCC) -DCUTLASS_ENABLE_CUBLAS=1 -DCUTLASS_NAMESPACE=cutlass $(INCLUDES)  -O3 -DNDEBUG -Xcompiler=-fPIE   -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 -DCUTLASS_TEST_LEVEL=0 -DCUTLASS_TEST_ENABLE_CACHED_RESULTS=1 -DCUTLASS_CONV_UNIT_TEST_RIGOROUS_SIZE_ENABLED=1 -DCUTLASS_DEBUG_TRACE_LEVEL=0 -Xcompiler=-Wconversion -Xcompiler=-fno-strict-aliasing -gencode=arch=compute_70,code=[sm_70,compute_70] -std=c++17 -x cu $< -o $@

conv-eval-tilesync: conv-eval-tilesync.cu
	$(NVCC) -DCUTLASS_ENABLE_CUBLAS=1 -DCUTLASS_NAMESPACE=cutlass $(INCLUDES)  -O3 -DNDEBUG -Xcompiler=-fPIE   -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 -DCUTLASS_TEST_LEVEL=0 -DCUTLASS_TEST_ENABLE_CACHED_RESULTS=1 -DCUTLASS_CONV_UNIT_TEST_RIGOROUS_SIZE_ENABLED=1 -DCUTLASS_DEBUG_TRACE_LEVEL=0 -Xcompiler=-Wconversion -Xcompiler=-fno-strict-aliasing -gencode=arch=compute_70,code=[sm_70,compute_70] -std=c++17 -x cu $<  -o $@

#VGG

vgg-rowsync: vgg.cu
	$(NVCC) -DCUTLASS_ENABLE_CUBLAS=1 -DCUTLASS_NAMESPACE=cutlass $(INCLUDES)  -O3 -DNDEBUG -Xcompiler=-fPIE   -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 -DCUTLASS_TEST_LEVEL=0 -DCUTLASS_TEST_ENABLE_CACHED_RESULTS=1 -DCUTLASS_CONV_UNIT_TEST_RIGOROUS_SIZE_ENABLED=1 -DCUTLASS_DEBUG_TRACE_LEVEL=0 -Xcompiler=-Wconversion -Xcompiler=-fno-strict-aliasing -gencode=arch=compute_70,code=[sm_70,compute_70] -std=c++17 -x cu ./vgg.cu -DROWSYNC -o $@

vgg-tilesync: vgg.cu
	$(NVCC) -DCUTLASS_ENABLE_CUBLAS=1 -DCUTLASS_NAMESPACE=cutlass $(INCLUDES)  -O3 -DNDEBUG -Xcompiler=-fPIE   -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 -DCUTLASS_TEST_LEVEL=0 -DCUTLASS_TEST_ENABLE_CACHED_RESULTS=1 -DCUTLASS_CONV_UNIT_TEST_RIGOROUS_SIZE_ENABLED=1 -DCUTLASS_DEBUG_TRACE_LEVEL=0 -Xcompiler=-Wconversion -Xcompiler=-fno-strict-aliasing -gencode=arch=compute_70,code=[sm_70,compute_70] -std=c++17 -x cu ./vgg.cu -DTILESYNC -o $@