include ../common.mk

ARCH_FLAGS=-gencode=arch=compute_70,code=[sm_70,compute_70]
XCOMPILER=-Xcompiler=-fPIE -Xcompiler=-Wconversion -Xcompiler=-fno-strict-aliasing
BUILD=build

INCLUDES=-I$(NV_CUTLASS)/include -I$(NV_CUTLASS)/examples/common -I$(NV_CUTLASS)/tools/util/include -I$(NV_CUTLASS) -I$(CUSYNC_CUTLASS)/include -I$(CUSYNC) -I.

DEFINES=-DCUTLASS_ENABLE_CUBLAS=1 -DCUTLASS_NAMESPACE=cutlass -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 -DCUTLASS_TEST_LEVEL=0 -DCUTLASS_TEST_ENABLE_CACHED_RESULTS=1 -DCUTLASS_CONV_UNIT_TEST_RIGOROUS_SIZE_ENABLED=1 -DCUTLASS_DEBUG_TRACE_LEVEL=0 

$(BUILD)/conv-rowsync: resnet.cu
	$(NVCC)  $(INCLUDES) $(DEFINES) -O3 -DNDEBUG $(XCOMPILER) $(ARCH_FLAGS) $< -DROWSYNC -o $@

$(BUILD)/conv-tilesync: resnet.cu
	$(NVCC)  $(INCLUDES) $(DEFINES) -O3 -DNDEBUG $(XCOMPILER) $(ARCH_FLAGS) $< -DTILESYNC -o $@

$(BUILD)/conv-eval-streamk: $(BUILD)/conv-eval-baseline.cu
	$(NVCC)  $(INCLUDES) $(DEFINES) -O3 -DNDEBUG $(XCOMPILER) $(ARCH_FLAGS) -DROWSYNC -DSTREAM_K  $< -o $@

$(BUILD)/conv-eval-baseline: $(BUILD)/conv-eval-baseline.cu
	$(NVCC)  $(INCLUDES) $(DEFINES) -O3 -DNDEBUG $(XCOMPILER) $(ARCH_FLAGS) -DROWSYNC  $< -o $@

$(BUILD)/conv-eval-rowsync: $(BUILD)/conv-eval-rowsync.cu
	$(NVCC)  $(INCLUDES) $(DEFINES) -O3 -DNDEBUG $(XCOMPILER) $(ARCH_FLAGS) $< -o $@

$(BUILD)/conv-eval-tilesync: $(BUILD)/conv-eval-tilesync.cu
	$(NVCC)  $(INCLUDES) $(DEFINES) -O3 -DNDEBUG $(XCOMPILER) $(ARCH_FLAGS) $<  -o $@

#VGG
$(BUILD)/vgg-rowsync: vgg.cu
	$(NVCC)  $(INCLUDES) $(DEFINES) -O3 -DNDEBUG $(XCOMPILER) $(ARCH_FLAGS)   ./vgg.cu -DROWSYNC -o $@

$(BUILD)/vgg-tilesync: vgg.cu
	$(NVCC)  $(INCLUDES) $(DEFINES) -O3 -DNDEBUG $(XCOMPILER) $(ARCH_FLAGS)   ./vgg.cu -DTILESYNC -o $@
