include ../common.mk

ARCH_FLAGS=-gencode=arch=compute_70,code=[sm_70,compute_70] -gencode=arch=compute_80,code=[sm_80,compute_80]
INCLUDES=-I$(NV_CUTLASS)/include -I$(NV_CUTLASS)/examples/common -I$(NV_CUTLASS)/tools/util/include -I$(CUSYNC_CUTLASS)/include/ -I$(CUSYNC) -I.

DEFINES=-DCUTLASS_ENABLE_CUBLAS=1 -DCUTLASS_NAMESPACE=cutlass -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 -DCUTLASS_TEST_LEVEL=0 -DCUTLASS_TEST_ENABLE_CACHED_RESULTS=1 -DCUTLASS_CONV_UNIT_TEST_RIGOROUS_SIZE_ENABLED=1 -DCUTLASS_DEBUG_TRACE_LEVEL=0

$(BUILD)/streamk: streamk.cu
	$(NVCC)  $(DEFINES) $(INCLUDES) $(ARCH_FLAGS)  -DNDEBUG ./streamk.cu -o $@ -Xptxas -v -lcublas -Xcompiler=-fopenmp -O3

$(BUILD)/streamk-eval: $(BUILD)/streamk-eval.cu
	$(NVCC)  $(DEFINES) $(INCLUDES) $(ARCH_FLAGS)  -DNDEBUG $(BUILD)/streamk-eval.cu -o $@ -Xptxas -v -lcublas -Xcompiler=-fopenmp -O3

$(BUILD)/mlp-batchedrow: mlp.cu common.h
	$(NVCC)  $(DEFINES) $(INCLUDES) $(ARCH_FLAGS)  -DNDEBUG ./mlp.cu -o $@ -Xptxas -v -lcublas -Xcompiler=-fopenmp -O3 -DBATCHEDROW

$(BUILD)/mlp-tilebatchsync: mlp.cu common.h
	$(NVCC)  $(DEFINES) $(INCLUDES) $(ARCH_FLAGS)  -DNDEBUG ./mlp.cu -o $@ -Xptxas -v -lcublas -Xcompiler=-fopenmp -O3 -DTILEBATCH

$(BUILD)/mlp-gpt3-rowsync: mlp.cu common.h
	$(NVCC)  $(DEFINES) $(INCLUDES) $(ARCH_FLAGS)  -DNDEBUG ./mlp.cu -o $@ -Xptxas -v -lcublas -Xcompiler=-fopenmp -O3 -DROWSYNC -DMLP_GPT3

$(BUILD)/mlp-gpt3-tilesync: mlp.cu common.h
	$(NVCC)  $(DEFINES) $(INCLUDES) $(ARCH_FLAGS)  -DNDEBUG ./mlp.cu -o $@ -Xptxas -v -lcublas -Xcompiler=-fopenmp -O3 -DTILESYNC -DMLP_GPT3

$(BUILD)/mlp-llama-rowsync: mlp.cu common.h
	$(NVCC)  $(DEFINES) $(INCLUDES) $(ARCH_FLAGS)  -DNDEBUG ./mlp.cu -o $@ -Xptxas -v -lcublas -Xcompiler=-fopenmp -O3 -DROWSYNC -DMLP_LLAMA

$(BUILD)/mlp-llama-tilesync: mlp.cu common.h
	$(NVCC)  $(DEFINES) $(INCLUDES) $(ARCH_FLAGS)  -DNDEBUG ./mlp.cu -o $@ -Xptxas -v -lcublas -Xcompiler=-fopenmp -O3 -DTILESYNC -DMLP_LLAMA

$(BUILD)/libmlp.so: mlp-lib.cu common.h
	$(NVCC)  $(DEFINES) $(INCLUDES) --shared -Xcompiler -m64,-fPIC,-Wconversion,-fno-strict-aliasing $(ARCH_FLAGS)  -DNDEBUG ./mlp-lib.cu -o $@ -Xptxas -v -lcublas -Xcompiler=-fopenmp -O3 -DROWSYNC

$(BUILD)/mlp-tilesync: mlp.cu common.h
	$(NVCC) $(DEFINES) $(INCLUDES) $(ARCH_FLAGS)  -DNDEBUG ./mlp.cu -o $@ -Xptxas -v -lcublas -Xcompiler=-fopenmp -O3 -DTILESYNC

$(BUILD)/mlp-eval-baseline: $(BUILD)/mlp-eval-baseline.cu common.h
	$(NVCC)  $(DEFINES) $(INCLUDES) $(ARCH_FLAGS)  -DNDEBUG $< -o $@ -Xptxas -v -lcublas -Xcompiler=-fopenmp -O3 -DROWSYNC

$(BUILD)/mlp-eval-rowsync: $(BUILD)/mlp-eval-rowsync.cu common.h
	$(NVCC)  $(DEFINES) $(INCLUDES) $(ARCH_FLAGS)  -DNDEBUG $< -o $@ -Xptxas -v -lcublas -Xcompiler=-fopenmp -O3

$(BUILD)/mlp-eval-tilesync: $(BUILD)/mlp-eval-tilesync.cu common.h
	$(NVCC) $(DEFINES) $(INCLUDES) $(ARCH_FLAGS)  -DNDEBUG $< -o $@ -Xptxas -v -lcublas -Xcompiler=-fopenmp -O3

$(BUILD)/mlp-eval: $(BUILD)/mlp-eval.cu common.h
	$(NVCC) $(DEFINES) $(INCLUDES) $(ARCH_FLAGS)  -DNDEBUG ./mlp-eval.cu -o $@ -Xptxas -v -lcublas -Xcompiler=-fopenmp -O3

$(BUILD)/attention-rowsync: attention.cu
	$(NVCC)  $(DEFINES) $(INCLUDES) -O3 $(ARCH_FLAGS)  -DNDEBUG ./attention.cu -o $@ -Xptxas -v -lcublas -Xcompiler=-fopenmp -DROWSYNC

$(BUILD)/attention-tilesync: attention.cu
	$(NVCC)  $(DEFINES) $(INCLUDES) -O3 $(ARCH_FLAGS)  -DNDEBUG ./attention.cu -o $@ -Xptxas -v -lcublas -Xcompiler=-fopenmp -DTILESYNC

$(BUILD)/attention-gpt3-stridedsync: attention.cu
	$(NVCC)  $(DEFINES) $(INCLUDES) -O3 $(ARCH_FLAGS)  -DNDEBUG ./attention.cu -o $@ -Xptxas -v -lcublas -Xcompiler=-fopenmp -DSTRIDEDSYNC -DGPT3

$(BUILD)/attention-llama-stridedsync: attention.cu
	$(NVCC)  $(DEFINES) $(INCLUDES) -O3 $(ARCH_FLAGS)  -DNDEBUG ./attention.cu -o $@ -Xptxas -v -lcublas -Xcompiler=-fopenmp -DSTRIDEDSYNC -DLLaMA

$(BUILD)/attention-eval-baseline: $(BUILD)/attention-eval-baseline.cu
	$(NVCC)  $(DEFINES) $(INCLUDES) -O3 $(ARCH_FLAGS)  -DNDEBUG $< -o $@ -Xptxas -v -lcublas -Xcompiler=-fopenmp -DTILESYNC

$(BUILD)/attention-eval-rowsync: $(BUILD)/attention-eval-rowsync.cu
	$(NVCC)  $(DEFINES) $(INCLUDES) -O3 $(ARCH_FLAGS)  -DNDEBUG $< -o $@ -Xptxas -v -lcublas -Xcompiler=-fopenmp -DROWSYNC

$(BUILD)/attention-eval-tilesync: $(BUILD)/attention-eval-tilesync.cu
	$(NVCC)  $(DEFINES) $(INCLUDES) -O3 $(ARCH_FLAGS)  -DNDEBUG $< -o $@ -Xptxas -v -lcublas -Xcompiler=-fopenmp -DTILESYNC

$(BUILD)/attention-gpt3-eval-stridedsync: $(BUILD)/attention-eval-stridedsync.cu
	$(NVCC)  $(DEFINES) $(INCLUDES) -O3 $(ARCH_FLAGS)  -DNDEBUG $< -o $(BUILD)/attention-eval-stridedsync -Xptxas -v -lcublas -Xcompiler=-fopenmp -DSTRIDEDSYNC -DGPT3

$(BUILD)/attention-llama-eval-stridedsync: $(BUILD)/attention-eval-stridedsync.cu
	$(NVCC)  $(DEFINES) $(INCLUDES) -O3 $(ARCH_FLAGS)  -DNDEBUG $< -o $(BUILD)/attention-eval-stridedsync -Xptxas -v -lcublas -Xcompiler=-fopenmp -DSTRIDEDSYNC -DLLaMA

$(BUILD)/attention-gpt3-eval: $(BUILD)/attention-eval.cu
	$(NVCC)  $(DEFINES) $(INCLUDES) -O3 $(ARCH_FLAGS)  -DNDEBUG ./attention-eval.cu -o $(BUILD)/attention-eval -Xptxas -v -lcublas -Xcompiler=-fopenmp -DGPT3

$(BUILD)/attention-llama-eval: $(BUILD)/attention-eval.cu
	$(NVCC)  $(DEFINES) $(INCLUDES) -O3 $(ARCH_FLAGS)  -DNDEBUG ./attention-eval.cu -o $(BUILD)/attention-eval -Xptxas -v -lcublas -Xcompiler=-fopenmp -DLLaMA

clean:
	rm -f $(BUILD)/mlp-* $(BUILD)/attention-*
