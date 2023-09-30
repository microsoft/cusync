NVCC=/usr/local/cuda-12.2/bin/nvcc
ROOT=../../../../
CUSYNC=$(ROOT)/src/include
NV_CUTLASS=$(ROOT)/cutlass/nvidia-cutlass
CUSYNC_CUTLASS=$(ROOT)/cutlass/cusync-cutlass
INCLUDES=-I$(NV_CUTLASS)/include -I$(NV_CUTLASS)/examples/common -I$(NV_CUTLASS)/tools/util/include -I$(CUSYNC_CUTLASS)/include/ -I$(CUSYNC)
DEFINES=-DCUTLASS_ENABLE_CUBLAS=1 -DCUTLASS_NAMESPACE=cutlass -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 -DCUTLASS_TEST_LEVEL=0 -DCUTLASS_TEST_ENABLE_CACHED_RESULTS=1 -DCUTLASS_CONV_UNIT_TEST_RIGOROUS_SIZE_ENABLED=1 -DCUTLASS_DEBUG_TRACE_LEVEL=0

all: mlp-rowsync mlp-tilesync

streamk: streamk.cu
	$(NVCC)  $(DEFINES) $(INCLUDES)  -Xcompiler=-Wconversion -Xcompiler=-fno-strict-aliasing -gencode=arch=compute_70,code=[sm_70,compute_70] -std=c++17 -x cu -DNDEBUG ./streamk.cu -o $@ -Xptxas -v -lcublas -Xcompiler=-fopenmp -O3

streamk-eval: streamk-eval.cu
	$(NVCC)  $(DEFINES) $(INCLUDES)  -Xcompiler=-Wconversion -Xcompiler=-fno-strict-aliasing -gencode=arch=compute_70,code=[sm_70,compute_70] -std=c++17 -x cu -DNDEBUG ./streamk-eval.cu -o $@ -Xptxas -v -lcublas -Xcompiler=-fopenmp -O3

mlp-batchedrow: mlp.cu common.h
	$(NVCC)  $(DEFINES) $(INCLUDES)  -Xcompiler=-Wconversion -Xcompiler=-fno-strict-aliasing -gencode=arch=compute_70,code=[sm_70,compute_70] -std=c++17 -x cu -DNDEBUG ./mlp.cu -o $@ -Xptxas -v -lcublas -Xcompiler=-fopenmp -O3 -DBATCHEDROW

mlp-tilebatchsync: mlp.cu common.h
	$(NVCC)  $(DEFINES) $(INCLUDES)  -Xcompiler=-Wconversion -Xcompiler=-fno-strict-aliasing -gencode=arch=compute_70,code=[sm_70,compute_70] -std=c++17 -x cu -DNDEBUG ./mlp.cu -o $@ -Xptxas -v -lcublas -Xcompiler=-fopenmp -O3 -DTILEBATCH

mlp-gpt3-rowsync: mlp.cu common.h
	$(NVCC)  $(DEFINES) $(INCLUDES)  -Xcompiler=-Wconversion -Xcompiler=-fno-strict-aliasing -gencode=arch=compute_70,code=[sm_70,compute_70] -std=c++17 -x cu -DNDEBUG ./mlp.cu -o $@ -Xptxas -v -lcublas -Xcompiler=-fopenmp -O3 -DROWSYNC -DMLP_GPT3

libmlp.so: mlp-lib.cu common.h
	$(NVCC)  $(DEFINES) $(INCLUDES) --shared -Xcompiler -m64,-fPIC,-Wconversion,-fno-strict-aliasing -gencode=arch=compute_70,code=[sm_70,compute_70] -std=c++17 -x cu -DNDEBUG ./mlp-lib.cu -o $@ -Xptxas -v -lcublas -Xcompiler=-fopenmp -O3 -DROWSYNC

mlp-tilesync: mlp.cu common.h
	$(NVCC) $(DEFINES) $(INCLUDES)  -Xcompiler=-Wconversion -Xcompiler=-fno-strict-aliasing -gencode=arch=compute_70,code=[sm_70,compute_70] -std=c++17 -x cu -DNDEBUG ./mlp.cu -o $@ -Xptxas -v -lcublas -Xcompiler=-fopenmp -O3 -DTILESYNC

mlp-eval-baseline: mlp-eval-baseline.cu common.h
	$(NVCC)  $(DEFINES) $(INCLUDES)  -Xcompiler=-Wconversion -Xcompiler=-fno-strict-aliasing -gencode=arch=compute_70,code=[sm_70,compute_70] -std=c++17 -x cu -DNDEBUG $< -o $@ -Xptxas -v -lcublas -Xcompiler=-fopenmp -O3 -DROWSYNC

mlp-eval-rowsync: mlp-eval-rowsync.cu common.h
	$(NVCC)  $(DEFINES) $(INCLUDES)  -Xcompiler=-Wconversion -Xcompiler=-fno-strict-aliasing -gencode=arch=compute_70,code=[sm_70,compute_70] -std=c++17 -x cu -DNDEBUG $< -o $@ -Xptxas -v -lcublas -Xcompiler=-fopenmp -O3

mlp-eval-tilesync: mlp-eval-tilesync.cu common.h
	$(NVCC) $(DEFINES) $(INCLUDES)  -Xcompiler=-Wconversion -Xcompiler=-fno-strict-aliasing -gencode=arch=compute_70,code=[sm_70,compute_70] -std=c++17 -x cu -DNDEBUG $< -o $@ -Xptxas -v -lcublas -Xcompiler=-fopenmp -O3

mlp-eval: mlp-eval.cu common.h
	$(NVCC) $(DEFINES) $(INCLUDES)  -Xcompiler=-Wconversion -Xcompiler=-fno-strict-aliasing -gencode=arch=compute_70,code=[sm_70,compute_70] -std=c++17 -x cu -DNDEBUG ./mlp-eval.cu -o $@ -Xptxas -v -lcublas -Xcompiler=-fopenmp -O3

attention-rowsync: attention.cu
	$(NVCC)  $(DEFINES) $(INCLUDES) -O3    -Xcompiler=-Wconversion -Xcompiler=-fno-strict-aliasing -gencode=arch=compute_70,code=[sm_70,compute_70] -std=c++17 -x cu -DNDEBUG ./attention.cu -o $@ -Xptxas -v -lcublas -Xcompiler=-fopenmp -DROWSYNC

attention-tilesync: attention.cu
	$(NVCC)  $(DEFINES) $(INCLUDES) -O3    -Xcompiler=-Wconversion -Xcompiler=-fno-strict-aliasing -gencode=arch=compute_70,code=[sm_70,compute_70] -std=c++17 -x cu -DNDEBUG ./attention.cu -o $@ -Xptxas -v -lcublas -Xcompiler=-fopenmp -DTILESYNC

attention-gpt3-stridedsync: attention.cu
	$(NVCC)  $(DEFINES) $(INCLUDES) -O3    -Xcompiler=-Wconversion -Xcompiler=-fno-strict-aliasing -gencode=arch=compute_70,code=[sm_70,compute_70] -std=c++17 -x cu -DNDEBUG ./attention.cu -o $@ -Xptxas -v -lcublas -Xcompiler=-fopenmp -DSTRIDEDSYNC -DGPT3

attention-llama-stridedsync: attention.cu
	$(NVCC)  $(DEFINES) $(INCLUDES) -O3    -Xcompiler=-Wconversion -Xcompiler=-fno-strict-aliasing -gencode=arch=compute_70,code=[sm_70,compute_70] -std=c++17 -x cu -DNDEBUG ./attention.cu -o $@ -Xptxas -v -lcublas -Xcompiler=-fopenmp -DSTRIDEDSYNC -DLLaMA

attention-eval-baseline: attention-eval-baseline.cu
	$(NVCC)  $(DEFINES) $(INCLUDES) -O3    -Xcompiler=-Wconversion -Xcompiler=-fno-strict-aliasing -gencode=arch=compute_70,code=[sm_70,compute_70] -std=c++17 -x cu -DNDEBUG $< -o $@ -Xptxas -v -lcublas -Xcompiler=-fopenmp -DTILESYNC

attention-eval-rowsync: attention-eval-rowsync.cu
	$(NVCC)  $(DEFINES) $(INCLUDES) -O3    -Xcompiler=-Wconversion -Xcompiler=-fno-strict-aliasing -gencode=arch=compute_70,code=[sm_70,compute_70] -std=c++17 -x cu -DNDEBUG $< -o $@ -Xptxas -v -lcublas -Xcompiler=-fopenmp -DROWSYNC

attention-eval-tilesync: attention-eval-tilesync.cu
	$(NVCC)  $(DEFINES) $(INCLUDES) -O3    -Xcompiler=-Wconversion -Xcompiler=-fno-strict-aliasing -gencode=arch=compute_70,code=[sm_70,compute_70] -std=c++17 -x cu -DNDEBUG $< -o $@ -Xptxas -v -lcublas -Xcompiler=-fopenmp -DTILESYNC

attention-gpt3-eval-stridedsync: attention-eval-stridedsync.cu
	$(NVCC)  $(DEFINES) $(INCLUDES) -O3    -Xcompiler=-Wconversion -Xcompiler=-fno-strict-aliasing -gencode=arch=compute_70,code=[sm_70,compute_70] -std=c++17 -x cu -DNDEBUG $< -o attention-eval-stridedsync -Xptxas -v -lcublas -Xcompiler=-fopenmp -DSTRIDEDSYNC -DGPT3

attention-llama-eval-stridedsync: attention-eval-stridedsync.cu
	$(NVCC)  $(DEFINES) $(INCLUDES) -O3    -Xcompiler=-Wconversion -Xcompiler=-fno-strict-aliasing -gencode=arch=compute_70,code=[sm_70,compute_70] -std=c++17 -x cu -DNDEBUG $< -o attention-eval-stridedsync -Xptxas -v -lcublas -Xcompiler=-fopenmp -DSTRIDEDSYNC -DLLaMA

attention-gpt3-eval: attention-eval.cu
	$(NVCC)  $(DEFINES) $(INCLUDES) -O3    -Xcompiler=-Wconversion -Xcompiler=-fno-strict-aliasing -gencode=arch=compute_70,code=[sm_70,compute_70] -std=c++17 -x cu -DNDEBUG ./attention-eval.cu -o attention-eval -Xptxas -v -lcublas -Xcompiler=-fopenmp -DGPT3

attention-llama-eval: attention-eval.cu
	$(NVCC)  $(DEFINES) $(INCLUDES) -O3    -Xcompiler=-Wconversion -Xcompiler=-fno-strict-aliasing -gencode=arch=compute_70,code=[sm_70,compute_70] -std=c++17 -x cu -DNDEBUG ./attention-eval.cu -o attention-eval -Xptxas -v -lcublas -Xcompiler=-fopenmp -DLLaMA

clean:
	rm -f mlp-rowsync mlp-tilesync attention-rowsync attention-tilesync
