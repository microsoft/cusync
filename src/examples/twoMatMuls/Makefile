include ../../../common.mk
all: matrixMul

matrixMul: matrixMul.cu
	$(NVCC) -I../../include/ matrixMul.cu ../../cusync.cu  -o matrixMul $(ARCH_CODE_FLAGS)
