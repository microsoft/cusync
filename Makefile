include common.mk

GOOGLE_TEST = tests/googletest
GOOGLE_TEST_BUILD = $(GOOGLE_TEST)/build
TEST_INCLUDE_DIRS = -Isrc/include/ -I$(GOOGLE_TEST)/googletest/include/ -L$(GOOGLE_TEST_BUILD)/lib/
TEST_LFLAGS = -lgtest -lpthread
GOOGLE_TEST_MAIN = $(GOOGLE_TEST)/googletest/src/gtest_main.cc
CUSYNC_SRC_FILES = src/cusync.cu

tests: run-simple-test

build-googletest: $(GOOGLE_TEST)
	mkdir -p $(GOOGLE_TEST_BUILD) && cd $(GOOGLE_TEST_BUILD) && cmake .. && make -j

simple-test: build-googletest $(shell find src/include/cusync -type f)
	$(NVCC) tests/$@.cu $(CUSYNC_SRC_FILES) $(TEST_INCLUDE_DIRS) $(TEST_LFLAGS) $(GOOGLE_TEST_MAIN) $(ARCH_CODE_FLAGS) -O3 -Xcompiler=-fopenmp,-O3,-Wall -o $@ -g -G

run-simple-test: simple-test
	./simple-test