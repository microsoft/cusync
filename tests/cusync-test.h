#include <vector>

#include <cuSync.h>

#pragma once

class CuSyncTest {
private:
  uint numStages;
  char* semValidArray;

public:
  CuSyncTest(int numStages_) : numStages(numStages_) {
    CUDA_CHECK(cudaMalloc(&semValidArray, numStages * sizeof(char)));
    CUDA_CHECK(cudaMemset(semValidArray, 1, numStages * sizeof(char)));
  }

  template<typename CuStage>
  __device__ 
  void setSemValue(uint stageIdx, dim3 tile, CuStage& custage) {
    if (!custage.isConsumer()) return;
    if (threadIdx.x == 0) {
      const uint index = custage.waitTileIndex(tile);
      char eq = (char)(custage.expectedWaitValue(tile) == custage.waitSemValue(index));
      semValidArray[stageIdx] = (bool) eq && ((bool) semValidArray[stageIdx]);
    }
  }

  bool allSemsCorrect() {
    char* hostSemValids = new char[numStages];

    CUDA_CHECK(cudaMemcpy(hostSemValids, semValidArray, numStages * sizeof(char), cudaMemcpyDeviceToHost));
    
    bool eq = true;
    for (uint i = 0; i < numStages; i++) {
      eq = eq && (bool)hostSemValids[i];
    }

    delete hostSemValids;
    return eq;
  }

  ~CuSyncTest() {
    // CUDA_CHECK(cudaFree(semValidArray));
  }
};