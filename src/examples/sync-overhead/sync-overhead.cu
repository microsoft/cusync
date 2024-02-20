#include<stdio.h>

#include <time.h>
#include <sys/time.h>


#define CUDA_CHECK(cmd) do {                        \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",       \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0);

static double convertTimeValToDouble(struct timeval _time) {
  return ((double)_time.tv_sec)*1e6 + ((double)_time.tv_usec);
}

static struct timeval getTimeOfDay () {
  struct timeval _time;

  if (gettimeofday (&_time, NULL) == -1) {
    fprintf (stderr, "gettimeofday returned -1\n");
    perror ("");
    abort ();
  }

  return _time;
}

static double timeInMicroSeconds() {
  return convertTimeValToDouble(getTimeOfDay());
}

static double getCurrentTime() {
  return timeInMicroSeconds();
}

__global__ void kernel1(float *in, int i, volatile int* sync, bool cansync) {
	int linearid = threadIdx.x + blockIdx.x * blockDim.x;
	in[linearid] = i;
  __syncthreads();
  if (cansync && threadIdx.x == 0)
    sync[blockIdx.x] += 1;
}

__global__ void kernel2(float *out, float *in, volatile int* sync, bool cansync, int iter) {
	if (cansync && threadIdx.x == 0) {
		for (int i = threadIdx.x; i < 1; i += blockDim.x) {
      while (sync[i] < iter + 1);
    }
      // sync[blockIdx.x] = 0;
	}
	__syncthreads();
	int linearid = threadIdx.x + blockIdx.x * blockDim.x;
	out[linearid] = in[linearid] + 1;
}

int main() {
  float* in, *out;
  size_t size = 1 << 20;
	CUDA_CHECK(cudaMalloc(&in, size));
  CUDA_CHECK(cudaMalloc(&out, size));
  int* sync;
  CUDA_CHECK(cudaMalloc(&sync, size));
  CUDA_CHECK(cudaMemset(sync, size * sizeof(int), 0));
  unsigned int threads = 128;
  dim3 grid = {80*2 * (1024/threads), 1, 1};
  dim3 block = {threads, 1, 1}; 
  cudaStream_t prodstream, constream;

  int highestPriority;
  int lowestPriority;
  
  CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&lowestPriority, &highestPriority));

  CUDA_CHECK(cudaStreamCreateWithPriority(&prodstream, 0, highestPriority));
  CUDA_CHECK(cudaStreamCreateWithPriority(&constream, 0, lowestPriority));
  CUDA_CHECK(cudaDeviceSynchronize());
  double sync_exec = 0;
  for (int i = 0; i < 110; i++) {
    double s = getCurrentTime();
    kernel1<<<grid,block,0,prodstream>>>(in, 0, sync, true);
    kernel2<<<grid,block,0,prodstream>>>(out, in, sync, true, i);
    CUDA_CHECK(cudaDeviceSynchronize());
    double t = getCurrentTime();
    if (i >= 10)
      sync_exec += t - s;
  }

  printf("exec with sync %lf\n", sync_exec);
  CUDA_CHECK(cudaDeviceSynchronize());

  double exec = 0;
  for (int i = 0; i < 100; i++) {
    double s = getCurrentTime();
    kernel1<<<grid,block>>>(in, 0, sync, false);
    kernel2<<<grid,block>>>(out, in, sync, false, i);
    CUDA_CHECK(cudaDeviceSynchronize());
    double t = getCurrentTime();
    exec += t - s;
  }
  printf("exec without sync %lf\n", exec);
  printf("Overhead %lf %%\n", (sync_exec - exec)/exec * 100.);
}