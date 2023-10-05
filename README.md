CuSync 1.0
------
CuSync is a framework to synchronize tile-based CUDA kernels in a fine-grained manner.
CuSync allows synchronization of dependent tiles, i.e. thread blocks, of a chain of producer and consumer kernels.
Synchronizing thread blocks instead of kernels allows concurrent execution of independent thread blocks. For more details please read https://arxiv.org/abs/2305.13450.

Clone the repo and its submodules using `git clone --recurse-submodules https://github.com/parasailteam/cusync.git`.

If already cloned and want to clone submodules, use `git submodule update --init --recursive`.

Usage
-------

### CUDA code
Each kernel is associated with a CuStage.
A CuStage is defined as follows:
```
using ProdCuStage = CuStage<CuStageType::Producer, RowMajorXYZ, TileSync>;
using ConsCuStage = CuStage<CuStageType::Consumer, RowMajorXYZ, TileSync>;
```
A CuStage is a producer, consumer or both. 
Defining a CuStage also requires tile processing order and synchronization policy.

All kernels takes the custage object associated with them. 
For each tile a consumer kernel waits for input tiles to be processed and post the status of processed output tile.
```
template <typename CuStageTy, int BLOCK_SIZE>
__global__ void kernel(CuStageTy custage, ... args) {
  dim3 tile = custage.tile();
  custage.wait(tile);
  //computation
  custage.post(tile);
}
```

### Initialization
In the host code, each custage is initialized with the grid sizes, tile sizes, and synchronization policy.
Finally, initialize the producer consumer pair.
```
TileSync sync;
dim3 tilesize = threads;
ProdCuStage prod(grid, tilesize, sync);
ConsCuStage cons(grid, tilesize, sync);
initProducerConsumer(prod, cons);
```

### Invoking kernels
We need to define two different streams `prod_stream` and `cons_stream`.
First the producer kernel is invoked by passing the ProdCuStage object.
Then invoke a wait kernel on the `cons_stream`.
Finally, invoke the consumer kernel by passing the ConsCuStage object and wait for execution to complete.

```
//Invoke producer kernel
kernel<ProdCuStage, 32>
  <<<grid, threads, 0, prod_stream>>>(prod, ... args);

//Invoke wait kernel
prod.invokeWaitKernel(cons_stream);

//Invoke consumer kernel
kernel<ConsCuStage, 32>
  <<<grid, threads, 0, cons_stream>>>(prod, ... args);

cudaDeviceSynchronize();
```

### Multiple runs
To run producer and consumer kernel multiple times, we need to increment an internal counter for semaphore.

```
prod.incrementIter();
cons.incrementIter();
```

### Examples
* Two dependent CUDA memcpy kernels: File `src/tests/simple-test.cu` contains an example of synchronizing two dependent CUDA kernels that copies one array to another.
* Two Dependent CUDA GeMMs:  The directory `src/examples/matrixMul` shows an example of synchronizing two dependent CUDA Matmuls.

Tests
------
Run tests using `make run-simple-test`