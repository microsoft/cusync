CuSync
---------------------

CuSync is a framework to synchronize tile-based CUDA kernels in a fine-grained manner.
With CuSync, a programmer can write policies to synchronize dependent tiles, i.e. thread blocks, of a chain of producer and consumer kernels.
Synchronizing thread blocks instead of kernels allows concurrent execution of independent thread blocks, thereby, improving the utilization during the last thread block waves on the GPU.
More details are available at https://arxiv.org/abs/2305.13450.

## Performance

The graphs below shows percentage improvement over GPT3 and LLAMA MLPs using optimized NVIDIA CUTLASS GeMMs on NVIDIA Tesla A100 and NVIDIA Tesla V100 GPUs for 8 way model parallelism GPT3 175B (H=12288) and LLAMA 65.2B (H=8192).
NVIDIA CUTLASS StreamK is another method to optimize the utilization during the last thread block wave.
PyTorch in the below experiments only performs GeMM and not the pointwise computations like GeLU, while CUTLASS implementations fuse these computations with the first GeMM. 

#### NVIDIA Tesla A100 SXM4 80GB with CUDA 12.2
![](https://github.com/parasailteam/cusync/blob/main/src/ml-bench/plots/mlp-gpt3-a100.png?raw=true)
![](https://github.com/parasailteam/cusync/blob/main/src/ml-bench/plots/mlp-llama-a100.png?raw=true)

#### NVIDIA Tesla V100 SXM2 32GB with CUDA 12.2

Coming Soon

![a](https://github.com/parasailteam/cusync/blob/main/src/ml-bench/plots/mlp-gpt3-v100.png?raw=true)
![a](https://github.com/parasailteam/cusync/blob/main/src/ml-bench/plots/mlp-llama-v100.png?raw=true)

## Usage

#### Clone
Clone the repo and its submodules using 

```git clone --recurse-submodules https://github.com/parasailteam/cusync.git```

If already cloned and want to clone submodules, use

```git submodule update --init --recursive```

#### Example
An example of synchronizing two dependent GeMMs is provided in the `src/example/`. Moreover, there are small tests in `tests/` that can be used as examples.

#### CuSync + CUTLASS

The repo also provides CUTLASS GeMM structs augmented with CuSync structures in `src/include/cusync-cutlass/`.
The MLP code in `src/ml-bench/transformer` provides a good way to use CUTLASS cusync.

## Tests

Run tests using `make tests`


