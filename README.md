CuSync
---------------------

CuSync is a framework to synchronize tile-based CUDA kernels in a fine-grained manner.
With CuSync, a programmer can write policies to synchronize dependent tiles, i.e. thread blocks, of a chain of producer and consumer kernels.
Synchronizing thread blocks instead of kernels allows concurrent execution of independent thread blocks, thereby, improving the utilization during the last thread block waves on the GPU.
More details are available at https://arxiv.org/abs/2305.13450.

## Performance

The graphs below shows percentage improvement over GPT3 and LLAMA MLPs using optimized NVIDIA CUTLASS GeMMs on NVIDIA Tesla A100 and NVIDIA Tesla V100 GPUs for 8 way model parallelism GPT3 175B (H=12288 FP16) and LLAMA 65.2B (H=8192 FP16).
NVIDIA CUTLASS StreamK is another method to optimize the utilization during the last thread block wave.
PyTorch in the below experiments only performs GeMM and not the pointwise computations like GeLU, while CUTLASS implementations fuse these computations with the first GeMM. 

#### NVIDIA Tesla A100 SXM4 80GB with CUDA 12.2
![](https://github.com/parasailteam/cusync/blob/main/src/ml-bench/plots/mlp-gpt3-a100.png?raw=true)
![](https://github.com/parasailteam/cusync/blob/main/src/ml-bench/plots/mlp-llama-a100.png?raw=true)

#### NVIDIA Tesla V100 SXM2 32GB with CUDA 12.2
![](https://github.com/parasailteam/cusync/blob/main/src/ml-bench/plots/mlp-gpt3-v100.png?raw=true)
![](https://github.com/parasailteam/cusync/blob/main/src/ml-bench/plots/mlp-llama-v100.png?raw=true)

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

## Evaluation

Instructions are in `src/ml-bench/README.md`.

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.


