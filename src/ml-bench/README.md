Evaluation
----------------

This directory contains instructions to run evaluation for results.

## Docker Image

The docker file with pre-requisites is in the main directory. Create the docker container using 

```
docker build -t cusync-cgo-24 .
docker run -it --gpus all cusync-cgo-24
cd /cusync
```

## Pre-requisites (Native Execution)

### Linux Installation
We recommend using Ubuntu 22.04 as the Linux OS. We have not tested our artifact with
any other OS but we believe Ubuntu 20.04 and 23.04 should
also work.

### Install Dependencies
Execute following commands to install dependencies.

```
sudo apt update
sudo apt install gcc linux-headers-$(uname -r) make g++ git python3 wget unzip python3-pip build-essential cmake
```

We use CUDA 12.2 in our experiments.

Install PyTorch using pip.
```
sudo pip3 install torch torchvision torchaudio
```

## Run Exeperiments

### MLP Results

Following commands will run all experiments to gather the
results

```
cd transformer
python3 eval_llm.py mlp gpt3
python3 eval_llm.py attention gpt3
python3 eval_llm.py mlp llama
python3 eval_llm.py attention llama
python3 allreduce_times.py
```

### Conv2D Results
Following commands will run all experiments to gather results

```
cd volta_conv2d
python3 eval_conv.py resnet
python3 eval_conv.py vgg
```

### Generate Plots
```
cd plots
make -j
```