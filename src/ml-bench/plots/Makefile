TRANSFORMER_RESULTS=../transformer/results/
CONV_RESULTS=../volta_conv2d/results/

mlp-gpt3-v100.pdf: $(TRANSFORMER_RESULTS)/mlp-gpt3-v100.csv
	python3 plotGPT.py $(TRANSFORMER_RESULTS)/mlp-gpt3-v100.csv mlp-gpt3-v100.pdf

mlp-gpt3-a100.pdf: $(TRANSFORMER_RESULTS)/mlp-gpt3-a100.csv
	python3 plotGPT.py $(TRANSFORMER_RESULTS)/mlp-gpt3-a100.csv mlp-gpt3-a100.pdf

mlp-llama-v100.pdf: $(TRANSFORMER_RESULTS)/mlp-llama-v100.csv
	python3 plotGPT.py $(TRANSFORMER_RESULTS)/mlp-llama-v100.csv mlp-llama-v100.pdf

mlp-llama-a100.pdf: $(TRANSFORMER_RESULTS)/mlp-llama-a100.csv
	python3 plotGPT.py $(TRANSFORMER_RESULTS)/mlp-llama-a100.csv mlp-llama-a100.pdf

all: mlp-gpt3-v100 mlp-gpt3-a100.pdf mlp-llama-v100.pdf mlp-llama-a100.pdf 

clean:
	rm -rf *.pdf
