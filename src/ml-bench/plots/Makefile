TRANSFORMER_RESULTS=../transformer/results/
CONV_RESULTS=../volta_conv2d/results/

.SECONDEXPANSION:

mlp-gpt3-v100: $(TRANSFORMER_RESULTS)/$$@.csv
	python3 plotGPT.py $(TRANSFORMER_RESULTS)/$@.csv $@.png

mlp-gpt3-a100: $(TRANSFORMER_RESULTS)/$$@.csv
	python3 plotGPT.py $(TRANSFORMER_RESULTS)/$@.csv $@.png

mlp-llama-v100: $(TRANSFORMER_RESULTS)/$$@.csv
	python3 plotGPT.py $(TRANSFORMER_RESULTS)/$@.csv $@.png

mlp-llama-a100: $(TRANSFORMER_RESULTS)/$$@.csv
	python3 plotGPT.py $(TRANSFORMER_RESULTS)/$@.csv $@.png

all: mlp-gpt3-v100 mlp-gpt3-a100.pdf mlp-llama-v100.pdf mlp-llama-a100.pdf 

clean:
	rm -rf *.pdf ; rm -rf *.png
