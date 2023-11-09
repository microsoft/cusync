import torch
import sys
import time

M = int(sys.argv[1])
N = int(sys.argv[2])
K = int(sys.argv[3])
L = int(sys.argv[4])
model = sys.argv[5]
assert model in ['gpt3', 'llama']

X = torch.ones((M, K), dtype=torch.half).cuda()
W1 = torch.ones((K, N), dtype=torch.half).cuda()
W2 = torch.ones((N, L), dtype=torch.half).cuda()
V = torch.ones((K, N), dtype=torch.half).cuda()

epochs = 20
for i in range(epochs):
    XW1 = X@W1
torch.cuda.synchronize()

start = time.time_ns()

if model == 'gpt3':
    for i in range(epochs):
        XW1 = X@W1
        out = XW1@W2
    torch.cuda.synchronize()
elif model == 'llama':    
    for i in range(epochs):
        XW1 = X@W1
        XV = X@V
        out = XW1@W2
    torch.cuda.synchronize()
end = time.time_ns()

print((end-start)/epochs/1e3)