import torch
import sys
import time

M = int(sys.argv[1])
model = sys.argv[2]
assert model in ['gpt3', 'llama']

if model == 'gpt3':
    H = 12288
    X = torch.ones((M, H), dtype=torch.half).cuda()
    W1 = torch.ones((H, H//2), dtype=torch.half).cuda()
    W2 = torch.ones((H//2, H), dtype=torch.half).cuda()
else:
    H = 8192
    H2 = ((H//3 + 127)//128)*128
    X = torch.ones((M, H), dtype=torch.half).cuda()
    W1 = torch.ones((H, 2*H2), dtype=torch.half).cuda()
    W2 = torch.ones((H2, H), dtype=torch.half).cuda()
    XW1_ = torch.ones((M, H2), dtype=torch.half).cuda()

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
        out = XW1_@W2
    torch.cuda.synchronize()
end = time.time_ns()

print((end-start)/epochs/1e3)