import torch
import sys
import time

M = int(sys.argv[1])
N = int(sys.argv[2])
K = int(sys.argv[3])
L = int(sys.argv[4])

X = torch.ones((M, K), dtype=torch.half).cuda()
QKV = torch.ones((K, N*3), dtype=torch.half).cuda()
W2 = torch.ones((N, L), dtype=torch.half).cuda()

for i in range(10):
    XQKV = X@QKV
    XQ = XQKV[:,0:N]
    XK = XQKV[:,N:2*N]
    XV = XQKV[:,2*N:3*N]
    
    XQDotXK = XQ*XK
    softmax = torch.softmax(XQDotXK, dim = 0)
    softmaxDotXV = softmax*XV
    dropout = torch.dropout(softmaxDotXV, 1.0, False)
    out = dropout@W2
torch.cuda.synchronize()

epochs = 20
start = time.time_ns()

for i in range(epochs):
    XQKV = X@QKV
    XQ = XQKV[:,0:N]
    XK = XQKV[:,N:2*N]
    XV = XQKV[:,2*N:3*N]
    
    # XQDotXK = XQ*XK
    # softmax = torch.softmax(XQDotXK, dim = 0)
    # softmaxDotXV = softmax*XV
    # dropout = torch.dropout(softmaxDotXV, 1.0, False)
    out = XQ@W2
torch.cuda.synchronize()
end = time.time_ns()

print((end-start)/epochs/1e3)