# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
import csv
from common import *
import math 
import matplotlib.ticker as mtick

csv_file = sys.argv[1]
pdf_name = sys.argv[2]

only_one_h = True
attention_or_mlp = "attention" if ("attention" in csv_file) else "mlp"
model = "gpt3" if "gpt3" in csv_file else "llama"
gpu = "a100" if "a100" in csv_file else ("v100" if "v100" in csv_file else "")

only_streamk = False
if len(sys.argv) > 3 and sys.argv[3] == "only_streamk":
    only_streamk = True
import math
import csv
mInd = 0
seqInd = 1
hInd = 2
syncTypeInd = 3
streamkInd = 4
torchInd = 4
baselineInd = 4
stdevBaselineInd = 5
# matmul1Ind = 6
# matmul2Ind = 7
# maxtbsInd = 8
# matmul1TbsInd = 9
# matmul2TbsInd = 10
overlapInd = 8
stdevOverlapInd = 9

def load_csv(csv_file):
    data = []
    with open(csv_file, 'r') as f:
        csv_reader = csv.reader(f,delimiter='&')
        for i, row in enumerate(csv_reader):
            row_new = []
            for e in row:
                row_new.append(e.strip())
            row = row_new
            data += [row]
    
    return data

data = load_csv(csv_file)

import matplotlib.pyplot as plt
import numpy as np
if attention_or_mlp == "attention":
    width = 0.3
else:
    width = 0.4

# fig = plt.subplots(figsize =(10, 7))
m = []
h = []
torchT = []
baseline = []
stdevBaseline = []
matmul1 = []
softmax = []
matmul2 = []
maxtbs = []
matmul1Tbs = []
matmul2Tbs = []
rowOverlap = []
stdevRowOverlap = []
tileOverlap = []
stdevTileOverlap = []
stridedTileOverlap = []
stdevStridedTileOverlap = []
maxSpeedup = [] 
analyticalOverlapTimes = []
streamK = []
stdevstreamk = []

rowIdx = 0

#Time t is in microseconds
def flops(model, m, t):
    t = t/1e6
    if model == "llama":
        H = 8192
        FFN = ((H+128-1)//128)*128
        return (2*(m * 2 * FFN * H + m*FFN*H)/t)/1e12
    elif model == "gpt3":
        H = 12288
        FFN = H/2
        return (2*(m * FFN * H + m*FFN*H)/t)/1e12

def flops_for_all_rows(model, batches, times):
    flops_list = []
    for m,t in zip(batches, times):
        flops_list += [flops(model, m, t)]
    return flops_list

while rowIdx < len(data):
    # print(rowIdx)
    row = data[rowIdx]
    i = 0
    while rowIdx < len(data) and i < (4 if attention_or_mlp == 'mlp' else 6):
        row = data[rowIdx]
        if row[syncTypeInd] == 'streamk':
            streamK += [float(row[streamkInd])]
        elif row[syncTypeInd] == 'rowsync':
            rowOverlap += [float(row[overlapInd])]
        elif row[syncTypeInd] == 'baseline':
            m += [int(row[mInd])]
            baseline += [float(row[baselineInd])]
        elif row[syncTypeInd] == 'tilesync':
            tileOverlap += [float(row[overlapInd])]
        elif row[syncTypeInd] == 'stridedsync':
            stridedTileOverlap += [float(row[overlapInd])]
        elif row[syncTypeInd] == 'torch':
            torchT += [float(row[torchInd])]
        rowIdx += 1
        i += 1

# baseline = flops_for_all_rows(model, m, baseline)
# streamK = flops_for_all_rows(model, m, streamK)
# rowOverlap = flops_for_all_rows(model, m, rowOverlap)
# tileOverlap = flops_for_all_rows(model, m, tileOverlap)

if __name__ == "__main__":
    # secFactor = 1e3 if (secs == "ms") else 1e6
    torchT = np.array(torchT)
    baseline = np.array(baseline)
    ind = np.arange(len(baseline))
    matmul1 = np.array(matmul1)
    matmul2 = np.array(matmul2)
    softmax = np.array(softmax)
    stdevBaseline = np.array(stdevBaseline)
    rowOverlap = np.array(rowOverlap)
    stdevRowOverlap = np.array(stdevRowOverlap)
    tileOverlap = np.array(tileOverlap)
    streamK = np.array(streamK)
    stdevTileOverlap = np.array(stdevTileOverlap)
    analyticalOverlapTimes = np.array(analyticalOverlapTimes)

    cutlassSpeedup = (torchT - baseline)/torchT*100
    cusync = np.minimum(rowOverlap, tileOverlap)
    cusyncSpeedup = (torchT - cusync)/torchT*100
    cusyncOverCUTLASS = (baseline - cusync)/baseline*100 
    if gpu == "a100":
        streamKSpeedup = (torchT - streamK)/torchT*100
    else:
        streamKSpeedup = np.array([0])
    
    cusyncSpeedup = np.clip(cusyncSpeedup, -5, 45)
    cutlassSpeedup = np.clip(cutlassSpeedup, -5, 45)
    streamKSpeedup = np.clip(streamKSpeedup, -5, 45)
    cusyncOverCUTLASS = np.clip(cusyncOverCUTLASS, -5, 45)

    # analyticalSpeedup = baseline/analyticalOverlapTimes
    fig, ax2 = plt.subplots(1,1,sharex=True)
    p0 = ax2.plot(ind, cutlassSpeedup, 'o', color=colors[0])
    p1 = ax2.plot(ind, cusyncOverCUTLASS, marker='+', color=colors[1])
    p2 = ax2.plot(ind, cusyncSpeedup, 'x', color=colors[2])
    if gpu == "a100":
        p3 = ax2.plot(ind, streamKSpeedup, 's',color=colors[3])

    # if attention_or_mlp == "attention":
    #     stridedTileSpeedup = (baseline - stridedTileOverlap)/baseline * 100
    #     p3 = ax2.plot(ind, stridedTileSpeedup,'v',color=colors[2])
    #     print(stridedTileSpeedup)
    #     for i, f in enumerate(np.maximum(np.maximum(rowSpeedup, tileSpeedup), stridedTileSpeedup)):
    #         ax2.text(i, f+1, "%.0f"%round(f, 0), color = 'black', ha = 'center', rotation=0)
    # else:
    # for i, f in enumerate(cusyncSpeedup):
    #     ax2.text(i*2+2, f+1, "%.0f"%round(f,0), color = 'black', ha = 'center', rotation=0)
    
    # p4 = ax2.plot(ind, streamKSpeedup, 'x',color=colors[3])
 
    # p3 = ax2.plot(list(range(0, len(data)//2)), analyticalSpeedup)
    
    # for bar1, d in zip(p1, cusyncSpeedup):
    #     ax2.text(bar1.get_x()+bar1.get_width()/2-0.05, bar1.get_height()+0.5, "%.0f"%(round(d,0)), 
    #     color = 'black', ha = 'center', va = 'center', rotation=0)

    # for bar1, speedup in zip(p3, fastkronspeedup):
    #     ax2.text(bar1.get_x()+bar1.get_width()/2+0.04, bar1.get_height()+0.05, r"%.2f$\times$"%(1/speedup), color = 'black', ha = 'center', va = 'center', rotation=0, fontsize='large')
    # if only_one_h and attention_or_mlp == True:
    #     plt.ylim(0.6, 1.3)
    #     plt.yticks([0.6+0.1*i for i in range(0, 7)])
    # else:
    # ax2.margins(0.02)
    max_speedup = max([np.amax(cusyncSpeedup), np.amax(streamKSpeedup), np.amax(cusyncOverCUTLASS)])
    print(max_speedup)
    max_speedup = int(((max_speedup+10-1)//10)*10)
    print(max_speedup)
    plt.ylim(-5, max_speedup)
    plt.yticks(ticks=[-5+5*i for i in range(0, max_speedup//5+1)],
               labels=["%d%%"%(-5+5*i) for i in range(0, max_speedup//5 + 1)])
    # ax2.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=None))
    # ax2.set_yticklabels(["%d%%"%(-5+5*i) for i in range(0, 9)])
    # plt.yticks(["%d%(-5+5*i) for i in range(0, 7)])
    # plt.xlim(-1,ind[-1]+1)
    # plt.title('Contribution by the teams')
    plt.axhline(0, color='black', ls='dotted')
    # plt.yticks(np.arange(0, 1.25, 0.25))
    if attention_or_mlp == "mlp":
        xt = list(m)
        plt.xticks(ind, xt, rotation=90)
        
        plt.ylabel('Percentage Improvement of X/Y')
        # ax2.get_yaxis().set_label_coords(-0.17,0.4)
        plt.xlabel("Number of Tokens in %s MLP on A100"%(model.upper()))
        # ax2.get_xaxis().set_label_coords(0.45,-0.4)
        if gpu == "a100":
            labels = (p0[0], p3[0], p2[0], p1[0])
            legends = ('CUTLASS/PyTorch', 'StreamK/PyTorch', 'CuSync/PyTorch', 'CuSync/CUTLASS')
        else:
            labels = (p0[0], p2[0], p1[0])
            legends = ('CUTLASS/PyTorch', 'CuSync/PyTorch', 'CuSync/CUTLASS')

        plt.legend(labels, legends,
                   loc='upper left', bbox_to_anchor=(-0.1, 1.20),
                   ncol=2,columnspacing=1,handlelength=1.7)
    else:
        ax2.get_yaxis().set_visible(False)
        ax2.get_xaxis().set_label_coords(0.45,-0.4)
        xt = list((2**i for i in range(0, len(ind))))
        if "attention" in csv_file:
            xt = ["512, 0", "1024, 0", "2048, 0", "1, 512", "2, 512", "4, 512", "1, 1024", "2, 1024", "4, 1024", "1, 2048", "2, 2048", "4, 2048"]
        plt.xticks(ind, xt, rotation=90)
        if attention_or_mlp == "attention" and model == "gpt3":
            plt.legend((p1[0], p2[0], p3[0], p4[0]), 
                    ('RowSync', 'TileSync+WRT', 'StridedTileSync+WRT', 'StreamK'),
                    loc='upper left', bbox_to_anchor=(-0.01, 1.16),
                    ncol=4,columnspacing=1,handlelength=1.7)
        plt.xlabel("B$\\times$S, S'")
        
    plt.rcParams["font.family"] = "libertine"
    #FIGURES_DIR = "./"
    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.1)
    # if only_one_h:
    # else:
    #     fig.set_size_inches(8.5, 2.5)
    # if attention_or_mlp == "mlp" and model == "gpt3":
    fig.set_size_inches(4, 3)
    # else:
    #     fig.set_size_inches(3.2, 2.4)
        # ax.set_xticks([])
    FIGURES_DIR = "./"
    fig.savefig(FIGURES_DIR+pdf_name,bbox_inches='tight',pad_inches=0)
    #plt.show()
