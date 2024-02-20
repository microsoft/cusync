# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import subprocess
import re
import sys
import os
import tile_sizes_db
import time

attention_or_mlp = sys.argv[1].lower()
model = sys.argv[2].lower()
arch = sys.argv[3].lower()

assert attention_or_mlp in ["attention", "mlp"]
assert arch.lower() in ["v100", "a100"]

baselineTimes = {}
cublasTimes = {}
overlappedTimes = {}
minimumTimes = {}
speedup = {}
maxspeedup = {}
import json
from statistics import stdev

def getAllTimes(s, START, END):
  '''Parse output of binaries to obtain list of times
  '''
  alltimes = {}
  assert START in s
  assert END in s
  s = s[s.find(START):s.find(END)]
  s = s[s.find("\n"):]
  alljsons = []
  for l in re.findall(r".+", s):
    j = json.loads(l)
    alljsons += [j]
  
  def sortkey(elem):
    return elem["Total"]
  
  alljsons.sort(key=sortkey)
  p = 0.9
  alljsons = alljsons[:int(len(alljsons)*0.9)]
  for j in alljsons:
    for k in j:
      if k not in alltimes:
        alltimes[k] = [] 
      alltimes[k] += [float(j[k])]

  return alltimes

def avg(l):
  return sum(l)/len(l)

def slurp(path):
  with open(path, "r") as f:
    return f.read()

def buildDir(f):
  return 'build/'+f

if not os.path.exists(buildDir("")):
  os.mkdir(buildDir(""))

def resultsDir(f):
  '''Results directory'''
  return 'results/'+f

'''Make results directory if not exists'''
if not os.path.exists(resultsDir("")):
  os.mkdir(resultsDir(""))

def getStreamKTimes(output):
  runtime = re.findall(r'\s*Avg runtime: ([\d\.]+)', output)
  return float(runtime[0])

def genAndMakeStreamK(batchInfo, gemmidx):
  inFile = "streamk.cu"
  outFile = buildDir("streamk-eval.cu")
  tilesCode = """using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<%d, %d, %d>;  
using ShapeMMAWarp = cutlass::gemm::GemmShape<%d, %d, %d>;"""
  tileSize = batchInfo[syncPolicy]["TileSizes"] if "TileSizes" in batchInfo["baseline"] else  batchInfo["TileSizes"]
  if len(tileSize) > 1:
    tilesCode = tilesCode % tuple(batchInfo["TileSizes"][gemmidx])
  else:
    tilesCode = tilesCode % tuple(batchInfo["TileSizes"])

  NumStages = batchInfo[syncPolicy]["NumStages"] if "NumStages" in batchInfo["baseline"] else  batchInfo["NumStages"]
  if isinstance(NumStages, list):
    NumStages = NumStages[gemmidx]

  numStagesCode = "const uint NumStages = %d;\n" % NumStages
  tilesCode += numStagesCode

  if model == "gpt3" and attention_or_mlp == "mlp" and gemmidx == 0:
    tilesCode += "#define MLP_GPT3_GEMM1"

  fileContents = slurp(inFile)
  tilesCodeStart = fileContents.find("//<eval tiles>") + len("//<eval tiles>")
  tilesCodeEnd = fileContents.find("//</eval tiles>")
  fileContents = fileContents[0:tilesCodeStart] + "\n" + tilesCode + "\n" + fileContents[tilesCodeEnd:]
  with open(outFile, "w") as f:
    f.write(fileContents)
  (s,o) = subprocess.getstatusoutput(f"rm -r {buildDir('streamk-eval')} ; make {buildDir('streamk-eval')}")
  if s != 0:
    print(o)
    sys.exit(0)

def deleteFiles(syncPolicies, attention_or_mlp):
  command = "rm -f "
  for policy in syncPolicies:
    if attention_or_mlp == 'attention' and policy == 'stridedsync':
      command += buildDir("%s-%s-eval-%s "%(attention_or_mlp, model, policy))
    else:
      command += buildDir("%s-eval-%s "%(attention_or_mlp, policy))
  
  (s,o) = subprocess.getstatusoutput(command)

  if s != 0:
    print(o)
    sys.exit(0)

def makeFiles(syncPolicies, attention_or_mlp):
  command = "make "
  for policy in syncPolicies:
    if attention_or_mlp == 'attention' and policy == 'stridedsync':
      command += buildDir("%s-%s-eval-%s "%(attention_or_mlp, model, policy))
    else:
      command += buildDir("%s-eval-%s "%(attention_or_mlp, policy))

  flags = "-j"
  command += flags
  (s,o) = subprocess.getstatusoutput(command)

  if s != 0:
    print(o)
    sys.exit(0)
  
def genFiles(batchInfo, syncPolicy, attention_or_mlp):
  inMLPFile = "mlp.cu" if attention_or_mlp == "mlp" else "attention.cu"
  outMLPFile = buildDir(attention_or_mlp + "-eval-" + syncPolicy + ".cu")
  tilesTemplate = """using ShapeThreadBlock%d = cutlass::gemm::GemmShape<%d, %d, %d>;  
using ShapeWarp%d = cutlass::gemm::GemmShape<%d, %d, %d>;"""
  tilesCode = ""

  tileSize = batchInfo[syncPolicy]["TileSizes"] if "TileSizes" in batchInfo[syncPolicy] else  batchInfo["TileSizes"]
  if len(tileSize) > 1:
    for i,tile in enumerate(tileSize):
      tilesCode += tilesTemplate % tuple([i+1] + tile[:3] + [i+1] + tile[3:])
      tilesCode += "\n"
  else:
    tilesTemplate = """using ShapeThreadBlock = cutlass::gemm::GemmShape<%d, %d, %d>;  
using ShapeWarp = cutlass::gemm::GemmShape<%d, %d, %d>;"""
    tilesCode = tilesTemplate % tuple(tileSize[0])

  NumStages = batchInfo[syncPolicy]["NumStages"] if "NumStages" in batchInfo[syncPolicy] else  batchInfo["NumStages"]
  numStagesCode = ""
  NumStagesTemplate = "const uint NumStages%d = %d;\n"
  if isinstance(NumStages, list) and len(NumStages) > 1:
    for i,num in enumerate(NumStages):
      numStagesCode += NumStagesTemplate % tuple([i+1, int(num)])
  else:
    numStagesCode = NumStagesTemplate %(1, int(NumStages)) + \
                NumStagesTemplate %(2, int(NumStages))
  tilesCode+=numStagesCode
  batchInfo = batchInfo["tilesync"] if syncPolicy == "stridedsync" or syncPolicy == 'baseline' else batchInfo[syncPolicy]
  if "SoftmaxRowTile" in batchInfo:
    tilesCode += "\nconst uint SoftmaxRowTile = %d;"%batchInfo["SoftmaxRowTile"]
  mlpFileContents = slurp(inMLPFile)
  tilesCodeStart = mlpFileContents.find("//<eval tiles>") + len("//<eval tiles>")
  tilesCodeEnd = mlpFileContents.find("//</eval tiles>")
  mlpFileContents = mlpFileContents[0:tilesCodeStart] + "\n" + tilesCode + "\n" + mlpFileContents[tilesCodeEnd:]
  optimizationsStart = mlpFileContents.find("//<OPTIMIZATIONS>") + len("//<OPTIMIZATIONS>")
  optimizationsEnd = mlpFileContents.find("//</OPTIMIZATIONS>")
  optimizationsCode = ""
  if model == "GPT3".lower():
    optimizationsCode += f"#define {attention_or_mlp.upper()}_GPT3\n"
  elif model == "LLAMA".lower():
    optimizationsCode += f"#define {attention_or_mlp.upper()}_LLAMA\n"

  if syncPolicy != 'baseline':
    if "AvoidCustomOrder" in batchInfo and batchInfo["AvoidCustomOrder"] == True:
      optimizationsCode += "#define AVOID_CUSTOM_ORDER"+"\n"
    else:
      optimizationsCode += "#undef AVOID_CUSTOM_ORDER"+"\n"
    if "AvoidWaitKernel" in batchInfo and batchInfo["AvoidWaitKernel"] == True:
      optimizationsCode += "#define AVOID_WAIT_KERNEL"+"\n"
    else:
      optimizationsCode += "#undef AVOID_WAIT_KERNEL"+"\n"
    if "ReorderTileLoads" in batchInfo and batchInfo["ReorderTileLoads"] == True:
      optimizationsCode += "#define REORDER_TILE_LOADS"+"\n"
    else:
      optimizationsCode += "#undef REORDER_TILE_LOADS"+"\n"
    if "NoAtomicAdd" in batchInfo and batchInfo["NoAtomicAdd"] == True:
      optimizationsCode += "#define NO_ATOMIC_ADD"+"\n"
    else:
      optimizationsCode += "#undef NO_ATOMIC_ADD"+"\n"

  optimizationsCode += "#define " + syncPolicy.upper() + "\n"
  optimizationsCode += "#define " + "EVAL_TILE_SIZES" + "\n"
  mlpFileContents = mlpFileContents[0:optimizationsStart] + "\n" + optimizationsCode + "\n" + mlpFileContents[optimizationsEnd:]
  if os.path.exists(outMLPFile):
    with open(outMLPFile, "r") as f:
      oldContents = f.read()
      if mlpFileContents == oldContents:
        return
  with open(outMLPFile, "w") as f:
    f.write(mlpFileContents)

tiles_field_str = f"{model}_{attention_or_mlp}_{arch}"
tiles = getattr(tile_sizes_db, tiles_field_str)

if model.lower() == "GPT3".lower():
  H = 12288
  FFN = int(4*H/8)
elif model.lower() == "llama".lower():
  H = 8192
  FFN = int(((8192/3+127)//128)*128)#int(2/3 * 4 * H/8)
else:
  print ("No Hidden dim for ", model)
  sys.exit(0)

policies = ['rowsync', 'tilesync', 'stridedsync']
if 'stridedsync' in policies and attention_or_mlp == 'mlp':
  policies.pop(policies.index('stridedsync'))

deleteFiles(policies+['baseline'], attention_or_mlp)

if attention_or_mlp == "mlp":
  cases = (([1,2,4,8,16,32,64,128,256]) if arch=='v100' else []) +\
          [512+256*i for i in range(0, 11)] 
else:
  #cases = [(0,256), (0,512), (0, 1024), (0, 2048), (1024,1), (1024,4), (2048,1), (2048,4)]
  cases = [(512,1),(512,2), (512,4), (1024,1), (1024,2), (1024,4), (2048,1), (2048,2), (2048,4)]

results_csv = ""

for case in cases:
  if attention_or_mlp == "attention":
    m = case[1]
    seq = case[0]
  else:
    m = case
    seq = 0

  caseTiles = None
  if attention_or_mlp == "attention":
    caseTiles = tiles[seq][m]
  else:
    if m > 2048:
      caseTiles = tiles[4096]
    else:
      caseTiles = tiles[m]

  if True:
    if attention_or_mlp == "attention":
      (s, o) = subprocess.getstatusoutput(f"python3 torch-baselines/torchAttention.py {m} {int(H/8)} {H} {H}")
    else:
      (s, o) = subprocess.getstatusoutput(f"python3 torch-baselines/torchmlp.py {m} {model}")
    
    if s == -1:
      print("error " + o)
    else:
      ctime = o
      cublasTimes[m] = ctime

    result_row = f'{m} & {seq} & {H} & {"torch"} & {"%.2f"%float(ctime)}'
    print(result_row)
    results_csv += result_row + "\n"

  if arch == "a100":
    genAndMakeStreamK(caseTiles, 0)
    streamk_command = buildDir("streamk-eval") + f" --m={m} --alpha=1 --beta=0 --iterations=20 "
    (s, o) = subprocess.getstatusoutput(streamk_command + f"--n={int(2*FFN if model=='llama' else FFN)} --k={H} " + f"--split={caseTiles['baseline']['split_ks'][0]}")
    if s != 0:
      print("StreamK Error")
      print(o)

    firstGeMMStreamK = getStreamKTimes(o)
    genAndMakeStreamK(caseTiles, 1)
    (s, o) = subprocess.getstatusoutput(streamk_command + f"--n={H} --k={int(FFN)} " + f"--split={caseTiles['baseline']['split_ks'][1]}")
    if s != 0:
      print("StreamK Error")
      print(o)

    secondGeMMStreamK = getStreamKTimes(o)
    total = firstGeMMStreamK + secondGeMMStreamK
    result_row = f'{m} & {seq} & {H} & {"streamk"} & {"%.2f"%(total*1000)} & {"%.2f"%(firstGeMMStreamK*1000)} & {"%.2f"%(secondGeMMStreamK*1000)}'
    print(result_row)
    results_csv += result_row + "\n"

  baselineDone = False
  bTimeTotal = 0

  for syncPolicy in (policies+['baseline']):
    genFiles(caseTiles, syncPolicy, attention_or_mlp)

  makeFiles(policies+['baseline'], attention_or_mlp)
  
  split_ks = caseTiles['baseline']['split_ks']
  splitKArgs = " " + " ".join([f"--split-k{i+1} {split_ks[i]}" for i in range(len(split_ks))])
  commandArgs = f" --batch {m} --check false --model {model.lower()}"
  if attention_or_mlp == "attention":
    commandArgs += f" --seqlen {(seq - m) if seq > m else seq}"
  baselineCommand = buildDir(f"{attention_or_mlp}-eval-baseline") + commandArgs + splitKArgs + " --policy baseline"
  (s, o) = subprocess.getstatusoutput(baselineCommand)
  # print(o)
  if "Invalid" in o:
    pass
  elif s != 0:
    print("error " + o)
  else:
    # print(o)
    baselinetimes = getAllTimes(o, 'START-BASELINE', 'END-BASELINE')
    bTimeTotal = baselinetimes["Total"]
    bTimeMatmul1 = baselinetimes["matmul1Time"]
    bTimeMatmul2 = baselinetimes["matmul2Time"]
    result_row = f'{m} & {seq} & {H} & baseline & {"%.2f"%avg(bTimeTotal)} & {"%.2f"%stdev(bTimeTotal)} & {"%.2f"%avg(bTimeMatmul1)} & {"%.2f"%avg(bTimeMatmul2)}'
    results_csv += result_row + "\n"
    print(result_row)
    baselineDone = True

  for syncPolicy in policies:
    split_ks = (caseTiles["tilesync"] if syncPolicy == "stridedsync" else caseTiles[syncPolicy])["split_ks"]
    splitKArgs = " " + " ".join([f"--split-k{i+1} {split_ks[i]}" for i in range(len(split_ks))])
    command = ""
    # if attention_or_mlp == 'attention' and syncPolicy == 'stridedsync':
    #   command += buildDir("%s-%s-eval-%s "%(attention_or_mlp, model, syncPolicy))
    # else:
    command += buildDir("%s-eval-%s "%(attention_or_mlp, syncPolicy))
    command += commandArgs + splitKArgs + " --policy cusync"
    (s, o) = subprocess.getstatusoutput(command)
  
    otime = -1
    if "Invalid" in o:
      pass
    elif s != 0:
      print("error " + o)
    else:
      overlaptimes  = getAllTimes(o, 'START-OVERLAPPED', 'END-OVERLAPPED')
      otime = overlaptimes["Total"]

    result_row = f'{m} & {seq} & {H} & {syncPolicy} & {"%.2f"%avg(bTimeTotal)} & {"%.2f"%stdev(bTimeTotal)} & {"%.2f"%avg(bTimeMatmul1)} & {"%.2f"%avg(bTimeMatmul2)} & {"%.2f"%avg(otime)} & {"%.2f"%stdev(otime)} & {"%.2f"%(100 - avg(otime)/avg(bTimeTotal)*100)}'
    results_csv += result_row + "\n"
    print(result_row)
  time.sleep(5)

with open(os.path.join(resultsDir(""), f"{attention_or_mlp}-{model}-{arch}.csv"), "w") as f:
  f.write(results_csv)
