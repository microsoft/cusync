import subprocess
import re
import sys
import os
attention_or_mlp = sys.argv[1]
model = sys.argv[2]

assert attention_or_mlp in ["attention", "mlp"]

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

def getStreamKTimes(output):
  runtime = re.findall(r'\s*Avg runtime: ([\d\.]+)', output)
  return float(runtime[0])

def genAndMakeStreamK(batchInfo):
  inFile = "streamk.cu"
  outFile = buildDir("streamk-eval.cu")
  tilesCode = """using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<%d, %d, %d>;  
using ShapeMMAWarp = cutlass::gemm::GemmShape<%d, %d, %d>;"""
  tilesCode = tilesCode % tuple(batchInfo["TileSizes"])
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
  if len(batchInfo["TileSizes"]) > 1:
    for i,tile in enumerate(batchInfo["TileSizes"]):
      tilesCode += tilesTemplate % tuple([i+1] + tile[:3] + [i+1] + tile[3:])
      tilesCode += "\n"
  else:
    tilesTemplate = """using ShapeThreadBlock = cutlass::gemm::GemmShape<%d, %d, %d>;  
using ShapeWarp = cutlass::gemm::GemmShape<%d, %d, %d>;"""
    tilesCode = tilesTemplate % tuple(batchInfo["TileSizes"][0])

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

if model == "gpt3" and attention_or_mlp == "attention":
  tiles = {
    0: {
      2048: {
        "TileSizes" : [[256, 256, 32, 128, 64, 32], [128, 128, 32, 64, 32, 32]],
        "baseline": {"split_ks": [1,1,1,1], "SoftmaxRowTile" : 1},
        "tilesync": {"split_ks": [1,1,1,1], "SoftmaxRowTile" : 1,
        "AvoidCustomOrder": False,
        "AvoidWaitKernel": False,
        "ReorderTileLoads": True},
        "rowsync": {"split_ks": [1,1,1,1], "SoftmaxRowTile" : 1}
      },
      1024: {"TileSizes" : [[256, 256, 32, 128, 64, 32], [256, 256, 32, 128, 64, 32]],
        "baseline": {"split_ks": [4,2,1,1], "SoftmaxRowTile" : 4},
        "tilesync": {"split_ks": [4,2,1,1], "SoftmaxRowTile" : 1,
        "AvoidCustomOrder": False,
        "AvoidWaitKernel": False,
        "ReorderTileLoads": True},
        "rowsync": {"split_ks": [4,2,1,1], "SoftmaxRowTile" : 1}
      },
      512: {"TileSizes" : [[256, 256, 32, 128, 64, 32], [128, 128, 32, 64, 64, 32]],
        "baseline": {"split_ks": [4,4,1,2], "SoftmaxRowTile" : 1},
        "tilesync": {"split_ks": [4,4,1,1], "SoftmaxRowTile" : 1,
        "AvoidCustomOrder": False,
        "AvoidWaitKernel": False,
        "ReorderTileLoads": True},
        "rowsync": {"split_ks": [4,4,1,1], "SoftmaxRowTile" : 1}
      },
      256: {"TileSizes" : [[128, 128, 32, 64, 64, 32], [128, 128, 32, 64, 64, 32]],
        "baseline": {"split_ks": [4,2,1,1], "SoftmaxRowTile" : 1},
        "tilesync": {"split_ks": [4,2,1,1], "SoftmaxRowTile" : 4,
        "AvoidCustomOrder": False,
        "AvoidWaitKernel": False,
        "ReorderTileLoads": True},
        "rowsync": {"split_ks": [4,2,1,1], "SoftmaxRowTile" : 4,
        "AvoidCustomOrder": False,
        "AvoidWaitKernel": False}
      },
      128: {"TileSizes" : [[128, 128, 32, 64, 64, 32]],
        "baseline": {"split_ks": [6,2], "SoftmaxRowTile" : 1},
        "tilesync": {"split_ks": [6,4], "SoftmaxRowTile" : 1,
        "AvoidCustomOrder": True,
        "AvoidWaitKernel": True,
        "ReorderTileLoads": True},
        "rowsync": {"split_ks": [6,4], "SoftmaxRowTile" : 1,
        "AvoidCustomOrder": True,
        "AvoidWaitKernel": True},
      },
      64: {"TileSizes" : [[64, 256, 32, 64, 64, 32]],
        "baseline": {"split_ks": [4,2,1,1], "SoftmaxRowTile" : 1},
        "tilesync": {"split_ks": [4,2,1,1], "SoftmaxRowTile" : 1,
        "AvoidCustomOrder": True,
        "AvoidWaitKernel": True,
        "ReorderTileLoads": True},
        "rowsync": {"split_ks": [4,2,1,1], "SoftmaxRowTile" : 1,
        "AvoidCustomOrder": True,
        "AvoidWaitKernel": True},
      },
      32: {"TileSizes" : [[32, 128, 32, 32, 32, 32]],
        "baseline": {"split_ks": [4,3,1,2], "SoftmaxRowTile" : 1},
        "tilesync": {"split_ks": [4,3,1,2], "SoftmaxRowTile" : 1,
        "AvoidCustomOrder": True,
        "AvoidWaitKernel": True,
        "ReorderTileLoads": True},
        "rowsync": {"split_ks": [4,3,1,2], "SoftmaxRowTile" : 1,
        "AvoidCustomOrder": True,
        "AvoidWaitKernel": True},
      },
      16: {"TileSizes" : [[32, 128, 32, 32, 32, 32]],
        "baseline": {"split_ks": [4,3,1,2], "SoftmaxRowTile" : 1},
        "tilesync": {"split_ks": [4,3,1,2], "SoftmaxRowTile" : 1,
        "AvoidCustomOrder": True,
        "AvoidWaitKernel": True,
        "ReorderTileLoads": True},
        "rowsync": {"split_ks": [4,3,1,2], "SoftmaxRowTile" : 1,
        "AvoidCustomOrder": True,
        "AvoidWaitKernel": True},
      },
      8: {"TileSizes" : [[32, 128, 32, 32, 32, 32]],
        "baseline": {"split_ks": [4,3,1,2], "SoftmaxRowTile" : 1},
        "tilesync": {"split_ks": [4,3,1,2], "SoftmaxRowTile" : 1,
        "AvoidCustomOrder": True,
        "AvoidWaitKernel": True,
        "ReorderTileLoads": True},
        "rowsync": {"split_ks": [4,3,1,2], "SoftmaxRowTile" : 1,
        "AvoidCustomOrder": True,
        "AvoidWaitKernel": True},
      },
      4: {"TileSizes" : [[32, 128, 32, 32, 32, 32]],
        "baseline": {"split_ks": [4,3,1,2], "SoftmaxRowTile" : 1},
        "tilesync": {"split_ks": [4,3,1,2], "SoftmaxRowTile" : 1,
        "AvoidCustomOrder": True,
        "AvoidWaitKernel": True,
        "ReorderTileLoads": True},
        "rowsync": {"split_ks": [4,3,1,2], "SoftmaxRowTile" : 1,
        "AvoidCustomOrder": True,
        "AvoidWaitKernel": True},
      },
      2: {"TileSizes" : [[32, 128, 32, 32, 32, 32]],
        "baseline": {"split_ks": [4,3,1,2], "SoftmaxRowTile" : 1},
        "tilesync": {"split_ks": [4,3,1,2], "SoftmaxRowTile" : 1,
        "AvoidCustomOrder": True,
        "AvoidWaitKernel": True,
        "ReorderTileLoads": True},
        "rowsync": {"split_ks": [4,3,1,2], "SoftmaxRowTile" : 1,
        "AvoidCustomOrder": True,
        "AvoidWaitKernel": True},
      },
      1: {"TileSizes" : [[32, 128, 32, 32, 32, 32]],
        "baseline": {"split_ks": [4,3,1,2], "SoftmaxRowTile" : 1},
        "tilesync": {"split_ks": [4,3,1,2], "SoftmaxRowTile" : 1,
        "AvoidCustomOrder": True,
        "AvoidWaitKernel": True,
        "ReorderTileLoads": True},
        "rowsync": {"split_ks": [4,3,1,2], "SoftmaxRowTile" : 1,
        "AvoidCustomOrder": True,
        "AvoidWaitKernel": True},
      }
    },
    1024: {
        1: {"TileSizes" : [[32, 128, 32, 32, 32, 32], [32, 128, 32, 32, 32, 32]],
            "baseline": {"split_ks": [4,16,2,2], "SoftmaxRowTile" : 1},
            "tilesync": {"split_ks": [4,8,2,2], "SoftmaxRowTile" : 1,
            "AvoidCustomOrder": True,
            "AvoidWaitKernel": True,
            "ReorderTileLoads": True},
            "rowsync": {"split_ks": [4,8,2,2], "SoftmaxRowTile" : 1,
            "AvoidCustomOrder": True,
            "AvoidWaitKernel": True},
        },
        4: {"TileSizes" : [[32, 128, 32, 32, 32, 32], [32, 128, 32, 32, 32, 32]],
            "baseline": {"split_ks": [4,16,2,2], "SoftmaxRowTile" : 1},
            "tilesync": {"split_ks": [4,8,2,2], "SoftmaxRowTile" : 1,
            "AvoidCustomOrder": True,
            "AvoidWaitKernel": True,
            "ReorderTileLoads": True},
            "rowsync": {"split_ks": [4,8,2,2], "SoftmaxRowTile" : 1,
            "AvoidCustomOrder": True,
            "AvoidWaitKernel": True},
        },
    },
    2048: {
        1: {"TileSizes" : [[32, 128, 32, 32, 32, 32], [32, 128, 32, 32, 32, 32]],
            "baseline": {"split_ks": [4,16,2,2], "SoftmaxRowTile" : 1},
            "tilesync": {"split_ks": [4,8,2,2], "SoftmaxRowTile" : 1,
            "AvoidCustomOrder": True,
            "AvoidWaitKernel": True,
            "ReorderTileLoads": True},
            "rowsync": {"split_ks": [4,8,2,2], "SoftmaxRowTile" : 1,
            "AvoidCustomOrder": True,
            "AvoidWaitKernel": True},
        },
        4: {"TileSizes" : [[32, 128, 32, 32, 32, 32], [32, 128, 32, 32, 32, 32]],
            "baseline": {"split_ks": [4,16,2,2], "SoftmaxRowTile" : 1},
            "tilesync": {"split_ks": [4,8,2,2], "SoftmaxRowTile" : 1,
            "AvoidCustomOrder": True,
            "AvoidWaitKernel": True,
            "ReorderTileLoads": True},
            "rowsync": {"split_ks": [4,8,2,2], "SoftmaxRowTile" : 1,
            "AvoidCustomOrder": True,
            "AvoidWaitKernel": True},
        }
      },
  }

elif model == "gpt3" and attention_or_mlp == "mlp":
    # Dictionary of tile sizes for each M
  tiles = {
    2048: {"TileSizes" : [[256, 256, 32, 128, 128, 32], [256, 256, 32, 128, 128, 32]],
      "baseline": {"split_ks": [1,1]},
      "rowsync": {"split_ks": [1,1]},
      "tilesync": {"split_ks": [1,1],
                  "AvoidCustomOrder": False,
                  "AvoidWaitKernel": False,
                  "ReorderTileLoads": True,
                  "NoAtomicAdd": True}
    },
    1024: {"TileSizes" : [[256, 256, 32, 128, 128, 32], [256, 256, 32, 128, 128, 32]],
      "baseline": {"split_ks": [2,1]},
      "rowsync": {"split_ks": [2,1]},
      "tilesync": {"split_ks": [2,1],
                  "AvoidCustomOrder": False,
                  "AvoidWaitKernel": False,
                  "ReorderTileLoads": True,
                  "NoAtomicAdd": True}
    },
    512: {"TileSizes" : [[256, 256, 32, 128, 128, 32], [256, 256, 32, 128, 128, 32]],
      "baseline": {"split_ks": [2,1]},
      "rowsync": {"split_ks": [2,1]},
      "tilesync": {"split_ks": [2,1],
                  "AvoidCustomOrder": False,
                  "AvoidWaitKernel": False,
                  "ReorderTileLoads": True,
                  "NoAtomicAdd": True}
    },
    256: {"TileSizes" : [[256, 128, 32, 128, 64, 32], [256, 128, 32, 128, 64, 32]],
      "baseline": {"split_ks": [4,2]},
      "rowsync": {"split_ks": [4,2],
                   "AvoidWaitKernel": True,
                   "AvoidCustomOrder": True},
      "tilesync": {"split_ks": [4,2],
                  "AvoidCustomOrder": True,
                  "AvoidWaitKernel": True,
                  "ReorderTileLoads": True,
                  "NoAtomicAdd": True}
    },
    128: {"TileSizes" : [[128, 256, 32, 64, 128, 32], [128, 256, 32, 64, 128, 32]],
      "baseline": {"split_ks": [3,3]},
      "rowsync": {"split_ks": [3,3],
                   "AvoidWaitKernel": True,
                   "AvoidCustomOrder": True},
      "tilesync": {"split_ks": [3,3],
                  "AvoidCustomOrder": True,
                  "AvoidWaitKernel": True,
                  "ReorderTileLoads": True,
                  "NoAtomicAdd": True}
    },
    64: {"TileSizes" : [[64, 256, 32, 32, 128, 32], [64, 256, 32, 32, 128, 32]],
      "baseline": {"split_ks": [6,3]},
      "rowsync": {"split_ks": [6,3],
                   "AvoidWaitKernel": True,
                   "AvoidCustomOrder": True},
      "tilesync": {"split_ks": [4,3],
                  "AvoidCustomOrder": True,
                  "AvoidWaitKernel": True,
                  "ReorderTileLoads": True,
                  "NoAtomicAdd": True}
    },
    32: {"TileSizes" : [[32, 256, 32, 32, 64, 32], [32, 128, 32, 32, 32, 32]],
      "baseline": {"split_ks": [4,3]},
      "rowsync": {"split_ks": [4,3],
                   "AvoidWaitKernel": True,
                   "AvoidCustomOrder": True},
      "tilesync": {"split_ks": [4,3],
                  "TileBatchSync":2,
                  "AvoidCustomOrder": True,
                  "AvoidWaitKernel": True,
                  "ReorderTileLoads": True,
                  "NoAtomicAdd": True}
    },
    16: {"TileSizes" : [[32, 256, 32, 32, 64, 32], [32, 128, 32, 32, 32, 32]],
      "baseline": {"split_ks": [4,3]},
      "rowsync": {"split_ks": [4,3],
                   "AvoidWaitKernel": True,
                   "AvoidCustomOrder": True},
      "tilesync": {"split_ks": [4,3],
                  "AvoidCustomOrder": True,
                  "AvoidWaitKernel": True,
                  "ReorderTileLoads": True,
                  "NoAtomicAdd": True}
    },
    8: {"TileSizes" : [[32, 256, 32, 32, 64, 32], [32, 128, 32, 32, 32, 32]],
      "baseline": {"split_ks": [4,3]},
      "rowsync": {"split_ks": [4,3],
                   "AvoidWaitKernel": True,
                   "AvoidCustomOrder": True},
      "tilesync": {"split_ks": [4,3],
                    "AvoidCustomOrder": True,
                    "AvoidWaitKernel": True,
                    "ReorderTileLoads": True,
                    "NoAtomicAdd": True}
    },
    4: {"TileSizes" : [[32, 256, 32, 32, 64, 32], [32, 128, 32, 32, 32, 32]],
      "baseline": {"split_ks": [4,3]},
      "rowsync": {"split_ks": [4,3],
                   "AvoidWaitKernel": True,
                   "AvoidCustomOrder": True},
      "tilesync": {"split_ks": [4,3],
                  "AvoidCustomOrder": True,
                  "AvoidWaitKernel": True,
                  "ReorderTileLoads": True,
                  "NoAtomicAdd": True}
    },
    2: {"TileSizes" : [[32, 256, 32, 32, 64, 32], [32, 128, 32, 32, 32, 32]],
      "baseline": {"split_ks": [4,3]},
      "rowsync": {"split_ks": [4,3],
                   "AvoidWaitKernel": True,
                   "AvoidCustomOrder": True},
      "tilesync": {"split_ks": [4,3], 
                  "AvoidCustomOrder": True,
                  "AvoidWaitKernel": True,
                  "ReorderTileLoads": True,
                  "NoAtomicAdd": True}
    },
    1: {"TileSizes" : [[32, 256, 32, 32, 64, 32], [32, 128, 32, 32, 32, 32]],
      "baseline": {"split_ks": [4,3]},
      "rowsync": {"split_ks": [4,3],
                   "AvoidWaitKernel": True,
                   "AvoidCustomOrder": True},
      "tilesync": {"split_ks": [4,3],
                   "AvoidCustomOrder": True,
                   "AvoidWaitKernel": True,
                   "ReorderTileLoads": True,
                   "NoAtomicAdd": True},
    },
  }
elif model == "llama" and attention_or_mlp == "mlp":
    # Dictionary of tile sizes for each M
  tiles = {
    2048: {
      "TileSizes" : [[256, 256, 32, 128, 64, 32], [256, 256, 32, 128, 64, 32]],
      "baseline": {"split_ks": [3,1]},
      "rowsync": {"split_ks": [1,1]},
      "tilesync": {"split_ks": [1,1],
                  "AvoidCustomOrder": False,
                  "AvoidWaitKernel": False,
                  "ReorderTileLoads": True,}
    },
    1024: {"TileSizes" : [[256, 256, 32, 128, 64, 32], [256, 256, 32, 128, 64, 32]],
      "baseline": {"split_ks": [2,1]},
      "rowsync": {"split_ks": [2,1]},
      "tilesync": {"split_ks": [2,1],
                  "AvoidCustomOrder": False,
                  "AvoidWaitKernel": False,
                  "ReorderTileLoads": True,}
    },
    512: {"TileSizes" : [[256, 256, 32, 128, 64, 32], [256, 256, 32, 128, 64, 32]],
      "baseline": {"split_ks": [3,1]},
      "rowsync": {"split_ks": [2,1]},
      "tilesync": {"split_ks": [2,1],
                  "AvoidCustomOrder": False,
                  "AvoidWaitKernel": False,
                  "ReorderTileLoads": True,}
    },
    256: {"TileSizes" : [[256, 128, 32, 128, 64, 32], [256, 128, 32, 128, 64, 32]],
      "baseline": {"split_ks": [6,4]},
      "rowsync": {"split_ks": [4,4], 
                  "AvoidWaitKernel": True,
                  "AvoidCustomOrder": True},
      "tilesync": {"split_ks": [4,4], "AvoidCustomOrder": True,
                                      "AvoidWaitKernel": True,
                                      "ReorderTileLoads": True},
    },
    128: {"TileSizes" : [[128, 128, 32, 64, 64, 32], [128, 128, 32, 64, 64, 32]],
      "baseline": {"split_ks": [8,2]},
      "rowsync": {"split_ks": [4,2], 
                   "AvoidWaitKernel": True,
                   "AvoidCustomOrder": True},
      "tilesync": {"split_ks": [4,2], "AvoidCustomOrder": True,
                                      "AvoidWaitKernel": True,
                                      "ReorderTileLoads": True},
    },
    64: {"TileSizes" : [[64, 128, 32, 32, 64, 32], [64, 128, 32, 32, 64, 32]],
      "baseline": {"split_ks": [8,8]},
      "rowsync": {"split_ks": [8,8],
                   "AvoidWaitKernel": True,
                   "AvoidCustomOrder": True},
      "tilesync": {"split_ks": [8,8], "AvoidCustomOrder": True,
                                      "AvoidWaitKernel": True,
                                      "ReorderTileLoads": True},
    },
    32: {"TileSizes" : [[32, 128, 32, 32, 32, 32], [32, 128, 32, 32, 32, 32]],
      "baseline": {"split_ks": [4,4]},
      "rowsync": {"split_ks": [4,4], 
                   "AvoidWaitKernel": True,
                   "AvoidCustomOrder": True},
      "tilesync": {"split_ks": [4,4], "AvoidCustomOrder": True,
                                      "AvoidWaitKernel": True,
                                      "ReorderTileLoads": True},
    },
    16: {"TileSizes" : [[32, 128, 32, 32, 32, 32], [32, 128, 32, 32, 32, 32]],
      "baseline": {"split_ks": [4,4]},
      "rowsync": {"split_ks": [4,4], 
                   "AvoidWaitKernel": True,
                   "AvoidCustomOrder": True},
      "tilesync": {"split_ks": [4,4], "AvoidCustomOrder": True,
                                      "AvoidWaitKernel": True,
                                      "ReorderTileLoads": True},
    },
    8: {"TileSizes" : [[32, 128, 32, 32, 32, 32], [32, 128, 32, 32, 32, 32]],
      "baseline": {"split_ks": [4,4]},
      "rowsync": {"split_ks": [4,4], 
                   "AvoidWaitKernel": True,
                   "AvoidCustomOrder": True},
      "tilesync": {"split_ks": [4,4], "AvoidCustomOrder": True,
                                      "AvoidWaitKernel": True,
                                      "ReorderTileLoads": True},
    },
    4: {"TileSizes" : [[32, 128, 32, 32, 32, 32], [32, 128, 32, 32, 32, 32]],
      "baseline": {"split_ks": [4,4]},
      "rowsync": {"split_ks": [4,4], 
                   "AvoidWaitKernel": True,
                   "AvoidCustomOrder": True},
      "tilesync": {"split_ks": [4,4], "AvoidCustomOrder": True,
                                      "AvoidWaitKernel": True,
                                      "ReorderTileLoads": True},
    },
    2: {"TileSizes" : [[32, 128, 32, 32, 32, 32], [32, 128, 32, 32, 32, 32]],
      "baseline": {"split_ks": [4,4]},
      "rowsync": {"split_ks": [4,4], 
                   "AvoidWaitKernel": True,
                   "AvoidCustomOrder": True},
      "tilesync": {"split_ks": [4,4], "AvoidCustomOrder": True,
                                      "AvoidWaitKernel": True,
                                      "ReorderTileLoads": True},
    },
    1: {"TileSizes" : [[32, 128, 32, 32, 32, 32], [32, 128, 32, 32, 32, 32]],
      "baseline": {"split_ks": [4,4]},
      "rowsync": {"split_ks": [4,4], 
                   "AvoidWaitKernel": True,
                   "AvoidCustomOrder": True},
      "tilesync": {"split_ks": [4,4], "AvoidCustomOrder": True,
                                      "AvoidWaitKernel": True,
                                      "ReorderTileLoads": True},
    },
  }
  
elif model == "llama" and attention_or_mlp == "attention":
  tiles = {
    2048: {
      "TileSizes" : [256, 128, 32, 128, 64, 32], "MaxTBsPerSM": 2, "Best-Policy": "Row-Sync",
      "baseline": {"split_ks": [2,2], "SoftmaxRowTile" : 1},
      "tilesync":   {"split_ks": [2,1], "SoftmaxRowTile" : 1,
      "AvoidCustomOrder": False,
      "AvoidWaitKernel": False,
      "ReorderTileLoads": True},
      "rowsync":   {"split_ks": [2,1], "SoftmaxRowTile" : 1},
    },
    1024: {"TileSizes" : [256, 128, 32, 128, 64, 32], "MaxTBsPerSM": 2, "Best-Policy": "Row-Sync",
      "baseline": {"split_ks": [2,2], "SoftmaxRowTile" : 1},
      "tilesync":   {"split_ks": [2,1], "SoftmaxRowTile" : 1,
      "AvoidCustomOrder": False,
      "AvoidWaitKernel": False,
      "ReorderTileLoads": True},
      "rowsync":   {"split_ks": [2,1], "SoftmaxRowTile" : 1},
    },
    512: {"TileSizes" : [256, 128, 32, 128, 64, 32], "MaxTBsPerSM": 2, "Best-Policy": "Row-Sync",
      "baseline": {"split_ks": [4,4], "SoftmaxRowTile" : 1},
      "tilesync":   {"split_ks": [2,1], "SoftmaxRowTile" : 1,
      "AvoidCustomOrder": False,
      "AvoidWaitKernel": False,
      "ReorderTileLoads": True},
      "rowsync":   {"split_ks": [2,1], "SoftmaxRowTile" : 1},
    },
    256: {"TileSizes" : [256, 128, 32, 128, 64, 32], "MaxTBsPerSM": 2, "Best-Policy": "Tile-Sync",
      "baseline": {"split_ks": [4,4], "SoftmaxRowTile" : 1},
      "tilesync":   {"split_ks": [6,1], "SoftmaxRowTile" : 1,
      "AvoidCustomOrder": False,
      "AvoidWaitKernel": False,
      "ReorderTileLoads": True},
      "rowsync":   {"split_ks": [6,1], "SoftmaxRowTile" : 1},
    },
    128: {"TileSizes" : [128, 128, 32, 64, 64, 32], "MaxTBsPerSM": 2, "Best-Policy": "Tile-Sync",
      "baseline": {"split_ks": [4,4], "SoftmaxRowTile" : 1},
      "tilesync":   {"split_ks": [6,1], "SoftmaxRowTile" : 1,
      "AvoidCustomOrder": False,
      "AvoidWaitKernel": False,
      "ReorderTileLoads": True},
      "rowsync":   {"split_ks": [6,1], "SoftmaxRowTile" : 1},
    },
    64: {"TileSizes" : [64, 256, 32, 32, 128, 32], "MaxTBsPerSM": 3, "Best-Policy": "Tile-Sync",
      "baseline": {"split_ks": [8,8], "SoftmaxRowTile" : 1},
      "tilesync": {"split_ks": [6,2], "SoftmaxRowTile" : 2,
      "AvoidCustomOrder": False,
      "AvoidWaitKernel": False,
      "ReorderTileLoads": True,},
      "rowsync": {"split_ks": [6,2], "SoftmaxRowTile" : 2,
      "AvoidCustomOrder": False,
      "AvoidWaitKernel": False}
    },
    32: {"TileSizes" : [32, 256, 32, 32, 64, 32], "MaxTBsPerSM": 3,
      "baseline": {"split_ks": [8,8], "SoftmaxRowTile" : 1},
      "tilesync": {"split_ks": [6,2], "SoftmaxRowTile" : 1,
      "AvoidCustomOrder": False,
      "AvoidWaitKernel": False,
      "ReorderTileLoads": True,},
      "rowsync": {"split_ks": [6,2], "SoftmaxRowTile" : 1,
      "AvoidCustomOrder": False,
      "AvoidWaitKernel": False}
    },
    16: {"TileSizes" : [32, 256, 32, 32, 64, 32], "MaxTBsPerSM": 3,
      "baseline": {"split_ks": [8,8], "SoftmaxRowTile" : 1},
      "tilesync": {"split_ks": [6,2], "SoftmaxRowTile" : 1,
      "AvoidCustomOrder": False,
      "AvoidWaitKernel": False,
      "ReorderTileLoads": True,},
      "rowsync": {"split_ks": [6,2], "SoftmaxRowTile" : 1,
      "AvoidCustomOrder": False,
      "AvoidWaitKernel": False}
    },
    8: {"TileSizes" : [32, 256, 32, 32, 64, 32], "MaxTBsPerSM": 3,
      "baseline": {"split_ks": [8,8], "SoftmaxRowTile" : 1},
      "tilesync": {"split_ks": [6,2], "SoftmaxRowTile" : 1,
      "AvoidCustomOrder": False,
      "AvoidWaitKernel": False,
      "ReorderTileLoads": True},
      "rowsync": {"split_ks": [6,2], "SoftmaxRowTile" : 1,
      "AvoidCustomOrder": False,
      "AvoidWaitKernel": False}
    },
    4: {"TileSizes" : [32, 256, 32, 32, 64, 32], "MaxTBsPerSM": 3,
      "baseline": {"split_ks": [8,8], "SoftmaxRowTile" : 1},
      "tilesync": {"split_ks": [6,2], "SoftmaxRowTile" : 1,
      "AvoidCustomOrder": False,
      "AvoidWaitKernel": False,
      "ReorderTileLoads": True},
      "rowsync": {"split_ks": [6,2], "SoftmaxRowTile" : 1,
      "AvoidCustomOrder": False,
      "AvoidWaitKernel": False}
    },
    2: {"TileSizes" : [32, 256, 32, 32, 64, 32], "MaxTBsPerSM": 3,
      "baseline": {"split_ks": [8,8], "SoftmaxRowTile" : 1},
      "tilesync": {"split_ks": [6,2], "SoftmaxRowTile" : 1,
      "AvoidCustomOrder": False,
      "AvoidWaitKernel": False,
      "ReorderTileLoads": True},
      "rowsync": {"split_ks": [6,2], "SoftmaxRowTile" : 1,
      "AvoidCustomOrder": False,
      "AvoidWaitKernel": False}
    },
    1: {"TileSizes" : [32, 256, 32, 32, 64, 32], "MaxTBsPerSM": 3,
      "baseline": {"split_ks": [8,8], "SoftmaxRowTile" : 1},
      "tilesync": {"split_ks": [6,2], "SoftmaxRowTile" : 1,
      "AvoidCustomOrder": False,
      "AvoidWaitKernel": False,
      "ReorderTileLoads": True},
      "rowsync": {"split_ks": [6,2], "SoftmaxRowTile" : 1,
      "AvoidCustomOrder": False,
      "AvoidWaitKernel": False}
    }
  }

if model.lower() == "BLOOM".lower():
  H = 14336
  FFN = 4*H/8
elif model.lower() == "GPT3".lower():
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
  cases = [1,2,4,8,16,32,64,128,256,512,1024,2048]
else:
  cases = [(0,256), (0,512), (0, 1024), (0, 2048), (1024,1), (1024,4), (2048,1), (2048,4)]

for case in cases:
  if attention_or_mlp == "attention":
    m = case[1]
    seq = case[0]
  else:
    m = case

  caseTiles = None
  if attention_or_mlp == "attention":
    caseTiles = tiles[seq][m]
  else:
    caseTiles = tiles[m]

  if False:
    if attention_or_mlp == "attention":
      (s, o) = subprocess.getstatusoutput(f"python3 torch-baselines/torchAttention.py {m} {int(H/8)} {H} {H}")
    else:
      (s, o) = subprocess.getstatusoutput(f"python3 torch-baselines/torchmlp.py {m} {int(FFN)} {H} {H} {model}")
    
    if s == -1:
      print("error " + o)
    else:
      ctime = o
      cublasTimes[m] = ctime

    print(f'{m} & {H} & {"pytorch"} & {"%.2f"%float(ctime)}')
  
  if False:
    genAndMakeStreamK(tiles[m])
    if model == 'gpt3' or (model == 'llama' and attention_or_mlp == 'attention'):
      streamk_command = buildDir("streamk-eval") + f" --m={m} --alpha=1 --beta=0 --iterations=20 "
      (s, o) = subprocess.getstatusoutput(streamk_command + f"--n={int(FFN)} --k={H} " + f"--split={tiles[m]['baseline']['split_ks'][0]}")
      if s != 0:
        print("StreamK Error")
        print(o)

      firstGeMMStreamK = getStreamKTimes(o)

      (s, o) = subprocess.getstatusoutput(streamk_command + f"--n={H} --k={int(FFN)} " + f"--split={tiles[m]['baseline']['split_ks'][1]}")
      if s != 0:
        print("StreamK Error")
        print(o)

      secondGeMMStreamK = getStreamKTimes(o)
      total = firstGeMMStreamK + secondGeMMStreamK
      print(f'{m} & {H} & {"streamk"} & {"%.2f"%(firstGeMMStreamK*1000)} & {"%.2f"%(secondGeMMStreamK*1000)} & {"%.2f"%(total*1000)}')
    elif model == 'llama' and attention_or_mlp == 'mlp':
      streamk_command = buildDir("streamk-eval") + f" --m={m} --alpha=1 --beta=0 --iterations=20 "
      (s, o) = subprocess.getstatusoutput(streamk_command + f"--n={int(FFN)} --k={H} " + f"--split={tiles[m]['baseline']['split_ks'][0]}")
      if s != 0:
        print("StreamK Error")
        print(o)

      firstGeMMStreamK = getStreamKTimes(o)

      (s, o) = subprocess.getstatusoutput(streamk_command + f"--n={int(FFN)} --k={H} " + f"--split={tiles[m]['baseline']['split_ks'][0]}")
      if s != 0:
        print("StreamK Error")
        print(o)

      secondGeMMStreamK = getStreamKTimes(o)

      (s, o) = subprocess.getstatusoutput(streamk_command + f"--n={H} --k={int(FFN)} " + f"--split={tiles[m]['baseline']['split_ks'][1]}")
      if s != 0:
        print("StreamK Error")
        print(o)

      thirdGeMMStreamK = getStreamKTimes(o)
      total = firstGeMMStreamK + secondGeMMStreamK + thirdGeMMStreamK
      print(f'{m} & {H} & {"streamk"} & {"%.2f"%(firstGeMMStreamK*1000)} & {"%.2f"%(secondGeMMStreamK*1000)} & {"%.2f"%(thirdGeMMStreamK*1000)} & {"%.2f"%(total*1000)}')
    continue

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
  baselineCommand = buildDir(f"{attention_or_mlp}-eval-baseline") + commandArgs + splitKArgs 
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
    print(f'{m} & {H} & baseline & {"%.2f"%avg(bTimeTotal)} & {"%.2f"%stdev(bTimeTotal)} & {"%.2f"%avg(bTimeMatmul1)} & {"%.2f"%avg(bTimeMatmul2)}')
    baselineDone = True

  for syncPolicy in policies:
    splitKs = caseTiles["tilesync"] if syncPolicy == "stridedsync" else caseTiles[syncPolicy]
    splitKArgs = " " + " ".join([f"--split-k{i+1} {split_ks[i]}" for i in range(len(split_ks))])
    command = ""
    # if attention_or_mlp == 'attention' and syncPolicy == 'stridedsync':
    #   command += buildDir("%s-%s-eval-%s "%(attention_or_mlp, model, syncPolicy))
    # else:
    command += buildDir("%s-eval-%s "%(attention_or_mlp, syncPolicy))
    command += commandArgs + splitKArgs
    (s, o) = subprocess.getstatusoutput(command)
  
    otime = -1
    if "Invalid" in o:
      pass
    elif s != 0:
      print("error " + o)
    else:
      overlaptimes  = getAllTimes(o, 'START-OVERLAPPED', 'END-OVERLAPPED')
      otime = overlaptimes["Total"]

    print(f'{m} & {H} & {syncPolicy} & {"%.2f"%avg(bTimeTotal)} & {"%.2f"%stdev(bTimeTotal)} & {"%.2f"%avg(bTimeMatmul1)} & {"%.2f"%avg(bTimeMatmul2)} & {"%.2f"%avg(otime)} & {"%.2f"%stdev(otime)} & {"%.2f"%(100 - avg(otime)/avg(bTimeTotal)*100)}')
      # btime = re.findall(r'START-BASELINE: ([\.\d]+)', o)
      # baselineTimes[m] = btime[0]
      # otime = re.findall(r'START-OVERLAPPED elapsedtime ([\.\d]+)', o)
      # overlappedTimes[m] = otime[0]
      # speedup[m] = float(btime[0])/float(otime[0])

      # print(f"{h} & {btime[0]} & {ctime} & {otime[0]}")

# print(baselineTimes)
# print(cublasTimes)
# print(overlappedTimes)
# print(minimumTimes)
# print("M & N & K & L & TBs & Baseline(us)  & Overlapped(us) & Minimum(us) & Speedup & MaxSpeedup")

for m in baselineTimes:
  print(f"{m} & {n} & {k} & {l} & {m//128*n//128} & {baselineTimes[m]} & {cublasTimes[m]} & {overlappedTimes[m]} & {minimumTimes[m]} & {speedup[m]} & {maxspeedup[m]}")


# tiles1 = {
#     2048: {
#       "TileSizes" : [256, 128, 32, 128, 64, 32], "MaxTBsPerSM": 2, "Best-Policy": "Row-Sync",
#       "split_ks": {8192:  [1,1],
#                    12288: [1,1]},
#     },
#     1024: {"TileSizes" : [256, 128, 32, 128, 64, 32], "MaxTBsPerSM": 2, "Best-Policy": "Row-Sync",
#       8192: {"split_ks": [1,1]},
#       12288: {"split_ks": [1,1]},
#     },
#     512: {"TileSizes" : [256, 128, 32, 128, 64, 32], "MaxTBsPerSM": 2, "Best-Policy": "Row-Sync",
#       8192: {"split_ks":   [2,1]},
#       12288: {"split_ks":  [2,1]},
#     },
#     256: {"TileSizes" : [256, 128, 32, 128, 64, 32], "MaxTBsPerSM": 2, "Best-Policy": "Tile-Sync",
#       6144: {"split_ks":   [4,1]},
#       8192: {"split_ks":   [4,1]},
#       10240: {"split_ks":  [4,1]},
#       12288: {"split_ks":  [3,2]},
#       14336: {"split_ks":  [4,1]},
#       16384: {"split_ks":  [4,1]},
#       20480: {"split_ks":  [4,1]},
#       25600: {"split_ks":  [4,1]},
#     },
#     128: {"TileSizes" : [128, 128, 32, 64, 64, 32], "MaxTBsPerSM": 2, "Best-Policy": "Tile-Sync",
#       6144: {"split_ks":   [12,3]},
#       8192: {"split_ks":   [12,3]},
#       10240: {"split_ks":  [12,3]},
#       12288: {"split_ks":  [12,3]},
#       14336: {"split_ks":  [12,3]},
#       16384: {"split_ks":  [2,1]},
#       20480: {"split_ks":  [12,3]},
#       25600: {"split_ks":  [12,3]},},
#     64: {"TileSizes" : [64, 256, 32, 32, 128, 32], "MaxTBsPerSM": 3, "Best-Policy": "Tile-Sync",
#       6144: {"split_ks":   [4,1], "TileBatch": 2},
#       8192: {"split_ks":   [4,1], "TileBatch": 4},
#       10240: {"split_ks":  [12,3]},
#       12288: {"split_ks":  [2,2], "TileBatch": 8},
#       14336: {"split_ks":  [3,1]},
#       16384: {"split_ks":  [2,1], "TileBatch": 8},
#       20480: {"split_ks":  [12,3]},
#       25600: {"split_ks":  [12,3]}},
#     32: {"TileSizes" : [64, 256, 32, 32, 128, 32], "split_ks": [4,1], "MaxTBsPerSM": 3, "Best-Policy": "Tile-Sync",
#       6144: {"split_ks":   [4,1], "TileBatch": 2},
#       8192: {"split_ks":   [4,1], "TileBatch": 4},
#       10240: {"split_ks":  [12,3]},
#       12288: {"split_ks":  [2,2], "TileBatch": 8},
#       14336: {"split_ks":  [3,1]},
#       16384: {"split_ks":  [2,1], "TileBatch": 8},
#       20480: {"split_ks":  [12,3]},
#       25600: {"split_ks":  [12,3]}},
#     16: {"TileSizes" : [64, 256, 32, 32, 128, 32], "split_ks": [4,1],
#     "MaxTBsPerSM": 3, "Best-Policy": "Tile-Sync",
#       6144: {"split_ks":   [4,1], "TileBatch": 2},
#       8192: {"split_ks":   [4,1], "TileBatch": 4},
#       10240: {"split_ks":  [4,1]},
#       12288: {"split_ks":  [2,2], "TileBatch": 8},
#       14336: {"split_ks":  [3,1]},
#       16384: {"split_ks":  [2,1], "TileBatch": 8},
#       20480: {"split_ks":  [4,1]},
#       25600: {"split_ks":  [4,1]}},
#     8: {"TileSizes" : [64, 256, 32, 32, 128, 32], "split_ks": [4, 1], 
#     "MaxTBsPerSM": 3, "Best-Policy": "Tile-Sync",
#       6144: {"split_ks":   [4,1], "TileBatch": 2},
#       8192: {"split_ks":   [4,1], "TileBatch": 4},
#       10240: {"split_ks":  [4,1]},
#       12288: {"split_ks":  [2,2], "TileBatch": 8},
#       14336: {"split_ks":  [3,1]},
#       16384: {"split_ks":  [2,1], "TileBatch": 8},
#       20480: {"split_ks":  [4,1]},
#       25600: {"split_ks":  [4,1]}},
#     4: {"TileSizes" : [64, 256, 32, 32, 128, 32], "split_ks": [4, 1],
#     "MaxTBsPerSM": 3, "Best-Policy": "Tile-Sync",
#       6144: {"split_ks":   [4,1], "TileBatch": 2},
#       8192: {"split_ks":   [4,1], "TileBatch": 4},
#       10240: {"split_ks":  [4,1]},
#       12288: {"split_ks":  [2,2], "TileBatch": 8},
#       14336: {"split_ks":  [3,1]},
#       16384: {"split_ks":  [2,1], "TileBatch": 8},
#       20480: {"split_ks":  [4,1]},
#       25600: {"split_ks":  [4,1]}},
#     2: {"TileSizes" : [64, 256, 32, 32, 128, 32], "split_ks": [4,1], "MaxTBsPerSM": 3, "Best-Policy": "Tile-Sync",
#       6144: {"split_ks":   [4,1], "TileBatch": 2},
#       8192: {"split_ks":   [4,1], "TileBatch": 4},
#       10240: {"split_ks":  [4,1]},
#       12288: {"split_ks":  [2,2], "TileBatch": 8},
#       14336: {"split_ks":  [3,1]},
#       16384: {"split_ks":  [2,1], "TileBatch": 8},
#       20480: {"split_ks":  [4,1]},
#       25600: {"split_ks":  [4,1]}},
#     1: {"TileSizes" : [64, 256, 32, 32, 128, 32], "split_ks": [4,1], "MaxTBsPerSM": 3, "Best-Policy": "Tile-Sync",
#       6144: {"split_ks":   [4,1], "TileBatch": 2},
#       8192: {"split_ks":   [4,2], "TileBatch": 4},
#       10240: {"split_ks":  [4,1]},
#       12288: {"split_ks":  [2,2], "TileBatch": 8}, #split_k: 3,1 64x256x32 32x128x32, tilebatch:8, load B before A
#       14336: {"split_ks":  [3,1]}, #TODO: Tile batch
#       16384: {"split_ks":  [2,1], "TileBatch": 8},
#       20480: {"split_ks":  [4,1]},
#       25600: {"split_ks":  [4,1]}}
#   }
