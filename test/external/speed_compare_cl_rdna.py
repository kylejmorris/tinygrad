import itertools
from tinygrad import Device
from tinygrad.engine.realize import CompiledRunner
from tinygrad.helpers import getenv, colored
from extra.optimization.helpers import load_worlds, ast_str_to_lin
from tinygrad.engine.search import bufs_from_lin
from tinygrad.runtime.ops_gpu import CLCompiler
from tinygrad.runtime.ops_amd import AMDCompiler
from tinygrad.renderer.assembly import RDNARenderer


from typing import Tuple
from tinygrad.ops import LazyOp
inf, nan = float('inf'), float('nan')

# kernel unpacker
from tinygrad.codegen.linearizer import Linearizer
def ast_str_to_ast(ast_str:str) -> Tuple[LazyOp,...]: return val if isinstance(val:=eval(ast_str), tuple) else (val,)
def ast_str_to_lin(ast_str:str, opts=None): return Linearizer(*ast_str_to_ast(ast_str), opts=opts)

# load worlds, a dataset of about 12k kernels
import gzip
from pathlib import Path
import random
from tinygrad.helpers import dedup
def load_worlds(filter_reduce=True, filter_noimage=True, filter_novariable=True):
  fn = Path(__file__).parent.parent / "datasets/sops.gz"
  ast_strs = dedup(gzip.open(fn).read().decode('utf-8').strip().split("\n"))
  if filter_reduce: ast_strs = [x for x in ast_strs if "ReduceOps" in x]
  if filter_noimage: ast_strs = [x for x in ast_strs if "dtypes.image" not in x]
  if filter_novariable: ast_strs = [x for x in ast_strs if "Variable" not in x]
  random.seed(1337)
  random.shuffle(ast_strs)
  return ast_strs

# move to helpers?
def colorize_float(x):
  ret = f"{x:7.2f}x"
  if x < 0.75: return colored(ret, 'green')
  elif x > 1.15: return colored(ret, 'red')
  else: return colored(ret, 'yellow')

if __name__ == "__main__":
  ast_strs = load_worlds(filter_reduce=False, filter_novariable=True)
  # ex: (LazyOp(op=BufferOps.STORE, src=(LazyOp(op=UnaryOps.SQRT, src=(LazyOp(op=UnaryOps.RECIP, src=(LazyOp(op=UnaryOps.CAST, src=(LazyOp(op=BufferOps.LOAD, src=(), arg=MemBuffer(idx=1, dtype=dtypes.ushort, st=ShapeTracker(views=(View(shape=(3,), strides=(1,), offset=0, mask=None, contiguous=True),)))),), arg=dtypes.float),), arg=None),), arg=None),), arg=MemBuffer(idx=0, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(3,), strides=(1,), offset=0, mask=None, contiguous=True),)))),)

  # # no bfloat16 for ptx at the moment
  # ast_strs = [x for x in ast_strs if "dtypes.bfloat16" not in x]
  dev = Device["GPU"]
  rdna = RDNARenderer(dev.arch)

  # NUM=112 python3 test/external/speed_compare_cuda_ptx.py

  single = getenv("NUM", -1)
  if single != -1: ast_strs = ast_strs[single:single+1]

  average_tm_cl, average_tm_rdna = 0, 0
  for num, ast in enumerate(ast_strs):
    # cl compile
    dev.compiler = CLCompiler(dev.arch)
    lin = ast_str_to_lin(ast, opts=dev.renderer)
    lin.hand_coded_optimizations()
    cl_prg = CompiledRunner(lin.to_program())

    bufs = bufs_from_lin(lin)

    # rdna compile
    dev.compiler = AMDCompiler(dev.arch)
    lin = ast_str_to_lin(ast, opts=rdna)
    lin.hand_coded_optimizations()
    lin.linearize()
    rdna_prg = CompiledRunner(lin.to_program())

    # warmup
    try:
      cl_prg(bufs, {}, wait=True)
    except RuntimeError:
      print("cuda failed ast:", num)
      continue
    rdna_prg(bufs, {}, wait=True)

    tm_cl, tm_rdna = [], []
    for i in range(5):
      tm_cl.append(cl_prg(bufs, {}, wait=True))
      tm_rdna.append(rdna_prg(bufs, {}, wait=True))
    average_tm_cl += min(tm_cl)
    average_tm_rdna += min(tm_rdna)
    ratio = min(tm_rdna)/min(tm_cl)
    print(f"{average_tm_rdna/average_tm_cl:5.2f}x -- {num:4d} {colorize_float(ratio)}  {min(tm_rdna)*1e6:7.2f} us", lin.name)
    if ratio > 1.5:
      def fix(x): return x.replace('\t', ' ').strip()
      ll1, ll2 = cl_prg.lib.decode().split('\n'), rdna_prg.lib.decode().split('\n')
      if single != -1:
        for ln, (l1, l2) in enumerate(itertools.zip_longest(ll1, ll2, fillvalue='')):
          print(f"{ln:5d} | {fix(l1):80s} | {fix(l2):80s}")
      print(len(ll1), len(ll2), "RATIO", ratio, "us", min(tm_rdna)*1e6)
