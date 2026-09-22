import os
os.environ['CUTE_DSL_KEEP_PTX']='1'
os.environ['CUTE_DSL_KEEP_CUBIN']='1'
from pathlib import Path
import cutlass.cute as cute
orig=cute.compile
def keep(*args,**kwargs):
 kwargs['options']='--keep-ptx'
 return orig(*args,**kwargs)
cute.compile=keep
exec(Path(__file__).with_name('probe.py').read_text())
from miniworld_engine.kernels.trimul_inproj.cute.parity_f567 import _COMPILE_CACHE
for fn in _COMPILE_CACHE.values():
 print('PTXTYPE',type(fn.__ptx__),flush=True)
 print('PTX_BYTES',len(fn.__ptx__ or ''),flush=True)
 ptx=fn.__ptx__
 if ptx is not None:
  if isinstance(ptx,dict): ptx='\n'.join(str(v) for v in ptx.values())
  Path(__file__).with_name('f567.ptx').write_text(str(ptx))
