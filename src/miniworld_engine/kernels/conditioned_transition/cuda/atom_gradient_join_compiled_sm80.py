"""Independent A100 BF16 gradient joins with original compiled left-associated F32 additions and one final BF16 rounding."""
import functools
from pathlib import Path
import torch
from ..._nvcc import ensure_cuda_home,host_flags,load_extension
@functools.lru_cache(maxsize=1)
def _ext():
 ensure_cuda_home();d=Path(__file__).parent
 return load_extension(name='atom_gradient_join_compiled_sm80',sources=[str(d/'atom_gradient_join_compiled_sm80.cu')],extra_cuda_cflags=[*host_flags(),'-O3','-std=c++17','-lineinfo','-Xptxas=-v','-gencode=arch=compute_80,code=sm_80'],extra_cflags=['-O3','-std=c++17'],verbose=True)
def run(a,b,c,out,planes=2,cfg=0):
 _ext().run(a,b,c,out,planes,cfg)
