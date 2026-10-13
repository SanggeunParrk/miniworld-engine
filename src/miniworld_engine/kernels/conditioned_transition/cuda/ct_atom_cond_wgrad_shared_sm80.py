"""CUDA conditioning weight gradients from physical BF16 fields, FP32 output."""
import functools
from pathlib import Path
import torch
from ..._nvcc import ensure_cuda_home,host_flags,load_extension
@functools.lru_cache(maxsize=1)
def _ext():
 ensure_cuda_home();d=Path(__file__).parent
 return load_extension(name='ct_atom_cond_wgrad_shared_sm80',sources=[str(d/'ct_atom_cond_wgrad_shared_sm80.cu')],extra_include_paths=[str(d)],extra_cuda_cflags=[*host_flags(),'-O3','-std=c++17','-lineinfo','-Xptxas=-v','-gencode=arch=compute_80,code=sm_80'],extra_cflags=['-O3','-std=c++17'],verbose=True)
def run(a0,a1,a2,b0,b1,partial,out,cfg=0,splits=32,planes=3):_ext().run(a0,a1,a2,b0,b1,partial,out,cfg,splits,planes)
