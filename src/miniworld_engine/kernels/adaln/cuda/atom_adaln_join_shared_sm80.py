"""E151 private atom AdaLN backward with original fanout joins in final stores."""
import functools
from pathlib import Path
import torch
from ..._nvcc import ensure_cuda_home, host_flags, load_extension

@functools.lru_cache(maxsize=1)
def _ext():
    ensure_cuda_home()
    path=Path(__file__).parent
    return load_extension(name='atom_adaln_join_shared_sm80',sources=[str(path/'atom_adaln_join_shared_sm80.cu')],extra_include_paths=[str(path/'sm80')],extra_cuda_cflags=[*host_flags(),'-O3','-std=c++17','-lineinfo','-Xptxas=-v','-gencode=arch=compute_80,code=sm_80'],extra_cflags=['-O3','-std=c++17'],verbose=True)

def result(dy,x,cond,xst,cst,lnw,ws,wb,sb,res,dc_transition,dc_attention,intermediate=False):
    ext=_ext();rows=ext.grid(x.shape[0]);dx,dc,dsc,aff=(torch.empty_like(x) for _ in range(4))
    psb=torch.empty(rows,128,device=x.device,dtype=torch.float32);plw=torch.empty_like(psb)
    ext.run(dy,x,cond,xst,cst,lnw,ws,ws.t().contiguous(),wb.t().contiguous(),sb,res,dc_transition,dc_attention,dx,dc,dsc,aff,psb,plw,intermediate)
    return dx,dc,dsc,aff,psb,plw
