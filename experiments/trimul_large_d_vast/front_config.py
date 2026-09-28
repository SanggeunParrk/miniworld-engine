"""Retile the existing saved-preactivation forward; retain all tensor lifetimes."""
import copy
import torch
from miniworld_engine.kernels.trimul_inproj.cuda import h100_wide_forward as F
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from wide_saved_front_overlap import SavedOverlapFront

def build(plan,cfg):
    original=plan.baseline_front
    base=copy.copy(original);d=plan.p.D;n=plan.p.n;a,b,slots,sk,mb=cfg[:5]
    base.cfg=list(cfg);base.smem=F.k1_smem(d,cfg)
    # Include the additional saved-preactivation staging in occupancy feasibility.
    if (base.smem+a*b//64*4096)*mb>232448:raise ValueError('saved-front shared-memory occupancy')
    base.threads=128*(a*b//64+1);tj=(n+b-1)//b;tiles=((n+a-1)//a)*tj
    base.grid=min(tiles,torch.cuda.get_device_properties(plan.p.x.device).multi_processor_count*mb)
    fields=original.params.fields.copy()
    operand=base.x if base.normalize else base.xn
    fields[0]=F.tm(operand,[64,b,a],[d,n,n],[d*2,n*d*2])
    fields[11:13]=[tj,tiles]
    base.params=T._launch_module().Struct(fields)
    result=SavedOverlapFront(base,plan.pre)
    drv=result.k.unit.drv;fn=drv.d.CUfunction(int(result.k.handle))
    result.occupancy=int(drv._unwrap('cuOccupancyMaxActiveBlocksPerMultiprocessor',drv.d.cuOccupancyMaxActiveBlocksPerMultiprocessor(fn,result.threads,result.smem)))
    if result.occupancy<mb:raise ValueError(f'actual occupancy {result.occupancy} < requested {mb}')
    return result
