"""D512-short input-weight split count, preserving GP/dX and forward arithmetic."""
import torch
from lt_contract import LtBmm
from wide_cached_input import CachedInput
from early_both_affine_init import IndependentInputLN

def attach(plan,splits,index=0):
    p=plan.p;d=p.D
    assert (d,p.n)==(512,384) and plan.joint_choice is None
    assert splits in (4,6,8,12,16,32) and p.M%splits==0
    step=p.M//splits
    partial=p.floats[7].reshape(-1)[3*d*d:].as_strided((splits,8*d,d),(11*d*d,d,1))
    a=p.gp_all.as_strided((splits,8*d,step),(step,p.M,1))
    b=p.xn.as_strided((splits,step,d),(step*d,d,1))
    workspace=torch.empty(64*1024*1024,device=p.x.device,dtype=torch.uint8)
    op=LtBmm(a,b,partial,workspace);op.index=index
    plan.input_dw=op;plan.input_splits=splits
    plan.dx.reduce_only=CachedInput(p,16,128,4,splits=splits)
    plan.dx.reduce_only=IndependentInputLN(plan,1,True)
    return op
