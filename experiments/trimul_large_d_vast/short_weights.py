"""Shape-scoped input-weight splitting, with joint gate gradients at D384."""
import torch
from lt_contract import LtBmm
from wide_cached_input import CachedInput
from wide_joint_input_reduce import JointInputReduce
from early_both_affine_init import IndependentInputLN

def attach(plan,splits,index=0):
    p=plan.p;d=p.D
    assert d in (384,512) and p.n==384 and splits<=32 and p.M%splits==0
    joint=plan.joint_choice is not None
    rows=(9 if joint else 8)*d;offset=(2 if joint else 3)*d*d
    src=plan.dx.input if joint else p.gp_all
    step=p.M//splits
    partial=p.floats[7].reshape(-1)[offset:].as_strided((splits,rows,d),(11*d*d,d,1))
    aa=src.as_strided((splits,rows,step),(step,p.M,1))
    bb=p.xn.as_strided((splits,step,d),(step*d,d,1))
    workspace=torch.empty(64*1024*1024,device=p.x.device,dtype=torch.uint8)
    op=LtBmm(aa,bb,partial,workspace);op.index=index
    plan.input_dw=op;plan.input_splits=splits
    # The staging constructor has a conservative split allowlist. Its parameters
    # are split-independent; IndependentInputLN recompiles the final reducer
    # below with the actual split count, including the exploratory two-way case.
    plan.dx.reduce_only=(JointInputReduce if joint else CachedInput)(p,16,128,4,splits=8 if joint else max(4,splits))
    plan.dx.reduce_only=IndependentInputLN(plan,1,not joint)
    return op
