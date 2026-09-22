"""Experimental L384 B7-B12 selection. Does not alter engine dispatch."""
from pathlib import Path
import importlib.util
R=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("b7_selected_lnreduce",R.parent/"trimul_b7_split_lnreduce_20260922/role_plan.py")
impl=importlib.util.module_from_spec(spec);spec.loader.exec_module(impl)
CONFIG=dict(split=True,splits=16,pc=1,dxctas=256,prod=32,cons=224,resident=0,slices=1,skip_restore=1,overlap=0,ln_mode=2,xn_early=0,mask_hoist=1,dw_mask_early=1,weight_prefetch=0,dx_pc=1,dw_pair=1,nextrow=0)
def make(d,dy,dl,dr,dg,xn,mask=None):
 if d["n"]!=384:raise ValueError("This experimental selection is validated only for L384")
 if xn is None:raise ValueError("Saved x_n is required")
 p=impl.Plan(d,dy,dl,dr,dg,xn=xn,**CONFIG)
 if mask is not None:p.mask=mask;p.bind(dl,dr,dg,dy,xn=xn)
 return p
