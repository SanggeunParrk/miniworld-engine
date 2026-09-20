"""Fixed-contract replay validation; new outputs never replace earlier evidence."""
from front_plan import *
import argparse
LIMITS=dict(dx=2e-5,dWL=5e-4,dWLg=5e-4,dWR=5e-4,dWRg=5e-4,dgamma=5e-6,dbeta=5e-6)
def check(p,a):
 out=p.outputs;ref=baseline(a);torch.cuda.synchronize();e=errors(out,ref)
 assert all(v['finite'] and v['relative_l2']<=LIMITS[k] for k,v in e.items()),e
 assert torch.equal(p.counts[:2],torch.zeros_like(p.counts[:2])),p.counts[:2]
 return e
def change(a,seed):
 torch.manual_seed(seed)
 for k in ('dl','dr','dg','dy'):a[k].copy_(torch.randn_like(a[k]))
 a['mask'].copy_((torch.rand_like(a['mask'].float())>.2).to(a['mask'].dtype))
 # Saved values update in place. Preserve valid positive inverse std.
 a['mu'].add_(torch.randn_like(a['mu'])*.001);a['rs'].mul_(1.001)
 a['d']['gi'].add_(torch.randn_like(a['d']['gi'])*.001)
