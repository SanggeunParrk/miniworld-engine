from core import *
import inspect,textwrap
from miniworld_engine.kernels.trimul_inproj.triton.backward_fused import input_ln_residual_bwd
s=textwrap.dedent(inspect.getsource(B._BidirBackHalfTriton.backward)).replace('@staticmethod\n','')
a=s.index('    # ② gate bwd');b=s.index('    # contraction bwd',a)
s=s[:a]+'''    d_glogit, dWg, d_tri, dLNo_w, dLNo_b, dWp = cuda_plan()
'''+s[b:]
ns=dict(vars(B));exec(s,ns);back_half=ns['backward']
def backward_cuda(d,saved,dy,plan):
 ns['cuda_plan']=plan;ctx,mu,rs=saved;r=back_half(ctx,dy);dx,dgi,dbi=input_ln_residual_bwd(r[0].reshape(-1,128),d['x'].reshape(-1,128),d['gi'],mu,rs,r[12],both_key(d['n']**2))
 return dx.reshape_as(d['x']),*[v.t() for v in r[1:6]],r[6],dgi,dbi,r[7],r[8]
