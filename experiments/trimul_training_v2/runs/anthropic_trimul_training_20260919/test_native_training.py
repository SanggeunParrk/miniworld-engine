import torch,json
from miniworld_engine.integrations.anthropic_training import *
from miniworld_engine.integrations.anthropic_training import _front,_back

def rel(a,b): return ((a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-20)).item()
def setup(n,direction):
 torch.manual_seed(439);c=128;h=c*(2 if direction=='bidirectional' else 1)
 z=torch.randn(1,n,n,c,device='cuda',dtype=torch.bfloat16,requires_grad=True)
 sizes=[(c,),(c,),(h,c),(h,c),(h,c),(h,c),(h,),(h,),(c,h),(c,c)]
 w={}
 for key,sh in zip(WEIGHT_KEYS,sizes):
  v=torch.randn(sh,device='cuda',dtype=torch.bfloat16 if len(sh)==2 else torch.float32)
  v=v/(sh[1]**.5) if len(sh)==2 else (v*.2+(1 if key.endswith('_w') else 0))
  w[key]=v.requires_grad_()
 mask=torch.rand(1,n,n,device='cuda')>.2
 ds=(torch.rand(1,1,n,c,device='cuda')>.25).bfloat16()/.75
 return z,w,mask,ds

def reference(z,w,mask,direction):
 ww=list(w.values());a,b=_front(z[0],ww,mask[0],1e-5)
 if direction=='bidirectional':
  h=a.shape[0]//2;t=torch.cat((a[:h]@b[:h].transpose(1,2),a[h:].transpose(1,2)@b[h:]))
 elif direction=='outgoing':t=a@b.transpose(1,2)
 else:t=a.transpose(1,2)@b
 return _back(z[0],t,ww,1e-5)[None]

def run(n,direction):
 z,w,m,ds=setup(n,direction);dy=torch.randn_like(z);leaves=(z,*w.values())
 y=triangle_multiplication_training(z,m,weights=w,direction=direction)
 if direction!='bidirectional':
  with torch.no_grad(): orig=native_ops().trimul(z,m,direction=direction,weights=w,residual=False,cache={})
  torch.testing.assert_close(y,orig,rtol=0,atol=0)
 out=z+y*ds;g=torch.autograd.grad(out,leaves,dy)
 r=z+reference(z,w,m,direction)*ds;rg=torch.autograd.grad(r,leaves,dy)
 vals={k:rel(a,b) for k,a,b in zip(['output','input',*WEIGHT_KEYS],[out,*g],[r,*rg])}
 print(n,direction,json.dumps(vals),flush=True)
 assert max(vals.values())<.02
 assert all(torch.isfinite(v).all() for v in g)

if __name__=='__main__':
 torch.backends.cuda.matmul.allow_tf32=False
 for d in ['outgoing','incoming','bidirectional']:
  for n in [64,65]:run(n,d)
