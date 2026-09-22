"""Experimental inference-only width adapter. No engine auto-dispatch changes.
Weights and inputs stay live; mutate their contents, not their storage, for graph replay.
"""
from native import *
import torch.nn.functional as F
OPTS=dict(fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
@torch.compile(**OPTS)
def pack_into(dst,wl,wlg,wr,wrg):
 d=wl.shape[-1];dst.copy_(torch.stack((torch.cat((wlg,wrg)).reshape(-1,32,d),torch.cat((wl,wr)).reshape(-1,32,d)),1).reshape(8*d,d))
@torch.compile(**OPTS)
def normalize_into(dst,x,g,b):dst.copy_(F.layer_norm(x.float(),(x.shape[-1],),g,b,1e-5).to(x.dtype))
@torch.compile(**OPTS)
def output(t,xn,wp,wg,go,bo,x):
 n=x.shape[1];h=go.numel();return B.trimul_back_triton(t.reshape(1,h,n,n),xn.reshape_as(x),wp.t().contiguous(),wg.t().contiguous(),go,bo,1e-5,x)
class Inference:
 def __init__(self,x,wl,wlg,wr,wrg,wg,wp,gi,bi,go,bo,mask):
  if torch.is_grad_enabled():raise RuntimeError('Width adapter is inference-only: construct and call under torch.no_grad()')
  assert x.ndim==4 and x.shape[0]==1 and x.shape[1]==x.shape[2] and x.dtype==torch.bfloat16
  assert x.is_contiguous() and x.is_cuda and torch.cuda.get_device_capability(x.device)==(9,0)
  assert all(w.is_contiguous() and w.dtype==x.dtype and w.device==x.device for w in (wl,wlg,wr,wrg,wg,wp))
  assert all(z.device==x.device and z.is_contiguous() for z in (gi,bi,go,bo,mask))
  D=x.shape[-1];n=x.shape[1];H=2*D;assert D in (64,256,384,512) and n in (384,768)
  assert all(w.shape==(H,D) for w in (wl,wlg,wr,wrg)) and wg.shape==(D,D) and wp.shape==(D,H)
  assert all(z.dtype==torch.float32 for z in (gi,bi,go,bo,mask)) and mask.is_contiguous() and mask.numel()==n*n
  assert gi.shape==bi.shape==(D,) and go.shape==bo.shape==(H,)
  self.x=x;self.weights=(wl,wlg,wr,wrg);self.wg=wg;self.wp=wp;self.gi=gi;self.bi=bi;self.go=go;self.bo=bo;self.mask=mask;self.D=D;self.H=H
  self.cfg=json.loads((R/'selection.json').read_text())[f'{D}-{n}'];self.separate=self.cfg['input_ln']=='separate';self.w1=x.new_empty((8*D,D));pack_into(self.w1,*self.weights)
  self.xn=torch.empty_like(x) if self.separate else None
  if self.separate:normalize_into(self.xn,x,gi,bi)
  self.front=Front((self.xn if self.separate else x)[0],self.w1,mask,gi,bi,self.cfg['k1'],emit_xn=self.cfg['emit_xn'],normalize=not self.separate);self.tri=x.new_empty((H,n,n))
  self.out=Output(self.tri,x[0],wp,wg,gi,bi,go,bo,self.cfg['k3']) if self.cfg['k3'] else None
 def __call__(self):
  if torch.is_grad_enabled():raise RuntimeError('This width adapter has no autograd; use under torch.no_grad()')
  pack_into(self.w1,*self.weights)
  if self.separate:normalize_into(self.xn,self.x,self.gi,self.bi)
  ab,xn=self.front();d=self.D;h=self.H
  torch.bmm(ab[:d],ab[h:h+d].transpose(-1,-2),out=self.tri[:d]);torch.bmm(ab[d:h].transpose(-1,-2),ab[h+d:],out=self.tri[d:])
  return self.out().reshape_as(self.x) if self.out else output(self.tri,self.xn if self.separate else xn,self.wp,self.wg,self.go,self.bo,self.x)
