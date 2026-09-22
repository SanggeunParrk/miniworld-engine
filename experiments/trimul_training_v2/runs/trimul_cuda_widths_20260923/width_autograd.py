"""Explicit all-width CUDA training entry, including the tuned D128 path.

No silent Triton backward fallback. Each forward owns its activations and
scratch; retained graphs and repeated invocations cannot overwrite each other.
"""
from width_plan import *
import importlib.util
@lru_cache(None)
def fixed_policy():
 path=R.parent/'trimul_ln_gradient_20260922/policy.py';s=importlib.util.spec_from_file_location('width_d128_fixed',path);m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m);return m
class D128:
 def __init__(self,leaves,mask,ds):
  x,wl,wlg,wr,wrg,wg,wp,gi,bi,go,bo=leaves;n=x.shape[1];dy=torch.zeros_like(x);weights=(wl,wlg,wr,wrg,wg)
  w1=torch.empty((1024,128),device=x.device,dtype=x.dtype);pack_into(w1,*weights[:4])
  d=dict(n=n,x=x,leaves=leaves,mask=mask.reshape(n,n).float(),ds=ds.reshape(n,128),wt=[w.t().contiguous() for w in weights],wp=wp,gi=gi,bi=bi,go=go,bo=bo,w1=w1)
  a=dict(d=d,dy=dy,dl=x.new_zeros((256,n,n)),dr=x.new_zeros((256,n,n)),dg=x.new_zeros((n*n,128)),mask=mask.bfloat16().reshape(-1))
  F=fixed_policy();self.model=F.Fixed(a) if n==384 else F.P.Regression(a);self.a=a;self.kept=None
 def forward(self):y,self.kept=self.model.forward();return y
 def backward(self,dy):self.a['dy']=dy.contiguous();return self.model.backward(self.kept)
class _TriMul(torch.autograd.Function):
 @staticmethod
 def forward(ctx,*args):
  leaves=args[:11];mask,ds=args[11:];x=leaves[0];D=x.shape[-1];n=x.shape[1]
  if D not in (64,128,256,384,512) or x.shape!=(1,n,n,D) or n not in (384,768):raise ValueError('CUDA training requires B1, D64/128/256/384/512, L384/768')
  if not x.is_cuda or torch.cuda.get_device_capability(x.device)!=(9,0):raise ValueError('CUDA width training requires Hopper sm90')
  if x.dtype!=torch.bfloat16 or any(t.dtype!=torch.bfloat16 for t in leaves[:7]):raise ValueError('Inputs and projection weights must be BF16')
  if x.device.index!=0:raise ValueError('Use one visible GPU per process for this development adapter')
  if ds.dtype!=torch.bfloat16:raise ValueError('Dropout scales must be BF16')
  if leaves[7].shape!=leaves[8].shape or leaves[7].shape!=(D,) or leaves[9].shape!=leaves[10].shape or leaves[9].shape!=(2*D,):raise ValueError('LN affine width mismatch')
  if any(t.dtype!=torch.float32 for t in leaves[7:]):raise ValueError('LN affine parameters must be FP32')
  if any(not t.is_contiguous() or t.device!=x.device for t in args):raise ValueError('All inputs must be contiguous and on the same device')
  if mask.numel()!=n*n or ds.numel()!=n*D:raise ValueError('Pair mask and row-broadcast dropout shape mismatch')
  if any(w.shape!=(2*D,D) for w in leaves[1:5]) or leaves[5].shape!=(D,D) or leaves[6].shape!=(D,2*D):raise ValueError('Bidirectional hidden width must be 2D')
  if D==128:ctx.model=D128(leaves,mask,ds)
  else:ctx.model=Training(*leaves,mask,ds,torch.empty_like(x))
  ctx.save_for_backward(*args)
  return ctx.model.forward()
 @staticmethod
 def backward(ctx,dy):
  T._launch_module()._make_context_current(dy.device.index)
  _=ctx.saved_tensors # Enforce PyTorch's in-place version checks.
  if isinstance(ctx.model,D128):grads=ctx.model.backward(dy)
  else:grads=ctx.model.backward(dy)
  return (*grads,None,None)
def bidirectional_trimul_cuda(x,wl,wlg,wr,wrg,wg,wp,gi,bi,go,bo,mask,dropscale):
 """All-width H100 training with caller-provided fixed/broadcast dropout scales."""
 return _TriMul.apply(x,wl,wlg,wr,wrg,wg,wp,gi,bi,go,bo,mask,dropscale)
