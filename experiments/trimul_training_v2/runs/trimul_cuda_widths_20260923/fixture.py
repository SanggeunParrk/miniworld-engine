import torch
import torch.nn.functional as F
from miniworld_engine.kernels.trimul_inproj.triton import bidirectional as B
def setup(D,L):
 H=2*D
 torch.manual_seed(20260922);dev='cuda';bf=torch.bfloat16
 x=torch.randn(1,L,L,D,device=dev,dtype=bf,requires_grad=True)
 weights=[(torch.randn(s,device=dev)/s[-1]**.5).to(bf).requires_grad_(True) for s in [(H,D)]*4+[(D,D),(D,H)]]
 affine=[(torch.ones(c,device=dev)+.1*torch.randn(c,device=dev) if i%2==0 else .05*torch.randn(c,device=dev)).requires_grad_(True) for i,c in enumerate([D,D,H,H])]
 leaves=(x,*weights,*affine);dy=torch.randn_like(x);mask=(torch.rand((1,L,L),device=dev)>.15).to(bf);ds=((torch.rand((1,1,L,D),device=dev)>.25).to(bf)*(4/3));names=['y','dx','dWL','dWLg','dWR','dWRg','dWgate','dWproj','dgamma_in','dbeta_in','dgamma_out','dbeta_out']
 def ref(x,wl,wlg,wr,wrg,wg,wp,gi,bi,go,bo,mask,ds):
  xn=F.layer_norm(x.float(),(D,),gi,bi,1e-5).to(x.dtype)
  left=(torch.sigmoid(F.linear(xn,wlg))*F.linear(xn,wl))*mask[...,None];right=(torch.sigmoid(F.linear(xn,wrg))*F.linear(xn,wr))*mask[...,None]
  out=torch.einsum('bikd,bjkd->bijd',left[...,:D],right[...,:D]);inc=torch.einsum('bkid,bkjd->bijd',left[...,D:],right[...,D:]);tri=torch.cat((out,inc),-1)
  norm=F.layer_norm(tri.float(),(H,),go,bo,1e-5).to(x.dtype);update=torch.sigmoid(F.linear(xn,wg))*F.linear(norm,wp)
  return x+(update if ds is None else update*ds)
 def triton(*args):return B.bidirectional_trimul_triton(*args[:-2],1e-5,1e-5,D,mask=args[-2],dropscale=args[-1])
 return leaves,dy,mask,ds,ref,triton,names
