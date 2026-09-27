"""Anthropic-derived K3: dropout, residual, all gradients and graph replay."""
import json,torch
from miniworld_engine import settings
import pytest

@pytest.fixture(autouse=True)
def sm90(monkeypatch):
 if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9,0):
  pytest.skip('SM90 required')
 monkeypatch.setattr(settings,'_ACTIVE',settings.current())
 settings.configure(engine_backend='triton',trimul_sm90_kernels=frozenset(),autotune_miss_cap=3)
 torch.backends.cuda.matmul.allow_tf32=False

from miniworld_engine.kernels.trimul_inproj.triton.bidirectional import bidirectional_trimul_triton
from miniworld_engine.kernels.trimul_inproj.triton.unidirectional import trimul_triton

def rel(a,b):return ((a.float()-b.float()).square().sum()/b.float().square().sum().clamp_min(1e-30)).sqrt().item()
def setup(N,kind):
 torch.manual_seed(341);C=128;H=C*(2 if kind=='bidir' else 1);dt=torch.bfloat16;dev='cuda'
 x=torch.randn(1,N,N,C,device=dev,dtype=dt,requires_grad=True)
 weights=[(torch.randn(H,C,device=dev,dtype=dt)/C**.5).requires_grad_() for _ in range(4)]
 weights += [(torch.randn(C,C,device=dev,dtype=dt)/C**.5).requires_grad_(),(torch.randn(C,H,device=dev,dtype=dt)/H**.5).requires_grad_()]
 norms=[torch.rand(C,device=dev,requires_grad=True),torch.randn(C,device=dev,requires_grad=True),torch.rand(H,device=dev,requires_grad=True),torch.randn(H,device=dev,requires_grad=True)]
 with torch.no_grad():norms[0][0]=0;norms[2][0]=0
 leaves=[x,*weights,*norms]
 mask=torch.rand(1,N,N,device=dev)>.2
 ds=(torch.rand(1,1,N,C,device=dev)>.25).to(dt)/.75
 def call(backend):
  args=(*leaves,1e-5,1e-5,C)
  if kind=='bidir':return bidirectional_trimul_triton(*args,mask=mask,dropscale=ds,output_backend=backend)
  return trimul_triton(*args,outgoing=kind=='outgoing',mask=mask,dropscale=ds,output_backend=backend)
 return leaves,call,mask,ds


def reference(leaves,kind,mask,ds):
 x,wl,wlg,wr,wrg,wg,wo,gi,bi,go,bo=leaves
 N=x.shape[1];C=x.shape[-1];H=wl.shape[0]
 xn=torch.nn.functional.layer_norm(x.float(),(C,),gi,bi,1e-5).bfloat16()
 a=(torch.sigmoid(xn.float()@wlg.float().T)*(xn.float()@wl.float().T)).bfloat16()*mask[...,None]
 b=(torch.sigmoid(xn.float()@wrg.float().T)*(xn.float()@wr.float().T)).bfloat16()*mask[...,None]
 a=a[0].permute(2,0,1).contiguous();b=b[0].permute(2,0,1).contiguous()
 if kind=='bidir':
  tri=torch.cat((a[:C]@b[:C].transpose(1,2),a[C:].transpose(1,2)@b[C:]),0)
 elif kind=='outgoing':tri=a@b.transpose(1,2)
 else:tri=a.transpose(1,2)@b
 norm=torch.nn.functional.layer_norm(tri.permute(1,2,0).float(),(H,),go,bo,1e-5).bfloat16()
 p=norm@wo.T;g=torch.sigmoid((xn@wg.T).float())
 return (p.float()*g*ds.float()+x.float()).bfloat16()


@pytest.mark.parametrize('kind',['outgoing','incoming','bidir'])
def test_full_gradients(kind):
 leaves,call,mask,ds=setup(64,kind);dy=torch.randn_like(leaves[0])
 y=call('anthropic_cuda');gg=torch.autograd.grad(y,leaves,dy)
 for fn,tol in [(lambda:call('triton'),.004),(lambda:reference(leaves,kind,mask,ds),.02)]:
  yr=fn();rg=torch.autograd.grad(yr,leaves,dy)
  for a,b in zip((y,*gg),(yr,*rg)):
   assert torch.isfinite(a).all()
   assert rel(a,b)<tol,rel(a,b)


@pytest.mark.parametrize('zero',['dropout','mask','projection'])
def test_identity_and_zero_gradients(zero):
 leaves,call,mask,ds=setup(64,'bidir');dy=torch.randn_like(leaves[0])
 with torch.no_grad():
  if zero=='dropout':ds.zero_()
  elif zero=='mask':mask.zero_();leaves[-1].zero_()
  else:leaves[6].zero_()
 y=call('anthropic_cuda');grads=torch.autograd.grad(y,leaves,dy)
 torch.testing.assert_close(y,leaves[0],rtol=0,atol=0)
 torch.testing.assert_close(grads[0],dy,rtol=0,atol=0)
 if zero=='dropout':
  assert all(torch.count_nonzero(g)==0 for g in grads[1:])
 else:
  # Wo or LN_out bias can have a gradient even when their current value is zero.
  assert all(torch.count_nonzero(g)==0 for g in grads[1:6])


def test_compile_graph_and_live_weights():
 stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
 with torch.cuda.stream(stream):
  leaves,call,mask,ds=setup(64,'bidir');dy=torch.randn_like(leaves[0])
  fn=torch.compile(lambda:call('anthropic_cuda'),dynamic=False,fullgraph=True)
  for _ in range(3):
   y=fn();expected=torch.autograd.grad(y,leaves,dy)
  graph=torch.cuda.CUDAGraph()
  with torch.cuda.graph(graph,stream=stream):
   actual=fn();grads=torch.autograd.grad(actual,leaves,dy)
  graph.replay()
  torch.testing.assert_close(actual,y,rtol=0,atol=0)
  for a,b in zip(grads,expected):assert rel(a,b)<.001
  with torch.no_grad():leaves[6].mul_(.7);ds[:, :, ::2].zero_()
  graph.replay()
  eager=call('anthropic_cuda');eg=torch.autograd.grad(eager,leaves,dy)
  torch.testing.assert_close(actual,eager,rtol=0,atol=0)
  for a,b in zip(grads,eg):assert rel(a,b)<.001
 torch.cuda.current_stream().wait_stream(stream)
 torch.cuda.synchronize()


@pytest.mark.parametrize('bidir',[False,True])
def test_module_entry(bidir):
 from miniworld_engine.modules.triangle_multiplication import TriangleMultiplication,BidirectionalTriangleMultiplication
 cls=BidirectionalTriangleMultiplication if bidir else TriangleMultiplication
 model=cls(128,implementation='triton',training_output_backend='anthropic_cuda',p_drop=.25).cuda().bfloat16()
 with torch.no_grad():
  for name,p in model.named_parameters():
   if p.ndim==2:p.normal_(std=p.shape[1]**-.5)
 x=torch.randn(1,64,64,128,device='cuda',dtype=torch.bfloat16,requires_grad=True)
 mask=torch.rand(1,64,device='cuda')>.25;dy=torch.randn_like(x)
 leaves=(x,*tuple(model.parameters()))
 torch.manual_seed(391);y=model(x,mask);grads=torch.autograd.grad(y,leaves,dy)
 model.training_output_backend='triton'
 torch.manual_seed(391);yr=model(x,mask);rg=torch.autograd.grad(yr,leaves,dy)
 for a,b in zip((y,*grads),(yr,*rg)):assert rel(a,b)<.004
 model.training_output_backend='anthropic_cuda';model.eval()
 with torch.no_grad():assert torch.isfinite(model(x,mask)).all()


@pytest.mark.parametrize('cfg',[(2,64,4,1,232,1),(1,64,4,1,232,1),(1,128,4,2,240,0)])
def test_padded_ragged_tma(cfg):
 from miniworld_engine.kernels.trimul_inproj.cuda.anthropic_training import output_training
 N=65;NP=72;C=128;H=256;dt=torch.bfloat16
 tri=torch.randn(H,NP,NP,device='cuda',dtype=dt)
 xn=torch.randn(N,N,C,device='cuda',dtype=dt)
 wp=torch.randn(C,H,device='cuda',dtype=dt)*H**-.5
 wg=torch.randn(C,C,device='cuda',dtype=dt)*C**-.5
 gamma=torch.rand(H,device='cuda');beta=torch.randn(H,device='cuda')*.2
 res=torch.randn(N*N,C,device='cuda',dtype=dt)
 ds=(torch.rand(N,C,device='cuda')>.25).bfloat16()/.75
 y,norm,mu,rs,proj,gate=output_training(tri,xn,wp,wg,gamma,beta,res,ds,1e-5,list(cfg))
 v=tri[:,:N,:N].permute(1,2,0).reshape(N*N,H).float()
 nr=torch.nn.functional.layer_norm(v,(H,),gamma,beta,1e-5).bfloat16()
 pr=nr@wp.T;gr=torch.sigmoid((xn.reshape(N*N,C)@wg.T).float())
 yr=(pr.float()*gr*ds.repeat(N,1).float()+res.float()).bfloat16()
 for a,b in [(y,yr),(norm,nr),(proj,pr),(gate,gr.bfloat16()),(mu,v.mean(-1)),(rs,(v.var(-1,unbiased=False)+1e-5).rsqrt())]:
  assert torch.isfinite(a).all()
  assert rel(a,b)<.003
