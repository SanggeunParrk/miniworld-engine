from pathlib import Path
E=Path('/home/psk6950/MiniWorld/runs/trimul_sm90_parity_20260917/engine')
s=Path('/home/psk6950/MiniWorld/runs/anthropic_trimul_training_20260919/check_full.py').read_text()
s=s[:s.index("if __name__=='__main__':")]
s=s.replace("settings.configure(engine_backend='triton',trimul_sm90_kernels=frozenset(),autotune_miss_cap=3)",'''import pytest

@pytest.fixture(autouse=True)
def sm90(monkeypatch):
 if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9,0):
  pytest.skip('SM90 required')
 monkeypatch.setattr(settings,'_ACTIVE',settings.current())
 settings.configure(engine_backend='triton',trimul_sm90_kernels=frozenset(),autotune_miss_cap=3)
 torch.backends.cuda.matmul.allow_tf32=False
''')
s+='''
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
'''
(E/'tests/numerics/test_trimul_anthropic_training_gpu.py').write_text('"""Anthropic-derived K3: dropout, residual, all gradients and graph replay."""\n'+s)
