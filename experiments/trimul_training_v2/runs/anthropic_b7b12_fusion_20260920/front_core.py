"""Fixed-saves B7-B12 reference and standalone comparison utilities."""
from pathlib import Path
import sys,json,statistics,torch
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R.parent/'anthropic_ln_equal_saves_20260919'))
import core_saved as C
from miniworld_engine.kernels.trimul_inproj.triton import bidirectional as B
from miniworld_engine.kernels.trimul_inproj.triton.back_fused import front_bwd_dW
from miniworld_engine.kernels.trimul_inproj.triton.backward_fused import input_dual_bwd,input_ln_residual_bwd
from miniworld_engine.autotune.shape_key import both_key

def setup(n,dropout=.25,seed=20260920):
 torch.manual_seed(seed+n);d=C.setup(n)
 d['ds'].copy_(((torch.rand_like(d['ds'].float())>=dropout)/(1-dropout)).to(torch.bfloat16))
 _,s=C.forward(d,True,(3,64,2,2,1),(1,1))
 dy=torch.randn_like(d['x']);ctx,mu,rs=s
 xn,wl,wlg,wr,wrg,wg,wp,go,pre,lf,rf,tri,norm,mo,ro,gate,proj=ctx.saved_tensors
 dp,dg=B.gate_elem_bwd_ew(dy.reshape(-1,128),proj,gate,d['ds'],n)
 dt,_,_,_,_=B._te_backward(dp,norm,tri.reshape(256,-1).t(),mo,ro,go,wp,False,shape_key=both_key(n*n))
 dl,dr=B.packed_backward(dt.t().reshape_as(tri),lf,rf,128)
 return dict(d=d,s=s,dy=dy,dl=dl.reshape(1,256,n,n),dr=dr.reshape(1,256,n,n),dg=dg,
   xn=xn,pre=pre,wl=wl,wlg=wlg,wr=wr,wrg=wrg,wg=wg,mu=mu,rs=rs,mask=d['mask'].to(torch.bfloat16).reshape(-1))

def baseline(a,debug=False):
 d=a['d'];n=d['n'];m=n*n
 dc,wl,wlg,wr,wrg,wstack=front_bwd_dW(a['dl'],a['dr'],a['pre'],a['xn'],a['wl'],a['wlg'],a['wr'],a['wrg'],pair_mask=a['mask'])
 dxn=input_dual_bwd(a['dg'],dc.t(),a['wg'].t(),wstack,n)
 dx,gi,bi=input_ln_residual_bwd(dxn,d['x'].reshape(m,128),d['gi'],a['mu'],a['rs'],a['dy'].reshape(m,128),both_key(m))
 out=(dx,wl,wlg,wr,wrg,gi,bi)
 return (out,dc,dxn) if debug else out

def rel(a,b):return ((a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-20)).item()
def errors(out,ref):
 names=('dx','dWL','dWLg','dWR','dWRg','dgamma','dbeta')
 return {k:dict(relative_l2=rel(x,y),max_absolute=(x.float()-y.float()).abs().max().item(),finite=bool(torch.isfinite(x).all())) for k,x,y in zip(names,out,ref)}
def capture(fn):
 st=torch.cuda.Stream();st.wait_stream(torch.cuda.current_stream())
 with torch.cuda.stream(st):
  for _ in range(2):fn()
 torch.cuda.current_stream().wait_stream(st);g=torch.cuda.CUDAGraph()
 with torch.cuda.graph(g,stream=st):fn()
 return g
def paired(graphs,warmup=20,iterations=200):
 for _ in range(warmup):
  for g in graphs.values():g.replay()
 torch.cuda.synchronize();events={k:[] for k in graphs};keys=list(graphs)
 for i in range(iterations):
  for k in (keys if i%2==0 else keys[::-1]):
   a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
   a.record();graphs[k].replay();b.record();events[k].append((a,b))
 torch.cuda.synchronize();out={}
 for k,pairs in events.items():
  ts=sorted(a.elapsed_time(b)*1000 for a,b in pairs)
  out[k]=dict(median_us=statistics.median(ts),p90_us=ts[int(.9*(len(ts)-1))],samples_us=ts)
 return out

if __name__=='__main__':
 rows=[]
 with torch.no_grad():
  for n in (384,768):
   a=setup(n);g=capture(lambda:baseline(a));t=paired({'baseline':g})
   print('BASELINE',n,{k:v['median_us'] for k,v in t.items()},flush=True)
   rows.append(dict(L=n,dropout=.25,times=t));(R/'baseline-initial.json').write_text(json.dumps(rows,indent=2))
