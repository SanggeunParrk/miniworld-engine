import json,torch,statistics,itertools,gc
from pathlib import Path
from miniworld_engine import settings
settings.configure(engine_backend='triton',trimul_sm90_kernels=frozenset(),autotune_miss_cap=24)
from miniworld_engine.kernels.trimul_inproj.cuda.anthropic_training import output_training,default_config,_kernel
from miniworld_engine.kernels.layernorm_linear.triton.te_style import _ln_materialize
from miniworld_engine.kernels.trimul_inproj.triton.output_fused import output_f567_train
from miniworld_engine.autotune.shape_key import both_key

def graph(fn):
 s=torch.cuda.Stream();s.wait_stream(torch.cuda.current_stream())
 with torch.cuda.stream(s):
  for _ in range(3):o=fn()
  g=torch.cuda.CUDAGraph()
  with torch.cuda.graph(g,stream=s):o=fn()
  g.replay()
 torch.cuda.current_stream().wait_stream(s)
 return g,o

def paired(gs,reps=40,rounds=8):
 times={k:[] for k in gs}
 for r in range(rounds):
  keys=list(gs);keys=keys if r%2==0 else keys[::-1]
  for k in keys:
   for _ in range(4):gs[k][0].replay()
   a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
   a.record()
   for _ in range(reps):gs[k][0].replay()
   b.record();b.synchronize();times[k].append(a.elapsed_time(b)*1000/reps)
 return {k:dict(median_us=statistics.median(v),min_us=min(v),max_us=max(v),samples_us=v) for k,v in times.items()}

if __name__=='__main__':
 torch.manual_seed(823);results=[]
 for H,N in itertools.product((128,256),(384,768)):
  C=128;M=N*N;kw=dict(device='cuda',dtype=torch.bfloat16)
  tri=torch.randn(H,N,N,**kw);xn=torch.randn(N,N,C,**kw)
  wp=torch.randn(C,H,**kw)/H**.5;wg=torch.randn(C,C,**kw)/C**.5;wgt=wg.T.contiguous()
  gamma=torch.rand(H,device='cuda');beta=torch.randn(H,device='cuda')*.2
  res=torch.randn(M,C,**kw);ds=(torch.rand(N,C,device='cuda')>.25).bfloat16()/.75
  def baseline():
   norm,mu,rs=_ln_materialize(tri.reshape(H,M).T,gamma,beta,1e-5,shape_key=both_key(M))
   y,p,g=output_f567_train(norm,xn.reshape(M,C),wp,wgt,res,ds,N)
   return y,norm,mu,rs,p,g
  gs={'triton':graph(baseline)}
  configs=[default_config(H),(2,64,4,2,232,1),(1,128,4,2,232,1),(1,64,4,1,232,1),(2,64,4,1,240,1),(2,64,4,1,232,0)]
  for cfg in dict.fromkeys(configs):
   k=','.join(map(str,cfg))
   fn=lambda cfg=cfg:output_training(tri,xn,wp,wg,gamma,beta,res,ds,1e-5,list(cfg))
   gs[k]=graph(fn)
   y=gs[k][1][0];yr=gs['triton'][1][0]
   error=((y.float()-yr.float()).norm()/yr.float().norm()).item()
   assert error<.005,(cfg,error)
   print('BUILT',N,H,cfg,_kernel(C,H,cfg,0).attrs(),error,flush=True)
  times=paired(gs);print('RESULT',json.dumps(dict(N=N,H=H,times=times)),flush=True)
  results.append(dict(N=N,H=H,times=times))
  Path('/home/psk6950/MiniWorld/runs/anthropic_trimul_training_20260919/k3-times.json').write_text(json.dumps(results,indent=2))
  del gs;gc.collect();torch.cuda.empty_cache()
