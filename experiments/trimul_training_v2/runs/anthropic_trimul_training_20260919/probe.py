import json,torch
from miniworld_engine.kernels.trimul_inproj.cuda.anthropic_training import output_training,default_config,_kernel

def rel(a,b):return ((a.float()-b.float()).square().sum()/b.float().square().sum().clamp_min(1e-30)).sqrt().item()
torch.manual_seed(812)
for H in (128,256):
 for N in (64,72,384):
  C=128;M=N*N;dev='cuda';dt=torch.bfloat16
  tri=torch.randn(H,N,N,device=dev,dtype=dt)
  xn=torch.randn(N,N,C,device=dev,dtype=dt)
  wp=torch.randn(C,H,device=dev,dtype=dt)/H**.5;wg=torch.randn(C,C,device=dev,dtype=dt)/C**.5
  gamma=torch.randn(H,device=dev);beta=torch.randn(H,device=dev)*.2
  res=torch.randn(M,C,device=dev,dtype=dt)
  ds=(torch.rand(N,C,device=dev)>.25).to(dt)/.75
  cfg=default_config(H)
  y,norm,mean,rs,proj,gate=output_training(tri,xn,wp,wg,gamma,beta,res,ds,1e-5,list(cfg))
  torch.cuda.synchronize()
  xx=tri.reshape(H,M).T.float();mr=xx.mean(-1);rr=((xx-mr[:,None]).square().mean(-1)+1e-5).rsqrt()
  nr=((xx-mr[:,None])*rr[:,None]*gamma+beta).to(dt)
  pr=nr@wp.T;gr=torch.sigmoid((xn.reshape(M,C)@wg.T).float())
  yr=(pr.float()*gr*ds.repeat(N,1).float()+res.float()).to(dt)
  errors={k:rel(a,b) for k,a,b in [('y',y,yr),('norm',norm,nr),('mean',mean,mr),('rstd',rs,rr),('proj',proj,pr),('gate',gate,gr.to(dt))]}
  print(json.dumps(dict(N=N,H=H,cfg=cfg,errors=errors,attrs=_kernel(C,H,cfg,0).attrs())),flush=True)
  assert all(v<.007 for v in errors.values()),errors
