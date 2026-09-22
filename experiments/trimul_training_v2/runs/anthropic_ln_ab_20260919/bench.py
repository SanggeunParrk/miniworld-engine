import sys,json,gc,torch
from pathlib import Path
from experiment import *
sys.path.insert(0,str(R.parent/'anthropic_trimul_training_20260919'))
from bench_k3 import graph,paired
from miniworld_engine.kernels.trimul_inproj.triton.contract import packed_forward
from miniworld_engine.kernels.layernorm_linear.triton.te_style import _ln_materialize
from miniworld_engine.autotune.shape_key import both_key
results=[]
for n in (384,768):
 torch.manual_seed(62);c=128;h=256;m=n*n
 x=torch.randn(1,n,n,c,device='cuda',dtype=torch.bfloat16);w=torch.randn(4*h,c,device='cuda',dtype=x.dtype)/c**.5
 mask=(torch.rand(n,n,device='cuda')>.2).float();g=torch.rand(c,device='cuda');b=torch.randn_like(g)
 gg=torch.rand(h,device='cuda');bb=torch.randn_like(gg);wp=torch.randn(c,h,device='cuda',dtype=x.dtype)/h**.5;wg=torch.randn(c,c,device='cuda',dtype=x.dtype)/c**.5
 ds=(torch.rand(n,c,device='cuda')>.25).bfloat16()/.75;tri0=torch.randn(h,n,n,device='cuda',dtype=x.dtype)
 configs={};ln_times={}
 for trans,inp,gw,gb in ((False,x,g,b),(True,tri0,gg,bb)):
  gs={};funcs={}
  for kind in ('vector','tma','tma-store'):
   for threads in ((128,256) if kind=='vector' else (128,)):
    for serial in (0,1):
     key=f'{kind}-{threads}-s{serial}';f=ln if kind=='vector' else ln_tma
     if kind=='tma-store':
      from functools import partial
      f=partial(ln_tma,bulk=1)
     funcs[key]=lambda z,a,b,f=f,threads=threads,serial=serial,trans=trans:f(z,a,b,trans=trans,threads=threads,serial=serial)
     gs[key]=graph(lambda key=key:funcs[key](inp,gw,gb))
  ref=gs[next(iter(gs))][1]
  for key,v in gs.items():
   for a,z in zip(v[1],ref):assert torch.equal(a,z),(key,'LN differs')
  tm=paired(gs,reps=30,rounds=8);best=min(tm,key=lambda k:tm[k]['median_us']);configs[trans]=funcs[best];ln_times[str(trans)]=dict(best=best,times=tm)
  print('LN',n,trans,best,tm[best]['median_us'],flush=True)
  del gs,funcs,ref;gc.collect()
 def ffront(fused):
  if fused:return front(x,w,mask,g,b)
  xn,mu,rs=configs[False](x,g,b)
  ab,pre=S.front_training(xn.reshape_as(x),w,mask,list(S.front_default(h)))
  return ab,pre,xn.reshape_as(x),mu,rs
 def ftail(tri,xn,fused):
  if fused:return T.output_training(tri,xn.reshape(n,n,c),wp,wg,gg,bb,x.reshape(m,c),ds,1e-5,[2,64,4,1,232,1])
  norm,mo,ro=configs[True](tri,gg,bb)
  yy,pp,gate=S.output_training(norm.reshape(n,n,h),xn.reshape(n,n,c),wp,wg,x,ds,[2,64,4,1,232])
  return yy.reshape(m,c),norm,mo,ro,pp.reshape(m,c),gate.reshape(m,c)
 def full(fi,fo):
  ab,pre,xn,mu,rs=ffront(fi)
  tri=packed_forward(ab[:h],ab[h:],h//2)
  outs=ftail(tri,xn,fo)
  return (*outs,ab,pre,xn,mu,rs,tri)
 gs={}
 for fi in (False,True):
  for fo in (False,True):
   name=f'input_{"fused" if fi else "separate"}_output_{"fused" if fo else "separate"}'
   gs[name]=graph(lambda fi=fi,fo=fo:full(fi,fo))
 ref=gs['input_separate_output_separate'][1]
 for name,v in gs.items():
  for i,(a,z) in enumerate(zip(v[1],ref)):
   assert torch.equal(a,z),(name,i,(a.float()-z.float()).abs().max().item())
 print('FULL_PARITY_PASS',n,flush=True)
 full_times=paired(gs,reps=16,rounds=8)
 del gs,ref;gc.collect();torch.cuda.empty_cache()
 xn=configs[False](x,g,b)[0].reshape_as(x)
 gs={f'front_{f}':graph(lambda f=f:ffront(f)) for f in (False,True)}
 gs.update({f'tail_{f}':graph(lambda f=f:ftail(tri0,xn,f)) for f in (False,True)})
 components=paired(gs,reps=24,rounds=8)
 row=dict(N=n,C=c,H=h,dropout=.25,mode='training_forward_all_saves',all_saved_values_bitwise_equal=True,ln=ln_times,components=components,full=full_times)
 results.append(row);(R/'results.json').write_text(json.dumps(results,indent=2));print('RESULT',json.dumps(row),flush=True)
 del gs;gc.collect();torch.cuda.empty_cache()
