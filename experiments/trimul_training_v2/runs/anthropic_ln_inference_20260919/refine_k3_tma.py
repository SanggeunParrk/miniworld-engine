import sys,json,gc,statistics,concurrent.futures
from core import *
sys.path.insert(0,str(R.parent/'anthropic_trimul_training_20260919'))
from bench_k3 import graph,paired
from miniworld_engine.kernels.trimul_inproj.triton.contract import packed_forward
s=(R/'tune.py').read_text();exec(s[s.index('def quick('):s.index('for N in (384,768):')])
rows=json.loads((R/'results.json').read_text());(R/'global-residual-results.json').write_text(json.dumps(rows,indent=2));outrows=[]
cs=list(k3_candidates())
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:list(pool.map(lambda c:build('k3_tma',False,c),cs))
for row in rows:
 N=row['N'];cf=row['configs'];lc=tuple(cf['ln_anthropic']);tc=tuple(cf['ln_triton']);kc=tuple(cf['k1_split']);oldoc=tuple(cf['k3_split']);F1=tuple(cf['k1_original']);F3=tuple(cf['k3_original'])
 torch.manual_seed(67);m=N*N;x=torch.randn(1,N,N,128,device='cuda',dtype=torch.bfloat16);g=torch.rand(128,device='cuda');b=torch.randn_like(g);go=torch.rand(256,device='cuda');bo=torch.randn_like(go)
 w=torch.randn(1024,128,device='cuda',dtype=x.dtype)/128**.5;wp=torch.randn(128,256,device='cuda',dtype=x.dtype)/16;wg=torch.randn(128,128,device='cuda',dtype=x.dtype)/128**.5;mask=(torch.rand(N,N,device='cuda')>.2).float()
 xn=ln(x,g,b,lc);ab=front(x,w,mask,g,b,True,F1);tri=packed_forward(ab[:256],ab[256:],128);yr=output(tri,x,wp,wg,g,b,go,bo,x,True,F3)
 oc,_=tune('k3-tma',cs,lambda cfg:output(tri,xn,wp,wg,g,b,go,bo,x,False,cfg,tma_residual=True),yr)
 gs={'global':graph(lambda:output(tri,xn,wp,wg,g,b,go,bo,x,False,oldoc)),'tma':graph(lambda:output(tri,xn,wp,wg,g,b,go,bo,x,False,oc,True))}
 rt=paired(gs,reps=30,rounds=12);use_tma=rt['tma']['median_us']<rt['global']['median_us'];oc=oc if use_tma else oldoc
 del gs;gc.collect()
 def full(mode):
  norm=x if mode=='original' else ln(x,g,b,lc) if mode=='split-anthropic-ln' else triton_ln(x,g,b,tc)
  a=front(norm,w,mask,g,b,mode=='original',F1 if mode=='original' else kc);t=packed_forward(a[:256],a[256:],128)
  return output(t,norm,wp,wg,g,b,go,bo,x,mode=='original',F3 if mode=='original' else oc,tma_residual=use_tma and mode!='original')
 gs={mode:graph(lambda mode=mode:full(mode)) for mode in ('original','split-anthropic-ln','split-triton-ln')}
 errs={mode:((v[1].float()-yr.float()).norm()/yr.float().norm()).item() for mode,v in gs.items()};assert max(errs.values())<.005
 times=paired(gs,reps=30,rounds=12);del gs;gc.collect()
 comp={'ln-triton':lambda:triton_ln(x,g,b,tc),'ln-anthropic':lambda:ln(x,g,b,lc),'k1-original':lambda:front(x,w,mask,g,b,True,F1),'k1-split':lambda:front(xn,w,mask,g,b,False,kc),'k3-original':lambda:output(tri,x,wp,wg,g,b,go,bo,x,True,F3),'k3-split':lambda:output(tri,xn,wp,wg,g,b,go,bo,x,False,oc,use_tma)}
 gs={name:graph(fn) for name,fn in comp.items()};ct=paired(gs,reps=30,rounds=12)
 row.update(full=times,components=ct,relative_l2=errs,residual_tma=use_tma,residual_comparison=rt);row['configs']['k3_split']=oc;row['counts']['k3_tma']=len(cs)
 outrows.append(row);(R/'results.json').write_text(json.dumps(outrows,indent=2));print('RESULT',json.dumps(row),flush=True)
 del gs,comp;gc.collect();torch.cuda.empty_cache()
