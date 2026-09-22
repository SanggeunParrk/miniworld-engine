import sys,time,gc,json,statistics,concurrent.futures
from core import *
sys.path.insert(0,str(R.parent/'anthropic_trimul_training_20260919'))
from bench_k3 import graph,paired
from miniworld_engine.autotune.configs import configs_for
from miniworld_engine.kernels.trimul_inproj.triton.contract import packed_forward
F1=(2,64,8,2,-1,232,2);F3=(2,64,4,1,1,1)
LN=[('vector',t,s,0) for t in (64,128,256,512) for s in (0,1)]+[('tma',128,s,b) for s in (0,1) for b in (0,1)]
TRI=sorted(set((c.kwargs['BLOCK_M1'],c.kwargs['BLOCK_K'],c.num_warps,c.num_stages) for c in configs_for('layernorm_fwd_saveact_triton')))
K1=list(k1_candidates());K3=list(k3_candidates())
print('CANDIDATES',dict(ln=len(LN),triton=len(TRI),k1=len(K1),k3=len(K3)),flush=True)
builds=[('ln',False,c) for c in LN]+[('k1',False,c) for c in K1]+[('k3',False,c) for c in K3]+[('k1',True,F1),('k3',True,F3)]
errors=[]
def compile_one(args):
 try:build(*args);return None
 except Exception as e:return dict(args=args,error=str(e))
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
 for i,result in enumerate(pool.map(compile_one,builds)):
  if result:errors.append(result)
  if i%10==0:print('COMPILED',i+1,'/',len(builds),flush=True)
(R/'compile-errors.json').write_text(json.dumps(errors,indent=2))
if errors:raise RuntimeError('compile errors; inspect record')
results=[]
def quick(fn):
 gg,out=graph(fn);a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);ts=[]
 for _ in range(3):
  a.record()
  for j in range(6):gg.replay()
  b.record();b.synchronize();ts.append(a.elapsed_time(b)*1000/6)
 return statistics.median(ts),out

def tune(kind,candidates,fn,ref,exact=True):
 path=R/f'tune-{N}-{kind}.json';records=[]
 for i,cfg in enumerate(candidates):
  try:
   us,out=quick(lambda:fn(cfg));torch.cuda.synchronize()
   if exact:valid=torch.equal(out,ref)
   else:valid=((out.float()-ref.float()).norm()/ref.float().norm()).item()<.003
   if not valid:raise AssertionError('numerical mismatch')
   records.append(dict(config=cfg,us=us));del out
  except Exception as e:records.append(dict(config=cfg,error=str(e)));print('FAILED',kind,cfg,str(e)[:200],flush=True)
  if i%10==0:
   path.write_text(json.dumps(records,indent=2));print('TUNED',N,kind,i+1,'/',len(candidates),flush=True)
 path.write_text(json.dumps(records,indent=2))
 good=sorted((r for r in records if 'us' in r),key=lambda r:r['us'])
 if not good:raise RuntimeError('no valid '+kind)
 # Recheck shortlist in alternating order with longer graph runs.
 top=good[:4];gs={str(r['config']):graph(lambda cfg=tuple(r['config']):fn(cfg)) for r in top}
 ts=paired(gs,reps=24,rounds=8);best=min(top,key=lambda r:ts[str(r['config'])]['median_us'])
 path.with_name(path.stem+'-final.json').write_text(json.dumps(ts,indent=2))
 print('BEST',N,kind,best['config'],ts[str(best['config'])]['median_us'],flush=True)
 return tuple(best['config']),records

for N in (384,768):
 torch.manual_seed(67);m=N*N;x=torch.randn(1,N,N,128,device='cuda',dtype=torch.bfloat16);g=torch.rand(128,device='cuda');b=torch.randn_like(g);go=torch.rand(256,device='cuda');bo=torch.randn_like(go)
 w=torch.randn(1024,128,device='cuda',dtype=x.dtype)/128**.5;wp=torch.randn(128,256,device='cuda',dtype=x.dtype)/16;wg=torch.randn(128,128,device='cuda',dtype=x.dtype)/128**.5;mask=(torch.rand(N,N,device='cuda')>.2).float()
 xn=ln(x,g,b,('tma',128,0,1));ab=front(x,w,mask,g,b,True,F1);tri=packed_forward(ab[:256],ab[256:],128)
 yr=output(tri,x,wp,wg,g,b,go,bo,x,True,F3)
 lc,lr=tune('ln-anthropic',LN,lambda cfg:ln(x,g,b,cfg),xn)
 tc,tr=tune('ln-triton',TRI,lambda cfg:triton_ln(x,g,b,cfg),xn,False)
 kc,kr=tune('k1-shapes',K1,lambda cfg:front(xn,w,mask,g,b,False,cfg),ab)
 # Retune register budget, scheduling and WG start offset around the three best geometries.
 best=sorted((r for r in kr if 'us' in r),key=lambda r:r['us'])[:3];refine=set()
 for r in best:
  bi,bj,slot,sk,_,rg,_=r['config']
  for regs,sched,offset in itertools.product(sorted(set((rg,max(96,rg-32),max(96,rg-64)))),(0,1),(0,2,4)):
   cf=(bi,bj,slot,sk,sched,regs,offset)
   try:k1_smem(cf)
   except ValueError:continue
   refine.add(cf)
 kc,krr=tune('k1-refine',sorted(refine|{kc}),lambda cfg:front(xn,w,mask,g,b,False,cfg),ab)
 oc,orr=tune('k3',K3,lambda cfg:output(tri,xn,wp,wg,g,b,go,bo,x,False,cfg),yr)
 def full(mode):
  norm=x if mode=='original' else ln(x,g,b,lc) if mode=='split-anthropic-ln' else triton_ln(x,g,b,tc)
  a=front(norm,w,mask,g,b,mode=='original',F1 if mode=='original' else kc)
  t=packed_forward(a[:256],a[256:],128)
  return output(t,norm,wp,wg,g,b,go,bo,x,mode=='original',F3 if mode=='original' else oc)
 gs={mode:graph(lambda mode=mode:full(mode)) for mode in ('original','split-anthropic-ln','split-triton-ln')}
 errs={mode:((v[1].float()-yr.float()).norm()/yr.float().norm()).item() for mode,v in gs.items()};assert max(errs.values())<.005,errs
 times=paired(gs,reps=30,rounds=12)
 del gs;gc.collect()
 comp={'ln-triton':lambda:triton_ln(x,g,b,tc),'ln-anthropic':lambda:ln(x,g,b,lc),'k1-original':lambda:front(x,w,mask,g,b,True,F1),'k1-split':lambda:front(xn,w,mask,g,b,False,kc),'k3-original':lambda:output(tri,x,wp,wg,g,b,go,bo,x,True,F3),'k3-split':lambda:output(tri,xn,wp,wg,g,b,go,bo,x,False,oc)}
 gs={name:graph(fn) for name,fn in comp.items()};ct=paired(gs,reps=30,rounds=12)
 row=dict(N=N,mode='inference_no_training_saves',dropout=0,residual=True,configs=dict(ln_anthropic=lc,ln_triton=tc,k1_split=kc,k3_split=oc,k1_original=F1,k3_original=F3),relative_l2=errs,full=times,components=ct,counts=dict(k1_initial=len(K1),k1_refine=len(refine|{kc}),k3=len(K3),ln_anthropic=len(LN),ln_triton=len(TRI)))
 results.append(row);(R/'results.json').write_text(json.dumps(results,indent=2));print('RESULT',json.dumps(row),flush=True)
 del gs,comp;gc.collect();torch.cuda.empty_cache()
