"""Only change K3 operand/implementation: reuse tuned split LN/K1; original K3 recomputes LN."""
import sys,json,gc,statistics
from core import *
sys.path.insert(0,str(R.parent/'anthropic_trimul_training_20260919'))
from bench_k3 import graph
from miniworld_engine.kernels.trimul_inproj.triton.contract import packed_forward
rows=json.loads((R/'results.json').read_text());results=[]
for row in rows:
 n=row['N'];cf=row['configs'];lc=tuple(cf['ln_anthropic']);tc=tuple(cf['ln_triton']);kc=tuple(cf['k1_split']);oc=tuple(cf['k3_split']);f1=tuple(cf['k1_original']);f3=tuple(cf['k3_original'])
 torch.manual_seed(67);x=torch.randn(1,n,n,128,device='cuda',dtype=torch.bfloat16);g=torch.rand(128,device='cuda');b=torch.randn_like(g);go=torch.rand(256,device='cuda');bo=torch.randn_like(go)
 w=torch.randn(1024,128,device='cuda',dtype=x.dtype)/128**.5;wp=torch.randn(128,256,device='cuda',dtype=x.dtype)/16;wg=torch.randn(128,128,device='cuda',dtype=x.dtype)/128**.5;mask=(torch.rand(n,n,device='cuda')>.2).float()
 def full(mode):
  original=mode=='original';recompute=original or mode.endswith('recompute')
  z=x if original else ln(x,g,b,lc) if mode.startswith('anthropic') else triton_ln(x,g,b,tc)
  a=front(z,w,mask,g,b,original,f1 if original else kc);t=packed_forward(a[:256],a[256:],128)
  return output(t,x if recompute else z,wp,wg,g,b,go,bo,x,recompute,f3 if recompute else oc,tma_residual=not recompute)
 keys=['original','anthropic-share','anthropic-recompute','triton-share','triton-recompute'];gs={k:graph(lambda k=k:full(k)) for k in keys};times={k:[] for k in keys}
 ref=gs['original'][1];errs={k:((v[1].float()-ref.float()).norm()/ref.float().norm()).item() for k,v in gs.items()}
 assert errs['anthropic-share']==errs['anthropic-recompute']==0,errs
 assert max(errs.values())<.005,errs
 print('VALIDATED',n,errs,flush=True)
 for _ in range(200):
  for k in keys:gs[k][0].replay()
 for r in range(20):
  order=keys[r%5:]+keys[:r%5]
  if (r//5)%2:order=order[::-1]
  for k in order:
   for _ in range(8):gs[k][0].replay()
   a=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);a.record()
   for _ in range(200):gs[k][0].replay()
   end.record();end.synchronize();times[k].append(a.elapsed_time(end)*1000/200)
 record=dict(N=n,configs=cf,relative_l2=errs,rounds=20,replays=200,training_saves=False,dropout=0,residual=True,weights_prepacked=True,times={k:dict(median_us=statistics.median(v),min_us=min(v),max_us=max(v),std_us=statistics.stdev(v),samples_us=v) for k,v in times.items()})
 for provider in ('anthropic','triton'):
  ds=[a-b for a,b in zip(times[provider+'-recompute'],times[provider+'-share'])]
  record[provider+'_recompute_minus_share']=dict(mean_us=statistics.mean(ds),std_us=statistics.stdev(ds),median_us=statistics.median(ds),recompute_wins=sum(d<0 for d in ds),samples_us=ds)
 results.append(record);(R/'recompute-results.json').write_text(json.dumps(results,indent=2))
 print('RESULT',n,{k:round(v['median_us'],3) for k,v in record['times'].items()},flush=True)
 del gs,ref;gc.collect();torch.cuda.empty_cache()
