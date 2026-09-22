"""Separate core/full timing domains, three paired blocks of200 samples.

Every block has20 warmups and alternating variant order. This prevents core
measurements from interleaving with unrelated full-backward graph tails.
Report every block and pooled medians, not the fastest run.
"""
from measure_experiment import *
from integrate import backward_cuda
sys.path.insert(0,str(R.parent/'anthropic_ln_equal_saves_20260919'))
import core_saved as C
records=[]
with torch.no_grad():
 for n in (384,768):
  d,dy,s=data(n);change_inputs(d,dy,.25,20260920+n)
  plans={k:Experiment(d,dy,s,132,2,k) for k in ['dual_balanced','dual_ln_prefetch','dual_pref_place13']}
  ref=baseline(d,dy,s);errors={k:check(p(),ref) for k,p in plans.items()}
  full_ref=C.backward(d,s,dy)
  for k,p in plans.items():
   err=[rel(x,y) for x,y in zip(backward_cuda(d,s,dy,p),full_ref)]
   assert max(err)<=5e-4;errors['full/'+k]=err
  domains={'core':dict(baseline=lambda:baseline(d,dy,s),**{k+'/part2':p for k,p in plans.items()}),
           'full':dict({'full/baseline':lambda:C.backward(d,s,dy)},**{'full/'+k+'/part2':lambda p=p:backward_cuda(d,s,dy,p) for k,p in plans.items()})}
  times={};blocks={}
  for domain,funcs in domains.items():
   graphs={k:capture(f) for k,f in funcs.items()}
   blocks[domain]=[paired_events(graphs) for _ in range(3)]
   for k in graphs:
    samples=sorted(t for b in blocks[domain] for t in b[k]['samples_us'])
    times[k]=dict(median_us=statistics.median(samples),p90_us=samples[int(.9*(len(samples)-1))],min_us=samples[0],max_us=samples[-1],samples_us=samples)
   print('RESULT',n,domain,[{k:round(v['median_us'],3) for k,v in b.items()} for b in blocks[domain]],flush=True)
  record=dict(L=n,dropout=.25,count=132,part=2,warmup_per_block=20,iterations_per_block=200,blocks_per_domain=3,errors=errors,times=times,blocks=blocks)
  records.append(record);(R/'sol-final-paired-results.json').write_text(json.dumps(records,indent=2))
