from measure_experiment import *
names=['load_and_B1','dW','LN_saved_stats','dnorm_and_row_reduction',
       'LN_epilogue_and_prefetch','dtri_TMA_and_join','partial_dump','grid_wait','reduce','reset']
results=[]
with torch.no_grad():
 for n in (384,768):
  d,dy,s=data(n);change_inputs(d,dy,.25,20260920+n)
  original=Experiment(d,dy,s,132,2,'dual_vec2');probe=Experiment(d,dy,s,132,2,'dual_detail')
  ref=baseline(d,dy,s);errors=check(probe(),ref)
  gs={k:capture(p) for k,p in {'original':original,'probe':probe,'baseline':lambda:baseline(d,dy,s)}.items()}
  times=paired_events(gs)
  records=[]
  for _ in range(20):
   probe();torch.cuda.synchronize();t=probe.timestamps.cpu().tolist()
   row={name:statistics.median(v[i+1]/1000 for v in t) for i,name in enumerate(names)}
   records.append(row)
  phases={k:statistics.median(r[k] for r in records) for k in names}
  result=dict(L=n,errors=errors,times=times,median_cta_phase_sum_us=phases,raw=records)
  print('CHECK',n,errors,flush=True)
  print('RESULT',n,{k:v['median_us'] for k,v in times.items()},phases,flush=True)
  results.append(result);(R/'detail-results.json').write_text(json.dumps(results,indent=2))
