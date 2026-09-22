"""Verify hoisted gamma and prefetched row statistics refresh on every replay."""
from measure_experiment import *
rows=[]
with torch.no_grad():
 for n in (384,768):
  for part in (1,2):
   d,dy,s=data(n);change_inputs(d,dy,.25,20260920+n)
   p=Experiment(d,dy,s,132,part,'dual_ln_prefetch')
   g=capture(p)
   saved=s[0].saved_tensors
   gamma,mean,rs=saved[7],saved[13],saved[14]
   errors=[]
   for replay in range(2):
    gamma.mul_(.9).add_(.01)
    mean.add_(.0001)
    rs.mul_(1.001)
    change_inputs(d,dy,.25,20260921+n+replay)
    ref=baseline(d,dy,s)
    g.replay();torch.cuda.synchronize()
    errors.append(check(p.outputs,ref))
    assert torch.count_nonzero(p.workspace[-1]).item()==0
   rows.append(dict(L=n,part=part,updated=['gamma','mean','rstd','dy','ds'],errors=errors))
   print('CHECK saved updates',n,part,flush=True)
   (R/'sol-saved-updates.json').write_text(json.dumps(rows,indent=2))
