"""Diagnose the extra L72 seed that narrowly exceeded the large-shape dtri bound."""
from check_experiment import *
results=[]
with torch.no_grad():
 d,dy,s=data(72)
 plans={name:Experiment(d,dy,s,132,1,name) for name in ('dual','dual_maskbits')}
 for seed in (20261150,20261151,20261152):
  change_inputs(d,dy,.25,seed);ref=baseline(d,dy,s)
  outputs={k:p() for k,p in plans.items()};torch.cuda.synchronize()
  row=dict(seed=seed,relative_l2={k:{name:rel(a,b) for name,a,b in zip(NAMES,o,ref)} for k,o in outputs.items()},
           original_new_dtri_bit_exact=torch.equal(outputs['dual'][2].view(torch.int16),outputs['dual_maskbits'][2].view(torch.int16)))
  results.append(row);print('CHECK',json.dumps(row),flush=True)
 (R/'extra72-results.json').write_text(json.dumps(results,indent=2))
