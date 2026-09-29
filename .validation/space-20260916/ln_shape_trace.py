import dataclasses,json,os
from pathlib import Path
import torch
from miniworld_engine.autotune import derive,builder,cache,configs
import triton.runtime.autotuner as am
os.environ['MINIWORLD_COMPILE_WRAP']='disable'
derive.require_environment();derive.target_arch('sm86');derive.install_no_calibration();derive.install_module_apply();derive.install_native_recorders()
original=derive.install_recorder
captured=[]
def install(sink):
 original(sink)
 base=am.Autotuner.run
 def run(self,*args,**kwargs):
  op=configs.op_of(getattr(self,'configs',None) or [])
  if op=='layernorm_bwd_atomic_triton':
   a=dict(zip(derive._arg_names(self),args));a.update(kwargs)
   captured.append({'M':int(a['M']),'N':int(a['N']),'rowscale':bool(a['HAS_ROWSCALE']),'dtype':cache.dtype_of_args(a)})
  return base(self,*args,**kwargs)
 am.Autotuner.run=run
derive.install_recorder=install
rows=[dataclasses.replace(r,lengths=(r.lengths[0],),modes=('train',)) for r in derive.module_rows() if 'train' in r.modes]
units=[u for u in derive.units(rows,arch='sm86') if u.option in (None, ('ln_bwd_path', 'atomic'))];case_by_name={c.name:c for c in builder.cases()}
out=[];errors=[]
for i,u in enumerate(units):
 captured.clear();launches,error=derive.record(u,case_by_name)
 if captured:out.append({'unit':dataclasses.asdict(u),'label':u.label,'launches':list(captured),'error':error})
 if error:errors.append({'unit':u.label,'error':error})
 if i%25==0:print(i,len(units),'hits',len(out),'errors',len(errors),flush=True)
p=Path(__file__).with_name('ln-shape-trace.json');p.write_text(json.dumps({'arch':'sm86','scope':'first registered length per row, default and forced-atomic training; fake launches, no GPU timing','units':len(units),'hits':out,'errors':errors},indent=2))
print('DONE',len(units),len(out),len(errors),p,flush=True)
