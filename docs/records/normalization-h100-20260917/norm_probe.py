import importlib,json,os,statistics
from pathlib import Path
import torch,triton
from miniworld_engine import settings
from miniworld_engine.autotune.run_all import check_one
from benchmarks.runners import bench
import baseline_norms as baseline
root=Path(__file__).parent
settings.configure(engine_backend='triton',transition_lnbwd_cuda=False)
specs=[
 ('transition.triton.fused','_transition_ln_bwd_kernel','transition:layernorm_bwd_foldstats_triton',('dx_ptr','dg_ptr','db_ptr'),('dg_ptr','db_ptr')),
 ('adaln.triton.training','_dgrad_condln_kernel','adaln:adaln_bwd_dx_dlnw',('DCond','DLNW'),('DLNW',)),
 ('layernorm_linear.triton.pair_bias','_layer_norm_linear_bwd','layernorm_linear:layernorm_linear_bwd_fp32_triton',('dx_ptr','dlnw_ptr','dpw_ptr'),('dlnw_ptr','dpw_ptr')),
 ('rmsnorm.triton.main','rmsnorm_bwd_kernel','rmsnorm:rmsnorm_bwd_triton',('DX','DW'),('DW',)),
 ('rmsnorm.triton.main','rmsnorm_adamod_bwd_kernel','rmsnorm_adamod:rmsnorm_adamod_bwd_triton',('DQ','DSD','DW'),('DW',)),
]
# Read the registry's exact checker names and tolerances rather than guessing aliases.
import csv
registry=list(csv.DictReader((Path(importlib.import_module('miniworld_engine').__file__).parent/'kernels/registry.csv').open()))
from miniworld_engine.autotune.run_all import declared_rtol
from miniworld_engine.kernels.drivers import DTYPE_MODE
report={'dtype':DTYPE_MODE,'L':os.environ.get('MINIWORLD_DRIVER_LENGTH'),'shape_mode':os.environ.get('MINIWORLD_SHAPE_MODE','aligned'),'kernels':[]}
for mod,name,checker,outs,reset in specs:
 module=importlib.import_module('miniworld_engine.kernels.'+mod);kernel=getattr(module,name);original=kernel.run;seen={}
 def observe(*args,**kw):
  result=original(*args,**kw);amap=dict(zip(kernel.arg_names,args));amap.update(kw)
  key=tuple((k,str(amap[k])) for k in kernel.keys)
  seen[key]=(args,kw.copy(),kernel.best_config)
  return result
 kernel.run=observe
 row=next(r for r in registry if r['fn']==name) if 'fn' in registry[0] else next(r for r in registry if name in r.values())
 if DTYPE_MODE not in row['dtypes'].split('|'):
  kernel.run=original;print('SKIP',name,DTYPE_MODE,flush=True);continue
 ok,detail=check_one(row['check'],declared_rtol(row,DTYPE_MODE));print('CHECK',name,ok,detail,flush=True)
 assert ok,(name,detail)
 kernel.run=original
 for args,kw,config in seen.values():
  grid=kw.pop('grid');kw.pop('warmup',None)
  for key in config.all_kwargs():kw.pop(key,None)
  amap=dict(zip(kernel.arg_names,args));amap.update(kw)
  outnames=[k for k in outs if k!='DW' or amap.get('HAS_WEIGHT',True)]
  outputs=[amap[k] for k in outnames]
  def launch(fn):
   for n in reset:
    if n=='DW' and not amap.get('HAS_WEIGHT',True):continue
    amap[n].zero_()
   return fn[grid](*args,**kw,**config.all_kwargs())
  launch(getattr(baseline,name));reference=[v.clone() for v in outputs]
  launch(kernel.fn);torch.cuda.synchronize()
  errs=[((v.float()-r.float()).norm()/r.float().norm().clamp_min(1e-8)).item() for v,r in zip(outputs,reference)]
  assert max(errs)<3e-4,(name,errs)
  graphs={}
  for tag,fn in [('old',getattr(baseline,name)),('relaxed',kernel.fn)]:
   binary=launch(fn);g=torch.cuda.CUDAGraph()
   with torch.cuda.graph(g):launch(fn)
   graphs[tag]=g
  samples={tag:[] for tag in graphs}
  for rnd in range(7):
   for tag in list(graphs)[::1 if rnd%2==0 else -1]:samples[tag].append(float(bench.bench_time(graphs[tag].replay,warmup=5,rep=35)['median_ms']))
  med={tag:statistics.median(v) for tag,v in samples.items()}
  report['kernels'].append({'name':name,'config':config.all_kwargs(),'flags':{k:amap[k] for k in kernel.keys},'ms':med,'speedup':med['old']/med['relaxed'],'samples_ms':samples,'old_new_errors':errs,'reference_check':detail})
  print('RESULT',json.dumps(report['kernels'][-1]),flush=True)
  # Capture physical workload metadata for a later targeted build, without claiming the whole grid.
  from miniworld_engine.autotune import cache
  report['kernels'][-1]['measurement']=cache.measurement_workload(row['kernel'],kernel,amap)
  del graphs,reference
 report['passed']=True
 (root/f"norm-{DTYPE_MODE}-{report['shape_mode']}-L{report['L']}.json").write_text(json.dumps(report,indent=2)+'\n')
