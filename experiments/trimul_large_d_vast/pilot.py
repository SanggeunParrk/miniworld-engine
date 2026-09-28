"""Frozen checkpoint reproduction and exact-output cuBLASLt tuning on Vast."""
import argparse, hashlib, json, os, pathlib, statistics, sys, time
p=argparse.ArgumentParser();p.add_argument('--width',type=int,required=True);p.add_argument('--length',type=int,required=True);p.add_argument('--tune',action='store_true');a=p.parse_args()
root=pathlib.Path('/workspace/experiments/trimul-large-d/runs')
pre=root/'trimul_d256_bwd_sol90_20260923';stage=root/'trimul_d256_bwd_sol90_stage2_20260923'
sys.path.insert(0,str(pre))
ns={'__file__':str(pre/'check.py')};exec(compile((pre/'check.py').read_text().split('ap=argparse.ArgumentParser()')[0],str(pre/'check.py'),'exec'),ns)
sys.path.insert(0,str(stage));setup,T,error=ns['setup'],ns['T'],ns['error']
import torch
from validate_engine import capture
os.environ.update(PREFIX_IMPL='blas',PREFIX_COPY='tma',CHECKPOINT_LN_THREADS='0')
if a.width==256:
 from d256_pool_checkpoint import Training,configuration,configure
 configure()
elif a.width==384:
 from wide_checkpoint23 import Training,configuration
else:
 from wide_checkpoint24 import Training,configuration
out=pathlib.Path('/workspace/vast-results/trimul-large-d');out.mkdir(exist_ok=True)
tag=f'D{a.width}-L{a.length}';dest=out/f'pilot-{tag}.json'
r={'width':a.width,'length':a.length,'complete':False,'tune':a.tune,'samples':{},'tuning':{},'script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}
def save():dest.write_text(json.dumps(r,indent=2))
def measure(g,count=25):
 for _ in range(4):g.replay()
 vals=[]
 for _ in range(count):
  s,e=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);s.record();g.replay();e.record();e.synchronize();vals.append(s.elapsed_time(e)*1000)
 return statistics.median(vals)
def paired(graphs):
 vals={k:[] for k in graphs}
 for rep in range(3):
  for i in range(25):
   for k in (list(graphs) if (i+rep)%2 else list(graphs)[::-1]):
    s,e=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);s.record();graphs[k].replay();e.record();e.synchronize();vals[k].append(s.elapsed_time(e)*1000)
 return {k:{'median_us':statistics.median(v),'samples_us':v} for k,v in vals.items()}
leaves,dy,mask,ds,ref,triton,names=setup(a.width,a.length)
with torch.no_grad(),T.native_context(leaves[0].device):
 plan=Training(leaves,mask,ds,dy);y,grads=plan();torch.cuda.synchronize()
 expected=[v.clone() for v in (y,*grads)]
 r['checkpoint_configuration']=configuration(plan)
 r['artifacts']=[{'path':str(x),'sha256':hashlib.sha256(pathlib.Path(x).read_bytes()).hexdigest()} for x in plan.artifacts]
 r['loaded_cublas']=sorted({line.split()[-1] for line in open('/proc/self/maps') if 'libcublas' in line})
 gbase,_=capture(plan);r['baseline_full_us']=measure(gbase,75)
 gbwd,_=capture(plan.backward);r['baseline_backward_us']=measure(gbwd,75);del gbwd
 print('BASELINE',tag,r['baseline_full_us'],r['baseline_backward_us'],flush=True);save()
 with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:plan()
 prof.export_chrome_trace(str(out/f'trace-{tag}.json'))
 events=[{'name':e.name,'us':e.device_time_total} for e in prof.events() if e.device_type==torch.autograd.DeviceType.CUDA]
 r['cuda_events']=events;save()
 if a.tune:
  ops=dict(plan.schedule.kernels)
  if hasattr(plan,'dx_lt'):ops['prefix_dx']=plan.dx_lt
  if hasattr(plan,'late_output_dw'):ops['late_dwp']=plan.late_output_dw.dwp
  r['selected']={}
  for name,op in ops.items():
   # Frozen selections, not unselected heuristic placeholders, form the baseline.
   if name in plan.schedule.kernels and name not in plan.schedule.selected:continue
   original=op.index
   op();torch.cuda.synchronize();expected_op=op.out.clone()
   records=[]
   for idx in op.indices:
    op.index=idx
    try:
     op();torch.cuda.synchronize()
     exact=torch.equal(op.out,expected_op)
     row={'index':idx,'exact':exact,'algo':list(op.heuristics[idx].algo.data)}
     if exact:
      g,_=capture(op);row['us']=measure(g,15);del g
     records.append(row)
    except RuntimeError as ex:
     records.append({'index':idx,'error':str(ex)[:200]})
   valid=[x for x in records if x.get('exact')]
   old=next(x for x in valid if x['index']==original)
   best=min(valid,key=lambda x:x['us'])
   # Avoid changing selection based on marginal timing noise.
   selected=best if best['us']<old['us']*.97 else old
   op.index=selected['index'];r['selected'][name]=selected
   if name in plan.schedule.kernels and selected['index']!=original:
    plan.schedule.selected[name]=selected
   r['tuning'][name]={'baseline':old,'selected':selected,'trials':records}
   print('TUNE',name,original,selected['index'],old['us'],selected['us'],flush=True);save()
  yc,gc=plan();torch.cuda.synchronize()
  errs={n:error(u,v) for n,u,v in zip(names,(yc,*gc),expected)}
  r['candidate_errors']=errs
  assert all(v<(2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,v in errs.items()),errs
  gcand,_=capture(plan);r['samples']=paired({'baseline':gbase,'candidate':gcand})
  r['speedup']=r['samples']['baseline']['median_us']/r['samples']['candidate']['median_us']
  print('PAIRED',tag,r['speedup'],{k:v['median_us'] for k,v in r['samples'].items()},flush=True)
 r['complete']=True;save()
