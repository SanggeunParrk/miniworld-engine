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
tag=f'D{a.width}-L{a.length}';dest=out/f'front-{tag}.json'
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

from front_config import build
leaves,dy,mask,ds,ref,triton,names=setup(a.width,a.length)
configs=([(1,64,2,2,2,2),(1,64,3,1,2,2),(1,64,4,1,2,2),(1,64,2,2,1,2),(2,64,2,2,1,2),(2,64,3,2,1,2),(2,64,4,1,1,2),(2,64,2,4,1,2)] if a.width==512 else [(1,64,2,3,2,2),(1,64,3,2,2,2),(1,64,4,1,2,2),(2,64,2,3,1,2),(2,64,3,2,1,2),(2,64,4,1,1,2)])
with torch.no_grad(),T.native_context(leaves[0].device):
 plan=Training(leaves,mask,ds,dy);original=plan.f.front
 y,grads=plan();expected=[v.clone() for v in (y,*grads)]
 saved=[v.clone() for v in (original.ab,original.xn,plan.pre)]
 gbase,_=capture(plan)
 for cfg in configs:
  key='-'.join(map(str,cfg))
  try:candidate=build(plan,cfg)
  except (RuntimeError,ValueError) as ex:
   r['tuning'][key]={'build_error':str(ex)[-1500:]};print('BUILD_REJECT',key,str(ex)[-150:],flush=True);save();continue
  plan.f.front=candidate;y,grads=plan();torch.cuda.synchronize()
  exact=[torch.equal(x,z) for x,z in zip((candidate.ab,candidate.xn,plan.pre),saved)]
  errs={n:error(u,v) for n,u,v in zip(names,(y,*grads),expected)}
  row={'saved_bitwise':exact,'errors':errs,'occupancy':candidate.occupancy,'smem':candidate.smem,'artifact':str(candidate.cubin),'artifact_sha256':hashlib.sha256(pathlib.Path(candidate.cubin).read_bytes()).hexdigest()}
  r['tuning'][key]=row;save()
  if all(exact) and all(v<(2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,v in errs.items()):
   g,_=capture(plan);times=paired({'baseline':gbase,'candidate':g});del g
   row.update(times=times,speedup=times['baseline']['median_us']/times['candidate']['median_us'])
   print('FRONT',key,row['speedup'],{k:v['median_us'] for k,v in times.items()},flush=True)
  else:print('NUMERIC_REJECT',key,exact,errs,flush=True)
  save();plan.f.front=original
 r['complete']=True;save()
