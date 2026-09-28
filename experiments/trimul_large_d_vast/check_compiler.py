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
tag=f'D{a.width}-L{a.length}';dest=out/f'compiler-{tag}.json'
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

from compiler_candidate import build
leaves,dy,mask,ds,ref,triton,names=setup(a.width,a.length)
with torch.no_grad(),T.native_context(leaves[0].device):
 plan=Training(leaves,mask,ds,dy);original_front=plan.f.front;original_contract=plan.contract_gp
 y,grads=plan();expected=[v.clone() for v in (y,*grads)]
 saved=[v.clone() for v in (original_front.ab,original_front.xn,plan.pre)]
 gbase,_=capture(plan)
 variants=['front','contract','both'] if a.width==512 and a.length==384 else ['front']
 built={}
 for variant in variants:
  try:
   for component in (['front','contract'] if variant=='both' else [variant]):
    if component not in built:built[component],r['candidate_compiler']=build(plan,component)
  except (RuntimeError,AssertionError) as ex:
   r['tuning'][variant]={'build_error':str(ex)[-1500:]};print('BUILD_REJECT',variant,str(ex)[-200:],flush=True);save();continue
  plan.f.front=built['front'] if variant in ('front','both') else original_front
  plan.contract_gp=built['contract'] if variant in ('contract','both') else original_contract
  y,grads=plan();torch.cuda.synchronize()
  exact=[torch.equal(x,z) for x,z in zip((plan.f.front.ab,plan.f.front.xn,plan.pre),saved)]
  errs={n:error(u,v) for n,u,v in zip(names,(y,*grads),expected)}
  row={'saved_bitwise':exact,'errors':errs,'artifacts':{k:{'path':str(v.cubin),'sha256':hashlib.sha256(pathlib.Path(v.cubin).read_bytes()).hexdigest()} for k,v in built.items()}}
  r['tuning'][variant]=row;save()
  if all(exact) and all(v<(2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta')) else 5e-4) for n,v in errs.items()):
   g,_=capture(plan);times=paired({'baseline':gbase,'candidate':g});del g
   row.update(times=times,speedup=times['baseline']['median_us']/times['candidate']['median_us'])
   print('COMPILER',variant,row['speedup'],{k:v['median_us'] for k,v in times.items()},flush=True)
  else:print('NUMERIC_REJECT',variant,exact,errs,flush=True)
  save()
 r['complete']=True;save()
