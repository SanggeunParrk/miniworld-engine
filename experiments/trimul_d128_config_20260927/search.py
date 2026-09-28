"""Process-local D128 B1/B7 schedule search; production files are never changed."""
import argparse, copy, gc, hashlib, itertools, json, pathlib, statistics, sys, types
import torch
from miniworld_engine.kernels.trimul_inproj.cuda import h100_training as H, h100_b1 as B1, h100_b7 as B7
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
ROOT=pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/trimul_training_v2/runs/trimul_cuda_widths_opt_20260923'))
from fixture import setup
p=argparse.ArgumentParser();p.add_argument('--length',type=int,required=True);p.add_argument('--mode',choices=['search','validate'],default='search');p.add_argument('--selection');p.add_argument('--sanitize',action='store_true');p.add_argument('--tag',default='search');a=p.parse_args()
out=pathlib.Path('/workspace/vast-results/trimul-d128-config-20260927');out.mkdir(parents=True,exist_ok=True)
dest=out/f'{a.tag}-L{a.length}.json'
record=dict(length=a.length,complete=False,rows=[],source_sha256={},scope='D128 bidirectional public CUDA autograd full F+B, fixed live dropscale')
for f in pathlib.Path(B1.__file__).parent.rglob('*'):
 if f.is_file() and f.suffix in ('.py','.cu','.cuh','.inc','.json'):record['source_sha256'][str(f.relative_to(ROOT))]=hashlib.sha256(f.read_bytes()).hexdigest()
record['script_sha256']=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()
base1=copy.deepcopy(T.read_config('b1/configs.json')[str(a.length)])
base7=dict(consumers=10,rings=12,clusters=10,mode=52)
original1,original7=B1.Plan,B7.Plan
modules={}
artifacts=set()
def install(config):
 artifacts.clear()
 c1=config.get('b1',base1);c7=config.get('b7',base7)
 # Packaged consumer_compute has eight 16-KiB phases and two barriers.
 # Other chunk sizes compile but do not match its consumer protocol.
 if not c7['mode'] & 4 or c7['mode'] & 8:
  raise ValueError('Packaged B7 consumer requires the 16-KiB ring chunk')
 key=(c7['consumers'],c7['rings'],c7.get('producer_regs',32))
 if key not in modules:
  source=pathlib.Path(B7.__file__).read_text()
  for field,value,old in [('consumers',key[0],10),('rings',key[1],12)]:
   needle=f"cfg_values.{field} = int(('{old}'))";assert source.count(needle)==1
   source=source.replace(needle,f'cfg_values.{field} = {value}')
  needle="\"-DB7_PRODUCER_REGS=\" + ('32')";assert source.count(needle)==1
  source=source.replace(needle,'"-DB7_PRODUCER_REGS=" + '+repr(str(key[2])))
  mod=types.ModuleType('d128_candidate_b7_'+str(key));mod.__file__=B7.__file__
  exec(compile(source,B7.__file__,'exec'),mod.__dict__);modules[key]=mod
 def plan1(*args,**kwargs):
  obj=original1(*args,**copy.deepcopy(c1));artifacts.add(obj.k.unit.cubin_path);return obj
 def plan7(*args,**kwargs):
  obj=modules[key].Plan(*args,**kwargs,clusters=c7['clusters'],mode=c7['mode']);artifacts.add(obj.k.unit.cubin_path);return obj
 B1.Plan=plan1;B7.Plan=plan7

def save():dest.write_text(json.dumps(record,indent=2)+'\n')
def run():
 with torch.enable_grad():
  y=H.bidirectional_trimul(*leaves,mask,ds)
  return (y,*torch.autograd.grad(y,leaves,dy))
def capture():
 s=torch.cuda.Stream();s.wait_stream(torch.cuda.current_stream())
 with torch.cuda.stream(s):
  for _ in range(3):run()
 torch.cuda.current_stream().wait_stream(s)
 g=torch.cuda.CUDAGraph()
 with torch.cuda.graph(g,stream=s):vals=run()
 g.replay();torch.cuda.synchronize();return g,vals

def errors(actual,expected,strict=True):
 result={}
 for name,v,r in zip(names,actual,expected):
  err=float((v.float()-r.float()).norm()/r.float().norm().clamp_min(1e-20))
  limit=(2e-5 if name=='dx' else 5e-6 if name.startswith(('dgamma','dbeta')) else 5e-4) if strict else (.005 if name=='y' else .01)
  result[name]=dict(relative_l2=err,passed=bool(v.isfinite().all()) and err<limit)
 return result

def paired(bg,cg,count=30,rounds=3):
 values={'baseline':[],'candidate':[]}
 for _ in range(5):bg.replay();cg.replay()
 for rep in range(rounds):
  for i in range(count):
   pairs=[('baseline',bg),('candidate',cg)]
   for k,g in pairs if (rep+i)%2 else reversed(pairs):
    s,e=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    s.record();g.replay();e.record();e.synchronize();values[k].append(s.elapsed_time(e)*1000)
 return {k:dict(median_us=statistics.median(v),samples_us=v) for k,v in values.items()}

leaves,dy,mask,ds,reference,_,names=setup(128,a.length)
record.update(device=torch.cuda.get_device_name(),torch=torch.__version__,baseline=dict(b1=base1,b7=base7))
with T.native_context(leaves[0].device):
 install({});bg,bvals=capture();torch.cuda.synchronize()
 expected=[v.clone() for v in bvals]
 candidates=[dict(label='baseline',b1=base1,b7=base7)]
 if a.mode=='search':
  for consumers,rings in itertools.product((6,8,10,12),(4,8,12,16)):
   maximum=264//(16+consumers)
   for clusters in sorted(set((max(1,maximum-2),maximum))):
    c=dict(consumers=consumers,rings=rings,clusters=clusters,mode=52)
    if c!=base7:candidates.append(dict(label=f'b7-c{consumers}-r{rings}-g{clusters}',b7=c))
  for count in (66,96,120):
   c=copy.deepcopy(base1);c['count']=count;candidates.append(dict(label=f'b1-count{count}',b1=c))
  for key,values in [('B1_STREAM_AFFINE_UNROLL',(1,2,4,8)),('B1_PARAM_UNROLL',(1,2,4,8)),('B1_DTRI_STORE_C',(16,32,64,128)),('B1_TMA_PRIORITY',(0,1,2,3,6)),('B1_GATE_DEMOTE',(0,1,2,3))]:
   for value in values:
    if base1['defines'][key]==value:continue
    c=copy.deepcopy(base1);c['defines'][key]=value;candidates.append(dict(label=f'{key}-{value}',b1=c))
 else:
  candidates=json.loads(pathlib.Path(a.selection).read_text())
 if a.selection:candidates=json.loads(pathlib.Path(a.selection).read_text())
 record['candidate_count']=len(candidates);save()
 for config in candidates:
  print('START',config['label'],flush=True)
  row=dict(config=config);record['rows'].append(row);save()
  try:
   install(config);cg,cvals=capture()
  except (RuntimeError,ValueError,AssertionError) as e:
   row.update(status='build_or_capture_reject',error=str(e)[-2400:]);save();print('REJECT',config['label'],row['error'][-160:],flush=True);continue
  row['artifacts']={str(p):hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() for p in artifacts}
  row['strict']=errors(cvals,expected)
  if all(v['passed'] for v in row['strict'].values()):
   # Graphs use identical live inputs, but separate saved activations and outputs.
   snaps=[v.detach().clone() for v in (*leaves,dy,mask,ds)]
   with torch.no_grad():
    leaves[0].mul_(.97);leaves[1].add_(.003);leaves[5].mul_(1.03);leaves[9][0]=0
    dy.mul_(.93);mask.copy_(mask.roll(1,1));ds.copy_(ds.roll(1,2))
   bg.replay();cg.replay();torch.cuda.synchronize()
   row['changed_graph']=errors(cvals,bvals)
   with torch.no_grad():
    for t,s in zip((*leaves,dy,mask,ds),snaps):t.copy_(s)
   del snaps
   if all(v['passed'] for v in row['changed_graph'].values()):
    if a.mode=='validate':
     row['edge_cases']={}
     for case in ('zero_gamma','mask_zero','dropout_zero'):
      target=leaves[7] if case=='zero_gamma' else mask if case=='mask_zero' else ds
      snap=target.detach().clone()
      with torch.no_grad():target.zero_()
      bg.replay();cg.replay();torch.cuda.synchronize();row['edge_cases'][case]=errors(cvals,bvals)
      with torch.no_grad():target.copy_(snap)
     # Independent PyTorch mathematical reference, allowing its different BF16 rounding contract.
     with torch.enable_grad():
      ry=reference(*leaves,mask,ds);rg=torch.autograd.grad(ry,leaves,dy)
     cg.replay();torch.cuda.synchronize();row['independent_reference']=errors(cvals,(ry,*rg),strict=False)
     del ry,rg
    row['times']=paired(bg,cg,1 if a.sanitize else 30 if a.mode=='search' else 60,1 if a.sanitize else 3 if a.mode=='search' else 5)
    row['speedup']=row['times']['baseline']['median_us']/row['times']['candidate']['median_us'];row['status']='measured'
   else:row['status']='graph_reject'
  else:row['status']='numeric_reject'
  if a.mode=='validate' and not a.sanitize:
   with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:cg.replay();torch.cuda.synchronize()
   prof.export_chrome_trace(str(out/f"{a.tag}-L{a.length}-{config['label']}-trace.json"))
  print('RESULT',config['label'],row['status'],row.get('speedup'),flush=True);save()
  del cg,cvals;gc.collect()
 record['complete']=True;save()
