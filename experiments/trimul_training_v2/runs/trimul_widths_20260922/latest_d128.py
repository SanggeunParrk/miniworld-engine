from pathlib import Path
import importlib.util,sys,os,json,torch,collections,hashlib,gc,argparse
R=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--length',type=int,required=True);p.add_argument('--split-b7',action='store_true');args=p.parse_args();N=args.length
s=importlib.util.spec_from_file_location('width_latest_fixed',R.parent/'trimul_ln_gradient_20260922/policy.py');F=importlib.util.module_from_spec(s);sys.modules[s.name]=F;s.loader.exec_module(F)
Q=F.Q
from miniworld_engine import settings
from miniworld_engine.kernels.trimul_inproj.triton import bidirectional as B
settings.configure(engine_backend='triton',trimul_sm90_kernels=(),autotune_miss_cap=24)
@torch.compile(fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
def baseline(*args):return B.bidirectional_trimul_triton(*args[:-2],1e-5,1e-5,128,mask=args[-2],dropscale=args[-1])
record=dict(D=128,L=N,job=os.environ.get('SLURM_JOB_ID'),dropout=.25,route='Latest cache-policy B1 + previous split CUDA B7' if args.split_b7 else 'Latest cache-policy B1 + corrected single B7',checks={},times={},traces={})
def save():(R/f'latest-D128-L{N}.json').write_text(json.dumps(record,indent=2))
def clone(o):return o[0].clone(),tuple(t.clone() for t in o[1])
def check(o,r,strict=False,same=False):
 es=F.H.errors(o,r)
 for k,v in es.items():
  lim=(0 if same else 0 if k=='forward' else 2e-5 if k=='dx' else 5e-6 if k.startswith(('dgamma','dbeta')) else 5e-4) if strict or same else (.005 if k=='forward' else .01)
  v['limit']=lim;v['valid']=v['finite'] and v['relative_l2']<=lim
 return dict(valid=all(v['valid'] for v in es.values()),errors=es)
with torch.no_grad():
 a=Q.setup(N);d=a['d'];m=F.P.Regression(a) if args.split_b7 else F.Fixed(a);reg=F.P.Regression(a)
 stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
 with torch.cuda.stream(stream):leaves=tuple(t.detach().clone().requires_grad_(True) for t in d['leaves']);mask=d['mask'].reshape(1,N,N).bfloat16();ds=d['ds'].reshape(1,1,N,128)
 torch.cuda.current_stream().wait_stream(stream)
 def bf():
  with torch.enable_grad():return baseline(*leaves,mask,ds)
 def bt():
  with torch.enable_grad():
   y=bf();return y,torch.autograd.grad(y,leaves,a['dy'])
 gs={};outs={}
 for name,fn in [('triton',bt),('cuda',m)]:
  gs[name],outs[name]=Q.capture_outputs(fn,stream=stream if name=='triton' else None);gs[name].replay();torch.cuda.synchronize()
 originals=[t.clone() for t in d['leaves']]
 for case in range(2):
  if case:
   d['x'].mul_(.97);d['leaves'][1].add_(.003);d['leaves'][5].mul_(1.03);d['go'][0]=0;d['bo'].add_(.017)
   for t,v in zip(leaves,d['leaves']):t.copy_(v)
  ref=clone(reg());eg=clone(m());old=clone(bt());gs['cuda'].replay();torch.cuda.synchronize()
  checks=dict(strict=check(eg,ref,strict=True),triton=check(eg,old),graph=check(outs['cuda'],eg,same=True));record['checks'][str(case)]=checks;save();print('CHECK',N,case,{k:v['valid'] for k,v in checks.items()},flush=True)
  assert all(v['valid'] for v in checks.values()),checks
 for t,v in zip(d['leaves'],originals):t.copy_(v)
 for t,v in zip(leaves,originals):t.copy_(v)
 for scope in ('training','training_forward'):
  if scope=='training_forward':gs={k:Q.capture_outputs(fn)[0] for k,fn in [('triton',bf),('cuda',m.forward)]}
  blocks=[Q.paired(gs,warmup=60,iterations=200) for _ in range(4)];record['times'][scope]=Q.pool(blocks);print('TIME',N,scope,{k:v['median_us'] for k,v in record['times'][scope].items()},flush=True);save()
  for k,g in gs.items():
   with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:g.replay();torch.cuda.synchronize()
   path=R/f'trace-latest-D128-L{N}-{scope}-{k}.json';prof.export_chrome_trace(str(path));events=json.loads(path.read_text())['traceEvents'];names=collections.Counter(v['name'] for v in events if v.get('cat')=='kernel');record['traces'][scope+'/'+k]=dict(names)
 if not args.split_b7:assert record['traces']['training/cuda'].get('b7_joint')==1
 assert record['traces']['training/cuda'].get('b1_fused')==1
 record['cubins']={name:[dict(path=k.unit.cubin_path,sha256=hashlib.sha256(Path(k.unit.cubin_path).read_bytes()).hexdigest()) for k in ([p.k] if hasattr(p,'k') else [v[0] for v in p.units])] for name,p in [('b1',m.p1),('b7',m.p7)]}
 record['complete']=True;save();print('DONE',N,flush=True)
