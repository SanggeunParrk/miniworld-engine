from width_plan import *
import argparse,statistics,gc,os,collections,time
from fixture import setup
from miniworld_engine import settings
settings.configure(engine_backend='triton',trimul_sm90_kernels=(),autotune_miss_cap=24)
a=argparse.ArgumentParser();a.add_argument('--width',type=int,required=True);a.add_argument('--length',type=int,required=True);a.add_argument('--sanitize',action='store_true');args=a.parse_args();D,N=args.width,args.length
if D==512:
 import dual_fixed
 from miniworld_engine.kernels.trimul_inproj.triton import backward_fused as BF
 BF._input_dual_bwd_kernel=dual_fixed._input_dual_bwd_kernel
leaves,dy,mask,ds,ref,triton,names=setup(D,N)
record=dict(D=D,L=N,job=os.environ.get('SLURM_JOB_ID'),checks={},times={},complete=False)
def save():(R/f'check-D{D}-L{N}.json').write_text(json.dumps(record,indent=2))
def clone(o):return o[0].clone(),tuple(x.clone() for x in o[1])
def check(o,r):
 out={}
 for n,t,v in zip(names,(o[0],*o[1]),(r[0],*r[1])):
  er=float((t.float()-v.float()).norm()/v.float().norm().clamp_min(1e-20));out[n]=dict(relative_l2=er,finite=bool(t.isfinite().all()),valid=bool(t.isfinite().all()) and er<(.005 if n=='y' else .01))
 return out
opts=dict(fullgraph=True,dynamic=False,options={'triton.cudagraphs':False});ref=torch.compile(ref,**opts);triton=torch.compile(triton,**opts)
def base(fn):
 with torch.enable_grad():
  y=fn(*leaves,mask,ds);return y,torch.autograd.grad(y,leaves,dy)
S=torch.cuda.Stream();S.wait_stream(torch.cuda.current_stream())
def capture(fn):
 s=S;s.wait_stream(torch.cuda.current_stream())
 with torch.cuda.stream(s):
  for _ in range(3):fn()
 torch.cuda.current_stream().wait_stream(s);g=torch.cuda.CUDAGraph()
 with torch.cuda.graph(g,stream=s):o=fn()
 g.replay();torch.cuda.synchronize();return g,o
with torch.cuda.stream(S),torch.no_grad():
 print('BUILD',D,N,flush=True);m=Training(*leaves,mask,ds,dy);print('RUN',D,N,flush=True)
 eg=clone(m());torch.cuda.synchronize();print('CUDA DONE',flush=True)
 if args.sanitize:
  m();torch.cuda.synchronize();print('SANITIZE_DONE',flush=True);sys.exit()
 for key,fn in [('pytorch',ref),('triton',triton)]:
  r=clone(base(fn));record['checks'][key]=check(eg,r);print('CHECK',key,{k:v['relative_l2'] for k,v in record['checks'][key].items()},flush=True);save()
 assert all(v['valid'] for c in record['checks'].values() for v in c.values()),record['checks']
 gs={};outs={};gs['cuda'],outs['cuda']=capture(m);gs['triton'],outs['triton']=capture(lambda:base(triton))
 # Live x and weights: graph uses the same buffers; params are not frozen constants.
 originals=[z.clone() for z in leaves]
 leaves[0].mul_(.97);leaves[1].add_(.003);leaves[5].mul_(1.03);leaves[9][0]=0
 eg=clone(m());gs['cuda'].replay();torch.cuda.synchronize();record['checks']['graph_mutation']=check(outs['cuda'],eg);record['checks']['mutated_reference']=check(eg,base(ref));save()
 assert all(v['valid'] for c in record['checks'].values() for v in c.values()),record['checks']
 for t,v in zip(leaves,originals):t.copy_(v)
 for scope in ('training','forward'):
  if scope=='forward':gs={'cuda':capture(m.forward)[0],'triton':capture(lambda:triton(*leaves,mask,ds))[0]}
  ev={k:[] for k in gs}
  for r in range(3):
   for g in gs.values():
    for _ in range(5):g.replay()
   for i in range(30):
    for k in list(gs) if (i+r)%2 else list(gs)[::-1]:
     a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record();gs[k].replay();b.record();ev[k].append((a,b))
  torch.cuda.synchronize();record['times'][scope]={k:dict(median_us=statistics.median(a.elapsed_time(b)*1000 for a,b in ee),samples_us=[a.elapsed_time(b)*1000 for a,b in ee]) for k,ee in ev.items()};print('TIME',scope,{k:v['median_us'] for k,v in record['times'][scope].items()},flush=True);save()
  if scope=='training':
   with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:gs['cuda'].replay();torch.cuda.synchronize()
   p=R/f'trace-D{D}-L{N}.json';prof.export_chrome_trace(str(p));tr=json.loads(p.read_text());record['trace']=dict(collections.Counter(e['name'] for e in tr['traceEvents'] if e.get('cat')=='kernel'));assert record['trace'].get('width_b1')==record['trace'].get('width_b7')==1
 record['complete']=True;record['cubin']=m.path;record['cubin_sha256']=hashlib.sha256(Path(m.path).read_bytes()).hexdigest();save();print('DONE',D,N,flush=True)
