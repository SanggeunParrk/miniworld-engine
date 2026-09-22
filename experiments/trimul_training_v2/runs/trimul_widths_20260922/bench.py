from pathlib import Path
import argparse,torch,json,os,sys,statistics,gc,collections
import torch.nn.functional as F
R=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--width',type=int,required=True);p.add_argument('--length',type=int,required=True);a=p.parse_args();D,L=a.width,a.length;H=2*D
from miniworld_engine import settings
from miniworld_engine.kernels.trimul_inproj.triton import bidirectional as B
settings.configure(engine_backend='triton',trimul_sm90_kernels=(),autotune_miss_cap=24)
if D==512:
 import dual_fixed
 from miniworld_engine.kernels.trimul_inproj.triton import backward_fused as BF
 BF._input_dual_bwd_kernel=dual_fixed._input_dual_bwd_kernel

torch.manual_seed(20260922);dev='cuda';bf=torch.bfloat16
x=torch.randn(1,L,L,D,device=dev,dtype=bf,requires_grad=True)
weights=[(torch.randn(s,device=dev)/s[-1]**.5).to(bf).requires_grad_(True) for s in [(H,D)]*4+[(D,D),(D,H)]]
affine=[(torch.ones(c,device=dev)+.1*torch.randn(c,device=dev) if i%2==0 else .05*torch.randn(c,device=dev)).requires_grad_(True) for i,c in enumerate([D,D,H,H])]
leaves=(x,*weights,*affine);dy=torch.randn_like(x);mask=(torch.rand((1,L,L),device=dev)>.15).to(bf);ds=((torch.rand((1,1,L,D),device=dev)>.25).to(bf)*(4/3));names=['y','dx','dWL','dWLg','dWR','dWRg','dWgate','dWproj','dgamma_in','dbeta_in','dgamma_out','dbeta_out']
def ref(x,wl,wlg,wr,wrg,wg,wp,gi,bi,go,bo,mask,ds):
 xn=F.layer_norm(x.float(),(D,),gi,bi,1e-5).to(x.dtype)
 left=(torch.sigmoid(F.linear(xn,wlg))*F.linear(xn,wl))*mask[...,None];right=(torch.sigmoid(F.linear(xn,wrg))*F.linear(xn,wr))*mask[...,None]
 out=torch.einsum('bikd,bjkd->bijd',left[...,:D],right[...,:D]);inc=torch.einsum('bkid,bkjd->bijd',left[...,D:],right[...,D:]);tri=torch.cat((out,inc),-1)
 norm=F.layer_norm(tri.float(),(H,),go,bo,1e-5).to(x.dtype);update=torch.sigmoid(F.linear(xn,wg))*F.linear(norm,wp)
 return x+(update if ds is None else update*ds)
def triton(*args):return B.bidirectional_trimul_triton(*args[:-2],1e-5,1e-5,D,mask=args[-2],dropscale=args[-1])
import cuequivariance_ops_torch as cueq
from cuequivariance_ops_torch.fused_layer_norm_torch import layer_norm_transpose
from cuequivariance_ops_torch.gated_gemm_torch import fused_sigmoid_gated_dual_gemm
cueq.init_triton_cache()
def cueq_fn(x,wl,wlg,wr,wrg,wg,wp,gi,bi,go,bo,mask,ds):
 xn=layer_norm_transpose(x,gi,bi,eps=1e-5,layout='bijd->bijd');ab=fused_sigmoid_gated_dual_gemm(xn,torch.cat((wlg,wrg)),torch.cat((wl,wr)),mask=mask,transpose_out=True);left,right=ab.chunk(2,dim=0)
 out=torch.einsum('dbik,dbjk->dbij',left[:D],right[:D]);inc=torch.einsum('dbki,dbkj->dbij',left[D:],right[D:]);tri=torch.cat((out,inc),0)
 norm=layer_norm_transpose(tri,go,bo,eps=1e-5,layout='dbij->bijd');update=torch.sigmoid(F.linear(xn,wg))*F.linear(norm,wp)
 return x+(update if ds is None else update*ds)
opts=dict(fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
fns={n:torch.compile(fn,**opts) for n,fn in [('pytorch',ref),('triton',triton),('cueq',cueq_fn)]}
record=dict(D=D,L=L,hidden_per_direction=D,gpu=torch.cuda.get_device_name(),job=os.environ.get('SLURM_JOB_ID'),dropout=.25,shapes=[list(t.shape) for t in leaves],times={},checks={},failures={},traces={})
def save():(R/f'bench-D{D}-L{L}.json').write_text(json.dumps(record,indent=2))
def capture(fn):
 s=torch.cuda.Stream();s.wait_stream(torch.cuda.current_stream())
 with torch.cuda.stream(s):
  for _ in range(3):fn()
 torch.cuda.current_stream().wait_stream(s);g=torch.cuda.CUDAGraph()
 with torch.cuda.graph(g,stream=s):out=fn()
 return g,out

def clone(v):return tuple(t.clone() for t in v)
def check(out,reference,same=False):
 es={}
 for n,t,r in zip(names,out,reference):
  val=float((t.float()-r.float()).norm()/r.float().norm().clamp_min(1e-20));limit=(5e-6 if n.startswith(('dgamma','dbeta')) else 0) if same else (.005 if n=='y' else .01)
  es[n]=dict(relative_l2=val,limit=limit,finite=bool(t.isfinite().all()),exact=torch.equal(t,r));assert es[n]['finite'] and val<=limit,(n,es[n])
 return es
for scope in ('inference','training_forward','training'):
 gs={};outs={};calls={};record['checks'][scope]={}
 for name,fn in fns.items():
  def run(fn=fn):
   with torch.set_grad_enabled(scope!='inference'):
    y=fn(*leaves,mask,None if scope=='inference' else ds)
    return (y,*torch.autograd.grad(y,leaves,dy)) if scope=='training' else (y,)
  try:
   print('CAPTURE',D,L,scope,name,flush=True);run();g,out=capture(run);gs[name]=g;outs[name]=out;calls[name]=run
  except Exception as e:record['failures'][scope+'/'+name]=repr(e);print('FAIL',scope,name,repr(e),flush=True);save()
 reference=clone(calls['pytorch']())
 for name in list(gs):
  try:
   gs[name].replay();torch.cuda.synchronize();record['checks'][scope][name]=check(outs[name],reference)
   with torch.no_grad():x.neg_()
   try:
    eg=clone(calls[name]());gs[name].replay();torch.cuda.synchronize();record['checks'][scope][name]['mutation']=check(outs[name],eg,same=True)
   finally:
    with torch.no_grad():x.neg_()
   print('PASS',scope,name,flush=True)
  except Exception as e:record['failures'][scope+'/'+name]=repr(e);print('CHECK_FAIL',scope,name,repr(e),flush=True);gs.pop(name)
  save()
 events={n:[] for n in gs}
 for r in range(4):
  for g in gs.values():
   for _ in range(20):g.replay()
  for i in range(100):
   for n in list(gs) if (i+r)%2==0 else list(gs)[::-1]:
    s,e=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);s.record();gs[n].replay();e.record();events[n].append((s,e))
 torch.cuda.synchronize();record['times'][scope]={n:dict(median_us=statistics.median(s.elapsed_time(e)*1000 for s,e in ev),samples_us=[s.elapsed_time(e)*1000 for s,e in ev]) for n,ev in events.items()};print('TIME',D,L,scope,{n:v['median_us'] for n,v in record['times'][scope].items()},flush=True)
 if 'triton' in gs:
  with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:gs['triton'].replay();torch.cuda.synchronize()
  path=R/f'trace-D{D}-L{L}-{scope}.json';prof.export_chrome_trace(str(path));trace=json.loads(path.read_text());sums=collections.defaultdict(float)
  for e in trace['traceEvents']:
   if e.get('cat')=='kernel':sums[e['name']]+=e['dur']
  record['traces'][scope]=dict(sums)
 save();del gs,outs,calls,reference,events;gc.collect();torch.cuda.empty_cache()
record['complete']=True;save()
