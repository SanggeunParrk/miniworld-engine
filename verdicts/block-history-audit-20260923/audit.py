from pathlib import Path
import torch, importlib.util, sys, os, json, statistics, threading, subprocess, hashlib, collections, time, ast, types
R=Path(__file__).resolve().parent;ROOT=R.parents[2];RUN=ROOT/'runs'
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
stream=torch.cuda.Stream();torch.cuda.set_stream(stream)
# Legacy research-only launch support was not shipped in v2. Extend only this
# CUDA package's search path; public modules and new H100 code remain current.
import miniworld_engine.kernels.trimul_inproj.cuda as cuda_package
cuda_package.__path__.append(str(RUN/'trimul_sm90_parity_20260917/engine/src/miniworld_engine/kernels/trimul_inproj/cuda'))
import miniworld_engine.integrations as integrations_package
integrations_package.__path__.append(str(RUN/'trimul_sm90_parity_20260917/engine/src/miniworld_engine/integrations'))
P=load('history_trimul',RUN/'trimul_full_latest_20260922/policy.py')
F=load('corrected_trimul',RUN/'trimul_ln_gradient_20260922/policy.py')
Q=P.Q
from miniworld_engine import settings
from miniworld_engine.modules import Transition
from miniworld_engine.modules.triangle_multiplication.bidirectional import BidirectionalTriangleMultiplication
from miniworld_engine.modules.exceptions import ImplementationType as I
settings.configure(engine_backend='auto')
import miniworld_engine.kernels.transition.cuda
old_native=load('miniworld_engine.kernels.transition.cuda.history_sm90a',R/'transition_snapshot/fused_sm90a.py')
torch.manual_seed(20260922)
trans=Transition(128,n=4,implementation=I.MINIWORLD).cuda().bfloat16()
with torch.no_grad():
 torch.manual_seed(20260922);trans.squeeze.weight.normal_(0,.02)
trans.ln_in.float();tp=tuple(trans.parameters())
with torch.no_grad():
 a=Q.setup(384);d=a['d'];old=P.Latest(a);fixed=F.Fixed(a);dy=a['dy'].clone()
 # Restore old fixture values after constructor warmup; none of these constructors updates weights.
 ds=d['ds'].reshape(1,1,384,128)
 # Both public and old research use exactly the same pair mask and numerical weights.
 mask=d['mask'].reshape(1,384,384).bool()
 native=BidirectionalTriangleMultiplication(128,implementation=I.MINIWORLD,p_drop=.25).cuda().bfloat16()
 front=(native.to_left.weight,native.to_left_gate.weight,native.to_right.weight,native.to_right_gate.weight,native.to_gate.weight,native.to_out.weight,native.ln_pair.weight,native.ln_pair.bias,native.ln_out.weight,native.ln_out.bias)
 for p,w in zip(front,d['leaves'][1:]):p.copy_(w)
x=d['x'];x.requires_grad_(True)
old_leaves=tuple(d['leaves']);new_leaves=(x,*front)
for t in old_leaves:t.requires_grad_(True)
# The historical fixture lazily imports validators that select Triton globally.
# Restore the production policy after fixture/plan construction.
settings.configure(engine_backend='auto',transition_fused_sm90a=True,transition_residual_fusion=True)
from miniworld_engine.kernels.transition.cuda import fused_sm90a as current_transition
assert current_transition.available(x,trans.expand_a.weight,trans.squeeze.weight)
opts=dict(fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
def hist_trans(x):return old_native.transition_fused_sm90a(x,trans.ln_in.weight,trans.ln_in.bias,trans.expand_a.weight,trans.expand_b.weight,trans.squeeze.weight,trans.ln_in.eps)
ot=torch.compile(hist_trans,**opts);nt=torch.compile(trans,**opts)
def research(model,tf):
 def step():
  with torch.no_grad():y,k=model.forward()
  with torch.enable_grad():
   z=y.detach().requires_grad_(True);out=tf(z);g=torch.autograd.grad(out,(z,*tp),dy)
  with torch.no_grad():a['dy']=g[0];gr=model.backward(k)
  return out,(*gr,*g[1:])
 return step
# A module wrapper preserves the public forward dispatch; direct update avoids the
# public annotation restricting a user mask to [B,L], since the original fixture
# contains a full [B,L,L] pair mask. This is the same native branch of forward.
from miniworld_engine.integrations.trimul_h100 import update
class Block(torch.nn.Module):
 def __init__(self,rng):super().__init__();self.tm=native;self.tr=trans;self.rng=rng
 def forward(self,x,mask):
  scale=self.tm._make_drop_row_scale(x,.25) if self.rng else ds
  return self.tr(update(self.tm,x,mask,scale))
def module_step(rng):
 fn=torch.compile(Block(rng),**opts)
 def run():
  out=fn(x,mask);return out,torch.autograd.grad(out,(*new_leaves,*tp),dy)
 return run
fns={'historical':research(old,ot),'historical_new_transition':research(old,nt),'corrected_research':research(fixed,nt),'current_fixed':module_step(False),'current_rng':module_step(True)}
record=dict(policy=vars(settings.current()),job=os.getenv('SLURM_JOB_ID'),gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),torch=torch.__version__,checks={},times={},samples={},telemetry=[],sources={},cubins={})
for name,obj in [('historical',old),('corrected',fixed)]:
 record['cubins'][name]={}
 for stage in ('p1','p7'):
  p=Path(getattr(obj,stage).k.unit.cubin_path)
  record['cubins'][name][stage]=dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
for rel in ('fused_sm90a.py','transition_fused_fwd_sm90a_kernel.cu','transition_fused_bwd_sm90a_kernel.cu'):
 orig=RUN/'minipairformer_block_baselines_20260922/transition_snapshot'/rel
 now=R.parents[1]/'src/miniworld_engine/kernels/transition/cuda'/rel
 record['sources'][rel]=dict(old=hashlib.sha256(orig.read_bytes()).hexdigest(),current=hashlib.sha256(now.read_bytes()).hexdigest())
def save():(R/'results.json').write_text(json.dumps(record,indent=2,default=lambda v:sorted(v) if isinstance(v,(set,frozenset)) else str(v))+'\n')
def capture(fn):
 for _ in range(4):fn()
 g=torch.cuda.CUDAGraph()
 with torch.cuda.graph(g,stream=stream):out=fn()
 return g,out
graphs={};outputs={}
for name,fn in fns.items():
 print('CAPTURE',name,flush=True);graphs[name],outputs[name]=capture(fn)
for g in graphs.values():g.replay()
torch.cuda.synchronize()
ref=outputs['corrected_research']
for name,out in outputs.items():
 if name=='current_rng':continue
 es=[float((p.float()-q.float()).norm()/q.float().norm().clamp_min(1e-20)) for p,q in zip((out[0],*out[1]),(ref[0],*ref[1]))]
 record['checks'][name]=es;print('ERRORS',name,es,flush=True);assert max(es)<.005,es
save()
phase=['idle'];stop=threading.Event();uid=record['gpu_uuid'];uid=uid if uid.startswith('GPU-') else 'GPU-'+uid
def watch():
 while not stop.is_set():
  p=subprocess.run(['nvidia-smi','-i',uid,'--query-gpu=clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True)
  record['telemetry'].append(dict(phase=phase[0],csv=p.stdout.strip()));stop.wait(.1)
th=threading.Thread(target=watch);th.start()
def measure(label,gs,rounds=12,iters=120):
 samples={n:[] for n in gs};names=list(gs)
 for r in range(rounds):
  order=names[r%len(names):]+names[:r%len(names)]
  if r%2:order=order[::-1]
  for name in order:
   phase[0]=label+'/'+name;g=gs[name]
   for _ in range(60):g.replay()
   a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record()
   for _ in range(iters):g.replay()
   b.record();b.synchronize();samples[name].append(a.elapsed_time(b)*1000/iters)
 record['samples'][label]=samples;record['times'][label]={n:statistics.median(v) for n,v in samples.items()};save();print('TIME',label,record['times'][label],flush=True)
try:
 measure('whole_block',graphs)
 # Recreate the original per-replay event/interleaving measurement, including
 # its memory-heavy PyTorch and cuEq peers, on the SAME GPU as current native.
 import torch.nn.functional as TF
 from cuequivariance_ops_torch.fused_layer_norm_torch import layer_norm_transpose
 from cuequivariance_ops_torch.gated_gemm_torch import fused_sigmoid_gated_dual_gemm
 import cuequivariance_ops_torch as cueq
 cueq.init_triton_cache()
 source=ast.parse((RUN/'minipairformer_block_baselines_20260922/bench.py').read_text())
 chosen={'torch_trimul','torch_transition','torch_block','cueq_forward','cueq_torch_block','cueq_cuda_block'}
 body=[n for n in source.body if isinstance(n,ast.FunctionDef) and n.name in chosen]
 refmodule=types.ModuleType('history_reference');sys.modules['history_reference']=refmodule
 ns=refmodule.__dict__;ns.update(torch=torch,F=TF,transition=trans,native_transition=hist_trans,layer_norm_transpose=layer_norm_transpose,fused_sigmoid_gated_dual_gemm=fused_sigmoid_gated_dual_gemm)
 exec(compile(ast.Module(body=body,type_ignores=[]),'<historical_reference_functions>','exec'),ns)
 peers={}
 for name,func in [('pytorch_eager',ns['torch_block']),('pytorch_compile',torch.compile(ns['torch_block'],**opts)),('cueq_torch',torch.compile(ns['cueq_torch_block'],**opts)),('cueq_cuda',torch.compile(ns['cueq_cuda_block'],**opts))]:
  def peer(fn=func):
   y=fn(*old_leaves,mask.to(torch.bfloat16),ds)
   return y,torch.autograd.grad(y,(*old_leaves,*tp),dy)
  print('CAPTURE peer',name,flush=True);peers[name]=capture(peer)[0]
 for label,gs in [('native_per_replay',{'historical':graphs['historical'],'current_fixed':graphs['current_fixed']}),('original_mix',dict(peers,historical=graphs['historical'],current_fixed=graphs['current_fixed']))]:
  phase[0]=label
  rr=[Q.paired(gs,warmup=100,iterations=250) for _ in range(5)]
  pooled=Q.pool(rr)
  record['times'][label]={n:v['median_us'] for n,v in pooled.items()}
  record['samples'][label]={n:v['samples_us'] for n,v in pooled.items()}
  save();print('TIME',label,record['times'][label],flush=True)

 # Isolate actual old/new Transition, common input/upstream and parameters.
 with torch.no_grad():tin=old.forward()[0].detach().requires_grad_(True)
 def tstep(fn):
  def run():
   y=fn(tin);return y,torch.autograd.grad(y,(tin,*tp),dy)
  return run
 tg={n:capture(tstep(fn))[0] for n,fn in [('historical',ot),('current',nt)]}
 measure('transition',tg)
 # Common upstream dy for the standalone TriMul arms.
 a['dy']=dy
 def raw(model):
  def run():
   with torch.no_grad():y,k=model.forward();return y,model.backward(k)
  return run
 tf=torch.compile(lambda x,mask:update(native,x,mask,ds),**opts)
 def tm_current():
  y=tf(x,mask);return y,torch.autograd.grad(y,new_leaves,dy)
 mg={n:capture(fn)[0] for n,fn in [('historical',raw(old)),('corrected',raw(fixed)),('current',tm_current)]}
 measure('trimul',mg)
 # Single-body comparisons. All use equivalent retained intermediates; no prep.
 from miniworld_engine.kernels.trimul_inproj.cuda import h100_training as H,h100_b1 as B1,h100_b7 as B7,_h100_runtime as RT
 with torch.no_grad():
  y,ab,tri,xn,stats,packed=H.forward(list(new_leaves),mask.bfloat16(),ds.reshape(384,128))
  data=H._data(list(new_leaves),mask.bfloat16(),ds,packed=packed,for_backward=True)
  cfg=json.loads((RT.SOURCES/'b1/configs.json').read_text())['384']
  p1=B1.Plan(dict(data,x=xn),dy,tri,stats,**cfg);dg,_,dt,*_=p1()
  dl=torch.empty_like(tri);dr=torch.empty_like(tri)
  torch.bmm(dt[:128],ab[256:384],out=dl[:128]);torch.bmm(dt[:128].transpose(-1,-2),ab[:128],out=dr[:128]);torch.bmm(ab[384:],dt[128:].transpose(-1,-2),out=dl[128:]);torch.bmm(ab[128:256],dt[128:],out=dr[128:])
  p7=B7.Plan(data,dy,dl,dr,dg,xn=xn)
  old();fixed()
  kg={n:capture(fn)[0] for n,fn in [('old_b1',old.p1),('current_b1',p1),('old_b7',old.p7),('fixed_b7',fixed.p7),('current_b7',p7)]}
 measure('kernel_bodies',kg,rounds=10,iters=200)
finally:stop.set();th.join();save()
# Prime CUPTI activity capture before replay, then repeat to expose missing events.
for name,g in graphs.items():
 with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as p:
  torch.cuda._sleep(10000000);torch.cuda.synchronize()
  for _ in range(5):g.replay();torch.cuda.synchronize()
 p.export_chrome_trace(str(R/(name+'-trace.json')))
record['complete']=True;save();print('DONE',flush=True)
