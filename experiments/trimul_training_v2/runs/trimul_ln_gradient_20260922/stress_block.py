"""Strict whole-TriMul and connected MiniPairformer checks; no tolerance changes."""
from pathlib import Path
import importlib.util,sys,os,json,torch,hashlib,collections
R=Path(__file__).resolve().parent

def load(n,p):
 sp=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(sp);sys.modules[n]=m;sp.loader.exec_module(m);return m
F=load('ln_stress_fixed',R/'policy.py');P=F.P;Q=P.Q
from miniworld_engine import settings
from miniworld_engine.modules import Transition
import miniworld_engine.kernels.transition.cuda
settings.configure(engine_backend='triton',transition_residual_fusion=True,autotune_miss_cap=24)
source=R.parent/'minipairformer_block_cuda_20260922/transition_snapshot/fused_sm90a.py'
N=load('miniworld_engine.kernels.transition.cuda.ln_fix_block',source)
torch.manual_seed(20260922)
tr=Transition(128,n=4,implementation='triton').cuda().bfloat16();tr.ln_in.float()
with torch.no_grad():tr.squeeze.weight.normal_(0,.02)
tp=tuple(tr.parameters())
def trans_native(x):return N.transition_fused_sm90a(x,tr.ln_in.weight,tr.ln_in.bias,tr.expand_a.weight,tr.expand_b.weight,tr.squeeze.weight,tr.ln_in.eps)
trans=torch.compile(trans_native,fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
with torch.no_grad():
 a=Q.setup(384);d=a['d'];models={'reference':P.Regression(a),'original_joint':P.Latest(a),'fixed':F.Fixed(a)};upstream=a['dy'].clone()
 names=['forward','dx','dWL','dWLg','dWR','dWRg','dWgate','dWproj','dgamma_in','dbeta_in','dgamma_out','dbeta_out']+['transition.'+n for n,_ in tr.named_parameters()]
 def clone(v):return v[0].clone(),tuple(x.clone() for x in v[1])
 def err(out,ref):
  es={}
  for n,x,y in zip(names,(out[0],*out[1]),(ref[0],*ref[1])):
   lim=0 if n=='forward' else 2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta','transition.ln_in.')) else 5e-4
   value=float((x.double()-y.double()).norm()/y.double().norm().clamp_min(1e-30))
   es[n]=dict(relative_l2=value,limit=lim,finite=bool(x.isfinite().all()),bit_exact=torch.equal(x,y));assert es[n]['finite'] and value<=lim,(n,es[n])
  return es
 def block(m):
  y,k=m.forward()
  with torch.enable_grad():
   z=y.detach().requires_grad_(True);out=trans(z);g=torch.autograd.grad(out,(z,*tp),upstream)
  a['dy']=g[0];return out,(*m.backward(k),*g[1:])
 def tri(m):a['dy']=upstream;return m()
 calls={'tri':{n:(lambda m=m:tri(m)) for n,m in models.items()},'block':{n:(lambda m=m:block(m)) for n,m in models.items()}}
 graphs={};outs={}
 for scope in calls:
  graphs[scope]={};outs[scope]={}
  for n,fn in calls[scope].items():
   fn();graphs[scope][n],outs[scope][n]=Q.capture_outputs(fn)
 tensors=list({t.data_ptr():t for t in [*d['leaves'],upstream,d['mask'],a['mask'],d['ds'],*tp]}.values());snapshot=[t.clone() for t in tensors]
 record=dict(job=os.environ.get('SLURM_JOB_ID'),L=384,cases={},times={},source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [R/'fixed/joint.cu',R/'fixed/single_wg.inc',R/'fixed/plan.py',R/'policy.py']})
 def save():(R/'stress_block.json').write_text(json.dumps(record,indent=2))
 for case in range(12):
  for t,v in zip(tensors,snapshot):t.copy_(v)
  torch.manual_seed(8100+case)
  if case:
   scale=[1.,.01,2.,.2][case%4];d['x'].normal_(0,scale);upstream.normal_(0,.1 if case%3==0 else 1.)
   for t in d['leaves'][1:7]:t.add_(torch.randn_like(t)*.001)
   d['gi'].normal_(1.,.15);d['bi'].normal_(0,.05);d['go'].normal_(1.,.15);d['bo'].normal_(0,.05)
   d['gi'][::17]=0;d['go'][::23]=0
   d['mask'].copy_((torch.rand_like(d['mask'].float())>.25).to(d['mask'].dtype));a['mask'].copy_(d['mask'].reshape(-1))
   d['ds'].copy_((torch.rand_like(d['ds'].float())>.25).to(d['ds'].dtype)*(4/3))
   tr.squeeze.weight.add_(torch.randn_like(tr.squeeze.weight)*.001)
  if case==10:d['mask'].zero_();a['mask'].zero_()
  if case==11:d['ds'].zero_()
  record['cases'][str(case)]={}
  for scope in calls:
   ref=clone(calls[scope]['reference']());eager=clone(calls[scope]['fixed']());graphs[scope]['fixed'].replay();graphs[scope]['fixed'].replay();torch.cuda.synchronize()
   checks=err(eager,ref);graph=err(outs[scope]['fixed'],ref);vs=err(outs[scope]['fixed'],eager);assert all(v['bit_exact'] for v in vs.values())
   record['cases'][str(case)][scope]=dict(eager=checks,graph=graph,graph_eager_bit_exact=True)
   print('PASS',case,scope,'lnmax',max(v['relative_l2'] for n,v in checks.items() if n.startswith(('dgamma','dbeta','transition.ln_in.'))),flush=True);save()
 for t,v in zip(tensors,snapshot):t.copy_(v)
 for scope in calls:
  gs={n:g for n,g in graphs[scope].items() if n!='reference'};rr=[Q.paired(gs,warmup=100,iterations=250) for _ in range(5)];record['times'][scope]=Q.pool(rr);print('TIME',scope,{n:v['median_us'] for n,v in record['times'][scope].items()},flush=True);save()
 with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:graphs['block']['fixed'].replay();torch.cuda.synchronize()
 prof.export_chrome_trace(str(R/'trace-block-fixed.json'));trace=json.loads((R/'trace-block-fixed.json').read_text());record['traces']=dict(collections.Counter(e['name'] for e in trace['traceEvents'] if e.get('cat')=='kernel'))
 assert record['traces'].get('b7_joint')==1 and record['traces'].get('transition_bwd_fused')==1
 record['complete']=True;save();print('DONE',flush=True)
