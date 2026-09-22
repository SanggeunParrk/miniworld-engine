from harness import *
def load(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
C=load('miniworld_engine.kernels.transition.cuda.transition_upgrade_block',R/'selected/fused_sm90a.py')
F=load('transition_upgrade_trimul_fixed',R.parent/'trimul_ln_gradient_20260922/policy.py');P=F.P;Q=P.Q
from miniworld_engine import settings
from miniworld_engine.modules import Transition
settings.configure(engine_backend='triton',transition_residual_fusion=True,autotune_miss_cap=24)
torch.manual_seed(20260922);tr=Transition(128,n=4,implementation='triton').cuda().bfloat16();tr.ln_in.float()
with torch.no_grad():tr.squeeze.weight.normal_(0,.02)
tp=tuple(tr.parameters())
def make(mod):
 def fn(x):return mod.transition_fused_sm90a(x,tr.ln_in.weight,tr.ln_in.bias,tr.expand_a.weight,tr.expand_b.weight,tr.squeeze.weight,tr.ln_in.eps)
 return torch.compile(fn,fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
trans={n:make(m) for n,m in [('baseline',N),('selected',C)]}
record=dict(job=os.environ.get('SLURM_JOB_ID'),L=384,dropout=.25,cases={},times={})
def save():(R/'block.json').write_text(json.dumps(record,indent=2))
with torch.no_grad():
 a=Q.setup(384);d=a['d'];m=F.Fixed(a);up=a['dy'].clone()
 names=['forward','dx','dWL','dWLg','dWR','dWRg','dWgate','dWproj','dgamma_in','dbeta_in','dgamma_out','dbeta_out']+['transition.'+n for n,_ in tr.named_parameters()]
 def call(n):
  y,k=m.forward()
  with torch.enable_grad():z=y.detach().requires_grad_(True);out=trans[n](z);g=torch.autograd.grad(out,(z,*tp),up)
  a['dy']=g[0];return (out,*m.backward(k),*g[1:])
 gs={};outs={}
 for n in trans:gs[n],outs[n]=capture(lambda n=n:call(n))
 tensors=list({t.data_ptr():t for t in [*d['leaves'],up,d['mask'],a['mask'],d['ds'],*tp]}.values());orig=[v.clone() for v in tensors]
 for case in range(8):
  for t,v in zip(tensors,orig):t.copy_(v)
  torch.manual_seed(8290+case)
  if case:
   d['x'].normal_(0,[1.,.01,2.,.2][case%4]);up.normal_()
   for t in d['leaves'][1:7]:t.add_(torch.randn_like(t)*.001)
   d['gi'].normal_(1.,.15);d['bi'].normal_(0,.05);d['go'].normal_(1.,.15);d['bo'].normal_(0,.05)
   d['mask'].copy_((torch.rand_like(d['mask'].float())>.25).to(d['mask'].dtype));a['mask'].copy_(d['mask'].reshape(-1));d['ds'].copy_((torch.rand_like(d['ds'].float())>.25).to(d['ds'].dtype)*(4/3))
  ref=tuple(v.clone() for v in call('baseline'));out=tuple(v.clone() for v in call('selected'));es={}
  for n,x,y in zip(names,out,ref):
   lim=0 if n=='forward' else 2e-5 if n=='dx' else 5e-6 if n.startswith(('dgamma','dbeta','transition.ln_in.')) else 5e-4
   val=float((x.double()-y.double()).norm()/y.double().norm().clamp_min(1e-30));es[n]=dict(relative_l2=val,limit=lim,bit_exact=torch.equal(x,y));assert bool(x.isfinite().all()) and val<=lim,(n,es[n])
  gs['selected'].replay();gs['selected'].replay();torch.cuda.synchronize();assert all(torch.equal(x,y) for x,y in zip(outs['selected'],out))
  record['cases'][str(case)]=dict(errors=es,graph_eager_exact=True);save();print('BLOCK_PASS',case,flush=True)
 for t,v in zip(tensors,orig):t.copy_(v)
 record['times']=paired(gs,250,5);print('BLOCK_TIME',{k:v['median_us'] for k,v in record['times'].items()},flush=True)
 with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:gs['selected'].replay();torch.cuda.synchronize()
 prof.export_chrome_trace(str(R/'trace-block.json'));record['complete']=True;save()
