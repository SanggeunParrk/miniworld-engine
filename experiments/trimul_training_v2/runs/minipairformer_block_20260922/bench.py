from pathlib import Path
import importlib.util,torch,json,platform,hashlib,collections,threading,subprocess,time,statistics,os,sys
R=Path(__file__).resolve().parent;ROOT=R.parent.parent

def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
P=load('block_latest_trimul',R.parent/'trimul_full_latest_20260922/policy.py');Q=P.Q
PIN=load('block_previous_configs',R/'pin_previous.py');settings=PIN.settings;B=PIN.B
from miniworld_engine.modules import Transition
from miniworld_engine.modules.exceptions import ImplementationType
settings.configure(engine_backend='triton',transition_residual_fusion=True,autotune_miss_cap=24)
# D128 Transition is kept identical for all arms; previous comparison is of
# the original TriMul backend plus this same residual-fused Transition.
torch.manual_seed(20260922)
transition=Transition(128,n=4,implementation=ImplementationType.TRITON).cuda().to(torch.bfloat16)
with torch.no_grad():
 torch.manual_seed(20260922)
 transition.squeeze.weight.normal_(0,.02)
transition.ln_in.float()
transition.train();tp=tuple(transition.parameters());trans_names=[n for n,_ in transition.named_parameters()]
trans=torch.compile(transition,fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
with torch.no_grad():
 a=Q.setup(384);d=a['d'];tm=P.Latest(a);block_dy=a['dy'].clone()
leaves=tuple(d['leaves'])
for x in leaves:x.requires_grad_(True)
mask=d['mask'].reshape(1,384,384).to(torch.bfloat16);ds=d['ds'].reshape(1,1,384,128)
LN=P.B1.BASE.OLD.LN
@torch.compile(fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
def pack(wl,wlg,wr,wrg):
 gate,proj=torch.cat((wlg,wrg),0),torch.cat((wl,wr),0)
 return torch.stack((gate.reshape(-1,32,128),proj.reshape(-1,32,128)),1).reshape(1024,128)

def old_fn(*args):return B.bidirectional_trimul_triton(*args[:-2],1e-5,1e-5,128,mask=args[-2],dropscale=args[-1],output_backend='triton')
old=torch.compile(old_fn,fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})

def set_backend(name):
 settings.configure(engine_backend='triton',trimul_sm90_kernels=('front','f567','dual_bwd','out_ln_bwd') if name=='previous_h100' else (),transition_residual_fusion=True,autotune_miss_cap=24)

def latest_train():
 with torch.no_grad():y,k=tm.forward()
 with torch.enable_grad():
  z=y.detach().requires_grad_(True);out=trans(z);grad=torch.autograd.grad(out,(z,*tp),block_dy)
 with torch.no_grad():
  a['dy']=grad[0];dg=tm.backward(k)
 return out,(*dg,*grad[1:])

def latest_training_forward():
 with torch.no_grad():y,k=tm.forward()
 with torch.enable_grad():return trans(y.detach().requires_grad_(True))

def latest_inference():
 with torch.no_grad():
  w1=pack(*d['leaves'][1:5]);ab=LN.I.front(d['x'],w1,d['mask'],d['gi'],d['bi'],True,(2,64,8,2,-1,232,2));tri=B.packed_forward(ab[:256],ab[256:],128)
  y=LN.I.output(tri,d['x'],d['wp'],d['leaves'][5],d['gi'],d['bi'],d['go'],d['bo'],d['x'],True,(2,64,4,1,1,1))
  return trans(y)

def old_train():
 with torch.enable_grad():
  out=trans(old(*leaves,mask,ds));return out,torch.autograd.grad(out,(*leaves,*tp),block_dy)

def old_training_forward():
 with torch.enable_grad():return trans(old(*leaves,mask,ds))

def old_inference():
 with torch.no_grad():return trans(old(*leaves,mask,None))

names=('previous_triton','previous_h100','latest_candidate')
record=dict(L=384,B=1,C=128,H_trimul=256,transition_expansion=4,blocks=1,use_single=False,host=platform.node(),job=os.environ.get('SLURM_JOB_ID'),gpu=torch.cuda.get_device_name(),dropout_training=.25,dropout_inference=0,transition='same D128 Triton full-K residual dispatch in every arm',baseline='preserved original pre-Anthropic TriMul routes, not historical whole installation',includes=['bidirectional TriMul','Transition','both residuals','pair mask','fixed training dropout','live weight packing','all block parameter gradients'],excludes=['optimizer','dropout RNG generation','compile','CPU dispatch'],latest_training_diagnostic=True,known_issue='latest B7 input LN gradient exceeds strict same-algorithm threshold in prior full TriMul test',times={},rounds={},checks={},traces={},transition_parameters=trans_names,engine=str(Path(PIN.B.__file__)),source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),R/'pin_previous.py',ROOT/'src/miniworld/modules/mini_pairformer.py']})
def save():(R/'results.json').write_text(json.dumps(record,indent=2))
def clone(v):return (v[0].clone(),tuple(t.clone() for t in v[1])) if isinstance(v,tuple) else v.clone()
def err(out,ref,train):
 xs=(out[0],*out[1]) if train else (out,);ys=(ref[0],*ref[1]) if train else (ref,)
 labels=['forward','dx','dWL','dWLg','dWR','dWRg','dWgate','dWproj','dgamma_in','dbeta_in','dgamma_out','dbeta_out',*['transition.'+n for n in trans_names]] if train else ['forward']
 ans={}
 for i,(n,x,y) in enumerate(zip(labels,xs,ys)):
  rel=float((x.float()-y.float()).norm()/y.float().norm().clamp_min(1e-20));bound=.005 if i==0 else .01
  ans[n]=dict(relative_l2=rel,limit=bound,finite=bool(x.isfinite().all()),bit_exact=torch.equal(x,y));assert rel<=bound and ans[n]['finite'],(n,ans[n])
 return ans

graphs={};outs={};fns={}
for scope in ('training','training_forward','inference'):
 transition.train(scope!='inference');graphs[scope]={};outs[scope]={};fns[scope]={}
 for name in names:
  set_backend(name);fn=(latest_train if scope=='training' else latest_training_forward if scope=='training_forward' else latest_inference) if name=='latest_candidate' else (old_train if scope=='training' else old_training_forward if scope=='training_forward' else old_inference)
  fns[scope][name]=fn;print('CAPTURE',scope,name,flush=True)
  # Run once outside capture to compile / tune all lazily built operations.
  fn();graphs[scope][name],outs[scope][name]=Q.capture_outputs(fn)
 for g in graphs[scope].values():g.replay()
 torch.cuda.synchronize()
 ref=clone(outs[scope]['previous_triton']);record['checks'][scope]={}
 for name in names:
  g=graphs[scope][name];g.replay();g.replay();torch.cuda.synchronize();record['checks'][scope][name]=err(outs[scope][name],ref,scope=='training')
  print('CHECK',scope,name,{k:v['relative_l2'] for k,v in record['checks'][scope][name].items()},flush=True)
 save()
# Validate a changed live input with graph/eager, keeping model modes explicit.
with torch.no_grad():
 x_saved=d['x'].clone();w_saved=transition.squeeze.weight.clone();d['x'].mul_(.97);transition.squeeze.weight.mul_(1.03)
record['mutation']={}
for scope in ('training','inference'):
 transition.train(scope=='training');record['mutation'][scope]={}
 for name in names:
  set_backend(name);eager=clone(fns[scope][name]());graphs[scope][name].replay();torch.cuda.synchronize()
  checks=err(outs[scope][name],eager,scope=='training');record['mutation'][scope][name]=checks;save();print('MUTATION',scope,name,{k:v for k,v in checks.items() if not v['bit_exact']},flush=True)
  for label,v in checks.items():
   # Existing LN kernels use FP32 atomic accumulation. Preserve the prior
   # 5e-6 same-algorithm LN bound; all other outputs must remain bit-exact.
   is_ln=label.startswith(('dgamma','dbeta','transition.ln_in.'))
   v['graph_eager_limit']=5e-6 if is_ln else 0.
   assert v['relative_l2']<=v['graph_eager_limit'],(scope,name,label,v)
  save()
with torch.no_grad():d['x'].copy_(x_saved);transition.squeeze.weight.copy_(w_saved)
del x_saved,w_saved
record['measurement_input_note']='Original inputs and weights restored exactly from clones after mutation checks.'
uuid=str(torch.cuda.get_device_properties(0).uuid);uuid=uuid if uuid.startswith('GPU-') else 'GPU-'+uuid;record['gpu_uuid']=uuid
phase=['idle'];stop=threading.Event();samples=[]
def monitor():
 while not stop.is_set():
  p=subprocess.run(['nvidia-smi','-i',uuid,'--query-gpu=clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True);samples.append(dict(phase=phase[0],csv=p.stdout.strip(),returncode=p.returncode));stop.wait(.05)
th=threading.Thread(target=monitor);th.start()
try:
 for scope in ('training','training_forward','inference'):
  phase[0]=scope;rr=[Q.paired(graphs[scope],warmup=100,iterations=250) for _ in range(5)];record['rounds'][scope]=rr;record['times'][scope]=Q.pool(rr);print('TIME',scope,{k:v['median_us'] for k,v in record['times'][scope].items()},flush=True);save()
finally:stop.set();th.join();record['telemetry']=samples;save()
for scope in ('training','inference'):
 record['traces'][scope]={}
 for name,g in graphs[scope].items():
  with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:g.replay();torch.cuda.synchronize()
  path=R/('trace-'+scope+'-'+name+'.json');prof.export_chrome_trace(str(path));trace=json.loads(path.read_text());record['traces'][scope][name]=dict(collections.Counter(e['name'] for e in trace['traceEvents'] if e.get('cat')=='kernel'))
assert record['traces']['training']['latest_candidate'].get('b7_joint')==1
assert record['traces']['inference']['latest_candidate'].get('infer_k3')==1
assert 'b7_joint' not in record['traces']['inference']['latest_candidate']
from miniworld_engine.kernels.transition.triton import b2b_residual
record['transition_config']={'b2b_best':str(getattr(b2b_residual._kernel,'best_config',None)), 'b2b_cache':{str(k):str(v) for k,v in getattr(b2b_residual._kernel,'cache',{}).items()}}
record['complete']=True;save();print('DONE',flush=True)
