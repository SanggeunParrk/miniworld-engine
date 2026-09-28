import importlib.util,json,statistics,sys
from pathlib import Path
import torch
from miniworld_engine.kernels.trimul_inproj.cuda import h100_training as new
R=Path(__file__).resolve().parent

def load(name,p,text=None):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m
    exec(compile(text if text else p.read_text(),str(p),'exec'),m.__dict__)
    return m
load('prep_before_width',R/'before_width.py')
old=load('prep_before_training',R/'before_training.py',(R/'before_training.py').read_text().replace('from miniworld_engine.kernels.trimul_inproj.cuda.h100_width import Training','from prep_before_width import Training').replace('trimul_h100_train_','prep_baseline_trimul_h100_train_').replace('                model.xn,','                model.xn.reshape_as(x),'))
records=[]
def rel(a,b):return float((a.detach().float()-b.detach().float()).norm()/b.detach().float().norm().clamp_min(1e-12))
for D in (64,256,384,512):
 for n in (384,768):
  torch.compiler.reset();torch.manual_seed(391)
  x=torch.randn(1,n,n,D,device='cuda',dtype=torch.bfloat16,requires_grad=True)
  weights=[(torch.randn(h,d,device='cuda',dtype=torch.bfloat16)*d**-.5).requires_grad_() for h,d in [(2*D,D)]*4+[(D,D),(D,2*D)]]
  ln=[(1+.1*torch.randn(c,device='cuda')).requires_grad_() if i%2==0 else (.05*torch.randn(c,device='cuda')).requires_grad_() for i,c in enumerate((D,D,2*D,2*D))]
  args=[x,*weights,*ln];mask=(torch.rand(n,n,device='cuda')>.15).bfloat16();ds=(torch.rand(n,D,device='cuda')>.25).bfloat16()*(4/3);dy=torch.randn_like(x)
  stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
  with torch.cuda.stream(stream):
   def run(fn):
    y=fn(*args,mask,ds);return [y,*torch.autograd.grad(y,args,dy)]
   a=run(old.bidirectional_trimul);b=run(new.bidirectional_trimul)
   errors=[rel(aa,bb) for aa,bb in zip(a,b)]
   assert max(errors)<2e-6,(D,n,errors)
   # Actual work counts for a full new invocation, not source-level guesses.
   from miniworld_engine.kernels.trimul_inproj.cuda import h100_width as W
   pack,norm=W.pack_into,W.normalize_into;counts={'pack':0,'norm':0}
   def pk(*a):counts['pack']+=1;return pack(*a)
   def nm(*a):counts['norm']+=1;return norm(*a)
   W.pack_into,W.normalize_into=pk,nm
   try:run(new.bidirectional_trimul)
   finally:W.pack_into,W.normalize_into=pack,norm
   assert counts=={'pack':1,'norm':int(D==512)},counts
   # A second forward must not overwrite the first one's retained pack/mask.
   first=new.bidirectional_trimul(*args,mask,ds)
   new.bidirectional_trimul(x*.9,*args[1:],1-mask,ds*.75)
   owned=torch.autograd.grad(first,args,dy)
   assert max(rel(aa,bb) for aa,bb in zip(owned,b[1:]))<2e-6
   graphs={};keep={}
   for label,fn in [('before',old.bidirectional_trimul),('after',new.bidirectional_trimul)]:
    compiled=torch.compile(fn,fullgraph=True,options={'triton.cudagraphs':False})
    for _ in range(3):run(compiled)
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g,stream=stream):out=run(compiled)
    graphs[label]=g;keep[label]=out
    g.replay()
    assert max(rel(aa,bb) for aa,bb in zip(out,b))<2e-6
   times={key:[] for key in graphs}
   for g in graphs.values():
    for _ in range(10):g.replay()
   for repeat in range(8):
    for key in (list(graphs) if repeat%2==0 else list(graphs)[::-1]):
     start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record()
     for _ in range(40):graphs[key].replay()
     end.record();end.synchronize();times[key].append(start.elapsed_time(end)/40)
   # Captured pack/mask normalization must still read updated live tensors.
   with torch.no_grad():x.mul_(.97);weights[0].mul_(.91);mask[::3]=0;ds[::3]=0
   graphs['after'].replay();expected=run(new.bidirectional_trimul)
   live=[rel(aa,bb) for aa,bb in zip(keep['after'],expected)]
   assert max(live)<2e-6,(D,n,live)
   row=dict(D=D,L=n,errors=errors,live_errors=live,calls=counts,times_ms={k:statistics.median(v) for k,v in times.items()},samples_ms=times)
   records.append(row);print('RESULT',json.dumps(row),flush=True)
   (R/'results.json').write_text(json.dumps(records,indent=2))
  torch.cuda.current_stream().wait_stream(stream)
  del graphs,keep,a,b,owned,first,expected,out,args,weights,ln,x,dy,mask,ds,compiled,g
  torch.cuda.synchronize();torch.cuda.empty_cache()
