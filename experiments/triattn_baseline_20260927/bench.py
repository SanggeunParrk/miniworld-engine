"""Current installed TriangleAttention baseline; no config or dispatch changes."""
import argparse,collections,gc,hashlib,json,pathlib,statistics
import torch
import miniworld_engine
from miniworld_engine.modules import TriangleAttention
p=argparse.ArgumentParser();p.add_argument('--length',type=int,required=True);a=p.parse_args()
root=pathlib.Path(__file__).resolve().parents[2]
out=pathlib.Path('/workspace/vast-results/triattn-baseline-20260927');out.mkdir(parents=True,exist_ok=True)
dest=out/f'baseline-L{a.length}.json'
record=dict(length=a.length,width=128,heads=4,head_dim=32,batch=1,complete=False,rows=[],dtype='bf16 activation/linear; fp32 LN',engine=miniworld_engine.__file__,torch=torch.__version__,gpu=torch.cuda.get_device_name(),gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),dropout=.25,mask_fraction=.1,compile='fullgraph dynamic=False',cudagraph=True,source_sha256={})
for base in ('src/miniworld_engine/kernels/triangle_attention','src/miniworld_engine/modules/triangle_attention'):
 for f in (root/base).rglob('*'):
  if f.is_file() and f.suffix in ('.py','.cu','.cuh','.json','.so'):record['source_sha256'][str(f.relative_to(root))]=hashlib.sha256(f.read_bytes()).hexdigest()
record['script_sha256']=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()
def save():dest.write_text(json.dumps(record,indent=2)+'\n')
def capture(fn):
 for _ in range(4):fn()
 torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
 with torch.cuda.graph(g,stream=stream):v=fn()
 g.replay();torch.cuda.synchronize();return g,v

def measure(graphs):
 vals={k:[] for k in graphs}
 for k,g in graphs.items():
  for _ in range(5):
   if k=='backward':graphs['forward'].replay()
   g.replay()
 for rep in range(3):
  for i in range(30):
   keys=list(graphs) if (i+rep)%2 else list(graphs)[::-1]
   for k in keys:
    if k=='backward':graphs['forward'].replay()
    s,e=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    s.record();graphs[k].replay();e.record();e.synchronize();vals[k].append(s.elapsed_time(e))
 return {k:dict(median_ms=statistics.median(v),samples_ms=v) for k,v in vals.items()}

def rel(a,b):return float((a.detach().float()-b.detach().float()).norm()/b.detach().float().norm().clamp_min(1e-8))
stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream());torch.cuda.set_stream(stream)
for starting in (True,False):
 torch.manual_seed(982)
 m=TriangleAttention(128,n_head=4,starting=starting,implementation='miniworld',p_drop=.25).cuda().bfloat16()
 m.ln_pair.float()
 with torch.no_grad():
  for name,t in m.named_parameters():
   if t.ndim==2:t.normal_(std=128**-.5)
   elif 'weight' in name:t.copy_(1+.1*torch.randn_like(t))
   else:t.normal_(std=.05)
 x=torch.randn(1,a.length,a.length,128,device='cuda',dtype=torch.bfloat16,requires_grad=True)
 mask=torch.rand(1,a.length,device='cuda')>.1;dy=torch.randn_like(x)
 leaves=(x,*m.parameters());names=['x',*dict(m.named_parameters())]
 row=dict(starting=starting,backend=str(m._backend),parameter_dtypes={n:str(t.dtype) for n,t in m.named_parameters()});record['rows'].append(row);save()
 m.train();compiled=torch.compile(m,fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
 def full():
  y=compiled(x,mask);return (y,*torch.autograd.grad(y,leaves,dy))
 fullg,fullout=capture(full)
 assert all(bool(v.isfinite().all()) for v in fullout)
 assert float((fullout[0]-x).float().norm())>0
 assert all(float(g.float().norm())>0 for g in fullout[1:])
 row['nonzero_gradients']=names
 old=fullout[0].detach().clone();fullg.replay();torch.cuda.synchronize()
 row['dropout_rng_replay_changes_output']=not torch.equal(old,fullout[0]);assert row['dropout_rng_replay_changes_output'];del old
 # Actual isolated forward and backward graphs; no inference subtraction.
 for _ in range(4):
  yy=compiled(x,mask);torch.autograd.grad(yy,leaves,dy)
 del yy
 fg,y=capture(lambda:compiled(x,mask))
 bg=torch.cuda.CUDAGraph()
 with torch.cuda.graph(bg,stream=stream):bout=torch.autograd.grad(y,leaves,dy)
 fg.replay();bg.replay();torch.cuda.synchronize()
 row['training']=measure({'forward':fg,'backward':bg,'full':fullg})
 with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
  fullg.replay();torch.cuda.synchronize()
 trace=out/f'train-L{a.length}-{"starting" if starting else "ending"}.json';prof.export_chrome_trace(str(trace))
 events=[e for e in json.loads(trace.read_text())['traceEvents'] if e.get('cat')=='kernel']
 counts=collections.Counter(e['name'] for e in events);dur=collections.defaultdict(float)
 for e in events:dur[e['name']]+=e.get('dur',0)
 row['kernels']={n:dict(calls=counts[n],total_us=dur[n]) for n in sorted(dur,key=dur.get,reverse=True)}
 print('TRAIN',a.length,starting,{k:v['median_ms'] for k,v in row['training'].items()},flush=True);save()
 del fullg,fullout,fg,bg,bout,y;gc.collect()
 # Deterministic changed-input graph check without stochastic dropout.
 m.p_drop=0
 dg,dout=capture(full)
 originals=[t.detach().clone() for t in (x,m.to_out.weight,dy,mask)]
 with torch.no_grad():x.mul_(.97);m.to_out.weight.mul_(.91);dy.mul_(.93);mask[:,::11]=False
 dg.replay();torch.cuda.synchronize();fresh=full()
 row['changed_graph_relative_l2']=[rel(u,v) for u,v in zip(dout,fresh)]
 assert max(row['changed_graph_relative_l2'])<.005
 with torch.no_grad():
  for t,v in zip((x,m.to_out.weight,dy,mask),originals):t.copy_(v)
 del dg,dout,fresh,originals
 m.eval()
 with torch.no_grad():
  infer=torch.compile(m,fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
  ig,iv=capture(lambda:infer(x,mask));row['inference']=measure({'forward':ig})['forward'];assert bool(iv.isfinite().all())
 print('INFER',a.length,starting,row['inference']['median_ms'],flush=True);save()
 del ig,iv,infer,compiled,m,x,mask,dy,leaves;gc.collect();torch.compiler.reset();torch.cuda.empty_cache()
record['complete']=True;save()
