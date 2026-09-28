"""Matched full-module backend comparison with interleaved CUDA Graph timings."""
import argparse,collections,gc,hashlib,importlib.metadata,json,os,pathlib,statistics
import torch
from miniworld_engine.modules import TriangleAttention
from miniworld_engine import settings
p=argparse.ArgumentParser();p.add_argument('--length',type=int,required=True);a=p.parse_args()
root=pathlib.Path(__file__).resolve().parents[2]
out=pathlib.Path('/workspace/vast-results/triattn-compare-20260927');out.mkdir(parents=True,exist_ok=True)
dest=out/f'compare-L{a.length}.json'
record=dict(length=a.length,width=128,heads=4,batch=1,dropout=.25,mask_fraction=.1,dtype='bf16 activation/linear; fp32 LN',complete=False,rows=[],script_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),torch=torch.__version__,gpu=torch.cuda.get_device_name(),gpu_uuid=str(torch.cuda.get_device_properties(0).uuid))
record['autotune_miss_cap']=settings.current().autotune_miss_cap
record['allow_tf32']=torch.backends.cuda.matmul.allow_tf32
record['versions']={k:importlib.metadata.version(k) for k in ('torch','triton','cuequivariance-ops-torch-cu12')}
record['source_sha256']={str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for base in ('src/miniworld_engine/kernels/triangle_attention','src/miniworld_engine/modules/triangle_attention') for f in (root/base).rglob('*') if f.is_file() and f.suffix in ('.py','.cu','.cuh','.json','.so')}
def save():dest.write_text(json.dumps(record,indent=2)+'\n')
def capture(fn):
 for _ in range(4):fn()
 torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
 with torch.cuda.graph(g,stream=stream):v=fn()
 g.replay();torch.cuda.synchronize();return g,v

def measure(graphs):
 vals={k:[] for k in graphs}
 for g in graphs.values():
  for _ in range(5):g.replay()
 for rep in range(3):
  for i in range(30):
   keys=list(graphs);offset=(rep+i)%len(keys);keys=keys[offset:]+keys[:offset]
   if (rep+i)%2:keys.reverse()
   for k in keys:
    s,e=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    s.record();graphs[k].replay();e.record();e.synchronize();vals[k].append(s.elapsed_time(e))
 return {k:dict(median_ms=statistics.median(v),samples_ms=v) for k,v in vals.items()}

def errors(actual,expected):
 return [float((u.detach().float()-v.float()).norm()/v.float().norm().clamp_min(1e-8)) for u,v in zip(actual,expected)]
stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream());torch.cuda.set_stream(stream)
for starting in (True,False):
 torch.manual_seed(982)
 template=TriangleAttention(128,n_head=4,starting=starting,implementation='miniworld',p_drop=0).cuda().bfloat16();template.ln_pair.float()
 with torch.no_grad():
  for name,t in template.named_parameters():
   if t.ndim==2:t.normal_(std=128**-.5)
   elif 'weight' in name:t.copy_(1+.1*torch.randn_like(t))
   else:t.normal_(std=.05)
 state={n:t.detach().clone() for n,t in template.state_dict().items()};del template
 x=torch.randn(1,a.length,a.length,128,device='cuda',dtype=torch.bfloat16,requires_grad=True);dy=torch.randn_like(x);mask=torch.rand(1,a.length,device='cuda')>.1
 row=dict(starting=starting,arms={});record['rows'].append(row);graphs={};objects=[];expected=None;save()
 for arm,impl in [('engine','miniworld'),('triton','triton'),('pytorch','pytorch'),('cueq','cuequivariance')]:
  print('BUILD',a.length,starting,arm,flush=True)
  os.environ['MINIWORLD_TRIATTN_TRAINING_FWD']='0' if arm=='triton' else '1'
  r={};row['arms'][arm]=r;save()
  try:
   if arm=='cueq':
    import cuequivariance_ops_torch
    cuequivariance_ops_torch.init_triton_cache()
   m=TriangleAttention(128,n_head=4,starting=starting,implementation=impl,p_drop=0).cuda().bfloat16();m.ln_pair.float();m.load_state_dict(state);m.train()
   if arm=='triton':
    for name in ('front','projection','gate','bias','dq'):setattr(m,'_fuse_'+name+'_backward',False)
   compiled=torch.compile(m,fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
   leaves=(x,*m.parameters())
   def step(compiled=compiled,leaves=leaves):
    y=compiled(x,mask);return (y,*torch.autograd.grad(y,leaves,dy))
   got=step();torch.cuda.synchronize();assert all(bool(v.isfinite().all()) for v in got)
   if expected is None:expected=[v.detach().clone() for v in got]
   r['nodrop_relative_l2_vs_engine']=errors(got,expected)
   assert max(r['nodrop_relative_l2_vs_engine'])<.03,r['nodrop_relative_l2_vs_engine']
   del got
   m.p_drop=.25
   graph,outputs=capture(step)
   assert all(bool(v.isfinite().all()) for v in outputs)
   previous=outputs[0].detach().clone();graph.replay();torch.cuda.synchronize()
   r['live_dropout_rng']=not torch.equal(previous,outputs[0]);assert r['live_dropout_rng'];del previous
   graphs[arm]=graph;objects.append((m,compiled,leaves,outputs,step));r['status']='ready'
   with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:graph.replay();torch.cuda.synchronize()
   trace=out/f'trace-L{a.length}-{starting}-{arm}.json';prof.export_chrome_trace(str(trace))
   r['kernels']=dict(collections.Counter(e['name'] for e in json.loads(trace.read_text())['traceEvents'] if e.get('cat')=='kernel'))
   print('READY',a.length,starting,arm,flush=True);save()
  except Exception as e:
   r.update(status='error',error=repr(e));print('FAILED',arm,repr(e),flush=True);save()
   if arm=='engine':raise
 timings=measure(graphs);row['training_full']=timings
 row['speedup_vs_engine']={k:v['median_ms']/timings['engine']['median_ms'] for k,v in timings.items()}
 print('TIMES',a.length,starting,{k:v['median_ms'] for k,v in timings.items()},flush=True);save()
 del graphs,objects,expected,state,x,dy,mask,m,compiled,leaves,outputs,step,graph;gc.collect();torch.compiler.reset();torch.cuda.empty_cache()
record['complete']=True;save()
