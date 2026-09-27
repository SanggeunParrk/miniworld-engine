"""Same GPU/data/graph timing of historical inference and current OPM routes."""
import os,json,statistics
from pathlib import Path
import torch
from miniworld_engine.modules.outer_product import OuterProductMean
from miniworld_engine.integrations import anthropic_msa,opm_train
R=Path(__file__).resolve().parent
os.environ['OPT_CORE_DIR']='/home/psk6950/ext/uplifting-biomolecular-modeling/common/opt_core'
torch.manual_seed(90323)
m=OuterProductMean(64,128,32,implementation='miniworld').cuda().bfloat16().eval()
with torch.no_grad():
 for n,p in m.named_parameters():
  if p.ndim>=2:p.normal_(std=p.shape[-1]**-.5)
  elif 'weight' in n:p.copy_(1+.1*torch.randn_like(p))
  else:p.normal_(std=.05)
x=torch.randn(1,1024,384,64,device='cuda',dtype=torch.bfloat16)
mask=torch.rand(1,1024,384,device='cuda')>.1
pair=torch.randn(1,384,384,128,device='cuda',dtype=torch.bfloat16)
variants={
 'legacy_raw':lambda:anthropic_msa.outer_product_mean(m,x,mask),
 'legacy_residual':lambda:anthropic_msa.outer_product_mean(m,x,mask)+pair,
 'current_raw':lambda:opm_train.outer_product_mean(m,x,mask),
 'current_residual':lambda:opm_train.outer_product_mean(m,x,mask)+pair,
 'current_compiled_residual':torch.compile(lambda:m(x,mask,residual=pair),fullgraph=True,dynamic=False,options={'triton.cudagraphs':False}),
}
graphs={};outs={};timings={k:[] for k in variants}
stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
with torch.no_grad(),torch.cuda.stream(stream):
 for name,fn in variants.items():
  for _ in range(4):fn()
  torch.cuda.synchronize()
  graph=torch.cuda.CUDAGraph()
  with torch.cuda.graph(graph,stream=stream):out=fn()
  graphs[name]=graph;outs[name]=out
 torch.cuda.synchronize()
torch.cuda.current_stream().wait_stream(stream)
for _ in range(100):
 for g in graphs.values():g.replay()
torch.cuda.synchronize()
for round_i in range(9):
 names=list(graphs);offset=round_i%len(names);names=names[offset:]+names[:offset]
 for name in names:
  start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
  start.record()
  for _ in range(100):graphs[name].replay()
  end.record();end.synchronize();timings[name].append(start.elapsed_time(end)/100)
results=dict(ms={k:statistics.median(v) for k,v in timings.items()},rounds_ms=timings,
 gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),job=os.getenv('SLURM_JOB_ID'),
 condition='B1 S1024 L384 MSA64 pair128 hidden32 BF16; same inputs, 10% mask; CUDA graph, 9x100 rotated; only current_compiled_residual uses torch.compile',
 relative_output_error={s:((outs['legacy_'+s].float()-outs['current_'+s].float()).norm()/outs['legacy_'+s].float().norm()).item() for s in ('raw','residual')})
(R/'opm-paths.json').write_text(json.dumps(results,indent=2)+'\n')
print(results['ms'],flush=True)
