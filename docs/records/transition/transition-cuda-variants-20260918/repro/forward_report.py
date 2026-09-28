import json,statistics,shutil
from pathlib import Path
r=Path(__file__).parent
record=r.parent/'trimul_sm90_parity_20260917/engine/docs/records/transition-cuda-variants-20260918'
text=['# Transition forward comparison','', 'Measured on node02 H100, B1 pair input [1,L,L,D], expansion4, nonzero squeeze, BF16 activations/weights and FP32 norm affine. All arms use static compile + manual CUDA Graph, two captures per row. Forward-only/inference, no optimizer. General Transition has no dropout.', '', '**Old Triton means the retained split algorithm rerun on the current checkout/runtime, not a restored historical commit.** Current Triton uses full-K b2b at D128/256 and split at D384/512. New CUDA is the faster measured streamed/full-K native variant at each shape, not the existing legacy H100 auto backend. PyTorch is the official module harness reference, compiled and graphed too. PyTorch runs followed the backend runs; they were not a single interleaved experiment. Near-unity differences should be treated cautiously.', '', '| D | L | PyTorch ms | Old Triton split ms | Current Triton ms | New CUDA ms | CUDA variant | Current Triton / PyTorch speedup | Current Triton / old speedup | New CUDA / PyTorch speedup | New CUDA / old speedup |', '|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|']
out=[]
for d in (128,256,384,512):
 for l in (384,768):
  m=json.loads((r/f'module-D{d}-L{l}.json').read_text());p=json.loads((r/f'pytorch-D{d}-L{l}.json').read_text())
  assert len(p['rows'])==2 and all(z['compiled'] and z['cudagraph']=='manual' and z['compiled_graphs']==1 for z in p['rows'])
  def time(arm):
   found=[z for z in m['rows'] if z['mode']=='inference' and z['arm']==arm];assert len(found)==2
   return statistics.median(z['value'] for z in found)
  py=statistics.median(z['value'] for z in p['rows']);old=time('triton_split');current=time('triton:full_k' if d<=256 else 'triton_split')
  variant=min(('streamed_k','full_k'),key=lambda v:time('cuda:'+v));native=time('cuda:'+variant)
  ratios=(py/current,old/current,py/native,old/native)
  text.append('| '+f'{d} | {l} | '+' | '.join(f'{v:.4f}' for v in (py,old,current,native))+' | '+variant+' | '+' | '.join(f'{v:.3f}x' for v in ratios)+' |')
  out.append(dict(D=d,L=l,pytorch_ms=py,old_triton_split_ms=old,current_triton_ms=current,new_cuda_ms=native,new_cuda_variant=variant,current_triton_vs_pytorch=ratios[0],current_triton_vs_old=ratios[1],new_cuda_vs_pytorch=ratios[2],new_cuda_vs_old=ratios[3]))
  shutil.copy2(r/f'pytorch-D{d}-L{l}.json',record)
text+=['', 'Speedup = baseline time / implementation time; below 1 means slower. The new CUDA variants pass numerical/graph checks but are not faster than the current Triton choices here. They are explicit experimental module options, not promoted to automatic dispatch.', '', 'See [RESULTS.md](RESULTS.md) for separate CUDA variants, full training, forward/backward breakdown and validation; [CONFIGS.md](CONFIGS.md) for sampled winners and limits.']
(record/'FORWARD_COMPARISON.md').write_text('\n'.join(text)+'\n');(record/'forward-summary.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
