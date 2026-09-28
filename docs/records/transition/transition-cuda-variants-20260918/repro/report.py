"""Summarize completed measurements; retain failed/excluded tuning records."""
import hashlib,json,shutil,statistics
from pathlib import Path
root=Path(__file__).parent
engine=root.parent/'trimul_sm90_parity_20260917/engine'
record=engine/'docs/records/transition-cuda-variants-20260918'
rows=[];text=['# Measured results\n','node02 H100; BF16 activations/weights, FP32 LN affine; residual ON; expansion 4. Static compile + manual CUDA Graph. Milliseconds, median of two harness repeats. No optimizer. L384 seed-sweep configs are reused at L768.\n']
for mode in ('inference','training'):
 text+=['## '+mode+' module\n','| D | L | Triton split | Triton streamed | CUDA streamed | Triton full | CUDA full |','|---:|---:|---:|---:|---:|---:|---:|']
 for d in (128,256,384,512):
  for length in (384,768):
   p=root/f'module-D{d}-L{length}.json';data=json.loads(p.read_text());r={}
   for arm in ('triton_split','triton:streamed_k','cuda:streamed_k','triton:full_k','cuda:full_k'):
    found=[x for x in data['rows'] if x['mode']==mode and x['arm']==arm];assert len(found)==2,(p,mode,arm,len(found))
    assert all(x['compiled'] and x['cudagraph']=='manual' for x in found)
    r[arm]=statistics.median(x['value'] for x in found)
   rows.append(dict(D=d,L=length,mode=mode,**r));text.append('| '+f'{d} | {length} | '+' | '.join(f'{x:.4f}' for x in r.values())+' |')
 text+=['']
text+=['## Direction timings and peak allocated memory\n','Separate fullgraph `autograd.grad` fixture with `donated_buffer=False` for retained-graph backward; backward is directly measured, not total minus inference. Peak extra allocation measures an eager invocation of the compiled step, including backward temporaries and returned gradients; it does not include graph private pools and excludes preexisting fixture allocations and reserved allocator memory.\n','| D | L | Arm | Training fwd | Bwd | Full step | Inference | Peak extra MiB |','|---:|---:|---|---:|---:|---:|---:|---:|']
for d in (128,256,384,512):
 for length in (384,768):
  data=json.loads((root/f'breakdown-D{d}-L{length}.json').read_text());assert len(data['rows'])==5
  for r in data['rows']:
   text.append('| '+f"{d} | {length} | {r['arm']} | "+' | '.join(f'{r[k]:.4f}' for k in ('training_fwd_ms','bwd_ms','training_total_ms','inference_fwd_ms'))+f" | {r['training_peak_extra_bytes']/2**20:.1f} |")
text+=['\n## Isolated common backward operations\n','Independent graph timings; do not add these to infer module latency.\n','| D | L | dh | dWs | dWab | dxn | weight cat |','|---:|---:|---:|---:|---:|---:|---:|']
for d in (128,256,384,512):
 for length in (384,768):
  r=json.loads((root/f'breakdown-D{d}-L{length}.json').read_text())['isolated_common_backward_ms']
  text.append('| '+f'{d} | {length} | '+' | '.join(f'{r[k]:.4f}' for k in ('dh','dWs','dWab','dxn','weight_cat'))+' |')
text+=['\n## Selected native machine instructions\n','Static SASS instruction-site counts for the selected direction, not executed instruction counts or traffic bytes. `UTMALDG` and `HGMMA` confirm explicit TMA/WGMMA. `LDL`/`STL` show remaining local-memory traffic in some wide configurations.\n','| Variant | D | Direction | UTMALDG | HGMMA | LDL | STL |','|---|---:|---|---:|---:|---:|---:|']
for r in json.loads((root/'sass-audit.json').read_text()):
 marker='ILb0E' if r['direction']=='forward' else 'ILb1E'
 k=next(k for k in r['kernels'] if marker in k['symbol'])
 text.append('| '+f"{r['variant']} | {r['D']} | {r['direction']} | "+' | '.join(str(k[a]) for a in ('tma_load','wgmma','local_load','local_store'))+' |')
text+=['\n## Validation\n','18 GPU checks passed, including both complete schedules, output and all six gradients, tail rows, residual identity, static compile/CUDA Graph and explicit module wiring. Unfiltered Compute Sanitizer: 10 numerical/module checks passed, 0 errors. Related import/dispatch checks: 47 passed, 1 skipped. Full logs and source hashes accompany this record.\n']
(record/'RESULTS.md').write_text('\n'.join(text)+'\n');(record/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
for pattern in ('pytest-final.log','pytest-final.exit','memcheck-final.log','memcheck-final.exit','integration-checks.log','selected-*-D*.txt','retry-build.log','tune-*-D*.json','module-D*-L*.json','breakdown-D*-L*.json','builds.json','sass-audit.json','norm-selections.json','validation-sources.json','builds-supplement.json','builds-supplement-fast.json','builds-large.json'):
 for p in root.glob(pattern):shutil.copy2(p,record/p.name)
files=['src/miniworld_engine/modules/transition/module.py','src/miniworld_engine/kernels/transition/cuda/variants.py','src/miniworld_engine/kernels/transition/cuda/transition_variants_kernel.cu','src/miniworld_engine/kernels/transition/cuda/transition_variant_norm.cu','src/miniworld_engine/kernels/transition/triton/segmented_b2b.py','src/miniworld_engine/kernels/transition/triton/wide_b2b.py','src/miniworld_engine/kernels/transition/triton/residual.py','src/miniworld_engine/kernels/transition/triton/main.py','src/miniworld_engine/kernels/transition/triton/fused.py','tests/numerics/test_transition_cuda_variants_gpu.py']
manifest={}
for name in files:
 p=engine/name;manifest[name]=hashlib.sha256(p.read_bytes()).hexdigest();dest=record/'sources'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
(record/'sources.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(rows,indent=2))

# Keep exact runners alongside the recorded measurements, without compiled artifacts.
for p in root.iterdir():
 if p.suffix in ('.py','.sh'):
  dest=record/'repro'/p.name;dest.parent.mkdir(exist_ok=True);shutil.copy2(p,dest)
validation=json.loads((root/'validation-sources.json').read_text())
for f,h in validation['sha256'].items():assert manifest[f]==h,(f,'changed since final validation')
assert (root/'pytest-final.exit').read_text().strip()=='0'
assert (root/'memcheck-final.exit').read_text().strip()=='0'
assert 'ERROR SUMMARY: 0 errors' in (root/'memcheck-final.log').read_text()
