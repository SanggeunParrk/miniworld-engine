import json,statistics,hashlib,shutil
from pathlib import Path
root=Path(__file__).parent;base=root.parents[1];engine=base/'runs/trimul_sm90_parity_20260917/engine'
record=engine/'docs/records/trimul-pytorch-compare-20260919';record.mkdir(parents=True,exist_ok=True)
rows=[];body=['# Bidirectional TriMul: PyTorch / current Triton / H100','', '2026-09-19, node02 H100 80GB, B1, D=hidden=128, BF16 activations/linear weights and FP32 norm affine. Official bidirectional module benchmark, identical nonzero parameters, input and token mask. All arms use static compile and manual CUDA Graph. No optimizer, loading or communication. This is a module comparison, not whole-model training.', '', 'The H100 arm explicitly selects front/f567/dual_bwd/out_ln_bwd over the Triton algorithm, matching the weekly closeout. It is not the untouched auto default. The experimental mapped B4 is excluded. In inference only forward-reachable overrides run; backward options do not add work.', '']
for mode in ('training','inference'):
 body+=['## '+mode, '', 'Training uses dropout=0.25 with fresh production RNG on every timed replay. Inference uses eval/dropout=0.', '', '| L | PyTorch ms | Triton ms | H100 mix ms | Triton/PyTorch speedup | H100/PyTorch speedup | H100/Triton speedup |', '|---:|---:|---:|---:|---:|---:|---:|']
 for length in (384,768):
  samples={a:[] for a in ('pytorch','triton','h100')};provenance=None
  for rep in (0,1):
   p=root/(f'current-L{length}-r{rep}.json' if mode=='training' else f'inference-L{length}-r{rep}.json');d=json.loads(p.read_text())
   if mode=='training':
    assert d['passed'];b=d['builds'][0]
    if provenance is None:provenance=b['provenance']
    assert provenance==b['provenance']
    for arm in samples:
     c=b['checks'][arm];v=b['fixture_results'][arm]
     assert c['all_gradients_present']==11 and c['cuda_rng_state_advanced'] and not c['timed_rng_reseed']
     assert v['compiled'] and v['compiled_graphs']==1 and v['input_dtype']=='bfloat16' and v['parameter_dtype']=='bfloat16+float32'
     assert v['output_rel_frob']<.02 and v['grad_rel_frob']<.02
     samples[arm]+=b['samples_ms'][arm]
   else:
    assert len(d['rows'])==3
    for arm in samples:
     v=d['rows'][arm];assert v['compiled'] and v['compiled_graphs']==1 and v['output_rel_frob']<.02
     samples[arm].append(v['value'])
  t={k:statistics.median(v) for k,v in samples.items()};ratios=[t['pytorch']/t['triton'],t['pytorch']/t['h100'],t['triton']/t['h100']]
  rows.append(dict(mode=mode,L=length,ms=t,ratios=dict(triton_vs_pytorch=ratios[0],h100_vs_pytorch=ratios[1],h100_vs_triton=ratios[2]),samples=samples))
  body.append('| '+str(length)+' | '+' | '.join(f'{t[a]:.6f}' for a in samples)+' | '+' | '.join(f'{v:.3f}x' for v in ratios)+' |')
 body+=['']
body+=['## Measurement and validation', '', '- Training: two independent processes/captures per shape, 12 rotated backend rounds in each, median of 24 graph timing samples. Reverse capture order in the second process. Inference: two independent captures with reversed backend order.', '- PyTorch is also compiled BF16, not an eager or FP32 baseline. FP32 norm parameters are retained across all backends. Every observed compiled forward has one graph.', '- Training fixture input/upstream-gradient/mask/parameter hashes match across backends and repetitions. The official paired-dropout FP32-reference checks run before timing; all output/input-gradient relative errors are below 0.02.', '- All 11 gradients are present and finite. RNG reset reproduces each backend output/gradients; leaving RNG advancing changes output and dx. Fresh-gradient overwrite and stable graph buffers are checked. Timed execution never resets RNG or substitutes a fixed mask.', '- Cross-backend graph RNG masks need not match for PyTorch versus custom ops. Their numerical reference comparison uses the official paired-dropout check; graph reset checks test each backend independently. H100 versus Triton additionally passes paired output/all-gradient comparison.', '- Measured F2/F567/B9+B10 and native B4 configs are pinned to the prior closeout manifests. Other cache misses retain repository bounded tuning; this is not an exhaustive config search. Source hashes and exact runners are included.', '- Some old Inductor cache entries could not be loaded and were recompiled during warmup. All accepted rows have positive compile evidence; compilation is outside graph replay timings.', '', 'Raw data, scripts and logs: '+str(root)+'.']
manifest=json.loads((root/'source-manifest.json').read_text())
for name,h in manifest['files'].items():assert hashlib.sha256((engine/name).read_bytes()).hexdigest()==h,name
(root/'summary.json').write_text(json.dumps(rows,indent=2)+'\n');(root/'README.md').write_text('\n'.join(body)+'\n')
for p in root.iterdir():
 if p.is_file() and p.suffix in ('.json','.py','.sh','.md'):shutil.copy2(p,record/p.name)
print(json.dumps([{k:v for k,v in r.items() if k!='samples'} for r in rows],indent=2))
