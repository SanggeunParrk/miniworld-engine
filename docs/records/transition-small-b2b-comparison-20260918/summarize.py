import csv,hashlib,json,shutil,statistics
from pathlib import Path
run=Path(__file__).resolve().parent;root=run.parents[1]
engine=root/'runs/trimul_sm90_parity_20260917/engine'
record=engine/'docs/records/transition-small-b2b-comparison-20260918'
record.mkdir(parents=True,exist_ok=True)
rows=[];checks=0
for d in (128,256):
 for length in (384,768):
  raw=json.loads((run/f'bench-L{length}-D{d}.json').read_text())
  assert raw['node'].startswith('node02') and len(raw['rows'])==12
  for mode in ('inference','training'):
   r=dict(D=d,L=length,mode=mode,samples={})
   for arm in ('split','triton_b2b','cuda'):
    selected=[v for v in raw['rows'] if v['backend']==arm and v['mode']==mode]
    assert sorted(v['repeat'] for v in selected)==[0,1]
    assert all(v['compiled_graphs']==1 for v in selected)
    for v in selected:
     for field in json.loads(v['execution_validation'])['graph_replay'].values():
      assert field['relative_frobenius']<=field['limit']
     checks+=1
    r['samples'][arm]=[v['value'] for v in selected]
    r[arm]=statistics.median(r['samples'][arm])
   r['cuda_over_triton_speedup']=r['triton_b2b']/r['cuda'];rows.append(r)
metrics={}
keys=['gpu__time_duration.sum','sm__throughput.avg.pct_of_peak_sustained_elapsed',
      'dram__throughput.avg.pct_of_peak_sustained_elapsed',
      'sm__warps_active.avg.pct_of_peak_sustained_active','lts__throughput.avg.pct_of_peak_sustained_elapsed',
      'launch__registers_per_thread','launch__shared_mem_per_block_allocated']
for d in (128,256):
 for backend in ('triton','cuda'):
  with (run/f'ncu-raw-{backend}-D{d}.csv').open() as f:raw=list(csv.DictReader(f))
  kernel=raw[-1];assert kernel['ID']=='0'
  metrics[f'{backend}-D{d}']={k:kernel[k] for k in keys}
summary=dict(node='node02',allocation=13276,rows=rows,graph_validation_rows=checks,ncu=metrics)
(record/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
table=['| D | L | Mode | Triton split ms | Triton b2b ms | CUDA b2b ms | CUDA speedup over Triton b2b |',
       '|---:|---:|---|---:|---:|---:|---:|']
table += [f"| {r['D']} | {r['L']} | {r['mode']} | {r['split']:.4f} | {r['triton_b2b']:.4f} | {r['cuda']:.4f} | {r['cuda_over_triton_speedup']:.3f}x |" for r in rows]
(record/'timings.md').write_text('\n'.join(table)+'\n')
for pattern in ('*.py','*.sh','*.json','*.csv','*.log'):
 for p in run.glob(pattern):shutil.copy2(p,record/p.name)
sources=['src/miniworld_engine/kernels/transition/triton/wide_b2b.py',
         'src/miniworld_engine/kernels/transition/cuda/transition_b2b_kernel.cu',
         'src/miniworld_engine/kernels/transition/triton/fused.py',
         'src/miniworld_engine/kernels/transition/hopper.py']
(record/'sources.json').write_text(json.dumps({p:hashlib.sha256((engine/p).read_bytes()).hexdigest() for p in sources},indent=2)+'\n')
print('\n'.join(table));print(json.dumps(metrics,indent=2))
