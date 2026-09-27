import hashlib,json,shutil,statistics
from pathlib import Path
root=Path(__file__).resolve().parent
engine=root.parent/'trimul_sm90_parity_20260917/engine'
record=engine/'docs/records/transition-triton-unified-forward-20260918'
record.mkdir(parents=True,exist_ok=True)
rows=[]
for p in sorted(root.glob('*bench-L*.json')):
 j=json.loads(p.read_text())
 for mode in ('inference','training'):
  subset=[r for r in j['rows'] if r['mode']==mode]
  values={arm:statistics.median(r['value'] for r in subset if r['backend']==arm) for arm in sorted({r['backend'] for r in subset})}
  rows.append(dict(D=j['width'],L=j['length'],mode=mode,ms=values,repeats=2))
summary={'node':'node02','rows':rows,'dispatch':{'default_triton_b2b':'SM90 BF16 n4 D128 M>=16384','split':'D256/384/512 and unsupported shapes'},'new_layout_attempts':{},'production_tuning_attempts':{}}
for variant in ('tune','concat','segmented'):
 for d in (384,512):
  p=root/f'{variant}-D{d}.json';j=json.loads(p.read_text());ok=[r for r in j['rows'] if r['status']=='ok']
  summary['new_layout_attempts'][f'{variant}-D{d}']={'attempts':len(j['rows']),'valid':len(ok),'best':min(ok,key=lambda r:r['ms'])}
for d in (128,256):
 j=json.loads((root/f'production-D{d}.json').read_text())
 summary['production_tuning_attempts'][str(d)]={'attempts':len(j['rows']),'valid':sum(r['status']=='ok' for r in j['rows'])}
(record/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
lines=['# Matched module timings','', 'Milliseconds; median of two captures in reverse arm order. BF16 activations/weights, FP32 LN affine; n4 B1, nonzero squeeze, static torch.compile (one graph observed), manual CUDA Graph. Training is forward+backward, no optimizer. General Transition has no dropout.','', '| D | L | Mode | split | full-output b2b | segmented b2b |','|---|---|---|---:|---:|---:|']
for row in rows:
 v=row['ms'];lines.append(f"| {row['D']} | {row['L']} | {row['mode']} | {v['split']:.4f} | {v.get('b2b',v.get('old_b2b')):.4f} | "+(f"{v['segmented']:.4f}" if 'segmented' in v else '—')+' |')
(record/'timings.md').write_text('\n'.join(lines)+'\n')
# All measurements plus exact scripts; exclude caches and verbose tuning logs.
for p in root.iterdir():
 if p.is_file() and p.suffix in ('.py','.sh','.json','.csv'):
  shutil.copy2(p,record/p.name)
for name in ('registry.log','final-dispatch.log','bench-small.log','validate-wide-final.log'):
 if (root/name).exists():shutil.copy2(root/name,record/name)
paths=['kernels/transition/triton/b2b_residual.py','kernels/transition/triton/wide_b2b.py','kernels/transition/triton/segmented_b2b.py','kernels/transition/triton/residual.py','kernels/transition/whole_op.py','modules/transition/module.py','settings.py','autotune/configs/grid/transition_b2b_residual_triton.csv']
manifest={}
for rel in paths:
 src=engine/'src/miniworld_engine'/rel;dest=record/'sources'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest);manifest[rel]=hashlib.sha256(src.read_bytes()).hexdigest()
(record/'sources.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(record)
print(summary['production_tuning_attempts'])
