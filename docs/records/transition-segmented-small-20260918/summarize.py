import hashlib,json,shutil,statistics
from pathlib import Path
root=Path(__file__).resolve().parent;engine=root.parent/'trimul_sm90_parity_20260917/engine'
record=engine/'docs/records/transition-segmented-small-20260918';record.mkdir(parents=True,exist_ok=True)
summary={'node':'node02','precision':'BF16 activations/weights, FP32 LayerNorm affine','default_rows':[],'candidate_rows':[],'search':{}}
for kind,pattern in [('default_rows','default-L*.json'),('candidate_rows','bench-L*.json')]:
 for p in sorted(root.glob(pattern)):
  j=json.loads(p.read_text());assert len(j['rows']) in (8,16)
  for mode in ['inference','training']:
   rows=[r for r in j['rows'] if r['mode']==mode]
   ms={k:statistics.median(r['value'] for r in rows if r['backend']==k) for k in sorted({r['backend'] for r in rows})}
   summary[kind].append({'D':j['width'],'L':j['length'],'mode':mode,'ms':ms})
for d in [128,256]:
 j=json.loads((root/f'tune-D{d}.json').read_text());summary['search'][str(d)]={'attempts':len(j['rows']),'valid':sum(r['status']=='ok' for r in j['rows']),'selected':j['selected']}
(record/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
lines=['# Default Triton module performance','', 'Milliseconds, median of two captures in reverse arm order. Static compile + manual CUDA Graph, B1 pair L384/768, n4, BF16 and FP32 affine, nonzero squeeze. Training = forward+backward, no optimizer.','', '| D | L | Mode | Previous split | New segmented default | Speedup |','|---|---|---|---:|---:|---:|']
for r in summary['default_rows']:
 ms=r['ms'];lines.append(f"| {r['D']} | {r['L']} | {r['mode']} | {ms['split']:.4f} | {ms['b2b']:.4f} | {ms['split']/ms['b2b']:.3f}x |")
lines+=['','## Full-output b2b and both segmented variants','','| D | L | Mode | split | full-output b2b | segmented LN separate | segmented LN fused |','|---|---|---|---:|---:|---:|---:|']
for r in summary['candidate_rows']:
 ms=r['ms'];lines.append(f"| {r['D']} | {r['L']} | {r['mode']} | {ms['split']:.4f} | {ms['old_b2b']:.4f} | {ms['segmented_separate']:.4f} | {ms['segmented_fused']:.4f} |")
(record/'timings.md').write_text('\n'.join(lines)+'\n')
for p in root.iterdir():
 if p.is_file() and p.suffix in ('.py','.sh','.json','.csv','.log'):shutil.copy2(p,record/p.name)
paths=['kernels/transition/triton/segmented_b2b.py','kernels/transition/triton/segmented_residual.py','kernels/transition/triton/wide_b2b.py','kernels/transition/triton/b2b_residual.py','kernels/transition/triton/residual.py','kernels/transition/whole_op.py','settings.py','modules/transition/module.py','autotune/configs/grid/transition_segmented_b2b_triton.csv']
sha={}
for rel in paths:
 src=engine/'src/miniworld_engine'/rel;dst=record/'sources'/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst);sha[rel]=hashlib.sha256(src.read_bytes()).hexdigest()
(record/'sources.json').write_text(json.dumps(sha,indent=2)+'\n')
print(record)
