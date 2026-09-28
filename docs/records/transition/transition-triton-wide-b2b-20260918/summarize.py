"""Archive final wide-b2b experiments without importing GPU libraries."""
import csv
import hashlib
import json
import shutil
import statistics
from pathlib import Path

run=Path(__file__).resolve().parent
root=run.parents[1]
engine=root/'runs/trimul_sm90_parity_20260917/engine'
record=engine/'docs/records/transition-triton-wide-b2b-20260918'
record.mkdir(parents=True,exist_ok=True)
rows=[];checks=[];tuning={}
for d in (384,512):
    candidates=[]
    for prefix in ('tune','tune-more','tune-refine','tune-ln','tune-retry'):
        candidates+=json.loads((run/f'{prefix}-D{d}.json').read_text())['rows']
    tuning[d]=dict(measurements=len(candidates),valid=sum(r['status']=='ok' for r in candidates),
                   unique_configs=len({tuple(sorted(r['config'].items())) for r in candidates}))
    for l in (384,768):
        raw=json.loads((run/f'bench-L{l}-D{d}.json').read_text())
        assert raw['node'].startswith('node02')
        assert len(raw['rows'])==16
        for mode in ('inference','training'):
            result=dict(D=d,L=l,mode=mode,samples_ms={})
            for arm in ('triton','auto','b2b','b2b_ln'):
                selected=[r for r in raw['rows'] if r['mode']==mode and r['backend']==arm]
                assert sorted(r['repeat'] for r in selected)==[0,1]
                assert all(r['compiled_graphs']==1 for r in selected)
                times=[r['value'] for r in selected]
                result[arm]=statistics.median(times);result['samples_ms'][arm]=times
                for r in selected:
                    for field in json.loads(r['execution_validation'])['graph_replay'].values():
                        assert field['relative_frobenius']<=field['limit']
                checks+=selected
            rows.append(result)
summary=dict(allocation=13275,node='node02',rows=rows,tuning=tuning,
             validation_rows=len(checks),output_rel_frob_max=max(r['output_rel_frob'] for r in checks),
             input_grad_rel_frob_max=max(r['grad_rel_frob'] or 0 for r in checks))
(record/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
table=['| D | L | Mode | Triton split ms | H100 CuTe ms | New b2b ms | New LN+b2b ms |',
       '|---:|---:|---|---:|---:|---:|---:|']
table += [f"| {r['D']} | {r['L']} | {r['mode']} | {r['triton']:.3f} | {r['auto']:.3f} | {r['b2b']:.3f} | {r['b2b_ln']:.3f} |" for r in rows]
(record/'timings.md').write_text('\n'.join(table)+'\n')
for pattern in ('*.py','*.sh','*.json','*.csv','*.log'):
    for p in run.glob(pattern):
        shutil.copy2(p,record/p.name)
sources=['src/miniworld_engine/kernels/transition/triton/wide_b2b.py',
         'tests/numerics/test_transition_wide_b2b_gpu.py',
         'src/miniworld_engine/kernels/transition/triton/residual.py',
         'src/miniworld_engine/kernels/transition/triton/fused.py',
         'src/miniworld_engine/kernels/transition/hopper.py']
(record/'sources.json').write_text(json.dumps({p:hashlib.sha256((engine/p).read_bytes()).hexdigest() for p in sources},indent=2)+'\n')
print('\n'.join(table));print(json.dumps(tuning))
