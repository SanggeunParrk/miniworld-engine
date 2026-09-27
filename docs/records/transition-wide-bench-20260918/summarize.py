"""Summarize saved node02 measurements; standard library only."""
import hashlib
import json
import shutil
import statistics
from pathlib import Path

root = Path(__file__).resolve().parents[2]
run = Path(__file__).resolve().parent
engine = root / 'runs/trimul_sm90_parity_20260917/engine'
record = engine / 'docs/records/transition-wide-bench-20260918'
record.mkdir(parents=True, exist_ok=True)
rows = []
all_measurements = []
for width in (384, 512):
    for length in (384, 768):
        path = run / f'bench-L{length}-D{width}-auto.json'
        data = json.loads(path.read_text())
        assert data['node'].startswith('node02'), data['node']
        assert len(data['rows']) == 8, (path, len(data['rows']))
        all_measurements.extend(data['rows'])
        for mode in ('inference', 'training'):
            times = {}
            for backend in ('triton', 'auto'):
                selected = [r for r in data['rows'] if r['mode'] == mode and r['backend'] == backend]
                assert sorted(r['repeat'] for r in selected) == [0, 1]
                assert all(r['compiled_graphs'] == 1 and r['cudagraph'] == 'manual' for r in selected)
                for r in selected:
                    validation = json.loads(r['execution_validation'])
                    for field in validation['graph_replay'].values():
                        assert field['relative_frobenius'] <= field['limit']
                times[backend] = [r['value'] for r in selected]
            t, h = [statistics.median(times[b]) for b in ('triton', 'auto')]
            rows.append(dict(L=length, D=width, mode=mode, triton_ms=t, h100_cute_ms=h,
                             speedup=t/h, samples_ms=times))
        shutil.copy2(path, record / path.name)
        shutil.copy2(run / f'L{length}-D{width}.log', record / f'L{length}-D{width}.log')

summary = dict(date='2026-09-18', allocation=13274, node='node02', rows=rows,
               aggregation='median of two captures with reversed backend order',
               scope='current auto versus forced Triton; no wide-D b2b implementation',
               output_rel_frob_max=max(r['output_rel_frob'] for r in all_measurements),
               input_grad_rel_frob_max=max(r['grad_rel_frob'] or 0 for r in all_measurements),
               replay_checks=len(all_measurements))
(record / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
table = ['| L | D | Mode | Triton ms | H100 CuTe ms | Triton / H100 |',
         '|---:|---:|---|---:|---:|---:|']
table += [f"| {r['L']} | {r['D']} | {r['mode']} | {r['triton_ms']:.4f} | {r['h100_cute_ms']:.4f} | {r['speedup']:.3f}x |" for r in rows]
(record / 'timings.md').write_text('\n'.join(table) + '\n')
sources = ['src/miniworld_engine/kernels/transition/hopper.py',
           'src/miniworld_engine/kernels/transition/cuda/transition_b2b_kernel.cu',
           'src/miniworld_engine/autotune/hopper_cuda_config.py',
           'src/miniworld_engine/kernels/transition/cute/gemm_transition_swiglu.py',
           'src/miniworld_engine/kernels/transition/cute/squeeze_residual.py',
           'src/miniworld_engine/kernels/transition/cute/fused.py',
           'src/miniworld_engine/kernels/transition/triton/fused.py',
           'src/miniworld_engine/settings.py', 'benchmarks/runners/bench.py']
(record / 'sources.json').write_text(json.dumps({p: hashlib.sha256((engine/p).read_bytes()).hexdigest() for p in sources}, indent=2) + '\n')
for name in ('measure.py', 'env.sh', 'summarize.py'):
    shutil.copy2(run / name, record / name)
print('\n'.join(table))
print(json.dumps({k: v for k, v in summary.items() if k != 'rows'}, indent=2))
