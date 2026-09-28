"""Require all eight qualified records before publishing the latency table."""
import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('results', type=Path)
p.add_argument('--d128-fixed', type=Path, help='Explicit replacement records for the D128 input-slot synchronization fix')
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
records = {}
for path in a.results.rglob('D*-L*.json'):
    r = json.loads(path.read_text())
    key = (r['L'], r['D'])
    assert key not in records, f'duplicate result for {key}'
    assert r['complete'] and r.get('times'), f'incomplete benchmark: {path}'
    assert r['dispatch_entry'].endswith('sm90a')
    kernels = r['native_cuda_kernel_names']
    expected = 'transition_fwd_fused' if r['D'] == 128 else ('wide_d256_fwd_kernel' if r['D'] == 256 else 'wide_lnsg_kernel')
    assert any(expected in k for k in kernels), (key, expected, kernels)
    records[key] = r
if a.d128_fixed:
    fixed = list(a.d128_fixed.rglob('D128-L*.json'))
    assert len(fixed) == 2
    for path in fixed:
        r = json.loads(path.read_text())
        assert r['complete'] and r['D'] == 128 and r.get('times')
        assert 'transition_bwd_fused' in r['native_cuda_kernel_names']
        records[(r['L'], 128)] = r
assert set(records) == {(l, d) for l in (384, 768) for d in (128, 256, 384, 512)}, records.keys()
lines = ['# Transition: PyTorch compiled versus installed native CUDA', '',
         'H100 SXM, BF16, B1, input `[1,L,L,D]`, expansion 4. Full fresh forward +',
         'backward, input and all five parameter gradients, residual included, no dropout.',
         'Both actual modules are compiled with fullgraph=True and measured in manual CUDA graphs.',
         '75 interleaved CUDA-event samples per implementation, medians below.', '',
         '| L | D | PyTorch (ms) | Native (ms) | Speedup |',
         '| ---: | ---: | ---: | ---: | ---: |']
for (l, d), r in sorted(records.items()):
    t = r['times']
    ref, native = t['pytorch']['median_ms'], t['native']['median_ms']
    lines.append(f'| {l} | {d} | {ref:.3f} | {native:.3f} | {ref/native:.2f}x |')
lines += ['', '## Optional saved activation', '',
          '`MINIWORLD_TRANSITION_WIDE_SAVE_H=1` reuses the forward SwiGLU activation at',
          'D384/512. It is already implemented and remains opt-in. Additional retained',
          'activation is per layer; activation checkpointing/layer count affect the total.', '',
          '| L | D | Default F+B (ms) | Save-h F+B (ms) | Gain | Retained h (MiB/layer) |',
          '| ---: | ---: | ---: | ---: | ---: | ---: |']
for (l, d), r in sorted(records.items()):
    if d < 384:
        continue
    native, candidate = [r['times'][k]['median_ms'] for k in ('native', 'save_h')]
    lines.append(f'| {l} | {d} | {native:.3f} | {candidate:.3f} | {(1-candidate/native)*100:.1f}% | {r["retained_h_bytes_per_layer"]/2**20:.0f} |')
lines += ['', '## Qualification', '',
          '- Native module dispatch observed directly, then CUDA kernel names recorded from compiled graph replay.',
          '- Nonzero randomized weights and nontrivial FP32 LN affine parameters.',
          '- Full output/input/all-parameter gradients pass the existing native-versus-Triton numerical gates.',
          '- Save-h passes the existing native-versus-native gates, without tolerance changes.',
          '- Compiled CUDA graphs agree with eager native execution and fresh compiled PyTorch execution.',
          '- Input, every weight, LN affine and upstream gradient changed in place; all graph replays match fresh calls.',
          '- Source hashes and raw sample lists are in each JSON. Sanitizer/pytest evidence is recorded separately in README.md.',
          '', 'PyTorch is the performance baseline. Engine Triton is only a numerical-contract check.',
          'PyTorch casts LN affine parameters to BF16; native retains FP32 affine math, so the two paths',
          'are numerically close rather than bit-identical. This table does not claim SOL or NCU results.', '']
if a.d128_fixed:
    lines += ['D128 uses the remeasured input-slot barrier fix (v2). The other six cells use',
              'the unchanged wide native kernels (v1). See README.md for the original',
              'race-instrumented gradient failure and the fix qualification.', '']
a.output.write_text('\n'.join(lines))
print(a.output)
