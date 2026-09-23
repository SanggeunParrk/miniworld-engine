"""Generate evidence notes for the qualified single-direction training port."""
import hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parent
repo=R.parents[1]
rows=json.loads((R/'legacy-bench.json').read_text())
assert len(rows)==4
lines=['''# Single-direction H100 TriMul training — 2026-09-23

Qualified production contract: `TriangleMultiplication(implementation="miniworld")`
with `engine_backend="auto"`, BF16, batch 1, D=hidden=128, L384/768, full 132-SM
H100, FP32 affine LN parameters and epsilon 1e-5. Outgoing and incoming both
select the new path. Other widths retain the existing general training route.
This is development-tree wiring after v2.0.0, not an update to the installed
training environment or a pushed release.

## Same-GPU module performance

Node01 H100 80GB; static `torch.compile(fullgraph=True)` + explicit CUDA graphs.
Training = forward + input and all parameter gradients. Forward columns are
**training forward**, including its retained activations, not inference.
Dropout 25% uses the same fixed row mask in all paths; masking, dropout scaling
and residual are included. RNG, compilation, first autotuning, CPU execution and
optimizer are excluded. Ten alternating rounds × 100 replays; median ms.
Legacy H100 means the former CuTe/v6 route, pinned with `implementation="cute"`.
Triton is pinned explicitly. Neither is an inferred historical time or another
version label. Some Triton cache entries were missing: its default heuristic
24-candidate search ran before timing. CuTe's front used its default full-grid
search. This comparison does not claim globally optimal baseline configurations.

| L | Direction | Triton train | Old H100 train | New CUDA train | vs old H100 | vs Triton | Old H100 fwd | New fwd |
|---:|:---|---:|---:|---:|---:|---:|---:|---:|''']
for r in rows:
 t=r['times_ms'];direction='outgoing' if r['outgoing'] else 'incoming'
 lines.append(f"| {r['L']} | {direction} | {t['triton']:.3f} | {t['legacy_cuda']:.3f} | {t['cuda']:.3f} | {t['legacy_cuda']/t['cuda']:.2f}x | {t['triton']/t['cuda']:.2f}x | {t['legacy_cuda_fwd']:.3f} | {t['cuda_fwd']:.3f} |")
lines.append('''
## Computation and retained tensors

- Front: Anthropic-derived native K1 with TMA/WGMMA, hidden=128 (not 256).
  Emits masked left/right in channel-major layout and affine input `x_n`.
- One outgoing `left @ right.T` or incoming `left.T @ right` cuBLAS contraction.
  LN_out sees exactly 128 channels; no duplicated direction or 2D normalization.
- Native K3 fuses output LN, projection, output gate, dropout and residual.
- Forward retains `x_n`, left/right, tri, and a 128-KiB packed front-weight tensor.
  No output-normalization activation, projection or gate arrays are retained.
- B1 recomputes output LN/projection/gate inside one CTA, applies dropout to the
  update gradient, accumulates dWgate/dWout on chip and emits only dGate and dTri
  plus small FP32 parameter partials. A separate small reduction completes dW/LN.
- Two direction-specific cuBLAS gradient contractions emit dLeft/dRight.
- B7 ports the selected bidirectional producer-consumer algorithm to H=128:
  eight 32-channel producers, eight consumers, sixteen groups, eight ring slots,
  four 16-KiB derivative planes per slot. dP/dG feed dX and dW from a bounded
  ring rather than full L² HBM derivative buffers. Input LN backward and residual
  gradient are fused. LN affine derivatives use the exact pre-affine normalized
  value; zero gamma is supported.
- Every forward owns its activations and packed weights. CUDA graph replay reads
  live parameters. Packed weights are reused by backward, never cached across
  optimizer steps. Custom-op outputs do not illegally alias: shared dW storage
  is returned once, then split outside the opaque boundary.

## Configuration search and limits

`selection.json` packages the measured D128/L384/768 schedules. Search per length:
6 K1 tilings/pipelines, 6 K3 schedules, 4 B1 CTA counts and 10 B7 consumer/ring
configurations. Selected K1=(1,64,4,2,2), K3=(2,64,8,1), B1=132 CTAs,
B7=(16 groups,8 consumers,8 ring slots). `tune.json` contains all measurements.
This is a bounded search; no SoL90 or exhaustive-config claim is made.
The standalone initial B1 retains register spills. Moving gate dW to a second
phase reduced register pressure but measured slower (~266 vs ~250 us at L384),
so that variant is retained only as rejected experiment evidence.

The initial width-generic port passed numerics but took 2.11 ms at L384 versus
1.01 ms Triton. It is not the production implementation. `initial-wide-bench.json`
records it; `initial_plan.py` is an isolated diagnostic oracle.
The selected B7 reduced a local initial 1001-us measurement to ~211 us, and the
selected K3 replaces the initial ~217-us output with ~66 us. Kernel microbench
numbers and full-module medians have different measurement loads; do not sum
these figures into the module table.

## Validation and reproducibility

- Job 16280: 11 GPU tests passed: four direction/length output+all-gradient
  comparisons, compile/replay/live-weight/mask/zero-gamma and saved-activation
  ownership for both directions, fully dropped update, strict LN derivatives,
  plus three existing bidirectional regression tests (D64/D128).
- PyTorch compiled BF16 oracle: output relative L2 <0.5%, every gradient <1%.
- Strict input/output LN affine gradients: FP64 oracle using actual incoming
  BF16 derivatives, including zero gamma; relative errors ~2–3e-7, threshold5e-6.
  Diagnostic variants only add derivative stores; production does not emit them.
- Job16281: compute-sanitizer memcheck, both directions/L384/L768, zero errors.
- Job16282: final three-way same-GPU benchmark, `legacy-bench.json`.
- Job16283: compute-sanitizer racecheck, both directions/L384/L768,
  zero hazards, zero errors and zero warnings.
- Earlier jobs16268–16279 preserve initial ports, rejected schedules and fixes;
  they are not substituted for the final evidence.

Run `sbatch verdicts/trimul-single-20260923/final.sbatch` for regressions and
native/Triton timing, `legacy.sbatch` for the three-way table, `sanitize.sbatch`
and `race.sbatch` for sanitizer checks. Sources/builders import only packaged
engine code. Diagnostic probes retain the initial generic implementation.

Anthropic Apache-2.0 attribution is preserved in derived CUDA sources and
`THIRD_PARTY_NOTICES.md`. The work extends that implementation to training.
''')
(R/'README.md').write_text('\n'.join(lines)+'\n')
paths=[repo/'src/miniworld_engine/integrations/trimul_h100.py',repo/'src/miniworld_engine/modules/triangle_multiplication/module.py',repo/'tests/integrations/test_trimul_single_h100_gpu.py']
paths+=list((repo/'src/miniworld_engine/kernels/trimul_inproj/cuda').glob('h100_single*.py'))
paths += [repo/'src/miniworld_engine/kernels/trimul_inproj/cuda'/name for name in ('h100_native.py','_h100_runtime.py')]
for name in ('single_b7','single_output','output','common','common_b7','b1','b7_384'):
 paths+=list((repo/'src/miniworld_engine/kernels/trimul_inproj/cuda/h100_sources'/name).glob('*'))
(R/'sources.json').write_text(json.dumps({str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()},indent=2)+'\n')
