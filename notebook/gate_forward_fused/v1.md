# Gate forward -> gate_elem_quack_fused (v1)

## Target
trimul TRAINING gate forward, single-dir (`v6_training_merged_sm100.py`) and bidir
(`bidir_training_sm100.py`). Replace `gate_elem_triton` (cuBLAS glogit GEMM +
`_gate_mul_kernel` sigmoid+mul + glogit HBM round-trip) with `gate_elem_quack_fused`:
`y = sigmoid(x_n@Wg) ⊙ proj` in ONE quack `gemm_act` launch (act(A@B)⊙C epilogue, C=proj).

## Hypothesis
Fusing GEMM+sigmoid+proj-mul into one launch removes the `_gate_mul_kernel` (~156us)
and the glogit (M,N) HBM write+read. Backward needs `gate=sigmoid(glogit)`; the fused
op has no free `gate`, so it stores the PREACT (glogit) and the bwd elementwise kernel
recomputes `gate=sigmoid(preact)` in-kernel (zero extra HBM: preact replaces the gate save).

## sm100 correctness bug found + fixed (impl-level, no algo change)
Initial swap gave y cos = 0.22 on B200. Root cause: `_gate_mul_patch.apply()` installed
the `act(A@B)⊙C` epilogue only on `GemmActSm90`. B200 uses `GemmActSm100`, which does
NOT subclass GemmActSm90 (MRO branches through GemmSm100), so it fell back to the STOCK
additive-C epilogue: y = sigmoid(glogit+proj), preact = glogit+proj. Fix: shadow
`epi_visit_subtile` on every concrete `GemmActSm{90,100,120}` present (mirrors _bdll_patch).
After fix: fused y cos 0.99999, preact=pure glogit, sigmoid(preact) vs gate cos = 1.0.

## Validation (QUACK_CACHE_ENABLED=0, B200, bf16, D=128, B=1)
- Isolated gate: fused y cos vs fp32 ref = 0.9999986 (== triton 0.9999985).
- Full-module grads vs fp32 pytorch ref (verify_v12 / verify_bidir_v3), L=384/768/1024:
  single worst grad cos = 0.999983-0.999984 (baseline 0.999982-0.999983); y fwd 0.999989.
  bidir  worst grad cos = 0.999982-0.999984 (baseline 0.999981-0.999983); y fwd 0.999989.
- CUDA-graph replay grads bit-exact vs eager: worst_cos = 1.000000 (single + bidir, all L).

## Performance (CUDA-graph replay, do_bench median rep=100)
Fair concurrent A/B on separate idle B200s @ L=1024 (baseline=mw-lnbwd @1626738):
| path   | baseline | fused  | speedup |
|--------|----------|--------|---------|
| single | 3.7026ms | 3.5799ms | 1.034x |
| bidir  | 6.1210ms | 5.9963ms | 1.021x |
(Non-concurrent single-GPU samples had ~10% cluster-contention noise; concurrent A/B is the fair number.)

## Result
CONFIRMED. gate_elem_quack_fused is correct on sm100 after the GemmActSm100 patch fix; the
fused gate removes _gate_mul + the glogit round-trip; single 3.4%, bidir 2.1% graph-time win
@L=1024; all accuracy/grad/bit-exact gates pass.
