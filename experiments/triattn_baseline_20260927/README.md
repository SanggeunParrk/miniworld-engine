# Current TriangleAttention H100 baseline — 2026-09-27

No production source, configuration or dispatch was changed. These are fresh
measurements of the installed current engine, before further optimization.

B=1, pair width=128, 4 heads, head dimension=32, BF16 activations/linear
weights, FP32 LayerNorm affine; starting and ending measured independently.
Training uses p_drop=0.25 with live dropout RNG and approximately 10% masked
tokens. Nonzero randomized output weights ensure a real update and nonzero
input/all-parameter gradients. Full module includes LN, projections, attention,
gating/output projection, residual/dropout and all gradients.

`torch.compile(fullgraph=True, dynamic=False, triton.cudagraphs=False)` followed
by explicit CUDA Graph capture. GPU0 measures L384; GPU1 measures L768.
Both are full H100 80GB HBM3, PyTorch 2.10.0+cu128. Three alternating blocks of
30 samples give 90 samples per timing category; compilation/warmup is excluded.

| L | Direction | Training forward ms | Backward ms | Full F+B ms | Eval inference ms |
|---|---|---:|---:|---:|---:|
| 384 | starting | 0.369 | 0.882 | 1.243 | 0.394 |
| 384 | ending | 0.395 | 0.870 | 1.260 | 0.416 |
| 768 | starting | 1.782 | 4.755 | 6.529 | 1.857 |
| 768 | ending | 1.883 | 4.697 | 6.589 | 1.970 |

Forward and backward are actual separate graph/event measurements, not a
subtraction from inference. Backward replays are preceded by the captured
training forward to refresh saved activations; that refresh is outside its
measured event interval. Full F+B is separately captured. The three medians
need not sum exactly because cache/launch conditions differ. Inference has
no dropout or backward saves and can select a different path, so it is not
the training-forward baseline.

## Dispatch and bottlenecks

All four training traces contain `qg_attention_fused<false>`,
`grouped_dkdv<8>`, `dq_tma`, `projection_ln_residual_tma<64>`,
`gate_delta_tma` and `grouped_wgrad`. Thus the active implementation uses
fused native Q+gate attention forward and the installed CUDA backward paths.
The generic module backend label `TRITON` does not describe these leaf kernels.
The QKV-forward implementation is restricted to L1024 and is not these paths.

Backward is approximately 70–73% of complete F+B. A single profiled L768
starting replay attributes about 2.048 ms to grouped dK/dV, 0.990 ms to dQ,
0.369 ms to projection/LN/residual, 0.313 ms to gate/delta, and 0.300 ms to
bias reduction. Grouped dK/dV and dQ together account for about 48% of the
trace's summed kernel duration. These are diagnostic activity-profile timings,
not hardware counters, SOL estimates or separately optimized stage benchmarks.
The first optimization targets should therefore be dK/dV and dQ, evaluating
fusion/tensor lifetimes together with bias reduction before launch tuning.

## Validation and provenance

Outputs and all gradients were finite and nonzero on initial execution.
Successive captured training replays changed output, confirming live dropout
RNG. Deterministic p_drop=0 changed-input/weight/dy/mask replay exactly matched
a fresh compiled invocation (relative-L2=0 for output and every gradient),
for both directions at both lengths. Original inputs/weights were restored
before inference timing. This baseline run is not a new independent numerical
reference or sanitizer qualification of the implementation.

Source/binary SHA256s and the exact script hash are in each baseline JSON;
all locally present source files matched the recorded remote hashes.
`experiments/triattn_baseline_20260927/evidence.json` hashes the result files.
The source script is `experiments/triattn_baseline_20260927/bench.py`;
`summary.csv` in that directory contains the precise timing values.

Raw results and traces:
`.bench/vast-20260927/results/triattn-baseline-20260927/`.
Remote counterpart: `/workspace/vast-results/triattn-baseline-20260927/`.
Final managed execution handles: 53429 and 10261; both finished successfully.
Logs: `/workspace/vast-results/triattn-baseline-v2-L384.log` and
`triattn-baseline-v2-L768.log`.

The initial harness attempted retained backward on a donated-buffer compiled
function and was rejected before timing. It was corrected to capture backward
once and replay the hardware graph after refreshing forward saves; compiler
buffer donation remains enabled. Original failure logs are retained as
`triattn-baseline-L384.log` and `triattn-baseline-L768.log`.

Run through `scripts/vast-sync.sh run 0|1 python
experiments/triattn_baseline_20260927/bench.py --length 384|768` after the explicit
source push, then `scripts/vast-sync.sh pull`. Respect the GPU/source locks.
