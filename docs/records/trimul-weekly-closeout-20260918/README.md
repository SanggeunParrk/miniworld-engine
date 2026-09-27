# TriMul development closeout and one-week comparison

## Scope

The verified implementation is the current Triton fusion algorithm with optional
SM90 F2/F567/B9+B10 and the registered B4 implementation. The experimental mapped
B4 and its small TMA reductions are **not promoted**. Kernel development stops at
this checkpoint. Running MiniWorld installations and training jobs are unchanged.
Native cache coverage remains partial; this is not a claim that every shape or
configuration has been tuned, or that whole-model convergence has been validated.

## One-week comparison

The baseline is the actual 2026-09-11 source commit
`fb6777417c0a34e383e0984978fec89169c18e28`, restored into an isolated directory.
It is rerun today with the same current runtime as the current source
`be17b1f4`, not inferred by combining old dynamic/no-dropout measurements.
The baseline is explicitly the **Triton backend**. The old legacy H100 auto path
is not measured here.

H100 80GB, B1, D=hidden=128, BF16 activations/linear weights and FP32 norm
parameters, masked input, dropout=0.25, one bidirectional module, static
`dynamic=False` compile plus stochastic manual CUDA graph. Timed work is forward
and backward with fresh gradient buffers captured for overwrite, without an
optimizer, data loading or communication.

|L|Sep11 Triton ms|Current Triton ms|Current H100 mix ms|Triton speedup vs Sep11|H100 speedup vs Sep11|H100 time reduction vs Sep11|
|---:|---:|---:|---:|---:|---:|---:|
|384|1.950904|1.586288|1.535416|1.2299x|1.2706x|21.30%|
|768|7.727528|6.162152|5.932240|1.2540x|1.3026x|23.23%|

Each value is the median of 24 measurements across two independent captures
(12 per capture). Per shape the revision order was old/current/current/old;
current Triton/H100 graphs were measured in alternating order in each current
process. Revisions run in separate processes, so this is not a single-process
alternating comparison of old and new graphs. L384/L768 run on two allocated
H100s in parallel.

The current H100 arm explicitly selects `front`, `f567`, `dual_bwd`, and
`out_ln_bwd`; it is not the untouched default. L384 B4 remains atomic, L768 B4
uses the registered TMA implementation. The newer mapped B4 experiment is excluded.
Current source uses the working H100 cache state plus explicitly pinned measured
F2/F567/B9+B10/B4 configurations recorded in each JSON. The Sep11 commit lacks
valid H100 cache entries for these workloads and uses its original heuristic
candidate fallback. Consequently this comparison includes implementation,
dispatch and tuning improvements; it does not isolate source-code changes alone
or establish superiority over an exhaustively retuned Sep11 implementation.

Input, upstream gradient, mask and every parameter SHA-256 match across all runs
at each length. Both versions use their official benchmark fixture; the matching
hashes verify the actual numerical workload rather than just nominal shapes.
Outputs/gradients are finite, all 11 gradients exist, reset RNG is reproducible,
and every timed replay advances real dropout randomness. Current native call
counters confirm the requested SM90 implementations ran. The fixture also reports
FP32-reference output/gradient error separately from speed.

## B4 correction and retained limitations

The earlier B4 1.187x claim was against the selected Triton persistent branch.
Forced updated atomic is faster than that TMA path at L768. The registered TMA
path remains available, but the 15% goal against the strongest measured Triton
baseline has not been met. The experimental mapped candidate measured ~1.023x
at L384 and ~1.080x at L768 at B4 level, with no consistent whole-module gain at
L384. It remains a research artifact, not part of the finalized figures above.
See [baseline correction and candidate evidence](../trimul-b4-l384-20260918/README.md).

The pure Triton and H100 columns distinguish total weekly improvement from the
additional gain of current H100 kernels. Component percentage gains must not be
summed to infer the full-module gain. These numbers are not full MiniWorld
training-step speedups.

## Reproduction

`measure.py` and all eight raw JSON results are included here. Use the same cu128
environment and set PYTHONPATH to the chosen revision root and its src directory.
Run `measure.py --length 384 --version old --round 0` with the Sep11 tree and
`--version current` with the current tree; repeat for L768 and round1. Current
config manifests are loaded from the normalization report path in the script.
The original logs and isolated Sep11 tree are retained in
`/home/psk6950/MiniWorld/runs/trimul_weekly_closeout_20260918/`.
