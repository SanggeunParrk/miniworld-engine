# TriMul SM90 round2

Baseline: engine commit `0f2d455b`. Scope: L384/L768 bidirectional training;
identical Triton fusion, rounding and declared configuration axes. F2 stays frozen.
The per-kernel target remains1.15× the strongest measured Triton configuration;
the eventual whole-module target is separate.

## Cleanup

Deleted66 untracked root-level `cutlass*.ptx` / `cutlass*.cubin` files, exactly the
prefix confirmed by the user. No source/cache/evidence-directory deletion.
[Deleted-file manifest](cleanup/root-dumps.json).
New compilation/profiling uses experiment-local working and dump directories.

## Completed investigation

- F567: source-correlated NCU of the final sigmoid checkpoint identifies inefficient
  dropout global sectors, while most long-scoreboard samples are TMA readiness
  waits. A capacity-derived spare shared tile permits dropout TMA on some configs.
- B9+B10: front TMA readiness dominates sampled long-scoreboard stalls. Separate
  producer/consumer warp groups did not beat the four-warp baseline, even with a
  two-CTA launch bound. Three tensor-to-L2 prefetch policies also failed to produce a stable
  gain. The dual backward production source is unchanged.
- Timing uses rotating CUDA graph measurements. NCU instrumented durations are
  diagnostic and do not replace benchmark medians.
- Numerical and sanitizer checks are required before candidate promotion.

Final F567 source is promoted: dropout TMA plus initial projection L2 prefetch.
Selected G1 config is measured at93.231/355.706us (L384/L768), versus strongest
Triton105.510/409.107us. L768 straddles the15% threshold across allocations;
no robust target-completion claim is made. Production35-case regression and
selected35-case memcheck/racecheck pass; whole-module output/all gradients pass.
Whole-module speed is1.033x/1.029x. Allocation13235 has been released.

[Consolidated report](../../docs/trimul-sm90/TMA_ROUND2.md) ·
[Final evidence](module/final-evidence.json).
