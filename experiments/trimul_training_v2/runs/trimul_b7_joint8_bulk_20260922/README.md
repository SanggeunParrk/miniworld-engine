# One genuine B7–B12 launch: diagnostic experiment

The selected validated development path remains `trimul_b7_nextrow_20260921`.
The three joint candidates below are **not numerically accepted and are not
selected**. Timing is an explicitly marked invalid diagnostic, not a valid
training speed comparison. Old split baseline timing in each run is not the
selected nextrow timing; do not conflate them.

## Why the selected path has two launches

Both paths recompute projection/gate and their BF16 GLU derivatives from saved
input xn. One kernel forms dW by reducing over pair positions; the other forms
dX by reducing over hidden channels, then performs input LN backward and adds
the residual. The two kernels do not exchange a materialized dP/dG HBM buffer.
Their cost is repeated reads and recomputation. A single true fused kernel can
share that work, but has to accommodate both reduction directions and resources.
Two launches are a design choice, not a proven impossibility of fusion.

## What was implemented

- `joint_cluster`: four CTA cluster, H128 per CTA, one producer WG, one
  GP/GLU/dX WG, two dW WGs. Shared xn and derivatives; FP32 dX partials summed
  across DSMEM. 864 B stack, 1544/1540 B compiler-reported spill stores/loads;
  WGMMA resource serialization warning. Diagnostic B7: 1908/7309 us.
- `joint8`: eight CTA cluster, H64 per CTA, one producer WG, one GP/GLU/dX WG,
  one dW WG. Register budgets32/224/248. **Zero spill and no WGMMA serialization
  warning**, but diagnostic B7 3056/12153 us. Direct remote scalar reads and
  cluster barriers remain; register spill was not the only problem.
- `joint8_bulk`: same eight CTA work split; replace scalar remote reads by
  three-stage shared-to-cluster bulk-copy tree reduction. No additional HBM
  activations. **Zero spill**. Diagnostic B7 2687/10531 us. NCU job14263.

All numbers above are L384/L768, microseconds, node01 H100, C128/H256,
BF16, dropout25%, mask and residual. All use one cooperative kernel launch,
not two independent kernels packaged behind one wrapper. All finish with
counter reset `[0,0]` on the tested inputs.

All three pass dx/dW relative-L2 limits on the initial input, but **fail the
unchanged input-LN parameter gradient limit5e-6** (errors roughly9e-6–2.34e-5).
Channel-partition accumulation changes floating-point order and is a suspected
cause, not a completed numerical diagnosis. No tolerance relaxation, no
promotion, no sanitizer/full-training pass claim. See report.json for exact
source hashes, limits and measurements. `timed_invalid_diagnostic_only=true`.

Current selected kernel hardware roofline remains51.6%/60.4%, not90%.
See `../trimul_b7_roofline_20260921/README.md` for the definition and caveats.

Bulk-copy instruction reference: NVIDIA PTX ISA8.8, CUDA12.9:
https://docs.nvidia.com/cuda/archive/12.9.0/parallel-thread-execution/index.html

## NCU result (job14263, completed)

Bulk-tree candidate: L384/L768 tensor-pipe activity4.80%/4.83%;
warp-active barrier stall metric54.25%/54.32%; long-scoreboard11.47%/11.04%.
These percentages are not additive fractions of kernel latency and are not SoL.
Combined with zero compiler spills, this supports synchronization/critical-path
cost as the next problem in this prototype, not HBM bandwidth saturation.
The bulk tree reduces time vs scalar DSMEM reads but remains much slower than
the split baseline. This experiment does not prove every single-kernel design
must lose; it rejects this channel-partitioned, per-row cluster-synchronizing
implementation. A better candidate must remove per-row all-CTA waits, or move
the consumer partition to avoid cross-CTA dX reductions. LN accuracy is also
still unresolved.
