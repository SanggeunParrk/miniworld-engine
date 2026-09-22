# B7 single launch, shared recomputation

User asks why two kernels and whether to merge; original SoL90 objective remains.
This prototype must share actual GP/GLU work, not just place disjoint roles in a
single launch. Selected production/development remains nextrow until validated.

Prototype: four-CTA Hopper cluster, ranks partition hidden channels: L[0:128],
L[128:256], R[0:128], R[128:256]. One cluster owns the same 64 pair rows.
Each CTA has four warpgroups (512 threads): producer (24 regs), GP/dX/LN (200),
two dW consumers (144 each). 128 CTAs / 32 clusters initially, one CTA per SM.
Check actual cluster cooperative occupancy; never oversubscribe a spinning grid.

Persistent per-CTA W1 weights: 64KiB, one rank-specific H128 projection+gate tile.
Shared: W1[0:65536], xn[65536:81920], dG[81920:98304],
dP[98304:114688], dx_partial_fp32[114688:147456], Wgate[147456:180224].
Raw x/residual/dGate reuse xn/dG/dP only after both dW consumers finish.
No new HBM activations. HBM partial dW remains 32*131072*4 = 16MiB;
LN partial is 32*256 floats. (Current split baseline also has 16MiB dW scratch.)

For each row tile: rank0 TMA multicasts xn; ranks TMA own dLeft/right slices.
GP/dX WG computes GP and GLU once for two H64 chunks, publishes shared dG/dP;
dW WGs accumulate dG.T@xn and dP.T@xn concurrently with dX contractions.
After CTA + cluster sync, rank0 GP/dX WG sums all four FP32 dX partials via DSMEM,
adds BF16-rounded dGate@Wgate.T, applies original LN derivative/residual, writes dx.
Rank0 producer supplies raw x, residual, dGate while peers wait.
Cluster sync before shared buffers are reused. Global cooperative sync at end,
then in-kernel weight and LN parameter reductions. One CUDA launch.

Numerical implications: shared gradients are BF16 identically to baseline.
Cross-CTA FP32 dx sums and 32-cluster dW partial reduction change accumulation
order. Check existing same-input dx<=2e-5, dW<=5e-4, LN<=5e-6 relative L2;
report bit-exact separately. Do not relax strict reference tolerances to win.

Potential costs: cluster sync and DSMEM, register pressure, one CTA/SM, stage
bubbles while rank0 performs gate/LN. Must benchmark before any selection.

Previous GLU role split diagnostics: 14236/14238 stalled in reduced-register
variants and were explicitly cancelled; not selected. 64/192 original variant
passed one input but was slower. Selected split path remains independently clean.
