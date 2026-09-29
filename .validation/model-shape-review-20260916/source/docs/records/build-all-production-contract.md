# Build-all production contract audit — 2026-09-11

The build input grid and FakeTensor derivation now share inference A=5, training A=48,
actual tensor ranks, token condition width 384, and the complete declared atom length ladder.
A source/architecture-specific verified plan is refreshed automatically before tuning starts.
A full build now checks native backend imports and SM90+ cuBLASDx headers before derivation
or tuning, so an incompatible FlashAttention/CUTLASS install fails early.
The SM100 bidirectional input projection and Transition backward no longer flatten tensors
before calling the LayerNorm wrapper that needs their original shape.

## Verified so far

| Check | Result |
|---|---|
| Current sm86 dispatch plan | 5,258 invocations; zero errors; 1,007 required keys |
| Current sm90 dispatch plan | 5,750 invocations; zero errors; 1,202 derived rows |
| Current sm100 dispatch plan | 5,750 invocations; zero errors; 1,229 derived rows |
| Targeted build/plan/config regression suite | 52 passed |
| Native dependency preflight and CLI regression suite | 20 passed; lint passed |
| Final page, stream, shape-key and grid validation | 41 passed |
| Final merged A6000 required-key coverage | 1,007 / 1,007 usable; zero missing; zero invalid |
| A5000 numerical suite | 82 passed |
| New A6000 LayerNorm top-five candidates | 8 keys, 40 candidates, 160 checks passed |

The A6000 candidate audit covers actual A48/L128–384 tensors, width 384, both contiguous and
m-major layouts, two random seeds, and the existing official FP32/BF16 error bands.
The A5000 ConditionedTransition BF16 regression had a FP32 reference discrepancy of
2.20e-7 relative to FP64. Its rounding-floor comparison now allows two FP32 epsilons of
reference arithmetic error; the raw numerical checker bands are unchanged. The final BF16
output matched the FP64 reference's nearest-BF16 rounding floor in that audit.

The A6000 build completed: 4 module invocations plus 608 alternative-kernel driver invocations,
612 successful units in total, zero empty units, and zero failed units. Jobs 1677612 (42m30s)
and 1677630 (9m57s, joining the remaining driver queue) both exited 0. The final merged cache
contains 2,954 entries, of which all 1,007 keys required by the current production plan are
usable. The 8 initial required-key holes are closed; required invalid keys are also zero.
See [the coverage artifact](build-all-a6000-coverage.json) and
[the new LayerNorm numerical audit](build-all-new-ln-numerics.json).

A5000 is still building: it started with 987 missing/stale required keys and selected 323 module
invocations. Job 1677616 uses its eight GPUs. Dependent job 1677633 resumes only after a
TIMEOUT/NODE_FAIL/PREEMPTED result, merging completed shards and reusing the same local compiler
cache and shared tuning-round cache. It uses 64 CPU threads (8 compile workers per GPU).
Other failures require inspection rather than blind repetition. No A5000 full-cache completion
is claimed; its 82 numerical tests are a separate result.

## Remaining qualification

Only A6000 and A5000 hardware is available in this cluster. The sm90/sm100 results above
are FakeTensor dispatch coverage, **not** H100/B200 execution or numerics. The installed
FlashAttention CuTe sources currently fail to import with the installed CUTLASS package;
SM90 native Transition additionally needs its cuBLASDx headers. These dependencies must be
validated in the target machine's environment.

Six CuTe config families have a cache resolver but their `sweep_and_cache` helper is not wired
into the builder: `transition_swiglu_fwd`, `transition_gate_bwd`, `dab_lnbwd`, `dgrad_lnbwd`,
`layernorm_linear_m1`, and `tm2_dual_fwd`. Full native-backend cache coverage is still open.
See [the exact command contract](../operations/dispatch-cache.md#what-build-all-verifies).

The existing L384 module benchmark table is unchanged: these build-contract changes do not
constitute a new benchmark. Its recorded compile/graph/dropout/augmentation provenance remains
in [the module table](a6000-l384-module-bench-latest.md).

Execution logs and machine-readable numerical evidence are in
`/home/psk6950/practice/miniworld-engine/.scratch/a6000-2026-09/mw-build-contract/`.

Nine historical `cpu.json` GPU-timing artifacts have been removed from the active cache tree
and preserved byte-for-byte in [an archive](archive/cpu-cache-artifacts/README.md). Their GPU
identity is unknown; none was relabelled as A6000 or A5000. The current manual merge command
already rejects an implicit CPU destination.

A broad earlier CPU run also exposed an existing ladder-policy failure: the
`gated_projection_bwd_dx_triton` grid offers warp counts 2/4/8, while the policy test demands
an additional lower neighbour (1). All recorded winning warp counts are offered. This audit
does not widen that tuning grid solely to satisfy the heuristic; a full-suite green result
is not claimed while that policy discrepancy and the A5000 cache refresh remain outstanding.
