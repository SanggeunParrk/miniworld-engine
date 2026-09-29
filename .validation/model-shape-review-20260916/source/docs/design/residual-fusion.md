# Residual fusion — follow-ups (future versions)

## Status at a glance

| op | residual owned by module | kernel-fused (default H100) | remaining |
|----|:---:|:---:|----|
| Transition (pair/msa/single) | ✅ | ✅ cuda-b2b / triton | — |
| TriangleMultiplication (single/bidir) | ✅ | ✅ cute **and** triton | sm100 (B200) back = explicit |
| TriangleAttention (start/end) | ✅ | ❌ explicit add | fuse into attn output epilogue |
| OuterProductMean (team-gm) | ✅ (`residual=pair`) | ❌ explicit add | cross-tensor; fuse in `to_out` |
| MSAPairWeightedAveraging / AttentionPairBias (team-gm) | ✅ | ❌ explicit add | self-residual; fuse in torch layers |
| AugmentedAttentionPairBias / ConditionedTransition (DiT) | mp_sum (block) | — | out of scope (magnitude-preserving) |


The residual connection (and its optional row/col dropout) is now **owned by the module** for
every residual-wrapped op (see the "ALWAYS APPLIES THE RESIDUAL" banners in each module). For
some ops the residual is **fused into the kernel epilogue** (the add rides the op's existing
output store — no separate elementwise kernel / HBM round-trip); for others it is currently an
**explicit add inside the module** (contract unified, but no speed win yet).

## Fused (residual in the kernel epilogue) — done
- `Transition` — folded into the b2b / triton squeeze epilogue.
- `TriangleMultiplication` (single-dir) — folded into the back-half store.
- `BidirectionalTriangleMultiplication` — folded into the gate/back store.
  (Measured fusion gain, compile+cudagraph: transition inf ~1.10–1.22x, trimul-single inf
  ~1.07–1.09x, bidir inf ~1.02–1.04x; training smaller.)
- **trimul TRITON backend** (`_forward_triton` / `bidirectional_trimul_triton`) — the residual +
  drop_row are now fused into the same `gate_elem` store epilogue as the cute dispatch, so the
  pre-Hopper (A100) default and any explicit `MINIWORLD_TRIMUL_IMPL=triton` path fuse too
  (verified cos 0.99999 fwd+grad, dropout deterministic).

## NOT kernel-fused yet — explicit add, revisit in a later version
These apply the residual (+ dropout) as an explicit `x + drop(op(x))` after the op. The contract
is unified (blocks just call `module(...)`), but fusing the add into the op's output kernel would
save a launch + an [B,L,L,D] HBM round-trip. Candidates, roughly by expected payoff:

- **sm100 (B200) trimul back paths** (`_forward_cute_free` / v6-sm100) — currently apply
  residual+dropout explicitly (`out + pair`); the tcgen05 back has no fused-residual store yet.
- **`TriangleAttention`** (miniworld-engine) — fuse residual + drop_row (starting) / drop_col
  (ending) into the attention output epilogue (`fused_gate_out` / the bias-only store). This is
  the highest-value miniworld-engine follow-up.
- **`OuterProductMean`** (team-gm) — CROSS-TENSOR residual (`pair += opm(msa)`; the residual is
  the running pair, not the module's own input). Passed explicitly as `residual=pair`. Fusing it
  means adding the pair in the OPM `to_out` epilogue — feasible but a different pattern than the
  self-residual ops.
- **`MSAPairWeightedAveraging`** / **`AttentionPairBias`** (team-gm) — self-residual + optional
  dropout, currently explicit in the team-gm torch layers.

## Out of scope (by design)
- **`AugmentedAttentionPairBias`** / **`ConditionedTransition`** (DiT / atom blocks) — these use a
  magnitude-preserving weighted residual (`mp_sum(x, branch, residual_t)`), a different contract
  than the plain `x + f(x)` add. Left as external `mp_sum` in the blocks.
- `single_to_msa` and similar trivial `x + Linear(y)` projections.
