# Trimul (trimul_inproj) batch generalization — investigation & decision

**Status:** investigated, **not shipped**. The triton trimul kernels keep their `B == 1`
assumption. B>1 was implemented and proven correct, but benchmarking showed a single-launch
batched path is *slower* than looping the tuned `B == 1` path over the batch, so it was reverted.
This note records why, so the experiment isn't repeated blindly.

## Background

The triton trimul paths (`trimul_inproj/triton/{front,back,bidirectional,unidirectional}.py`)
assume `B == 1`. The pipeline is: `LN_in → gated BDLL front → triangle contraction(s) →
LN_out+@Wp → output gate`. Intermediates (`left`/`right`) are stored **bdll** `(B, D, L, L)` —
batch OUTSIDE the channel dim — which is why the host-side `reshape`/`view` shortcuts hard-assume
`B == 1` (e.g. `lr.view(2D, M)`, `tri.reshape(H, M)`). The compute itself (row-parallel GEMMs,
`torch.bmm`/`einsum` contraction) is batch-agnostic.

Note: `_cute_eligible` already routes `B > 1` to triton (cute is `B == 1` only), so today
**B>1 raises the assert** — there is no working B>1 path until a caller loops (see below).

## What was tried (correct, but reverted)

A full B>1 generalization of the **triton** paths:

- **front kernels** (`_bidir_front_kernel`, `_lr_kernel`, `_back_kernel`): added a batch grid
  axis (`grid = (cdiv(L*L, BM), B)`); `pid_b` offsets into batch-outer bdll `left/right` and the
  channel-outer `preact (4H, B*L*L)`. `B == 1 ⇒ pid_b == 0`, collapsing to the original code
  (zero regression, verified).
- **contraction**: rewritten from slice+`reshape(B*h,L,L)`+`bmm` to `einsum(..., '->bijh')` with
  **channel-last output**, so the `(M, H)` view for `LN_out` is a free reshape (no permute-copy).
- **`front_bwd_dW`**: `d_left.permute(1,0,2,3).reshape(H*M)` to feed the channel-outer
  `_dconcat_kernel` correctly for B>1 (no-op view at `B == 1`).

**Correctness:** vs the pytorch reference module (shared weights), `B ∈ {1,2,4}`, full/partial
mask, both directions — forward cos `0.99999`, input-grad `0.99998`, weight-grad `1.00000`.
Batch-independence (`out[b] == out[b:b+1]`) exact. Full pytest suite green.

## Why it was NOT shipped — speed

Benchmarked **under CUDA graphs** (the deployment regime; `bench.py` uses `cudagraph=manual`.
Eager numbers are misleading — they include host/launch/alloc overhead graphs remove).

Forward inference, A100, bf16, d=128 (ms; `bat/loop = batched ÷ loop-of-B×(B=1)`):

| path | L | B | tri_batched | tri_loop | pytorch(compiled\*) | bat/loop | bat/py |
|------|---|---|-------------|----------|---------------------|----------|--------|
| bidir | 384 | 8 | 18.11 | **12.83** | 33.40 | **1.41** | 0.54 |
| bidir | 256 | 8 | 7.49  | **5.90**  | 13.78 | **1.27** | 0.54 |
| uni   | 384 | 8 | 8.39  | **6.36**  | 19.19 | **1.32** | 0.44 |
| uni   | 256 | 8 | 3.34  | **2.98**  | 7.92  | **1.12** | 0.42 |

`B == 1`: batched ≈ loop (same path). Both beat pytorch ~2×. **But for B>1 the loop is 10–40%
faster than the batched single launch.**  (\*pytorch = eager in the table above; the triton path
is `@torch.compiler.disable`, so `compile` is a no-op for it and does not change bat/loop.)

## Root cause — L2 thrashing of large batched intermediates (NOT a slow kernel)

This is the counter-intuitive part, resolved by per-stage timing **in the same (cudagraph) regime**:

- Every individual kernel is **faster** batched (ratio 0.86–0.97), and the **sum of isolated
  stages** is faster batched (uni L=384: batched 5.75 vs loop 6.09).
- Yet the **chained full forward** is *slower* batched (8.27 vs 6.17). For the loop, chained ≈
  sum-of-stages (6.17 ≈ 6.09); for batched, chained ≫ sum (8.27 ≫ 5.75) — a ~2.5 ms penalty that
  appears **only when stages are chained**.

The penalty is **memory locality**, not any kernel. A100 L2 = **40 MB**. One intermediate
`left`/`right` = `(B, H, L, L)`:

- batched (B=8, L=384): `8·128·384·384·2 B` ≈ **302 MB** ≫ L2 → each chained stage reloads its
  input cold from HBM.
- loop (per-batch, L=384): `1·128·384·384·2 B` ≈ **38 MB** ≈ L2 → adjacent stages reuse L2.

Prediction check: the gap must grow with L (bigger intermediates). It does — uni bat/loop
1.12 (L=256) → 1.32 (L=384); the batched full-vs-sum gap 0.73 ms (L=256) → 2.5 ms (L=384).

Isolated per-stage benchmarks hide this (no chaining), which is why "every kernel is faster
batched, yet combined it's slower" looked paradoxical — it's the working-set size of the
chained intermediates, not the kernels.

## Decision & recommendation

- **Keep `B == 1`** in the triton kernels (reverted the generalization).
- **If B>1 is needed, loop the tuned `B == 1` path over the batch** at the dispatch layer
  (`modules/triangle_multiplication`, or `whole_op`). It is the fastest option measured here
  (≈2× pytorch) and needs no kernel changes.
- A single-launch batched path only wins if the forward is **fused** so intermediates stay in
  SRAM/registers instead of round-tripping HBM (the raison d'être of fused kernels). That is a
  separate, larger effort.
- **cute (sm90/sm100) paths**: not generalized. B200-only, unverifiable on this A100 cluster,
  and the CuTe TMA epilogue bakes the bdll `(2D, M)` layout (see `front_sm100.py` — it documents
  silent half-tile TMA corruption for the wrong store layout). Same loop recommendation applies.

## Reproduction

Scripts under `submits/` (A100): `_trimul_speed_cudagraph.py` (batched/loop/pytorch under graphs),
`_uni_full_cg.py` (per-stage vs full-forward reconciliation, the L2-thrashing evidence),
`_bidir_bwise_verify.py` / `_trimul_uni_fb_verify.py` (B>1 correctness). Data: 2026-07-16,
NVIDIA A100 80GB PCIe (sm80), bf16.
