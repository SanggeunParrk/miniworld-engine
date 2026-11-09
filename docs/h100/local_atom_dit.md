# Local atom DiT (AF3 block-local attention) on H100

## Scope

The AF3 atom transformer block with block-local attention (AF3 Alg. 23 with the 32 x 128 trunking of Alg. 24), one block per
instance, in `modules/atom_dit.py`. `AtomDiT(model=...)` builds the model's own class:

| class | models | K / V source | default `key_window` | backends |
|---|---|---|---|---|
| `AtomDiTProtenix` | protenix, protenix_v2, opendde | chained (`ada_ln_kv(ada_ln_in(a))`) | `pad` | pytorch, anthropic, cuda, triton |
| `AtomDiTOpenFold3` | af3, openfold3, openfold3_preview2, intellifold2 | parallel (`ada_ln_kv(a)`) | `slide` | pytorch, anthropic, cuda, triton |
| `AtomDiTBoltz2` | boltz2 | shared (one AdaLN), up-gated transition | `pad` | pytorch, anthropic, cuda, triton |
| `AtomDiTRoseTTAFold3` | rosettafold3 | shared, flat q / k LayerNorm, parallel schedule | `pad` | pytorch, cuda, triton |
| `AtomDiTChai1` | chai1 | shared, identity AdaLN, no gate / out-projection, conformer mask `uid` | `pad` | pytorch, cuda, triton |

Requirements rows (`requirements/requirements_module.csv`): stream `atom_single`, lengths 1024-8192, `d_single = d_cond = 128`,
`d_pair = 16`, 4 heads x 32, transition `n = 2`; dtypes `float32 | master`, modes `eval | train`. Extra rows: Protenix /
Boltz-2 / RF3 with `key_window=slide`, IntelliFold2 with `key_window=slide_qblock`, Chai-1 with `adaln_eps=1e-05;key_window=slide`.

Inputs: `single` / `cond` `[A, B, N, 128]` (A = samples on the augmentation axis), the trunked pair `[B, nwin, 32, 128, 16]`
(`nwin = ceil(N / 32)`), a key `mask [B, N]` with the real atoms first. Query window `w = i // 32` sees 128 keys: `pad` =
`[32 w - 48, 32 w + 80)` clipped to `[0, N)`; `slide` = the same window shifted back inside the real atoms (AF3); `slide_qblock` =
`slide` against the real count rounded up to 32 (IntelliFold2 torch release).

This page covers the H100 kernels behind the two `"cuda"` paths and the `"triton"` window core. How accuracy and speed are judged:
[../measure.md](../measure.md); backend selection: [../modules.md](../modules.md); kernels and launch configs: [../kernels.md](../kernels.md).

## Which code serves which backend

- `"cuda"`, `AtomDiTProtenix` / `AtomDiTOpenFold3`: the whole block on hand-written sm_90a kernels,
  `kernels/integrations/local_dit_h100.py` (`run_block`). Row kernels `kernels/augmented_attention/cuda/sm90_atom/`, window attention
  `kernels/augmented_attention/cuda/sm100_atom_local/lattn90.cu` / `lattn90_tf32.cu` (the directory name is historical; these are the
  sm_90a mma.sync kernels), cuBLAS for the weight-gradient GEMMs.
- `"cuda"`, `AtomDiTBoltz2` / `AtomDiTRoseTTAFold3` / `AtomDiTChai1`: `modules/dit_variant_cuda.py` on `kernels/dit_variants`
  (row kernels + cuBLAS), with the same `lattn90` window attention (`attn_fwd_mma`, `attn_fwd_mma32`, `attn_bwd_mma`,
  `pair_bias_mma`). The row kernels of that family are documented in [dit_variants.md](dit_variants.md), not here.
- `"triton"` (every class): only the window softmax core runs on `kernels/augmented_attention/triton/window.py`; AdaLN, projections,
  gates and the transition are the PyTorch statements.
- `"anthropic"`: release ops (af3_torch `atom_window` / `dtk_kernels`, Boltz-2 `launch_win_attn`), a comparison baseline, eval only.

## Kernels (H100 block, `augmented_attention` family)

Names are the `requirements_kernel.csv` names without the `augmented_attention_` prefix and `_sm90` suffix.

| name | source (function) | role | what it does |
|---|---|---|---|
| `atom_pack` | `sm90_atom/pack90.cu` (`pack90`) | forward | every forward / backward weight form of the 27 parameters in one launch: cast, LayerNorm-gamma fold, transpose, lane / K-slot permutation, tf32 rounding |
| `atom_cond_fwd`, `atom_cond_fwd_split` | `cond90.cu` (`cond90_fwd`, `cond90_split`) | forward | conditioning tables `mod [M, 768] = [s1 | bi1 | so | s2 | bi2 | st]`, `mkv [M, 256]` from `LN(c)` (gamma folded into the weights) |
| `atom_pre_fwd`, `atom_pre_fwd_split` | `pre90.cu` (`pre90_fwd`, `pre90_split`) | forward | AdaLN 1, `q | sigmoid(gate)`, K / V AdaLN, `k | v` |
| `local_pair_bias_fwd` | `lattn90.cu` (`h_pair_bias`) | forward | `bias2[h, w, i, j] = LN(z) . Wb[h] * log2(e)`, `-inf` at keys outside `[0, N)` or masked |
| `local_attn_fwd`, `local_attn_fwd_tf32` | `lattn90.cu` (`h_attn_fwd`), `lattn90_tf32.cu` (`h_attn_fwd_tf32`) | forward | window attention, O and the row LSE (log2 units) |
| `atom_post_fwd` | `post90.cu` (`post90_fwd`) | forward | gated out-projection, residual, AdaLN 2, SwiGLU (hidden 256), squeeze, gated residual |
| `atom_tr_bwd` | `tr_bwd90.cu` | backward | transition output gate + SwiGLU backward (`a`, `b` recomputed from saved `x2`) |
| `atom_post_bwd` | `post_bwd90.cu` (plain and `-DSPLITK=1` builds) | backward | AdaLN 2, residual and gated out-projection backward; `D = rowsum(dO * O)` per head |
| `local_attn_bwd_dq(_tf32)` | `lattn90*.cu` (`h_attn_dq*`) | backward | dQ, query-centric |
| `local_attn_bwd_dkv(_tf32)` | `lattn90*.cu` (`h_attn_dkv*`) | backward | dK, dV and per-sample-chunk dbias partials, key-centric |
| `local_bias_bwd_samples` | `lattn_dkv.cu` (`local_bias_samples_sm90`) | backward | sums the dbias partials over the sample chunks in fixed order |
| `local_pair_bias_bwd`, `local_pair_bias_bwd_fin` | `lattn90.cu` (`h_pair_bias_bwd`), `lbias.cu` (`local_bias_fin`) | backward | dz of the trunked pair; per-block partials of dgamma / dWb finished in fixed order |
| `atom_pre_bwd` | `pre_bwd90.cu` | backward | q / gate / k / v projection and both AdaLN backwards |
| `atom_cond_bwd`, `atom_cond_bwd_shared` | `cond_bwd90.cu` (plain and split-K), `cond_bwd_sh90.cu` | backward | conditioning-table backward: per-sample rows, or the grouped partials of a shared conditioning summed in fp32 |
| `atom_finalize_bwd` | `finalize90.cu` | backward | every parameter gradient from the GEMM outputs and column sums, one launch, written in each parameter's dtype |

`requirements_kernel.csv` also lists the Triton window kernels `augmented_attention_fwd_window`, `_bwd_window`,
`_bwd_reduce_window` (the `"triton"` backend of all five classes).

## Design

**Pipeline.** Inference: `cond90` (hoisted) -> `pre90` -> pair bias (hoisted) -> window attention -> `post90`. Training forward runs
the same kernels with `save=True` and keeps `bias2, mod, mkv, q, k, v, sg, x1, xkv, o, lse, u, a2, x2, t`; backward:
`tr_bwd90` -> `post_bwd90` -> `h_attn_dq` / `h_attn_dkv` / sample-sum -> pair-bias backward -> `pre_bwd90` -> `cond_bwd90` (or the
shared form) -> ten cuBLAS GEMMs `a^T b` -> `finalize90`. The forward and backward are two opaque ops (`local_dit_h100_fwd` / `_bwd`)
inside one `autograd.Function`, so they work eager, under CUDA graphs and under `torch.compile`. Atoms are padded to a multiple of
128 inside (padded keys masked); the trunked pair is padded with zero windows.

**Row kernels (`sm90_atom`).** A CTA is one warpgroup = 64 rows (`NWARP = 4`; the fp32 `tr_bwd90` uses 8 warps); a warp owns a
16 x 128 tile as accumulator fragments. Products are `wgmma` with the A operand in registers (`wg90.inc`: `m64n{32,64,128}k16` bf16,
`k8` tf32): the previous GEMM's accumulator is the next GEMM's A operand with no data movement. Weights stream through shared memory as
swizzled K-major tiles via cp.async. Every activation access is one 16-byte vector per row: a lane's columns are permuted (`pcol`)
and the weights' row and K axes are permuted the same way when packed (`sm90_atom._lperm`). The LayerNorm weights of the conditioning
are folded into the weights (`W' = W diag(gamma)`); `dgamma` is recovered in `finalize90` as `sum dW' * W`. Every kernel is launched
with programmatic dependent launch (`griddepcontrol.wait` before reading a predecessor's output). Register targets: bf16 `cond`,
`cond_bwd` ask for 3 CTAs per SM (`_MINB`).

**Small row counts.** When `A * N` gives few 64-row tiles, `_split_for` splits a tile's work over `grid.y` to reach about 264 CTAs:
`cond90_split` / `pre90_split` by weight unit, `tr_bwd90` by hidden chunk (disjoint outputs), `post_bwd90` / `cond_bwd90` as separate
`-DSPLITK=1` builds with fp32 partials and a last-CTA finish in fixed order (run-to-run identical). `post90` forward and `pre_bwd90`
are not split.

**Shared conditioning.** When `cond` is an expanded view (stride 0 over the samples, `A > 1`), the tables are built once per atom and
read modulo `N`. In the backward, if `A % 4 == 0`, a CTA group covers four samples (one warp each) and their modulation gradients
are summed in fp32 through shared memory before one store (otherwise per-sample rows); `cond_bwd_sh90` then sums the partials, runs
the six products and the LayerNorm backward on the `N` shared rows, and writes fp32 sums for the weight-gradient GEMMs. `dcond` is
returned as `1 / A` of the sum per copy (the expand's autograd sums it back). A contiguous `cond` takes the per-sample path; there is
no value comparison.

**Window attention (`lattn90.cu`, `lattn90_tf32.cu`).** `mma.sync` (m16n8k16 bf16 / m16n8k8 tf32), not wgmma: 32-wide heads and
128-key windows are softmax / latency bound. Forward and dQ: CTA = 64 queries (two windows) x one head x a chunk of samples, 4 warps
of 16 queries; K / V are the 160-key union of the two windows. dK / dV: CTA = 64 keys x one head x a sample chunk, a warp owns 16 keys,
which see exactly 4 query windows (128 queries), no masked waste; Q / dO are the 192-query union. The pair bias (log2 domain, key mask
folded in as `-inf`) is staged in shared memory once per CTA and reused by every sample of the chunk; tiles are double-buffered with
cp.async; dbias accumulates in registers over the chunk (one fp32 partial per chunk, summed in fixed order). Chunk size `_chunk`:
16 samples, halved until the grid has about one CTA per SM.

**AF3 slid window (`AtomDiTOpenFold3`, inference only).** `pre90` takes a `parallel_kv` flag (K / V AdaLN of the raw atoms). The slid
window equals the clipped one except for windows 0, 1 and the last three real windows; `_slid_edges` recomputes those five windows in
torch (fp32) from the kernels' q / k / v and overwrites the attention output. The five window indices come from the real atom count
on the device, so there is no host read and the call can be captured.

**Inference hoist.** Without `save`, the conditioning tables, the pair bias and the slid-edge tables are made once per input tensor
(storage address, in-place version, shape) and weight version, via `kernels/_capture.hoisted_table` (scoped to CUDA-graph capture);
training recomputes them every call. Weight packs are cached per weight version and capture scope.

## Numerics

- bf16 path: bf16 activations and MMA operands, fp32 accumulation, fp32 LayerNorm statistics and softmax.
- fp32 path (`-DTF32=1` twins): MMA operands rounded to nearest tf32 (`cvt.rna`): weights once per version on the host side,
  activations when they feed an MMA. Residual stream, LayerNorms, softmax, saved activations and every gradient stay fp32. A TF32
  `wgmma` takes K-major operands only, so the K slots of each k8 step are permuted (`[0 2 4 6 1 3 5 7]`) to make the accumulator the A
  fragment in place.
- Weight-gradient GEMMs (`_mmT`): TF32 for the fp32 block, bf16 products with fp32 output for the bf16 block, exact fp32 for the fp32
  operands of the shared-conditioning path.
- The training step is bitwise repeatable: every reduction is a fixed-order partial sum, no float atomics.

## Launch configs

The CUDA kernels have fixed geometry (64-row tiles, split factors from `_split_for`, sample chunks from `_chunk`); they take no entry in
`kernels/augmented_attention/configs/h100_sm90.json`. The Triton window kernels are `@configured` as
`augmented_attention_{fwd,bwd,bwd_reduce}_window_triton`; the H100 table has no entry for them, so they launch their defaults
(`BLOCK_N = 128`, 4 warps, 2 stages; fold `BLOCK_D = 32`) and warn once.

## Limits (`_LocalAtomDiT._supports_cuda`, Protenix / OpenFold3)

- H100 (sm_90) CUDA tensors; `single` / `cond` / `pair` all bf16 or all fp32. fp32 needs fp32 parameters and no autocast; bf16 takes
  bf16 or fp32 parameters.
- Not the shared-AdaLN classes (`kv_norm == "shared"`), not `key_window="slide_qblock"`.
- `single` `[A, 1, N, 128]` (one pair per call), `cond` `[A, 1, N, 128]`, pair `[1, nwin, 32, 128, 16]`, mask bool `[1, N]` or none.
- Geometry 128 / cond 128 / pair 16, 4 heads x 32, transition hidden 256, a query bias; no q / k norm; every LayerNorm eps `1e-5`;
  contiguous parameters.
- Training: chained K / V AdaLN and the clipped `pad` window only (so `AtomDiTProtenix` with the default window). `AtomDiTOpenFold3`
  and `AtomDiTProtenix(key_window="slide")` are eval only on `"cuda"`.

`"triton"` (`_supports_triton`): CUDA, bf16 / fp32, `[A, B, N, d]` inputs, head dim <= 64, trunked pair, `mask` / `uid` `[B, N]`.

Variant classes (`VariantAtomCuda._atom_serving`): H100, `key_window` `pad` or `slide`, no autocast, all-bf16 or all-fp32,
`[A, 1, N, 128]`, 4 heads, pair 16, transition `n = 2`, pair `[1, N / 32, 32, 128, 16]` (N a multiple of 32; fp32 and training: of 64),
one AdaLN epsilon for the block and conditioning LayerNorm eps `1e-5`.

## Rejected approaches (brief)

- `mma.sync` row kernels with shared-memory B fragments: shared-memory bandwidth / bank-conflict bound; replaced by register-A wgmma
  with 16-byte lane vectors.
- 128-row CTAs: wave quantisation; only the fp32 `tr_bwd90` keeps 8 warps.
- TMA weight ring (persistent or not): slower than the cp.async ring, especially for 64-row K-major TF32 boxes.
- Fusing dQ into the key-centric dK / dV kernel: a query's keys span 2-3 key tiles, so dQ partials leave the CTA (large fp32
  reductions) while dbias is summed over samples in the kernel; not built.
- Fusing the conditioning tables into `pre90` / `post90`: inference already hoists them; training needs them again in the backward.

Historical (from the 2026-10-15 note, not re-measured under the current measurer): A = 48, 4096 atoms, CUDA-graph training step
bf16 3.50 ms, fp32 5.81 ms.

## History

Old page: `git show HEAD:docs/gpus/h100/local_dit/local_dit.md`.
