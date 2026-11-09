# SWA atom DiT on H100

## Scope

The pair-free sliding-window atom block of ESMFold2 (Algorithm 8, `SWAAtomBlock`), which MiniWorld adopted, in
`modules/atom_dit.py`:

| class | what | backends |
|---|---|---|
| `AtomDiTMiniWorld` | one block: adaLN-Zero (`W silu(c)` -> shift / scale / gate for attention and FFN), affine-free RMSNorm, `SWA3DRoPEAttention`, `SwiGLUFFN` | pytorch, anthropic, cuda |
| `AtomDiTESMFold2` | the same block under ESMFold2's constructor (`half_window = 64`) | pytorch, anthropic, cuda |
| `SWAAtomTransformer` | `n_block` `AtomDiTMiniWorld` blocks (`half_window = swa_window_size // 2`) | pytorch, anthropic, cuda |
| `InputFeatureEmbedder` (`modules/embedder.py`) | a composition holding an `SWAAtomTransformer`; its `"cuda"` sets the transformer's | pytorch, anthropic, cuda, triton (refused: the transformer has no `"triton"`) |

Block math: `x = x + gate_a * attn(rms(x) (1 + scale_a) + shift_a)`, `x = x + gate_f * ffn(rms(x) (1 + scale_f) + shift_f)`;
attention = `Wqkv` (no bias) -> per-head RMSNorm of q, k (fp32 eps, affine-free) -> 3D RoPE (`cos` / `sin` per atom) -> window
`|rank_i - rank_j| <= half_window` among the valid atoms (a negative half window, or one >= 65536, is global attention) ->
`sigmoid(gate_proj(x))` gate -> `out_proj`. `d_atom = d_cond = 128`, 4 heads x 32, SwiGLU hidden 256.

Requirements rows: `AtomDiTESMFold2` and `SWAAtomTransformer` (`swa_window_size` 128, and 1000000 = global) at 1024-8192 atoms,
`float32 | master`, `eval | train`; `InputFeatureEmbedder` (3 blocks, `swa_window_size` 128 or 1000000, `features` legacy / esmfold2,
`token_pool` relu / linear). There is no `"triton"` backend for these classes.

Accuracy / speed judging: [../measure.md](../measure.md); backends: [../modules.md](../modules.md); kernels: [../kernels.md](../kernels.md).

## Path selection (`AtomDiTMiniWorld._forward_cuda`, `kernels/swa_dit/dispatch.py`)

The module flattens `cond` to rows and computes the modulation `mod = W silu(c)` [rows, 768] fp32 itself
(`dispatch.native_h100_mod_fwd`: a silu + cast copy kernel, then a GEMM -- bf16 operands with fp32 output for a bf16 block, cuBLAS
fp32 for an fp32 block); with a gradient through `SWADiTModulationH100`. It then calls the block with `B = N` (one modulation row per
sequence row), so the augment count the kernels see is 1. Inference calls `swa_dit_block_fwd` directly; training goes through
`autograd.SWADiTBlockFunction` (forward with `save=True`, 13 saved intermediates, `swa_dit_block_bwd`).

`swa_dit_block_fwd` / `_bwd` pick one of two H100 implementations:

| call | stages |
|---|---|
| bf16, every composed extension builds | **composed**: qkvg (slab or wgmma) -> `swa_attn.cu` -> out-projection + FFN (slab or wgmma) |
| fp32, matmul precision `high` / `medium` (TF32 allowed) | **composed fp32**: `swa_fp32.cu` (`fwd` / `bwd`) |
| fp32 under `highest`, a composed extension missing, or global attention | **native**: `native_rows` / `native_copy` row kernels, cuBLAS projections (`native_owned_mm`), `tiled_attention` (or `native_attention`); global attention on FlashAttention-4 |

Global attention (`InputFeatureEmbedder` rows with `swa_window_size = 1000000`) runs on the external library FlashAttention-4
(`flash_attn.cute` `_flash_attn_fwd` / `_flash_attn_bwd`, `swa_dit/cuda/sm100.attn_global_fwd` / `_bwd`) between the native row
stages; without FlashAttention-4 installed `_supports_cuda` refuses the call.

## Kernels (`swa_dit` family, `kernels/swa_dit/cuda`, sm_90a)

| source (extension) | entry points | what it does |
|---|---|---|
| `swa_slab.cu` (`slab`) | `qkvg_fwd`, `oproj_fwd`, `ffn_fwd`, `ffn_bwd`, `oproj_bwd`, `qkvg_bwd` | every row stage as "slab" kernels: mma.sync m16n8k16, persistent CTAs of two 4-warp groups, a 16-row slab per group with its GEMMs split 4 ways, weights resident in shared memory |
| `swa_qkvg_fwd.cu` (`qkvg`, `qkvg_small`) | `qkvg_fwd` | RMSNorm / adaLN + q / k / v / g projections + head RMSNorm + RoPE on wgmma: two consumer warpgroups ping-pong on tiles of 8 augments x 8 atoms, a TMA producer, a weight ring |
| same file | `native_rows`, `native_copy`, `native_attention`, `tiled_attention`, `fused_oproj_bwd`, `fused_qkvg_bwd`, `native_ffn_pack`, `native_owned_mm` | the native stages: row kernels (one warp per row, a stage number per step), copy / cast / silu / weight-layout copies, an untiled reference window attention, the tiled window attention (TMA tiles, wgmma, persistent CTAs of 64 rows), fused A = 1 backward stages, cuBLAS GEMMs |
| `swa_ffn_fwd.cu` (`fwd`, `fwd_small`) | `ffn_fwd` | gated out-projection + residual + adaLN + SwiGLU + gated residual on wgmma (tiles of 8 augments x 8 atoms) |
| `swa_ffn_bwd.cu` (`bwd`, `bwd_overwrite`) | `ffn_bwd` | FFN-half backward on wgmma; writes the bf16 operands of the FFN weight gradients |
| `swa_attn.cu` (`attn`) | `attn_fwd`, `attn_bwd` | the window attention, bf16 mma.sync; backward dq + dk + dv in one kernel |
| `swa_fp32.cu` (`fp32`) | `fwd`, `bwd`, `prep` | the whole fp32 block on tf32 wgmma (weight gradients excluded) |
| `swa_wgrad_wg.cu` (`wgwg`), `swa_wgrad.cu` (`wgrad`) | `wgrad`, `wgrad_f32` | the five weight gradients of the backward in one launch (TMA + wgmma; mma.sync fallback) |
| `swa_wgrad_wg_f32.cu` (`wgwg32`) | `wgrad_f32` | the same on fp32 operands as bf16 hi / lo planes |
| `swa_mod.cu` (`mod`) | `mod_fwd`, `mod_bwd` | built (the composed path requires it) but not launched by any current code path |

`requirements_kernel.csv` lists 43 `swa_dit` kernels: `swa_dit_*_sm90` names for the composed and fp32 stages (`inproj`, `softmax`,
`output_swiglu`, `swiglu`, `output`, `dw`, `transpose`, plus `_rows` and `_fp32` variants), `swa_dit_native_{rows,copy,mod_bwd}_sm90`,
`swa_dit_tiled_attn_{fwd,bwd_dq,bwd_dkdv}_sm90`, `swa_dit_fused_{oproj,qkvg}_bwd_sm90`, and the Triton kernel names without suffix
(`swa_dit_inproj_fwd`, `swa_dit_softmax_fwd`, ...; `kernels/swa_dit/triton/`).

## Design

**Composed bf16 forward.** Four kernels, no torch elementwise op between them: qkvg -> window attention -> out-projection + gated
residual -> FFN + gated residual (the slab path runs `oproj_fwd` and `ffn_fwd` separately). `_rows_cuda` picks the wgmma kernels when
`A % 8 == 0` and rows >= 5120 (qkvg, forward) / 20480 (FFN backward), else the slab kernels; from the module `A = 1`, so the slab
kernels serve. Q / K / V are written head-major `[N, H, S, D]`, G row-major.

**Window attention (`swa_attn.cu`).** Every 64-query tile reaches exactly the 192 keys `[i0 - 64, i0 + 128)` and one warp's 16 queries
144 of them, so the forward is a single pass (no online softmax): each warp holds its 16 x 144 scores in registers. Forward CTA =
(64-query tile, head), 4 warps; K / V 192 rows via cp.async, zero-filled outside `[0, seqused)`; O leaves through shared memory as
16-byte stores. Backward: one kernel, CTA = (64-row tile, head), 8 warps: warps 0-3 own the tile's queries (dq), warps 4-7 its keys
(dk, dv, computed transposed so `p^T` / `ds^T` are register A fragments); dq / dk / dv go row-major into columns of a `[M, 4C]` buffer
`dP` that the qkvg backward and the weight-gradient kernel read. Tiles use an XOR swizzle of 16-byte chunks for conflict-free
ldmatrix.

**Backward chain (bf16).** FFN backward (`dq1`, `dffn`, `h`, `[da | db]`) -> out-projection backward (`datt`, `gated`, `dO`, `dG`,
`D`) -> attention -> qkvg backward (dp_q, dp_k in place, dx, dq) -> one weight-gradient launch: 11 output tiles of 128 x 128, a
cluster of 8 CTAs per tile splits K = rows 8 ways through a 4-stage TMA / mbarrier ring, two consumer warpgroups run wgmma with both
operands MN-major from 128 B-swizzled shared memory, and after a cluster barrier each CTA sums its 16 rows over the eight partials in
rank order through distributed shared memory (deterministic, no workspace). Modulation gradients: slab kernels pre-sum over a slab's
augments with shuffles, then one `red.add` per (atom, channels), or a plain store when one slab holds every augment (A in 1, 2, 4, 8, 16;
then `dmod` needs no zero fill).

**fp32 block (`swa_fp32.cu`).** A 64-row tile (SP augments x AT atoms) runs on a pair of one-warpgroup CTAs, persistent over tiles; the
pair splits the GEMM work and each streams half the weights through its own ring of 16 KB TMA stages; row tensors move by TMA both
ways; partial sums split over K cross to the partner by `st.async` on the partner's mbarrier. A tf32 accumulator is an A operand once
the K axis is permuted inside each group of 8 (`prep` permutes those weights).

**Native stages.** Generic row kernels (one warp per row) around cuBLAS GEMMs, for the cases the composed stages do not take. The
tiled attention keeps all three 64 x 64 score tiles of a query block in registers until the window maximum is known; each persistent
CTA owns 64 rows, two TMA slots overlap adjacent items; the backward owns one dQ or one (dK, dV) tile (no atomics).

**Programmatic dependent launch** (`cuda/pdl.cuh`; slab, attention and weight-gradient kernels): a kernel may be scheduled while its
predecessor still runs, copies its weights into shared memory, then blocks at `pdl_gate()` before reading anything a predecessor
writes. Also under CUDA-graph capture.

## Numerics

- bf16 composed: bf16 operands, fp32 accumulation, the rounding points of the Triton kernels (each kernel header lists them); the
  modulation is fp32 `[rows, 768]`.
- fp32 composed (`swa_fp32.cu`): residual stream, norms, modulation, gates, RoPE, SwiGLU and every saved tensor fp32; projection
  GEMMs one tf32 pass (round to nearest, ties away); the SwiGLU FFN GEMMs split the weights into tf32 hi + lo (two passes); the window
  attention on bf16 q / k / v / P / dO / dS with fp32 softmax. Served only while TF32 is allowed.
- fp32 weight gradients: bf16 hi / lo planes, `hi*hi + hi*lo + lo*hi` (~2^-16 relative) up to 65536 rows (`swa_wgrad_wg_f32.cu`; the
  mma.sync kernel up to 8192 where it does not build); above, torch GEMMs at the process's matmul precision.
- Native under `highest`: exact fp32 cuBLAS projections, bf16 attention operands.
- Determinism: output, dq and the block-weight gradients are bitwise repeatable; `dmod` accumulated with `red.add` (A not in
  1, 2, 4, 8, 16) is not.

## Launch configs

The CUDA kernels have fixed tiles and launch shapes; `kernels/swa_dit` has no `configs/h100_sm90.json`. The Triton kernels are
`@configured` (`swa_dit_*_triton`) and would launch their defaults on H100; with every composed extension built, the H100 dispatch does
not reach them.

## Limits (`AtomDiTMiniWorld._supports_cuda`; `SWAAtomTransformer` uses its first block's)

- H100 (sm_90) only; `attention_params` the 6-tuple of `build_attention_params`.
- `x` / `cond` bf16 or fp32, the same dtype; `x` `[N, S, 128]` with 4 heads; `cond` `[N, S, 128]`.
- `half_window` 64, or global (FlashAttention-4 installed).
- Bias-free plain `Linear` weights (no magnitude-preserving layers), SwiGLU hidden 256, weights in the activation dtype or fp32.
- `cos` / `sin` fp32 `[..., 16]` without gradient; `seqused` int32 `[N]`.

## Rejected approaches (brief)

- Thread-block-cluster and TMA weight paths for the row kernels were not the bottleneck: the row kernels are bound by RMSNorm / RoPE /
  modulation-gradient work and atomics per row, not by the MMA instruction set.
- A bf16-plane fp32 modulation kernel was more accurate than the cuBLAS TF32 GEMM but not faster.
- Tying the tiled attention to the B200-style fused row stages sent every sampling step to the untiled attention; the tiled attention
  now serves every native call with S a multiple of 64.

## History

Old page: `git show HEAD:docs/gpus/h100/swa_atom_dit/swa_atom_dit.md`.
