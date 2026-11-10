# Transition and ConditionedTransition on H100 (sm90)

Scope. `modules.Transition` (`modules/transition.py`):

    h = LN(x)                                   # ln_in, affine, fp32-pinned weight / bias
    y = x + W_s (silu(W_a h) * (W_b h))         # expand_a, expand_b, squeeze; no biases; hidden = n * d

Backends: `pytorch` (reference), `cuda` (the sm_90a kernels below), `triton` (the split Triton Transition), `anthropic` (external
baseline, inference only; not described here). `PairformerBlock` hands its backend to its pair / single `Transition`. Requirements rows
(`requirements/requirements_module.csv`): pair stream d 64 / 128 / 256 / 384 / 512 with n 2 and 4, single stream d 384 (n 2, 4) and
d 768 (n 2), MSA stream d 64 / 128 / 256 (n 4); dtypes `float32` and `master` (fp32 parameters, bf16 activations under autocast);
modes eval and train. Row counts are L^2 (pair, L = 128 .. 768) or L (single), so every pair row count is a whole number of 128-row
tiles.

`ConditionedTransition` / `UpGatedConditionedTransition` (AF3 Alg. 25 and Boltz-2's up-gated form) have only the `pytorch` backend. The
`conditioned_transition` kernel family is launched by the token DiT classes (`TokenDiT*`), not by the module; see the last section.

Judging: accuracy and speed of every backend come from `miniworld_engine.measure` / `judge` ([measure](../measure.md),
[modules](../modules.md), [kernels](../kernels.md)).

## Kernels

Names are the `requirements/requirements_kernel.csv` rows of family `transition` (role in brackets); the code symbol follows.

| name | role | what it does |
|---|---|---|
| `transition_fwd_residual_sm90` | forward | D128 fused forward, `transition_fwd_fused` (`cuda/transition_fused_fwd_sm90a_kernel.cu`): LN + expand + SwiGLU + squeeze + residual in one persistent kernel; the saving build also writes xn, rstd, c1 |
| `transition_bwd_residual_sm90` | backward | D128 fused backward, `transition_bwd_fused`: two CTA roles (weight-gradient role, dx role) in one launch |
| `transition_bwd_residual_reduce_sm90` | backward | `reduce_partials`: sums the per-CTA dW / dgamma / dbeta partials in a fixed order |
| `transition_prepare_weights_sm90` | forward | `prepare_k` (`transition_fused_sm90a.cu`): bf16 copies of non-bf16 / non-contiguous weights and Ws^T, one launch |
| `transition_pair_layout_sm90` | forward | `pair_layout_k`: [Wa; Wb] interleaved in row blocks (one GEMM tile holds a and b of the same hidden units), optionally transposed |
| `transition_fwd_d64_sm90` | forward | D64 fused forward (`wide/kernels/transition_fwd_d64.cu`); inference also has 64-row-tile builds (`d64_fwd_small_f2/f3.cu`) |
| `transition_bwd_d64_sm90`, `transition_bwd_d64_reduce_sm90` | backward | D64 two-role fused backward and its partial reduction (`transition_bwd_d64.cu`) |
| `transition_fwd_d256_sm90` | forward | D256 fused forward (`transition_fwd_d256.cu`) |
| `transition_bwd_dxn_layernorm_d256_sm90` | backward | D256 `dxn_lnbwd`: d_xn = [dA\|dB][Wa;Wb] + LayerNorm backward + residual, per-CTA affine partials |
| `transition_wide_affine_reduce_sm90` | backward | `affine_reduce_k`: fixed-order sum of the D256 dgamma / dbeta partials |
| `transition_layernorm_expand_swiglu_wide_sm90` | forward | `ln_swiglu_gemm` (D384 / 512): LayerNorm inside the expand GEMM, xn tile resident, h written |
| `transition_squeeze_residual_wide_sm90` | forward | `squeeze_gemm`: y = h Ws^T + x |
| `transition_expand_swiglu_saveact_wide_sm90` | forward | split expand of the saved-a/b path (D768, D512 at the saved row counts): h, a, b from a precomputed xn |
| `transition_wide_layernorm_fwd_sm90` | forward | `ln_fwd_k`: the standalone LayerNorm of that split forward (xn, rstd, c1) |
| `transition_bwd_gate_wide_sm90` | backward | `gate_gemm2`: dh = dy Ws, [a\|b] recomputed (or read when saved), SwiGLU backward -> h, [dA\|dB] |
| `transition_wide_layernorm_bwd_sm90`, `..._compact_sm90`, `..._reduce_sm90` | backward | `ln_bwd_k` / `ln_bwd_compact{384,512}_k` / `ln_reduce_k`: LayerNorm backward + residual from d_xn, per-warp affine partials, fixed-order reduce |
| `transition_generic_layernorm_fwd_sm90` | forward | `ln_fwd_kernel` (`generic_sm90a/tr90.cuh`): warp per row, xn (+ rstd, c1) |
| `transition_generic_expand_swiglu_sm90` | forward | `gemm_kernel<MODE_SWIGLU>`: h = silu(xn Wa^T) * (xn Wb^T) |
| `transition_generic_expand_swiglu_saveact_sm90` | forward_train | `gemm_kernel<MODE_SWIGLU_SAVE>`: also stores [a\|b] (bf16 only) |
| `transition_generic_squeeze_residual_sm90` | forward | `gemm_kernel<MODE_RES>`: y = h Ws^T + x, residual tile TMA-loaded into the staging buffer |
| `transition_generic_bwd_gate_saveact_sm90` | backward | `gemm_kernel<MODE_GATE>`: dh = dy Ws with the saved a, b, SwiGLU backward in the epilogue |
| `transition_generic_bwd_gate_recompute_sm90` | backward | `gate_kernel`: dh, a, b over K = D, SwiGLU backward -> h, [dA\|dB] |
| `transition_generic_bwd_dxn_sm90` | backward | `gemm_kernel<MODE_OUT>`: d_xn = [dA\|dB][Wa;Wb] |
| `transition_generic_layernorm_bwd_sm90`, `..._reduce_sm90` | backward | `ln_bwd_kernel` + `reduce_cols_kernel` (PDL dependent): LN backward + residual, fixed-order affine sum |
| `transition_generic_prepare_weights_sm90` | forward | `prep_bwd_kernel`: Ws^T, [Wa;Wb]^T and TF32-rounded copies, chosen by a job mask, one launch |
| `transition_generic_h16_absmax_sm90`, `transition_generic_h16_apply_sm90` | forward | the fp16-intermediate scales of the large-row fp32 forward (see Numerics) |
| `transition_generic_swiglu_fp32_sm90`, `transition_generic_bwd_swiglu_fp32_sm90` | forward_train / backward | elementwise SwiGLU / SwiGLU backward of the small-row fp32 cuBLAS plan |
| `transition_fwd_f32io_sm90` | forward_infer | `transition_fwd_f32io` (`wide/kernels/transition_fwd_f32io.cu`, bound by `wide/f32io.cu`): the fp32 inference forward at D 64 / 128 (n 2, 4) in one launch (see below) |
| `transition_expand_swiglu` | forward | Triton `transition_fwd_kernel` (`triton/main.py`) |
| `transition_squeeze_residual`, `transition_squeeze_residual_splitk_reduce` | forward | Triton `_squeeze_residual_kernel` and its split-K sum `_split_k_residual_kernel` (`triton/residual.py`) |
| `transition_bwd_squeeze_swiglu` | backward | Triton `_squeeze_swiglu_bwd_kernel` (`triton/training.py`) |

The weight-gradient GEMMs of the wide and generic paths, the generic small-row plans and the Triton backward's dx / dW products are
cuBLAS (`torch.mm` / `addmm`), not engine kernels.

## Design: the `cuda` backend

`kernels/transition/cuda/__init__.py::transition_sm90a` takes the first build that serves the call:

1. `fused_generic_sm90a`, if `prefers_generic` (bf16 only; at few rows the row-tile-per-CTA builds leave SMs idle). Row limits per D,
   (inference, training): D64 (3072, 3072), D128 (6144, 3072), D256 (6144, 12288), D384 (16384, 16384), D512 (16384, 12288). D64
   inference never prefers it; D128 inference prefers it up to 512 rows (n 2) / 4096 rows (n 4).
2. `fused_sm90a`: D128, hidden 256 or 512, bf16, rows a multiple of 128.
3. `fused_wide_sm90a`: D 64 / 256 / 384 / 512 (n 2, 4) and D768 (n 2), bf16, rows a multiple of 128.
4. `fused_generic_sm90a`: D a multiple of 64 up to 1024, hidden a multiple of 128, any row count, bf16, or fp32 when TF32 is allowed.

`None` (no build compiled) becomes `BackendUnsupported` in the module. Inference vs training is decided at the entry (grad mode and an
operand that requires grad), not inside the autograd Function, so a no-grad call never runs the saving build.

### D128 fused (`fused_sm90a`)

Forward: a persistent CTA per 128-row tile (grid = SM count, compiled in), two consumer warpgroups of 64 rows. Weights stream through a
two-slot TMA ring of 64-unit hidden chunks ([Wa_j; Wb_j] 32 KB + Ws^T_j 16 KB per slot). Per chunk: `[a|b] = xn [Wa_j; Wb_j]^T` as one
`m64n128` chain; `h_j = bf16(silu(a) b)` is formed directly in the wgmma A-fragment registers (the m64n64 C fragment is the m64k64 A
fragment); `acc += h_j Ws^T_j` as an RS `m64n128` accumulated over all chunks; `out = bf16(x + acc)`. The [M, n*128] SwiGLU activation
never reaches HBM. The inference build (`_FWD_TUNING`): LayerNorm in the A-fragment layout with xn kept in registers (`LNFRAG=1`; the
hidden-chunk loop must be unrolled, `RSMASK` bit 8, or ptxas aliases the loop-carried RS operand), the x tile's TMA issued before the
weight chunks (`XFIRST`), and a warpgroup ping-pong (`DECOUPLE`: whichever warpgroup retires a slot last refills it; `SKEW`: warpgroup 1
starts half a chunk late). The saving (training) build turns those off, keeps the shared-memory LayerNorm and writes xn, rstd and
`c1 = mean * rstd`. Recomputing xn in the backward instead of saving it was measured slower and dropped.

Backward: one launch with two roles, no cluster. DW CTAs (2n hidden slices x `dw_repl` replicas) keep one slice's Ws / Wa / Wb resident
(48 KB), recompute dh, a, b for their rows, and accumulate dWa_s, dWb_s and dWs^T_s (96 fp32 accumulators per thread). DX CTAs stream
all hidden chunks through a two-slot ring, form dA / dB as bf16 in A-fragment registers and accumulate d_xn with an RS `m64n128` over
the packed [Wa; Wb]; the LayerNorm backward, the residual and the dgamma / dbeta partials follow in the same fragment layout. Partials
are one private row per warp / CTA, summed by `reduce_partials`: no atomics, a replay is bit-identical. A cluster design that split the
hidden axis and reduce-scattered d_xn through DSMEM was 2.5x slower and was dropped.

### Other widths (`fused_wide_sm90a`)

| D | forward | backward |
|---|---|---|
| 64 | one fused kernel (2 CTAs / SM); inference at a multiple of 64 rows: 64-row tile builds, LN in fragment layout, 3 or 2 CTAs / SM (`_d64_variant`) | fused two-role kernel + partial reduction (26 weight replicas at n 2, 16 at n 4) |
| 256 | one fused kernel | `gate` -> fp32-output cuBLAS dW GEMMs -> `dxln_partials` (d_xn + LN backward) -> `reduce_affine` |
| 384, 512 | `ln_swiglu_gemm` (LN inside the expand GEMM, xn tile resident per 128-row block) + `squeeze_gemm` | `gate` -> cuBLAS dW -> cuBLAS d_xn -> `ln_bwd` (compact build at the D512 saved row counts) |
| 768 (n 2) | `ln_fwd` + split expand that saves h, a, b + `squeeze` | `gate_saved` (dh GEMM only) -> cuBLAS dW -> cuBLAS d_xn -> `ln_bwd` |

At D >= 256 a single fused backward does not fit (the dW accumulators exceed the register file), so h and [dA|dB] round-trip HBM.
D384 keeps the forward's h for the backward (`save_h_enabled`); D512 keeps a / b only at 384^2 and 768^2 rows (`save_ab_enabled`).

### fp32 inference at D 64 / 128 (`fused_wide_sm90a.transition_f32_sm90a`, f32io)

One fused kernel, the D64 forward's structure (LNFRAG) with fp32 x / y: a persistent CTA per 64-row tile per consumer warpgroup; the
fp32 x tile is TMA-loaded as 32-column 128-B-swizzled blocks, the LayerNorm runs in fp32 in the wgmma A-fragment layout and leaves xn in
registers as fp16; [Wa_j; Wb_j] and Ws^T_j (fp16) stream through a two-slot TMA ring; SwiGLU on the fp32 accumulators goes straight into
the A fragments of the squeeze; y = x + acc in fp32 over the x tile, one TMA store per half. HBM traffic is x in and y out (the three-launch
TF32 path also moved xn and h). Builds (`wide/f32io.cu` switch; `_f32_variant` picks 1): D64 `<HS 64, 1 WG, 1 x buffer, 3 CTAs / SM>`,
D128 `<HS 64, 2 WGs, 2 x buffers, 1 CTA / SM>`. Training keeps the generic path.

### Generic (`fused_generic_sm90a`, sources `generic_sm90a/`)

One persistent GEMM skeleton (`tr90.cuh`): a producer warpgroup (`setmaxnreg` 40) TMA-loads 128-B-swizzled K-major slabs through an
mbarrier ring; two consumer warpgroups (`setmaxnreg` 232, 64 rows each) run wgmma SS with fp32 accumulators. bf16 uses wgmma k16, fp32
uses TF32 wgmma k8. The ragged tail of the last 128-row tile is zero-filled by TMA on load and clipped on store. Output tile width is
`pick_width` (256 / 192 / 128 / 64 columns: the fewest waves) so a few rows still fill the GPU. bf16 outputs are staged in shared memory
and stored by TMA; fp32 outputs are stored directly. Launches are chained with programmatic dependent launch.

- Forward: `ln_fwd` -> SwiGLU GEMM -> squeeze GEMM with the residual. bf16 training uses `SWIGLU_SAVE` (keeps [a|b] and h from the fp32
  accumulators). LayerNorm inside the expand GEMM was built and measured 5-25 % slower at 768 .. 12288 rows; it stays a separate kernel.
- Backward, bf16 (a|b saved): `MODE_GATE` (dh = dy Ws, Ws read as stored, MN-major) -> d_xn (cuBLAS over a concatenated [Wa; Wb] at
  <= 12288 rows, else `gemm<OUT>` reading Wa / Wb as stored) -> `ln_bwd` + `reduce_cols`. The dW GEMMs (dWs = dy^T h, [dWa; dWb] =
  [dA|dB]^T xn) run on a side stream at small rows (`_DW_ASIDE_ROWS`; `MINIWORLD_TRANSITION_GENERIC_DW_ASIDE=0` runs them in line).
- Backward, fp32 (recompute): `prep_bwd` (Ws^T, [Wa;Wb]^T, rounded copies; fp32 training makes all copies once per step in the forward)
  -> `gate_kernel` -> `gemm<OUT>` -> `ln_bwd` + `reduce_cols`; dW by cuBLAS. TF32 wgmma cannot read an M-major operand, which is why dW
  stays cuBLAS.
- fp32 plan switches by shape: training with rows <= min(d, 2048) and d >= 512 uses cuBLAS TF32 for [a|b] and the squeeze around the
  hand `ln_fwd` / `swiglu_f32` (and `swiglu_bwd` in the backward); fp32 inference uses the squeeze as cuBLAS `addmm` at <= 512 rows.

### `triton` backend

`transition_residual` (`triton/residual.py`): the input LayerNorm that also returns the residual (`trimul_inproj` `input_ln_residual`,
so the residual and LN gradients meet in one LN backward), the expand-SwiGLU kernel over a row-interleaved [Wa; Wb] (one dot of N =
2*BLOCK_N produces a and b side by side), and the squeeze with the residual in its epilogue (split-K plus an ordered sum when the tiles do
not fill the SMs). Training stores a and b in the activation dtype (fp32 activations above 4096 rows: fp16 divided by a device scale
derived from the same bound as the CUDA h16 path, `residual.half_scale`); the
backward kernel reads them, computes dh and the gate, and writes [dA|dB] so dx and [dWa; dWb] are single stacked cuBLAS GEMMs. Any width;
bf16 / fp16 / fp32 activations; parameters cast to the activation dtype.

## Numerics

- LayerNorm statistics in fp32; the engine kernels round xn to the activation dtype before the GEMMs (as the reference).
- The D128 / D64 kernels compute sigmoid as `rcp.approx(1 + ex2.approx(-a log2 e))`; the wide widths use a `tanh.approx` sigmoid.
- h goes into the squeeze rounded once from the fp32 accumulator. dW accumulate in fp32 and are rounded once to the weight dtype (fp32
  for fp32 masters). dgamma / dbeta are summed in a fixed order everywhere on the `cuda` backend: replays are bit-identical.
- fp32 activations: every operand the TF32 MMA reads next (xn, h, dA / dB, the weights) is rounded to nearest TF32 where it is produced;
  truncation by the tensor core was a one-sided error. fp32 needs TF32 allowed (`allow_tf32` or matmul precision not `"highest"`);
  the measurement's PyTorch reference runs with TF32 allowed too.
- fp32 inference at D 64 / 128 (f32io): the products run on fp16 tensor cores with fp32 accumulation. fp16 has TF32's 10-bit mantissa and
  rounds to nearest, so xn, Wa, Wb, h and Ws are rounded as the TF32 path rounds them; four device scales (`_f32_forms`, made once per
  parameter version: xn, [Wa, Wb], h, Ws) keep each fp16 operand below 30000 (all 1 for ordinary weights).
- fp32 at >= 16384 rows keeps h in fp16 between the two GEMMs (`_fwd_h16`): `h16_absmax` / `h16_apply` compute device scales from
  `bound = (sqrt(D) max|gamma| + max|beta|) * max row L1 of Wa, Wb` so h and Ws stay inside fp16's range (scale 1 for ordinary weights).
- The fp32 small-row training plan (cuBLAS) and the hand-kernel inference are two TF32 GEMM plans: the no-grad forward is not bit-equal to
  the training forward at those shapes.
- The Triton fp32 path uses IEEE dots unless torch allows TF32, then TF32 operands rounded once (`kernels/_tf32.py`).

## Launch configs

`kernels/transition/configs/h100_sm90.json`: `transition_fwd_residual_sm90_cuda` / `transition_bwd_residual_sm90_cuda` (D128 fused:
`ctas` and `dw_repl` per (M bucket, ND, SMS); candidates from `transition_candidates`, default `ctas = SMs`, `dw_repl` 8 at n 4, 15 at
n 2) and the three Triton ops `transition_expand_swiglu_triton`, `transition_squeeze_residual_triton`,
`transition_bwd_squeeze_swiglu_triton`. Top 4 per (dtype, shape bucket), rank 1 by default. The wide and generic builds have no table:
their schedules are compile-time constants and the shape rules above.

## Limits (`_supports_cuda`, `_supports_triton`)

- `cuda`: H100 (sm_90) CUDA tensors; `x [..., d_hidden]`, non-empty; affine LayerNorm; bf16 or fp32 activations; the three projections
  share one dtype (bf16 or fp32). Otherwise some build must take the shape: generic needs d a multiple of 64 up to 1024 and hidden a
  multiple of 128, and fp32 needs TF32 allowed.
- `triton`: CUDA tensors, bf16 / fp16 / fp32, same affine / shape checks; any width.
- A `dead` instance returns the input in inference on every backend.

## ConditionedTransition kernels (family `conditioned_transition`)

`ConditionedTransition` itself is PyTorch-only. Its math inside the H100 token DiT block runs on the `cond_transition_token_*` rows:
`token_dit_rows.cu` (inference row passes: AdaLN rows, residual-gate + AdaLN, SwiGLU rows, QK-norm, pair LayerNorm / bias, weight
packing) and `token_dit_train_rows.cu` (training forward / backward row passes and their fixed-order partial reductions), plus
`cond_transition_gemm_swiglu_sm90` (`gemm_swiglu_sm90.cu`: the expand GEMM with SwiGLU in its epilogue, bf16 or TF32, persistent,
producer + two consumer warpgroups, `m64n256`, [Wa|Wb] packed per 256-row tile). Products not listed are cuBLAS. These are part of the
token DiT design ([token_dit.md](token_dit.md)). Two rows of the family serve other modules: `cond_transition_caller_grad_sum_sm90` /
`..._add_sm90` (`caller_sum90_k` / `caller_add90_k`) replace autograd's gradient accumulation on the leaves of the H100 fused Transition,
OPM, PWA and TriMul outputs (`integrations/token_dit_train._native_caller_output` / `_native_caller_hooks`).
`kernels/conditioned_transition/configs/h100_sm90.json` holds the Triton ops of that family.

## History

Old note: `git show HEAD:docs/gpus/h100/transition/transition.md` (per-shape timings against anthropic / PyTorch, the D128 forward
tuning log, the n = 2 replica sweeps, rejected experiments: 2 CTAs / SM, 2-CTA clusters with multicast, `addmm` for the residual,
cuBLAS dh + elementwise SwiGLU backward, all-chunks-resident weights, `SPLIT2`, `NEWTON`, `SUBCH`, `STAGGER`, `CPX`).
