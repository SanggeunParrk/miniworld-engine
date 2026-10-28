# H100 module dispatch

`implementation="miniworld"` with the default `force_portable_kernels=False` selects the
packaged H100 implementations below. No research-directory import, external
TriMul payload, or MSA opt-in environment variable is required. Configure the
policy before constructing/compiling models.

| Module | Inference | Training | Native contract |
|---|---|---|---|
| Bidirectional TriMul | fp32 on TF32, D = hidden = 128 (`integrations/trimul_h100_f32.py`, below): LN_in → wgmma-TF32 gated front → two cuBLAS TF32 contractions → LN_out + transpose → wgmma-TF32 back; bf16: Anthropic-derived native K1 → two cuBLAS contractions → K3/residual; D256/384/512 at every L (multiples of 128): B200-style port (`h100_b200style`), L384 / L768 D512 L384 keep the preserved K1/K3 | fp32 on TF32 (D128): the same forward saving x_n, planes, tri, on, gate logit, projection; gate backward, LN_out backward as wgmma, four cuBLAS contraction gradients, `front_bwd` (front GEMM again + gradient epilogue), cuBLAS dW / dx_n, LN_in backward; bf16 D128: selected CUDA forward + B1–B4 + four cuBLAS contractions + B7–B12. D256/384/512: flattened port of the qualified large-width research plans (below), B200-style port at the other lengths | fp32 (TF32, no autocast, matmul precision not "highest"): D = hidden = 128, inference any L >= 16, training L a multiple of 4, B samples one after the other (checkpointed above half the device memory), bitwise run to run. BF16, batch B <= `max_batch(D)` = 48 at every width (samples b-major in one call; D256/384/512 B > 1 on the B200-style port, inference at L % 128 != 0 padded to 128 on the host; `MAX_BATCH` = the D64 value); D64/128: inference any L >= 16 (one sample pads to 16 inside K1 / K3, B samples at L % 16 != 0 are padded on the host), training L a multiple of 128 up to 10240 (`MAX_TRAINING_LENGTH`); D256/384/512: L a multiple of 128 (above L1024 the B200-style port); B L^2 < 2^31; direction hidden=D, D64/128/256/384/512; D128 requires a full 132-SM H100 |
| Single-direction TriMul | D64/128: packaged K1 → contraction → K3/residual; D256: the same K1/K3 tiles at every L except 384 (B200-style port at L384); D384/512: B200-style port; D512 L384/768 may use `h100_uni_wide_inference` | D64/128: native K1/K3, fused CUDA B1, two cuBLAS gradient contractions, streamed producer-consumer B7. D256/384/512: the wide path `h100_uni_wide_training` (front LN → K1 → cuBLAS contraction → LN → projection / gate GEMMs → output epilogue; backward gate, dn GEMM, output-LN backward, `contract_gp`, weight GEMMs, input-LN backward) | BF16, batch B <= `max_batch(D)` = 48 at every width (D256/384/512 B > 1 on the B200-style port at multiples of 128, inference at other L padded to 128 on the host), D=hidden=64/128/256/384/512 (or hidden = 2D at D64 / D128: the bidirectional D64 / D128 kernels with a one-direction contraction, `trimul_h100._hidden_twice`); D64/128: inference any L >= 16, training L a multiple of 128 up to 10240; D256 / D384 inference: the packaged K1/K3 at any L except where the B200-style port serves (D256 L384, D384 multiples of 128 up to 1024); D512 and wide training: L a multiple of 128 (training: wide path L384/768 and the B200-style port elsewhere); both outgoing and incoming; D128 requires a full 132-SM H100 |
| TriangleAttention (`use_self_attention=True`) | `integrations/triattn_h100_sm90.py`: d_pair 128 / 4 x 32 fused wgmma front → attention core → tail; other widths LayerNorm → cuBLAS projection → core (gate in its epilogue) → cublasLt out-projection + residual | The same forward saving q / k / v / g, o, LSE; gate backward → key-side and query-side core backward → projection / LayerNorm backward, weight gradients (fused: tensor-core kernels; generic: cuBLAS) | BF16, starting and ending node, d_pair a multiple of 8 up to 512 with heads of 16 / 32 channels (at most 16), L a multiple of 16 (padded to 128); B samples per call: up to `max_batch` = 48 per launch, more (or more than fit in memory) in chunks of `plan(...)` (shapes, mode, the device's total memory), never a size-driven fallback; B L^2 < 2^31 per launch. QK-norm, the bias-only variant, `BidirectionalTriangleAttention` and heads of 64+ channels: PyTorch statements |
| Transition | Residual-fused hand-CUDA path (tuned fused kernels, else the generic wgmma kernels) | Same forward saving xn / rstd / c1, native backward (dW: cuBLAS) | BF16 or FP32 on TF32 (needs `set_float32_matmul_precision("high")` or `allow_tf32`); tuned: n=4 D64/128/256/384/512 and n=2 D64/128/256 with rows a multiple of 128; generic: any D multiple of 64 up to 1024, hidden a multiple of 128, any row count |
| OuterProductMean | Packaged OPM with residual in the CUDA epilogue, no LN statistics saved | Residual-fused forward and native backward; residual gradient is passed through | BF16, batch 1, hidden32; (MSA, pair) 64/128 (MiniWorld), 128/256 (Protenix-v2, ESMFold2) or 128/384 (OpenDDE); any L and MSA depth (padded to 64 / 256 with masked rows); either normalization order; no interchain masking |
| MSAPairWeightedAveraging | Packaged forward without training saves; N ≥ 512: cuBLAS contraction + `pwa_gateout` | Packaged forward/backward, residual and row dropout; N ≥ 512: same split forward; dv on cuBLAS, read tile by tile by `dgv_bwd`; no pair-LayerNorm beta (the module has none) | BF16, batch 1. MiniWorld: MSA64/pair128, 8 heads × 32, L multiple of 128, even MSA depth. Wide (`pwa_wide`): MSA 64 or 128, pair 128 / 256 / 384, 8 heads × 8 or 16 (ESMFold2, Protenix-v2, OpenDDE), L ≤ 1024 (padded to 16), any depth |
| AdaptiveLayerNorm | One entry (`kernels/adaln/cuda.adaln`): `ln_rows` (LN(cond), statistics of x) + one wgmma GEMM over [Wscale \| Wbias] with the gate and the LayerNorm of x in its epilogue (`adaln_sm90.cu`), the second launch under PDL; **atom width (d = c = 128): ONE launch (`adaln_atom_k`)**, LN(cond) and the statistics of x computed in shared memory on the TMA-staged tiles; cast / packed weights cached per weight pack | The same forward saving the pre-activation and the statistics (bf16; fp32 at the atom width: `adaln_atom_k` with the saves), the row-kernel backward `adaln_rows.cu` + cuBLAS (fp32 token width: one-launch weight packing `pack_f32_k`, and with a cond of 768 columns the feature-major row pass `adaln_bwd_t_k` feeding the TF32 weight-gradient GEMM); atom width: `adaln_dgrad_atom_k` (dgrad + the LayerNorm backward of cond in one wgmma kernel) | bf16, or fp32 on TF32; widths 128 .. 1024, `d_hidden` a multiple of 64 for the wgmma tile |
| ConditionedTransition | AdaLN above, then `ct_sm90.cu`: expand + SwiGLU as one wgmma GEMM, squeeze + gate + residual as one (second GEMM under PDL; 64-column output tiles while a 128-column tiling would leave most SMs idle, i.e. M <= ~1300 rows) | AdaLN above, then `ct_rows.cu` row kernels between cuBLAS GEMMs (`ct_train.py`; fp32: the K-major weight copies are packed by one launch) | bf16, or fp32 on TF32; d 768, hidden 1536 (any d multiple of 64 / hidden multiple of 128 for inference) |
| Token DiT | Fused step (`integrations/token_dit.py`): CUDA rows, CUDA gated core (bf16 `attn_fwd.cu` GATED, fp32 TF32 `attn_tf32.cu` with a v^T GEMM and a key-permuted pair bias), cuBLAS GEMMs; capture-scoped weight-pack and pair-bias caches | One opaque fwd / bwd per block (`integrations/token_dit_train.py`): CUDA rows (bf16 or fp32-operand build), cuBLAS GEMMs, bf16 `attn_fwd` + `attn_dqb`/`attn_dkv` or fp32 TF32 `attn_tf32` fwd + `dqb` + `dkv` | BF16/FP32, batch 1, single768/condition384/pair128, 16 heads, expansion1536, L multiple of 128, QK norm on or off, key mask [1, L]; inference: shared or per-sample conditioning ([token_dit/token_dit.md](token_dit/token_dit.md)) |
| Bias-only token DiT (`BiasOnlyDiTBlock`) | B = 1: `integrations/bias_only_dit.py` -> `FusedBiasOnlyDiT` (hoisted P per pair, `pvx` / `pvx32`, cuBLAS, CUDA rows). A pair of its own per sample (single [1, B, L, 768], pair [B, L, L, 128] or `bias=` [B, H, L, L], B > 1): `serves_psb` / `update_psb`: one-pass pair LayerNorm + bias (`pair_bias_k` / `pair_bias_tc_k`), `psb_fwd` (bf16) / `psb32i_fwd` (fp32 TF32) core, the 12-launch step, B split into chunks past 2 GiB of bias | One opaque fwd / bwd per block (`integrations/bias_only_dit_train.py`): shared bias, or `block_psb` for a bias of its own per sample (`psb_fwd` / `psb_bwd`, `psb32_*`), cuBLAS weight gradients, hoisted pair bias (`bias_only_dit_hoist.py`) | BF16/FP32 (TF32), 768 / cond 384 / pair 128, 16 x 48 (also 24 x 32, 12 x 64, 16 x 64), L multiple of 128 (inference B = 1 up to 3072; per-sample pair and training up to 768), key mask [1, L] (or [B, L] per-sample); [bias_only_dit/bias_only_dit.md](bias_only_dit/bias_only_dit.md) |
| Local atom DiT (AF3 windowed, cross-attention) | `integrations/local_dit_h100.py`: hoisted conditioning tables and windowed pair bias, `pre90` (AdaLN 1 + q / gate + K / V AdaLN + k / v), `mma.sync` window attention, `post90` (gate, out-projection, AdaLN 2, SwiGLU, residuals); wgmma row kernels, TF32 or bf16 | One opaque fwd / bwd per block: row kernels `cond90` `pre90` `post90` / `tr_bwd90` `post_bwd90` `pre_bwd90` `cond_bwd90` (wgmma, register A), `mma.sync` attention fwd / dq / dkv, cuBLAS weight gradients, `finalize90` (gradient packing); ESMFold2 variant (bf16, shared conditioning): `local_dit_h100._esm_train`, opaque fwd / bwd over the DiT-variant row kernels (`train_rows.cu`, RMS + raw-gate switch), cuBLAS, `lattn90_esm.cu` attention fwd / dq / dkv, `qknorm_rope_bwd` | fp32 activations + fp32 parameters (no autocast, TF32) or bf16 activations (fp32 master or bf16 parameters, autocast allowed); B = 1, d128 / cond 128 / pair 16, 4 x 32 heads, n = 2, any A, any N (padded to 128 inside), key mask [1, N]; `local_dit_hoist` ([local_dit/local_dit.md](local_dit/local_dit.md)); few 64-row tiles: weight packs in one launch (`pack90`), `cond90` / `pre90` / `tr_bwd90` / `post_bwd90` / `cond_bwd90` split over CTAs (`grid.y`), shared conditioning: four samples per CTA group and `cond_bwd_sh90` |
| DiT variants (Boltz-2, RoseTTAFold3, Chai-1; token and atom stacks) | `kernels/dit_variants/interface.py` (bf16, no grad): per-step conditioning tables, one pair LayerNorm + GEMM for every block's bias, CUDA rows (`rows.cu`), cuBLAS, the token DiT / local DiT attention cores | `kernels/dit_variants/cuda/train.py`: one opaque fwd / bwd per block, CUDA rows (`train_rows.cu`, `edge.cu`), cuBLAS, the sm_90a attention cores (token `attn_fwd` / `attn_dqb` / `attn_dkv` or TF32 `attn_tf32`, atom `lattn90`); the conditioning tables and the pair bias are autograd ops outside the block, so their gradients chain through the modules' own layers | token: 768 wide, 16 x 48 heads, L a multiple of 128, conditioning `[L, c]` shared by the samples, pair 128 / 256 wide, bf16 or fp32 (TF32), any number of samples; atom: 128 wide, 4 x 32 heads, N a multiple of 64, conditioning `[N, 128]`, trunked pair 16 wide, slid or padded key window, Chai-1's `uid` mask; inference is bf16 only ([model_variants/model_variants.md](model_variants/model_variants.md)) |
| AttentionPairBias fp32 (TF32) | `kernels/augmented_attention/cuda/apb_tf32.py` (`modules/attention_pair_bias`): cuBLAS q/k/v/g GEMM, `apb_prep`, `pb_fwd` (pair LayerNorm + head projection on mma.sync TF32, key-permuted bias, mask folded in), the TF32 wgmma core `attn_tf32.cu`, `apb_gate`, cuBLAS to_out + residual | One opaque fwd / bwd: forward as inference saving q+bias, the packed projections and the bias; backward `apb_bwd_prep`, exact-fp32 cuBLAS GEMMs for S / dP / dq / dk / dv around `apb_bwd_rows` (softmax rows, D, dS), TF32 cuBLAS dx / dW, `pb_bwd_tc` (d_pair 128) or `pb_bwd_rows` (d_pair 256) | FP32 on TF32 (`set_float32_matmul_precision("high")` or `allow_tf32`; else PyTorch), batch 1, L a multiple of 128, heads x width 8x48 / 12x32 / 16x24 / 16x32 / 24x16, d_pair 128 (8x48 also 256), no QK norm, key mask [1, L], `gate_bias` ([attention_pair_bias/attention_pair_bias.md](attention_pair_bias/attention_pair_bias.md)) |

A call outside these contracts runs the PyTorch reference (next section). `auto` selects the bidirectional CUDA training path at
D64/128/256/384/512 (D64: its own fused path `h100_d64_b200gate`). Inference: the packaged K1/K3 table (D64/128, uni D256/384), the
wide bidirectional K1/K3 `h100_wide_inference` (D256/384/512) and the single-direction D512 path `h100_uni_wide_inference`.
Single-direction training serves D64/128/256/384/512 in both directions. Per-shape status: `docs/gpus/h100/h100.md`; comparison
matrix: `docs/records/trimul_h100_comparison_20261015.md`. Measurements and audits of the v2.2.0 integration that this page used to
carry (production-module audit, wide training against the portable path, the master-parameter summary):
[docs/records/h100_dispatch_20260923_20261005.md](../../records/h100_dispatch_20260923_20261005.md).

## FP32 (TF32) bidirectional TriMul, D128 (2026-10-27)

`BidirectionalTriangleMultiplication.forward` asks `integrations/trimul_h100_f32.serves(module, pair, mask, dropscale)` before the bf16 H100 path. Served when:
`implementation="miniworld"`, `force_portable_kernels` is off, the module is the bidirectional one with `d_pair == d_hidden == 128`,
`pair` is a contiguous fp32 `[B, L, L, 128]` CUDA tensor on a sm_90 card, no autocast, TF32 allowed (`torch.get_float32_matmul_precision() != "highest"` or
`torch.backends.cuda.matmul.allow_tf32`), the parameters are fp32 (the D128 module's column-major front matrices included), `L >= 16` (training: `L % 4 == 0`), `mask` a bool
`[B, L]`, the dropout scale `[B, 1, L, 128]`. B samples run one after the other; when their saved state (5.65 kB per pair token) would pass half the device's total memory every sample
is checkpointed (`CHECKPOINT_FRACTION`; a function of the shapes and the card, so compile and graph capture see the same plan). Anything else takes the next path: the PyTorch
statements on H100. Kernels, stages and numbers: [trimul/trimul.md](trimul/trimul.md#fp32-activations-on-tf32-tensor-cores--bidirectional-d128-2026-10-27).

## No portable-kernel fallback; PyTorch where no CUDA path serves the call (intentional)

H100 (sm_90) and B200 (sm_100) run hand-written CUDA and cuBLAS. **There is no portable-kernel fallback on either card, and that is deliberate.**
A `MINIWORLD` call that no CUDA path serves (a ragged length, an unbuilt width, fp32 without TF32, a dtype or layout outside a kernel's
contract) is not routed to the portable kernel family -- untuned on these cards, unmeasured against the CUDA path, easy to mistake for it in a
benchmark -- and it is not refused either: it runs the **PyTorch reference** (`torch.compile` fuses it where the caller compiles), always
with a correct result. `modules.dispatch.restricted_card(device)` is the single test; each module takes its PyTorch leg when it is true.

* LayerNorm (`modules.LayerNorm` MINIWORLD, `architecture.primitives.LayerNorm` MINIWORLD_ENGINE) runs the general CUDA LayerNorm
  (`kernels/layernorm/cuda/general`: any width up to 64K bf16 / 32K fp32, any row count, bf16 / fp16 / fp32, forward and backward,
  bit-identical replays); a multi-dimension `normalized_shape` or a CPU tensor runs the PyTorch statements.
* RMSNorm (`modules.RMSNorm` MINIWORLD) runs the general CUDA RMSNorm (`kernels/rmsnorm/cuda`, the LayerNorm kernel without mean
  and bias; same widths, dtypes and determinism; returns the input dtype); a multi-dimension `normalized_shape` or a CPU tensor runs
  the PyTorch statements.
* AdaptiveLayerNorm and ConditionedTransition run their hand-written sm_90a kernels on H100 (`dispatch._cuda_on_h100`, `KernelBackend.CUDA`: `kernels/adaln/cuda`,
  `kernels/conditioned_transition/cuda`; a call they do not serve runs the PyTorch reference) and resolve to PyTorch on B200 (open work there).
* TriMul (single / bidirectional), TriangleAttention (single / bidirectional / bias-only), AttentionPairBias, AugmentedAttention,
  OuterProductMean, MSAPairWeightedAveraging, Transition and the SWA RMS norm take the CUDA path where it exists and the PyTorch statements
  otherwise. The former `ValueError` refusals of those modules ("H100 ... requires the native CUDA contract") are gone.
* The architecture blocks that call kernels directly rather than through the module dispatch obey it too: `SWAAtomBlock` / `SwiGLUFFN`
  (`architecture/blocks/swa_atom_transformer.py`: the RMSNorm-adaLN modulation and the SwiGLU FFN) and `SWA3DRoPEAttention` (QK norm +
  RoPE, gate, output projection) run their PyTorch statements; the token DiT's fp32 Hopper path declines (instead of raising) a call
  outside its buckets. `tests/gpu/sm90/test_architecture_no_triton_gpu.py` records every Triton launch through the atom transformer
  (windowed, AF3 and global attention), the Pairformer and the diffusion transformer: none, bf16 and fp32, inference and training. A
  MiniWorld phase-1a training run with `train.forbid_triton` on ran 67 steps on H100 without a Triton launch (2026-10-18).
* The engine `SWADiTBlock` (`modules/swa_dit`) does the same: its leg for a contract the fused block refuses, and its `SwiGLUFFN`, run
  the PyTorch statements on these cards rather than the Triton RMS-adaLN / gated-residual / SwiGLU ops.
* SWA atom DiT in fp32 under matmul precision "highest" (torch's default): the native H100 stages serve it (exact-fp32 cuBLAS
  projections); a call they do not take (B200, `swa_dit_h100_native=False`) runs the PyTorch statements, never the sm_100a TF32 or the
  Triton fp32 kernels (`kernels/swa_dit/interface.py:highest_fp32_on_hopper_blackwell`, `tests/gpu/sm90/test_swa_fp32_highest_gpu.py`).
* Other cards: not this page. `miniworld` runs a CUDA path where one serves the call, else the portable kernel family only on a card the engine has a record for, else a warning and the PyTorch reference: [../../dispatch.md](../../dispatch.md).
* `implementation` is pytorch / miniworld / cuequivariance / anthropic; `miniworld` routes per card. The A/B switch brings the portable kernels back on H100 / B200:
  `settings.force_portable_kernels=True`, or `settings.portable_on_hopper_blackwell=True` (A/B work, building portable-kernel caches). `b200_portable_kernels` does the same on B200.
* Committed H100 / B200 portable-kernel autotune caches are not needed and are not maintained.

Verified with a Triton-launch recorder over every module, bf16 and fp32, inference and training, ragged lengths (L = 100, 128, 200, 384),
widths 64 / 100 / 128 / 192 / 256 / 768: zero Triton launches, zero errors, and the output agrees with the module's
`implementation="pytorch"` on the same weights (bf16: relative error <= 4e-3, fp32: exact).

## Retained values and execution

D128 TriMul training retains input `x_n`, left/right, `tri`, and packed front weights.
Bidirectional D128 additionally retains output-LN statistics; single-direction
D128 recomputes output-LN statistics in B1; D128 projection/gate intermediates are
recomputed in backward. The D256/384/512 saved set is listed in the wide section below. Every forward
owns its saved tensors, and original parameters participate in PyTorch's
saved-tensor version checking. Parameter packing is live on every call, including
CUDA graph replay. There is no process-global activation or weight snapshot.

The module entry points have opaque operators with shape-only implementations,
so supported paths work under `torch.compile(fullgraph=True)`. MSA dropout depends
on `module.training`, including a training-mode call under `torch.no_grad()`.
CUDA sources and includes ship in the wheel; nvcc builds into the user cache,
never into the installed source tree. Warm up each shape before CUDA capture.

OPM has no dropout in the model. Its BF16 pair residual is added after rounding
the projected update, preserving the separate add's numerics. Other residual
dtypes or broadcast shapes retain the general add. PWA applies dropout only to
the update: its forward residual/dropout was already fused; backward now masks
the TMA-loaded gradient tiles in shared memory before both input and weight
gradient GEMMs consume them. The residual gradient stays unmasked. This removes
the separate `[S,N,64]` masked-gradient allocation and HBM write/read.

## Policy and tuning

`force_portable_kernels=True` (or `settings.portable_on_hopper_blackwell=True`) is the A/B opt-out of these automatic paths: it brings the
portable kernel family back on H100, including the residual-fused Transition.
`settings.opm_train=0`, `settings.pwa_train=0` and
`settings.pwa_infer=0` disable their respective integrations.
`trimul_h100_training_widths` can restrict the connected CUDA training widths;
its default is `(64, 128, 256, 384, 512)`.

Native TriMul ships measured tile selections; Token DiT retains its local first-use GEMM selection. The H100 CUDA paths need no
autotune cache (Triton caches matter only for explicit portable-kernel A/B runs). Derived build plans model these native
composites without invoking nvcc on fake tensors.

The research histories remain available under `experiments/`; source hashes for
the selected CUDA bodies are in
`kernels/trimul_inproj/cuda/h100_sources/PROVENANCE.json` within the package.
Anthropic attribution and Apache-2.0 terms apply as described in
[third-party notices](../../../licenses/THIRD_PARTY_NOTICES.md).

## Validation

Validation record (`archive/docs-20260928:docs/records/verdicts/h100-module-wiring.json`): 46 H100 test
executions passed, including a separate wheel installation on CUDA device 1,
all ten TriMul training width/length combinations, masked/dropout gradients,
live-weight CUDA graph replay, token DiT inference graphs, and MSA repeated
compiled calls without recompilation, including noncontiguous inputs. Source/config provenance hashes were
verified. The record distinguishes the initial CPU audit failures from the
successful checks after fixing their static analysis.

## D128 TriMul parameter layout and optimizer resume

The four front matrices of a `miniworld` bidirectional D128/H128 module retain
legacy parameter names and `[out, in]` shapes, with column-major storage. Native
backward reads their transposes as views and returns gradients in the same
storage order. Forward's 256 KiB packed-weight tensor is saved per invocation
and reused by backward. No weights are cached across training steps. Output
projection/gate layouts remain unchanged; their necessary conversions remain.
Existing row-major weights (including `load_state_dict(assign=True)` and external
weights-as-arguments callers) still execute correctly through preparation.

Model checkpoints preserve names, shapes and values. Ordinary `load_state_dict`
copies into the module's new strides. **Old optimizer moments need a one-time
layout migration**, especially with fused AdamW:

```python
from miniworld_engine.integrations.optimizer import align_optimizer_state_layout_

model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
align_optimizer_state_layout_(optimizer)  # before capturing any training graph
```

This only copies same-shape state tensors whose strides differ from their
parameter; it preserves values and scalar step counters. Fresh optimizers
already initialize state with the right layout. See the
paired preparation benchmark (`archive/docs-20260928:docs/records/verdicts/trimul-preparation-20260923/`).

## Single-direction training

Both outgoing and incoming use the native route automatically under the contract
above. A single invocation performs one forward contraction and two backward
contractions; it never computes the other direction or normalizes across 2D.
K1 saves affine `x_n`; K3 fuses output LN, projection, gate, dropout and residual.
B1 recomputes output LN/projection/gate on chip and emits dTri, dGate and parameter
gradients. B7 reuses the bidirectional producer-consumer design with eight
32-channel producers (H=128), four 16-KiB derivative planes and eight consumers
per group. Its bounded ring buffer is not a full L² activation cache.

The input/output LN derivatives are separately checked against FP64 using the
actual incoming BF16 gradients, including zero affine scales. Both directions
also cover static compile, CUDA graph replay with changed weights/masks,
outstanding-forward ownership and fully dropped updates.

Selected schedules live in `h100_sources/single_output/selection.json`.
The measured local search covers 6 K1 tiles, 6 K3 schedules, 4 B1 CTA counts and
10 B7 producer/consumer/ring combinations per length. This is a bounded search,
not a claim of exhaustive configuration tuning or 90% speed-of-light.
D256/384/512 **single-direction training** runs the wide path (`h100_uni_wide_training`) described in the next section.
See the measurements and validation at `archive/docs-20260928:docs/records/verdicts/trimul-single-20260923/`.

## Single-direction wide path (D256/384/512), 2026-10-15

`h100_uni_wide_training` serves outgoing and incoming (`GP_DIR` 1 / 2 in `contract_gp`; the contraction picks the transposed operand).
D512 front: `ln_in_d512.cu` writes `x_n` (one warp per row, fp32 two-pass statistics), then `front_pre_ss.cu` (derived from the frozen
`front_pre_d384.cu`) reads the normalised operand and issues wgmma with A and B from shared memory (no 128-register A fragment set);
tile (1, 64, 4, 4, 1). A 2-CTA multicast variant and an N-split two-warpgroup variant were measured and removed
(3.14 / 3.19 ms against 3.18 ms for the default: no gain). Weight pack, LN-affine fold and input LN of the wide paths
use the CUDA aux kernels (`pack_into`, `fold_into`, `normalize_into`) whenever they run eagerly; the Triton versions only serve compiled
regions. A kernel-name scan of D256/384/512 at L384/768 (36 cells) finds no Triton kernel.

Inference selection (`_legacy_native_faster`): single direction D256 takes the tmn K1/K3 tiles at every L except 384; bidirectional
keeps the preserved K1/K3 at (D384, L768) and (D512, L384).

## Wide bidirectional training (D256/384/512)

`kernels/trimul_inproj/cuda/h100_wide_training.py` is a flattened port of the
qualified research selections recorded in
`archive/wip-main-20260928:experiments/trimul_large_d_vast` (September 27): D256 `d256_pool_checkpoint`,
D384 `wide_checkpoint23`, D512 `wide_checkpoint24`, and D512/L384 with the 4-way
input-weight split. The research checkpoint chains, runtime text patching and
`quack`/CuTe imports are gone: each of the 29 kernels is a frozen `.cu` file under
`h100_sources/wide_train` (the final generated text of its chain, hashes in
`PROVENANCE.json`). The D256 chain constructed a CuTe/quack GEMM that the selected
path never launched; the port has no CuTe/quack/cutlass-DSL dependency.

Stages per call (grids, shared memory and tensor maps match the research plans):

- Forward: packed front K1 (D256 also writes input-LN stats; D384/D512 also save
  channel-major projection pre-activations; D512 normalizes `x` first), two
  contraction GEMMs, output LN (D512: packed normalization plus exact-scalar patch
  records), cuBLASLt projection and gate GEMMs, gate/dropout/residual epilogue.
- Backward: D512 patch application, fallback and row-compacted re-projection;
  output-gate kernel (L384 also zeroes the LN affine sums); dN GEMM; dWproj/dWgate
  (cuBLASLt, split FP32 with a native reduction at D512/L768, torch.mm where the plan
  used it); output-LN backward (cooperative at L768); contraction plus
  gate/projection gradients (D256: native contraction + register-budget or bulk-mask
  source; D384/D512: fused contraction/GP); split-K input-weight partials; one dX GEMM;
  input-LN backward with the ordered partial reductions. D256/L384 overlaps
  dWproj/dWgate with the source kernel and D512/L768 overlaps the joint input dW with
  the dX GEMM on a side stream (graph-capturable fork/join).
- D256/L384's source kernel uses `setmaxnreg` 32/208; its cubin's `.nv.info` launch
  register count is lowered to 120 (two resident CTAs) only after a control-flow walk
  of the SASS (`cuobjdump`) proves every register operand fits each allocation state.
  Without `cuobjdump` or on a failed check the original cubin runs (one CTA/SM).

Saved per forward (owned by that call): `ab`, `tri`, `x_n`, the normalized output-LN
rows, projection and gate products, output-LN mean/rstd; plus D256 packed front
weights, or D384/D512 channel-major pre-activations `[8D, L²]` (plus D512 patch
records). Measured forward-retained memory (saves plus output) equals the Triton
path's at D384/D512 (D512/L768 11.26 vs 11.27 GiB) and is 40% lower at D256
(L768 3.38 vs 5.64 GiB); the removed port retained about 4.5 GiB at D512/L768.
Peak fwd+bwd step memory is within -7%/+10% of Triton (D512/L768 20.7 vs 19.4 GiB).

**cuBLASLt algorithms.** The research plans froze heuristic indices and asserted the
algorithm words under cuBLASLt 12.8.4. torch 2.13+cu129 loads cuBLASLt 12.9.1, whose
lists differ and whose algorithm word 4 is runtime metadata, so those assertions
could not pass unchanged. `h100_sources/wide_train/lt_selection.json` keeps each
frozen algorithm; at first use the port selects the heuristic whose other seven words
match exactly and otherwise uses cuBLASLt's first heuristic with a one-time
`RuntimeWarning` (currently only D256/L384 dWgate). The compacted D512 re-projection
reuses the projection's algorithm after `cublasLtMatmulAlgoCheck`. Validation under
12.9.1: with the research plans' indices both implementations produce bitwise-equal
output, dX and weight gradients; all launched kernels have identical SASS. Reselection
check: the matched frozen configurations are 0.4–3.6% faster end to end than taking
cuBLASLt's first heuristic for every GEMM (paired graph replay, all six cells), so they
remain selected. `h100_wide_training.LT_SELECTION` switches the policy for such checks.

## H100 FP32 master parameters

The qualified BF16-mixed paths keep FP32 master parameters and unrounded FP32 matrix gradients with BF16 compute; weight casts happen
inside the autograd functions and run afresh on every CUDA graph replay. Qualification of 2026-10-05 (per-shape F+B latency, dispatch,
validation): [docs/records/h100_master_weights_20261005.md](../../records/h100_master_weights_20261005.md).

## AttentionPairBias (2026-10-13)

`modules/attention_pair_bias` -> `integrations/attention_pair_bias_b200.py` (the B200 file; sm_90 branches inside). Served when:
`implementation=miniworld`, `force_portable_kernels` off, bf16 `single` / `pair`, B = 1, contiguous, (heads, d_single) in
{(8, 384), (12, 384), (16, 384), (24, 384), (16, 512)}, d_pair 128, no QK-norm, key mask `[1, L]` or none, capability 9.0 or 10.0;
inference `L % 16 == 0`, training `L % 128 == 0` (H100). Anything else takes the module path: the PyTorch reference on H100 (no Triton
fallback, see above), the portable path on other cards.

| switch (`settings`) | default | meaning |
|---|---|---|
| `apb_h100_exact_dwb` | **off** | recompute the `to_bias.weight` gradient in fp32 from bf16-rounded inputs (needs CUTLASS 4.2 headers via `CUTLASS_PATH`); 1.7e-2 -> 0.8e-2 on that one gradient, 2.4x the step |

Saved tensors (training): xa, (mean, rstd), q|k|v|g, bias, O fp32, LSE, og, Wf, packed W q|k|v|g, bf16 Wo, bf16 Wb; the original
parameters are kept for autograd version checks. Parameter gradients are fp32 and are cast to the parameter dtype only when the
parameter is not fp32. Weight packs are cached per parameter version and scoped to the CUDA-graph capture (`kernels/_capture.py`).
