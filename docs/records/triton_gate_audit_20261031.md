# Triton record-gate audit (2026-10-31)

Base: `main` 47b07a1d. Branch: `fix/triton-gate`. The question: on a card the engine holds no validated record for (anything but H100 /
B200 in practice: A100, A5000, A6000, ...), is every Triton kernel a public entry point launches behind the record gate
(`docs/dispatch.md`), and does every op count the portable kernels it launches, for the dtypes and modes it declares?

## Method

1. Every Triton row of `kernels/registry/registry.csv` (backend `triton`, `developed=yes`) against the patterns of
   `autotune.records.OP_KERNELS`.
2. What each op's portable path actually LAUNCHES: `kernels/registry/registry_kernel.units.json` (`dev derive`, sm_86: every
   `registry_module.csv` row run on fake tensors, every Triton launch recorded, per dtype and eval / train), compared with
   `records.triton_kernels(op, dtype, mode)`. This comparison is now a CPU test
   (`tests/cpu/compile/test_triton_gate_audit.py::test_every_derived_launch_is_counted_by_its_op`).
3. Every call site in `modules/`, `ops/`, `architecture/`, `integrations/` that reaches a `kernels/**/triton/**` function, a
   `kernels.triton_*` export or an `ops.*` facade: which gate (op, mode, dtype) runs before it on a card other than H100 / B200.
4. Every `@triton.jit` kernel that is launched (`name[grid](...)`) against the registry's `file` / `symbol` columns.

The emulated-card GPU test (`tests/gpu/any/test_triton_gate_gpu.py`) checks each rewired site both ways on a real card: no record -> one
`UnrecordedCardWarning` and no launch of the gated kernels; a record -> silent, the Triton kernels launch, and the result agrees with the
PyTorch statements.

## Ungated launches found, and what was wired

"Before" is `main`; "after" is this branch. Dtypes / modes are what the site runs (registry_module.csv where it has a row).

| entry point | dtypes x modes | Triton kernels it launches | gate before | gate after |
|---|---|---|---|---|
| `ops.rms_norm_modulation` (`registry_module`: `rms_norm_modulation`, bf16, eval / train) | bf16, fp32 x inference, training | `rmsnorm_adamod_fwd_triton`, `rmsnorm_adamod_bwd_triton` | **none**; family in no `OP_KERNELS` entry | op `rms_norm_modulation` (`rmsnorm_adamod/*`), call precision and mode; else `rmsnorm_adamod_reference` |
| SWADiTBlock per-op path (`modules/swa_dit`) | bf16 x both | the above + `ops.gated_residual` | `swa_dit` (the FUSED block's kernels, not these) | the facades' own gates (`rms_norm_modulation`, `gated_residual`) under the existing `swa_dit` check |
| ESMFold2 `SWAAtomBlock` (`architecture/blocks/swa_atom_transformer.py`) | bf16 / fp32 (autocast) x both | `rmsnorm_adamod_*` via `kernels.triton_rmsnorm_adamod` | **none** (`restricted_card` only) | `rms_norm_modulation` at the compute precision; else the block's PyTorch statements under `fallback_autocast` |
| ESMFold2 `SwiGLUFFN` (same file) | bf16 / fp32 x both | `transition_expand_swiglu_triton`, `transition_bwd_squeeze_swiglu_triton` | **none** | `swiglu_ffn` at the compute precision; else PyTorch under `fallback_autocast` |
| ESMFold2 `SWAAtomTransformer` fused block (`config.fused_triton`, set by `InputFeatureEmbedder`) | bf16, fp32 x both | the 9 bf16 / 5 fp32 `swa_dit_*` kernels | **none** (`swa_dit` never asked) | `swa_dit` in `_fused_ok`; else the per-block path, each part gated |
| `ops.gated_residual` | bf16 / fp32 x both | `gated_projection/triton/residual.py:_forward`, `_backward` (**no registry row**) | **none** | `records.UNREGISTERED_OPS["gated_residual"]`: never recorded, the PyTorch statement off H100 / B200 (same rounding) |
| `ops.layer_norm_linear` (`registry_module`: `layernorm_linear_native`, `layernorm_linear_atom_output`) | bf16 x both | `layernorm_linear_fwd_fp32_triton`, `layernorm_linear_bwd_fp32_triton` | **none** | op `layer_norm_linear`; else `F.layer_norm` + `F.linear` |
| `ops.conditioned_transition` (architecture `ConditionedTransition`, MINIWORLD_ENGINE) | bf16 / fp32 x both | `transition_expand_swiglu_triton`, `transition_bwd_squeeze_swiglu_triton` | **none** | op `swiglu_ffn`; else PyTorch statements |
| `ops.triangle_multiplicative_update`, `ops.bidirectional_triangle_multiplicative_update` | bf16 x both (kernels bf16 only) | the TriMul module's set (below) | **none** | op `triangle_multiplication`; a non-bf16 input is a `KernelRefusedWarning`; else a PyTorch reference of the whole update (`trimul_inproj/whole_op.py::_trimul_pytorch`, checked against the module reference on CPU and against the Triton path on GPU) |
| `ops.triangle_attention` QK norm | bf16 x both | `rmsnorm_fwd_triton`, `rmsnorm_bwd_triton` | `triangle_attention` (does not count the rmsnorm family) | also `rmsnorm`; a non-bf16 core is a refusal |
| `ops.swiglu_ffn` | bf16 x training | + `transition_bwd_squeeze_swiglu_triton` | `swiglu_ffn` counted the forward only | `swiglu_ffn` counts the backward |
| `BiasOnlyAttention` training core (`modules/bias_only_dit`) | bf16 / fp32 x training | `gated_projection_gate_flat_triton`, `gated_projection_bwd_gate_flat_triton` (+ `_cast.py` helper) | **none** | op `bias_only_core`; else the einsum branch |
| `BiasOnlyAttention` pair bias, `pair_bias_all` portable leg | bf16 / fp32 x training (and hoist inference) | `bias_only_attention/triton/pair_bias.py:_ln_bias_fwd_kernel`, `_ln_bias_bwd_kernel` (**no registry row**) | **none** | `UNREGISTERED_OPS["bias_only_pair_bias"]`: off H100 / B200 the PyTorch LayerNorm + projection |
| `Transition` on A100 built with the sm_80 fused kernels (d = 128, n = 4) | bf16 x both | `transition_*`, `layernorm_fwd_saveact_triton`, ... when `fused_sm80` turns a call down (rows % 256, dtype, build) | **none** (`native_cuda=True` skipped the record at build) | `portable_blocked("transition")` before `transition_residual` |
| `LayerNorm`, `RMSNorm` modules (`modules/primitives.py`) | bf16 / fp32 x both | `layernorm_*`, `rmsnorm_*` | inference bf16 at build only | also per call: training and fp32 |
| `AdaptiveLayerNorm`, `ConditionedTransition`, `AugmentedAttention` (registry_module declares bfloat16 \| float32) | fp32 x both | their fp32 kernels | bf16 rows only | `dispatch.fp32_gate`: an fp32 call asks fp32 rows |
| `BidirectionalTriangleMultiplication` | non-bf16 pair | bf16-only TriMul kernels | no `guard_dtype` | `guard_dtype` (as the unidirectional module) |
| ProteinMPNN `auto` (`kernels/mpnn_message`, `kernels/mpnn_edge_mlp` interfaces) | bf16 x both | message: `mpnn_message_*` (+ the unregistered no-grad kernel); edge MLP (sm_86): `mpnn_edge_mlp_*` | **none** (`is_cuda` / contract only) | `mpnn_message` / `mpnn_edge_mlp` records (see below) |

### Ops whose `OP_KERNELS` entry missed kernels they launch (from the derived units)

| op | kernels added | effect on the recorded cards |
|---|---|---|
| `triangle_multiplication` | `layernorm_fwd_saveact_triton`, `layernorm_bwd_split_triton`, `layernorm_fwd_saveact_strided_triton`, `layernorm_bwd_atomic_strided_triton`, `layernorm_bwd_split_mmajor_triton` | none (all `ok` on A100 / A5000 / A6000); training 3 of 13 missing (was 3 of 8) |
| `adaptive_layernorm` | the strided LayerNorm pair (family `layernorm_linear`) | none; 2 of 10 |
| `conditioned_transition` | strided LayerNorm pair, `gated_projection_gate_flat_triton`, `gated_projection_bwd_gate_flat_triton` | none |
| `augmented_attention` | `layernorm_fwd_saveact_triton`, `layernorm_bwd_atomic_triton`, strided pair, `layernorm_linear_fwd_fp32_triton`, `layernorm_linear_bwd_fp32_triton`, `layernorm_linear_fwd_triton` (AttentionPairBias inference) | none |
| `swa_atom_attention` | `gated_projection_gate_flat_triton`, `gated_projection_bwd_gate_flat_triton` | none |
| `swiglu_ffn` | `transition_bwd_squeeze_swiglu_triton` | **training now warns on A100 / A5000 / A6000** (no row for that kernel there) |
| `local_dit`, `dit_variant`, `dit_variant_atom` | `layernorm_fwd_saveact_triton` (an fp32 pair's pair-bias LayerNorm) | none |
| (rule) | `records.kernel_mode("layernorm_fwd_saveact_triton")` is now `inference`: every eval unit launches it, so no inference gate counted the LayerNorm forward that ran | none (`ok` everywhere) |

New entries: `rms_norm_modulation`, `layer_norm_linear`, `bias_only_core`, and the six MPNN families. `test_every_registered_family_is_counted_by_some_op`
holds that every developed Triton row is now counted by some op.

### Decisions that changed on the packaged manifests (bf16; H100 untouched)

| card | change |
|---|---|
| A100, A5000, A6000 | `swiglu_ffn` training: portable -> warns (backward kernel unrecorded) |
| A100 | `rms_norm_modulation`, `layer_norm_linear`, `bias_only_core`: portable (rows `ok`); MPNN `auto` message reduction: warns (no MPNN rows) |
| A5000 | `rms_norm_modulation`: warns (1 of 1 / 2 of 2); MPNN `auto`: portable (every message / edge-MLP row `ok`) |
| A6000 | `rms_norm_modulation`: warns; MPNN `auto`: warns (no message / edge-MLP rows; the A6000 manifest has the edge-tail, node-message and relative-position rows only) |
| all three | `gated_residual` and the bias-only pair-bias kernel: warn (no registry row) |
| A6000, fp32 | `layernorm` fp32 inference: portable (A6000 holds fp32 rows; the inference set used to be empty, so it warned) |

fp32 calls through the facades are now asked at fp32: A100 and A5000 hold no fp32 rows, so an fp32 facade call there warns and runs PyTorch
(before: asked at bf16).

## ProteinMPNN: the decision

MPNN is a separate model family. Its kernels are chosen by backend options (`message_backend`, `edge_mlp_backend`, `edge_dropout_backend`,
`edge_tail_backend`, `node_message_backend`, `relative_position_backend`), never by `implementation`. Evidence:

* `auto` for the message reduction is Triton on EVERY card where the call fits the kernel (`_should_use_triton(supported, "auto")`),
  including H100, and the edge MLP's `auto` is Triton on sm_86 only (`_select_backend`: "Keep unmeasured devices ... on PyTorch"). Dropout
  `auto` is PyTorch; edge-tail / node-message / relative-position have no `auto` (opt-in switches).
* The manifests: A5000 holds `ok` rows for every message and edge-MLP kernel; A6000 holds only the edge-tail / node-message / relative-position
  rows; A100 none. The edge-tail and node-message backward kernels are `failed` on every card, H100 included.

Decision: only `auto`'s choice is gated, and only off H100 / B200 (`portable_blocked`, which passes those through): on another card the
message reduction (and the sm_86 edge MLP) goes to Triton only where the card holds the record (ops `mpnn_message`, `mpnn_edge_mlp`, bf16,
mode from the call); otherwise one `UnrecordedCardWarning` and the PyTorch statements -- the same function `backend="pytorch"` computes, so
no numerics change. An explicit backend name is the caller's request and is honoured as before, as an explicit `implementation` is in the
modules. H100 keeps the Triton message reduction it ran by default. All six families are in `OP_KERNELS` so `records.missing` reports them.

## tm1 / tm2 and `triangle_multiplication`

* `tm1` (`trimul_gemm_gate_triton`, `trimul_bwd_gate_recompute_triton`) and `tm2` (`trimul_outproj_gemm_gate_triton`,
  `trimul_outproj_bwd_gate_recompute_triton`): their jit symbols are launched only from `kernels/tm1|tm2/triton/main.py` (`triton_tm1` /
  `triton_tm2`, exported raw by `kernels/__init__.py`), `kernels/drivers/` and `kernels/checks/`. No module, `ops` facade or architecture block
  calls them, and `registry_kernel.csv` derives no unit for them. Nothing reaches them at runtime. Their `developed=yes` was plainly wrong by
  the registry's own definition (`tests/cpu/registry/test_a_built_kernel_is_one_something_launches.py`: a kernel kept deliberately with no
  path to it is `developed=no` with a reason) -- the reachability test's name-level over-approximation hid it. Set to `developed=no` with the
  reason in `undeveloped.csv`; no code removed.
* `triangle_multiplication` family (3 rows, `developed=no`): declared in `modules/triangle_multiplication/baseline_dtv1.py`, imported only by
  `kernels/checks/triangle_multiplication.py`. Correctly `no`.

## SWA atom attention (global): the ladder's "no portable rung by design"

The 2026-10-29 ladder row `swa_attn` benchmarks `SWA3DRoPEAttention` (target `swa_atom_attention`): the attention core is flash-attention
(FA4 on sm_90+, FA2 on sm_80+), not Triton, so there is no portable rung to measure; the pieces around it (QK RMSNorm + 3D RoPE, the
sigmoid gate) are the portable `rope` / `gated_projection` kernels, gated by `swa_atom_attention`. On a non-H100 / B200 card:

* the module: FA2 core + those pieces when the card holds the `swa_atom_attention` record, else FA2 + PyTorch pieces with one warning
  (A100 / A5000 / A6000 all warn today);
* the ESMFold2 `SWAAtomTransformer` with a global window: the fused global block is served on sm_90 / sm_100 only (`_fused_ok`), so other
  cards take the per-block path -- FA2 with a huge window, the gated pieces above, and (new) the gated `rms_norm_modulation` / `swiglu_ffn`.
  Before this branch the last two ran Triton with no gate;
* `block_style="af3"` + `global_attn`: FA2 core, AdaLN / RMSNorm / ConditionedTransition modules, each gated.

Consistent with the rule now: nothing Triton runs there without a record, and the flash core is not a Triton kernel.

## Triton kernels with no registry row

| kernel | reached from | covered by |
|---|---|---|
| `gated_projection/triton/residual.py:_forward`, `_backward` | `ops.gated_residual` | `UNREGISTERED_OPS` (always PyTorch off H100 / B200) |
| `bias_only_attention/triton/pair_bias.py:_ln_bias_fwd_kernel`, `_ln_bias_bwd_kernel` | BiasOnlyAttention, `pair_bias_all` | `UNREGISTERED_OPS` |
| `_cast.py:_cast_kernel` | bias-only core, AdaLN / ConditionedTransition training (fp32 masters under bf16) | the caller's op gate |
| `_tf32.py:_grid_many_kernel` | fp32 AdaLN / ConditionedTransition / Transition / SwiGLU | the caller's op gate |
| `transition/triton/residual.py:_split_k_residual_kernel` | Transition residual path | `transition` |
| `mpnn_message/triton/inference.py:_message_inference_kernel`, `main.py:_zero_bias_grad_kernel` | MPNN message reduction | `mpnn_message` (`auto`); counted through the registered forward kernels |
| `conditioned_transition/triton/token_dit_kernels.py` (5), `token_dit_attn.py` (2) | `integrations/token_dit.py` | B200 fallback only (`serves()` is False off sm_90 / sm_100) |
| `augmented_attention/cuda/__init__.py:_prep_do_kernel`, `_scale_bf16_kernel`, `cuda/fused_pair.py:_wf_kernel`, `trimul_inproj/cuda/_h100_pack.py:_pack`, `h100_wide_inference.py:_fold`, `integrations/opm_train.py:_opm_prologue_kernel`, `integrations/pwa_train.py:_pair_fwd_kernel`, `_pair_bwd_kernel` | the H100 / B200 CUDA paths | H100 / B200 only |

## Left as it is (with the reason)

* **H100 / B200 still launch Triton in places** (the rule says they should not; the brief says H100 defaults must not change): every `ops.*`
  facade (`portable_blocked` is False there), the bias-only DiT training core / pair bias when the CUDA block declines, B200 `swa_dit` fp32
  training and bf16 fallback, the small Triton helpers of the H100 CUDA paths. Changing them is a separate decision for the maintainer.
* **`LocalDiTBlock`, `DiTVariantStack`, `SWA3DRoPEAttention`** still ask bf16 rows for an fp32 call (`portable_leg_allowed` has no dtype).
  Nothing changes on today's manifests (their bf16 rows already warn, except `dit_variant`); passing the dtype is the next step.
* **`augmented_attention_bwd_reduce_triton`** is declared fp32 only (it reduces fp32 partials) but runs in a bf16 training call, so no bf16
  gate requires it; **`augmented_attention_bwd_atomic_triton`** is `developed=no` but `kernel_type="memory_efficient"` launches it. Both need a
  registry decision (the row's dtype declaration describes the kernel's own buffers), not a gate change.
* **Over-strict entries** (count kernels a call does not launch): `transition` inference counts the b2b / layernorm-expand kernels,
  `triangle_attention` / `gated_linear` count `gated_projection_gate_triton`, `swa_atom_attention` counts `rope_fwd_triton` and `rmsnorm/*`,
  `layernorm` counts `layernorm_fwd_mmajor_triton`. Conservative (they can only block), left as they are.
* **Build-time resolve** uses the current device, not the tensor's; a module built on a CPU host keeps the label (documented limit).
* **`gated_residual` and the bias-only pair-bias kernels** need registry rows (driver + check + `run_all`) before any card can be recorded
  for them.

## Tests (H100 node, 2026-10-31)

* Code graph (`dev code-graph --affected`, sm90 bundle manifest) on every changed file: reached by 0 of 874 known artifacts (no native
  rebuild). Native caches restamped (7 `*_cuda` op identities); `dev derive --arch sm86` (1220 units, 0 errors) and the sweep page re-run.
* CPU suite (`tests/cpu`): 4458 passed, 2 failed -- the two known on `main` (one-sided key flags, stale sm86 Triton caches). New:
  `tests/cpu/compile/test_triton_gate_audit.py` (derived launches counted per op, unregistered ops never recorded, call precision,
  facades' PyTorch statements against their references, H100 / B200 untouched).
* GPU, emulated unrecorded / recorded card (`tests/gpu/any/test_triton_gate_gpu.py`): 16 passed -- rms_norm_modulation and
  layer_norm_linear (inference + training), gated_residual, both TriMul facades, conditioned_transition, triangle_attention QK norm,
  ESMFold2 SWAAtomTransformer (per-block and fused gate), bias-only training core, MPNN `auto` vs explicit, LayerNorm per-call training gate,
  AdaLN fp32 rows.
* H100 defaults (real card, no emulation), 25 files: architecture_no_triton (28), bias_only_dit portable / pair256 / psb / psb_infer / h100
  (41 / 44 / 95 / 40 / 99), bodit_ct_training_portable (36), mpnn_message_kernel (30), mpnn_numerical (34), normalization_atomic_gradients
  (12), swa_dit_kernels (13), swa_dit_fused (80), swa_fp32_highest (6), h100_module_wiring (34), force_portable_option (8),
  pure_triton_routing (6), op_contracts (5), input_feature_embedder (46), engine_layernorm_mixed_fallback (4), portable_fp32_accuracy (37),
  norm_dispatch_cache (4), conditioned_transition_precision (18), reference_purity_and_bidir (4), portable_ct_adaln_residual (9): all passed.
