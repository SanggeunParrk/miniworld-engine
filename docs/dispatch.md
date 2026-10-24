# Where `miniworld` runs an op, card by card

The public backends are `pytorch`, `miniworld`, `cuequivariance` and `anthropic`. **Triton is not one of them.** It is the
engine's portable kernel family, used behind `miniworld` on a card the engine has evidence for, and as an A/B switch.

`implementation="miniworld"` resolves **once per (op, card), when the module is built** (`modules.dispatch.resolve`; the modules
that resolve per call ask the same function):

| card | what `miniworld` does |
|---|---|
| **H100 (sm_90), B200 (sm_100)** | The native CUDA path; a call no CUDA path serves runs the **PyTorch reference**. Never Triton, never a warning. Unchanged: [gpus/h100/dispatch.md](gpus/h100/dispatch.md) (`restricted_card`, `settings.triton_on_hopper_blackwell`, `b200_engine_triton` are A/B switches). |
| **any other card** | (a) a hand-written CUDA path of the engine that serves the call: use it (A100: the fused sm_80 Transition at d = 128, n = 4). (b) else, if the engine holds a **record for this exact card** that validates the op's Triton kernels: the Triton family. (c) else a `UnrecordedCardWarning` (once per op and card) and the **PyTorch reference**. |
| no GPU (CPU build) | the Triton family is kept as the label; the module runs its reference on CPU tensors. Nothing to warn about. |

## What "recorded" means

`python -m miniworld_engine.autotune.run_all` drives every kernel of `kernels/registry/registry.csv` on the card, checks it against
its reference and writes `autotune/manifests/<card name> (smXX).csv` (the `#provenance` row says when and at which commit). An op is
**recorded on a card for a call** when every Triton kernel *that call launches* has an `ok` row in that card's manifest at the dtype in use:

* the kernels of an op are the `developed == yes` Triton rows of the registry selected by `autotune.records.OP_KERNELS`
  (family and kernel-name patterns per op; the undeveloped A/B alternatives are not held against an op);
* **the call's mode decides which of them**: an *inference* call (no backward recorded: `torch.no_grad`, `eval()` with detached inputs and
  no parameter gradient) launches the forward kernels only; a *training* call launches those and the backward ones. A kernel is a
  training kernel when its name carries a `bwd`, `saveact`, `train` or `dw` word (`records.kernel_mode`); the rest run in an inference
  forward too. A missing backward kernel therefore blocks training calls only: TriMul on an Ampere card runs Triton for inference and
  the PyTorch reference (one warning) for a call that needs a backward;
* `failed` and `untested` rows, a kernel with no row, and a row at another precision are not evidence;
* the card is identified by its **exact name and capability**: `NVIDIA RTX A6000 (sm86)`. An A40 or an L40S (same sm_86 / sm_89
  family, other shared memory, bandwidth and clocks) is not recorded: it warns and runs PyTorch;
* the per-card autotune caches (`autotune/data/<kernel>/<card>.json`) tune the kernels; they are not the evidence that a kernel runs
  correctly there, the manifest is.

Modules resolve the inference side when they are built (`dispatch.resolve`) and keep a `_train_gate` for the training side, consulted per
call (`dispatch.call_backend`, plain attributes, so a compiled forward reads constants); the modules that resolve per call ask with
`training=`. The lookup is `autotune.records` (stdlib only, cached, no GPU needed; `records.use_manifest_dir(path)` is the test hook).
`dispatch.route_for_card(op, name, capability, mode=)` is the pure rule; the tests fake the card (`tests/fake_card.py`).

## A kernel that refuses the call

A Triton kernel can refuse a call by a shape or dtype guard (`d_hidden != d_pair` in TriMul, a width that is not a power of two in
OuterProductMean / MSAPairWeightedAveraging, value_dim < 16, ...). Under `miniworld` that is never an error and never silent: the module
emits one `KernelRefusedWarning` per (op, card, guard text) and runs the PyTorch reference (`dispatch.warn_refused`). A CPU tensor is not
a refusal (there is no kernel to run) and does not warn. H100 / B200 are not affected: their modules take the CUDA or PyTorch leg before
a Triton guard is reached.

## The recorded cards today

Generated from the manifests in the tree (`records.missing(op, card, mode=)`, `bf16`). `Triton` = every kernel the call launches has an
`ok` row; otherwise the call warns and runs PyTorch (counts are kernels without an `ok` row).

| op | A100 80GB PCIe (sm80) | RTX A5000 (sm86) | RTX A6000 (sm86) | H100 / B200 | any other card |
|---|---|---|---|---|---|
| Transition | CUDA at d=128, n=4; else warns (1 of 7) | warns (1 of 7) | warns (1 of 7) | CUDA, else PyTorch | warns, PyTorch |
| TriangleMultiplication (+ bidirectional) | Triton inference; training warns (3 of 8) | Triton inference; training warns (3 of 8) | Triton inference; training warns (3 of 8) | CUDA, else PyTorch | warns, PyTorch |
| TriangleAttention (+ bidirectional, bias-only) | Triton | Triton | Triton | CUDA, else PyTorch | warns, PyTorch |
| AdaptiveLayerNorm | warns (2 of 8) | warns (2 of 8) | warns (2 of 8) | PyTorch (no CUDA path yet) | warns, PyTorch |
| ConditionedTransition | warns (2 inference, 3 training of 17) | warns (2 / 3 of 17) | warns (2 / 4 of 17) | PyTorch (no CUDA path yet) | warns, PyTorch |
| AugmentedAttention / AttentionPairBias | warns (2 of 11) | warns (2 of 11) | warns (2 of 11) | CUDA, else PyTorch | warns, PyTorch |
| OuterProductMean | warns (2 inference, 5 training of 5) | same | same | CUDA, else PyTorch | warns, PyTorch |
| MSAPairWeightedAveraging | warns (3 inference, 7 training of 7) | same | same | CUDA, else PyTorch | warns, PyTorch |
| RMSNorm | Triton | warns (1 inference, 2 training of 2) | same | CUDA | warns, PyTorch |
| LayerNorm | Triton forward, CUDA / Triton backward | same | same | CUDA (general kernel) | warns, PyTorch |
| SWA atom attention (QK norm + RoPE, gate) | warns (3 / 4 of 6) | warns (4 / 6 of 6) | warns (4 / 6 of 6) | CUDA, else PyTorch | warns, PyTorch |
| SWADiTBlock | warns (3 / 9 of 11) | warns (4 / 11 of 11) | warns (4 / 11 of 11) | CUDA, else PyTorch | warns, PyTorch |

Why so many "warns": the Ampere manifests are from 2026-08-31 / 09-01 (v1.0.0), before the OuterProductMean, MSAPairWeightedAveraging,
SWA and several TriMul / AdaLN / Transition kernels existed, so those kernels have no row. The routing is one function for every card;
**recording a card is what moves its ops from the warning column to Triton**: run `python -m miniworld_engine.autotune.run_all` on it and
commit the manifest. These Ampere cards have no validated record for OuterProductMean, MSAPairWeightedAveraging, AdaptiveLayerNorm,
ConditionedTransition, AugmentedAttention, Transition, the SWA ops: `miniworld` warns and runs PyTorch there, by the rule, until they do.
Forward kernels with no `ok` row on A100 / A5000 / A6000: `transition_squeeze_residual`, `adaln_fwd_gate`, `adaln_gemm_gate`,
`outer_product_mean_layernorm_gemm` / `_epilogue`, `pair_weighted_averaging_layernorm_gemm_softmax` / `_layernorm_gemm` / `_gate_gemm`,
the SWA kernels; on A5000 / A6000 also `rmsnorm_fwd`.

## The `ops.*` entry points

The weights-as-args facade follows the same rule where a bare Triton unit sat behind it: `ops.transition`, `ops.gated_linear`,
`ops.augmented_attention_pair_bias`, `ops.swiglu_ffn`, `ops.layer_norm` and the attention core of `ops.triangle_attention` call
`dispatch.triton_blocked(op, device, *tensors)` first (CUDA path, recorded Triton, else the PyTorch statements and the same warning). H100
and B200 keep what they did. **Not routed** (they call their Triton kernels directly, as before): `ops.triangle_multiplicative_update`,
`ops.bidirectional_triangle_multiplicative_update`, `ops.conditioned_transition`, `ops.layer_norm_linear`, `ops.rms_norm_modulation`,
`ops.gated_residual`, and the QK-norm of `ops.triangle_attention`'s fallback; so are the raw `kernels.*` functions.

## Overrides

| | |
|---|---|
| `settings.force_triton_kernels=True` | Internal A/B switch (formerly `engine_backend="triton"`): the Triton family on every card, recorded or not, H100 and B200 included, with every `*_cuda` stage off. How a Triton cache is built and how Triton is compared with CUDA. Not a production setting. |
| `settings.capture=True` (an autotune build) | A build may use Triton on a card without a record: it is how the card's configs are captured. |
| `settings.trimul_impl` | Pins the TriMul backend (`pytorch`, `triton`, `cuda`, ... as `KernelBackend`), for A/B work. |
| `python -m miniworld_engine.autotune.run_all` | Records the card. |

`implementation="triton"` and `implementation="cuda"` still parse, with a `DeprecationWarning`, as `miniworld` (on H100 and B200 that
is the CUDA path; forcing Triton is only possible through the switch above). They are removed in the next minor release, as is the
`settings.engine_backend` alias of `force_triton_kernels` and the hidden `build --backend` spelling of `build --kernels`.

## Limits of the rule as implemented

* The route is resolved when the module is built, from the current device (or the `device` the resolver is given). A module built on a
  CPU host and moved to a card keeps the Triton family label; its Triton leg is then not gated. Under `torch.compile` an uncached card
  reads as "no card" (the driver query cannot be traced): build or run the module eagerly once before compiling.
* The native-CUDA test at build time is per op and shape-agnostic except for Transition (A100: d = 128, n = 4). Ragged rows or a dtype the
  fused sm_80 kernel does not take still reach the Triton kernels of that card.
* The architecture blocks that call kernels directly (`architecture.blocks.swa_atom_transformer`, the standalone `LayerNorm` of
  `architecture.primitives`) obey the H100 / B200 rule (`restricted_card`) but are not gated by records.
* dtype: the module-level resolve checks bf16, the dtype the fused kernels run (`guard_dtype` sends the rest to PyTorch).
* LayerNorm is gated by the forward kernels only (all four `layernorm` kernels are `ok` on every recorded card); its backward choice
  (CUDA vs Triton) stays kernel-internal.
