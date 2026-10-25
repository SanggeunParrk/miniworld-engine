# Where `miniworld` runs an op, card by card

The public values of `implementation` are `pytorch`, `miniworld`, `cuequivariance` and `anthropic`. **`miniworld` routes per card automatically**:
you never choose a kernel family. Behind it the engine keeps a portable kernel family (source: `kernels/*/triton/`), used on a card the
engine has evidence for, and reachable explicitly only through the A/B switch `settings.force_portable_kernels`.

`implementation="miniworld"` resolves **once per (op, card), when the module is built** (`modules.dispatch.resolve`; the modules
that resolve per call ask the same function):

| card | what `miniworld` does |
|---|---|
| **H100 (sm_90), B200 (sm_100)** | The native CUDA path; a call no CUDA path serves runs the **PyTorch reference**. Never the portable kernels, never a warning. Unchanged: [gpus/h100/dispatch.md](gpus/h100/dispatch.md) (`restricted_card`, `settings.portable_on_hopper_blackwell`, `b200_portable_kernels` are A/B switches). |
| **any other card** | (a) a hand-written CUDA path of the engine that serves the call: use it (A100: the fused sm_80 Transition at d = 128, n = 4). (b) else, if the engine holds a **record for this exact card** that validates the op's portable kernels: the portable kernel family. (c) else a `UnrecordedCardWarning` (once per op and card) and the **PyTorch reference**. |
| no GPU (CPU build) | the portable kernel family is kept as the label; the module runs its reference on CPU tensors. Nothing to warn about. |

## What "recorded" means

`python -m miniworld_engine.autotune.run_all` drives every kernel of `kernels/registry/registry.csv` on the card, checks it against
its reference and writes `autotune/manifests/<card name> (smXX).csv` (the `#provenance` row says when and at which commit). An op is
**recorded on a card for a call** when every portable kernel *that call launches* has an `ok` row in that card's manifest at the dtype in use:

* the kernels of an op are the `developed == yes` portable rows of the registry selected by `autotune.records.OP_KERNELS`
  (family and kernel-name patterns per op; the undeveloped A/B alternatives are not held against an op);
* **the call's mode decides which of them**: an *inference* call (no backward recorded: `torch.no_grad`, `eval()` with detached inputs and
  no parameter gradient) launches the forward kernels only; a *training* call launches those and the backward ones. A kernel is a
  training kernel when its name carries a `bwd`, `saveact`, `train` or `dw` word (`records.kernel_mode`); the rest run in an inference
  forward too. A missing backward kernel therefore blocks training calls only: TriMul on an Ampere card runs the portable kernels for inference and
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

A portable kernel can refuse a call by a shape or dtype guard (a TriMul `d_hidden` / `d_pair` outside the served range, a width that is not a power of two in
OuterProductMean / MSAPairWeightedAveraging, value_dim < 16, ...). Under `miniworld` that is never an error and never silent: the module
emits one `KernelRefusedWarning` per (op, card, guard text) and runs the PyTorch reference (`dispatch.warn_refused`). A CPU tensor is not
a refusal (there is no kernel to run) and does not warn. H100 / B200 are not affected: their modules take the CUDA or PyTorch leg before
a portable-kernel guard is reached.

## The recorded cards today

Generated from the manifests in the tree (`records.missing(op, card, mode=)`, `bf16`). `portable` = every kernel the call launches has an
`ok` row; otherwise the call warns and runs PyTorch (counts are kernels without an `ok` row).

| op | A100 80GB PCIe (sm80) | RTX A5000 (sm86) | RTX A6000 (sm86) | H100 / B200 | any other card |
|---|---|---|---|---|---|
| Transition | CUDA at d=128, n=4; else warns (1 of 5 inference, 2 of 9 training) | warns (1 / 2 of 9) | warns (1 / 2 of 9) | CUDA, else PyTorch | warns, PyTorch |
| TriangleMultiplication (+ bidirectional) | portable inference; training warns (3 of 8) | portable inference; training warns (3 of 8) | portable inference; training warns (3 of 8) | CUDA, else PyTorch | warns, PyTorch |
| TriangleAttention (+ bidirectional, bias-only) | portable | portable | portable | CUDA, else PyTorch | warns, PyTorch |
| AdaptiveLayerNorm | warns (2 of 8) | warns (2 of 8) | warns (2 of 8) | PyTorch (no CUDA path yet) | warns, PyTorch |
| ConditionedTransition | warns (2 inference, 3 training of 17) | warns (2 / 3 of 17) | warns (2 / 4 of 17) | PyTorch (no CUDA path yet) | warns, PyTorch |
| AugmentedAttention / AttentionPairBias | warns (2 of 11) | warns (2 of 11) | warns (2 of 11) | CUDA, else PyTorch | warns, PyTorch |
| OuterProductMean | warns (2 inference, 5 training of 5) | same | same | CUDA, else PyTorch | warns, PyTorch |
| MSAPairWeightedAveraging | warns (3 inference, 7 training of 7) | same | same | CUDA, else PyTorch | warns, PyTorch |
| RMSNorm | portable | warns (1 inference, 2 training of 2) | same | CUDA | warns, PyTorch |
| LayerNorm | portable forward, CUDA / portable backward | same | same | CUDA (general kernel) | warns, PyTorch |
| SWA atom attention (QK norm + RoPE, gate) | warns (3 / 4 of 6) | warns (4 / 6 of 6) | warns (4 / 6 of 6) | CUDA, else PyTorch | warns, PyTorch |
| SWADiTBlock | warns (3 / 9 of 11) | warns (4 / 11 of 11) | warns (4 / 11 of 11) | CUDA, else PyTorch | warns, PyTorch |
| La-Proteina pair builders (`pair_embed_geom`, `pair_embed_ca`, `pair_embed_adaln`) | warns (2 inference, 5 training of 5 for the encoder / decoder pair; 1 / 4 of 4 for the initial pair) | same | same | CUDA on H100, PyTorch on B200 | warns, PyTorch |
| Token-pair initialisation (`token_pair_init`) | warns (1 inference, 4 training of 4) | same | same | CUDA | warns, PyTorch |
| smooth lDDT (`kernels.smooth_lddt`, the diffusion loss; called directly, not a module) | warns (no record) | warns | warns | CUDA | warns, PyTorch |
| LocalDiTBlock window attention (AF3 / Protenix / OpenFold3 / OpenDDE / ESMFold2 atom blocks) | warns (1 inference, 3 training of 4) | same | same | CUDA (bf16 / fp32 inference), else PyTorch | warns, PyTorch |
| DiTVariantStack (Boltz-2 / RoseTTAFold3 / Chai-1 token stacks) | portable | portable | portable | CUDA (bf16 inference), else PyTorch | warns, PyTorch |
| AtomDiTVariantStack (their atom stacks) | warns (1 inference, 3 training of 4) | same | same | CUDA (bf16 inference), else PyTorch | warns, PyTorch |

**Parts that route through other ops.** `BiasOnlyDiTBlock` builds its AdaptiveLayerNorm, ConditionedTransition and pair LayerNorm with the
block's implementation, so it has no row of its own: on a card it takes the portable kernels of those three rows when each is recorded
(the attention core is the batched einsum on every card) and warns per unrecorded part otherwise; H100 / B200 run the CUDA block
paths and the PyTorch leg of the parts. `DiTBlock(extra_pair_bias=)` rides the attention's bias, so it follows the `AugmentedAttention` row.
**Transition fp32** (TF32 tensor cores) takes the portable residual path only where torch allows TF32 (`set_float32_matmul_precision` not
`"highest"` / `allow_tf32`, the switch the H100 generic kernels follow; full-precision fp32 keeps the PyTorch statements), and a card needs
**fp32** `ok` rows for what the call launches (`records.missing("transition", ..., "fp32", mode)`; the Ampere manifests hold none yet, so
fp32 warns and runs PyTorch there until `run_all` has been run on the card). **TriMul with `d_hidden != d_pair`** (Protenix / FoldForge
template: pair 64, hidden 128) is served from the weights up to pair 256 / hidden 512 (both multiples of 16,
`trimul_inproj.triton.contract.hidden_refusal`); outside that range it is a `KernelRefusedWarning` and PyTorch. **TriangleAttention variants**
take fp32 master parameters under bf16 activations (cast inside, gradients in the master dtype).

The three DiT rows are the portable attention cores ([docs/records/triton_dit_portable_20261024.md](records/triton_dit_portable_20261024.md)):
the window kernels (`augmented_attention_{fwd,bwd,bwd_reduce}_window_triton`) have no row on any card yet, so the atom paths warn and run
PyTorch until `run_all` has been run there; the token stacks launch the token attention kernels and the `layernorm_linear` forward, which the
Ampere manifests do record, so by the rule they already take the portable kernels there -- on the strength of those old rows, not of a run of the stack.

Why so many "warns": the Ampere manifests are from 2026-08-31 / 09-01 (v1.0.0), before the OuterProductMean, MSAPairWeightedAveraging,
SWA and several TriMul / AdaLN / Transition kernels existed, so those kernels have no row. The routing is one function for every card;
**recording a card is what moves its ops from the warning column to the portable kernels**: run `python -m miniworld_engine.autotune.run_all` on it and
commit the manifest. These Ampere cards have no validated record for OuterProductMean, MSAPairWeightedAveraging, AdaptiveLayerNorm,
ConditionedTransition, AugmentedAttention, Transition, the SWA ops: `miniworld` warns and runs PyTorch there, by the rule, until they do.
Forward kernels with no `ok` row on A100 / A5000 / A6000: `transition_squeeze_residual`, `adaln_fwd_gate`, `adaln_gemm_gate`,
`outer_product_mean_layernorm_gemm` / `_epilogue`, `pair_weighted_averaging_layernorm_gemm_softmax` / `_layernorm_gemm` / `_gate_gemm`,
the SWA kernels; on A5000 / A6000 also `rmsnorm_fwd`.

## The La-Proteina pair ops

`pair_embed_geom` / `pair_embed_ca` (the autoencoder's backbone and CA-distance pair), `pair_embed_adaln` (the denoiser's initial pair) and `token_pair_init` are not modules:
callers ask `refusal(...)` and then call the op, and the op routes itself (`kernels/_pair_route.py`, one rule for the three): the CUDA kernels where one exists (H100; for the
token-pair initialisation also B200); any other card with a record for every kernel the call launches (`OP_KERNELS`: `pair_embed_geom`, `pair_embed_ca`, `pair_embed_adaln`,
`token_pair_init`; a call that needs a gradient also needs the backward kernels) runs the portable kernels; otherwise one `UnrecordedCardWarning` and the PyTorch reference.
H100 / B200 never run the portable kernels, except under the internal A/B switch. A CPU tensor is the PyTorch reference. `refusal` returns the reason for a restricted card without a CUDA kernel
of the op (B200, for the pair builders) and None otherwise; it never warns.

## The `ops.*` entry points

The weights-as-args facade follows the same rule where a bare portable unit sat behind it: `ops.transition`, `ops.gated_linear`,
`ops.augmented_attention_pair_bias`, `ops.swiglu_ffn`, `ops.layer_norm` and the attention core of `ops.triangle_attention` call
`dispatch.portable_blocked(op, device, *tensors)` first (CUDA path, recorded portable, else the PyTorch statements and the same warning). H100
and B200 keep what they did. **Not routed** (they call their portable kernels directly, as before): `ops.triangle_multiplicative_update`,
`ops.bidirectional_triangle_multiplicative_update`, `ops.conditioned_transition`, `ops.layer_norm_linear`, `ops.rms_norm_modulation`,
`ops.gated_residual`, and the QK-norm of `ops.triangle_attention`'s fallback; so are the raw `kernels.*` functions.

## The Anthropic payload

`implementation="anthropic"` names the kernels of Anthropic's uplifting-biomolecular-modeling release. It is wired into the modules that
have a branch for it: `TriangleMultiplication` (the TriMul payload, `TRIMUL_NATIVE_BUILD_DIR`), `TriangleAttention` (use_self_attention=True,
transposed bias; `integrations.anthropic`, `MINIWORLD_ANTHROPIC_ROOT`), `OuterProductMean` and `MSAPairWeightedAveraging` (the `opt_core` cells,
`OPT_CORE_DIR`; inference, bf16, sm_90). `modules.dispatch.resolve` accepts the option for the first two (`_ANTHROPIC_OPS`) and rejects it
for every other op; the two MSA modules route inside themselves. An explicit `anthropic` request that the release's path cannot serve
raises with the reason (`PayloadUnavailable`), it never reroutes; the engine's own H100 OPM / PWA paths (`opm_train`, `pwa_train`) do not
answer to it. The epilogue of the OPM path divides by the pair count before the output projection and adds the bias undivided, so an
`OuterProductMean(normalize_before_proj=False)` (ESMFold2) is refused.

**`miniworld` uses the Anthropic payloads automatically when their variables are set** (`OPT_CORE_DIR` for OPM / PWA, `TRIMUL_NATIVE_BUILD_DIR`
for TriMul; `TriangleAttention` has no such automatic use) and a call fits them; with the variables unset `miniworld` never touches them.
This is the behaviour of every release so far and is left as it is; whether `miniworld` should ask for the payload only when
`implementation="anthropic"` is a decision for the maintainer. To keep a process off the payloads, leave the variables unset.

## Overrides

| | |
|---|---|
| `settings.force_portable_kernels=True` | Internal A/B switch: the portable kernel family on every card, recorded or not, H100 and B200 included, with every `*_cuda` stage off. How a portable-kernel cache is built and how the portable kernels are compared with the native CUDA path. Not a production setting. |
| `settings.portable_on_hopper_blackwell=True`, `settings.b200_portable_kernels=True` | The same for H100 / B200 only (A/B switches; B200 norm kernels). |
| `settings.capture=True` (an autotune build) | A build may use the portable kernels on a card without a record: it is how the card's configs are captured. |
| `settings.trimul_impl` | Pins the TriMul route for A/B work: `"portable"`, `"native"` or `"pytorch"` (`None`: `miniworld` decides). Any other value is rejected. |
| `python -m miniworld_engine.autotune.run_all` | Records the card. |

An `implementation` outside the four values raises a `ValueError` that lists them. `settings.configure` rejects an unknown setting, and names the
replacement when the name is one that 2.3.12 removed (see the CHANGELOG).

## Limits of the rule as implemented

* The route is resolved when the module is built, from the current device (or the `device` the resolver is given). A module built on a
  CPU host and moved to a card keeps the portable kernel family label; its portable leg is then not gated. Under `torch.compile` an uncached card
  reads as "no card" (the driver query cannot be traced): build or run the module eagerly once before compiling.
* The native-CUDA test at build time is per op and shape-agnostic except for Transition (A100: d = 128, n = 4). Ragged rows or a dtype the
  fused sm_80 kernel does not take still reach the portable kernels of that card.
* The architecture blocks that call kernels directly (`architecture.blocks.swa_atom_transformer`, the standalone `LayerNorm` of
  `architecture.primitives`) obey the H100 / B200 rule (`restricted_card`) but are not gated by records.
* dtype: the module-level resolve checks bf16, the dtype the fused kernels run (`guard_dtype` sends the rest to PyTorch). Transition fp32
  is the exception: it asks for fp32 rows per call (`dispatch.dtype_gate`).
* LayerNorm is gated by the forward kernels only (all four `layernorm` kernels are `ok` on every recorded card); its backward choice
  ((CUDA vs portable)) stays kernel-internal.
