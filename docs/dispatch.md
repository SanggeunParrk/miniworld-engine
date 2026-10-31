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
  (family and kernel-name patterns per op; the undeveloped A/B alternatives are not held against an op). Each entry is checked against
  what the op's portable path launches -- the units `dev derive` records per `registry_module.csv` row, dtype and eval / train
  (`kernels/registry/registry_kernel.units.json`; `tests/cpu/compile/test_triton_gate_audit.py`);
* **the call's mode decides which of them**: an *inference* call (no backward recorded: `torch.no_grad`, `eval()` with detached inputs and
  no parameter gradient) launches the forward kernels only; a *training* call launches those and the backward ones. A kernel is a
  training kernel when its name carries a `bwd`, `saveact`, `train` or `dw` word (`records.kernel_mode`; one exception, the LayerNorm
  forward `layernorm_fwd_saveact_triton`, which every inference forward launches too); the rest run in an inference forward too. A missing backward kernel therefore blocks training calls only: TriMul on an Ampere card runs the portable kernels for inference and
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
| Transition | CUDA at d=128, n=4 (a call those kernels turn down asks the record); else warns (1 of 6 inference, 3 of 10 training) | warns (1 / 3 of 10) | warns (1 / 3 of 10) | CUDA, else PyTorch | warns, PyTorch |
| TriangleMultiplication (+ bidirectional, `ops.triangle_multiplicative_update` and its bidirectional form) | portable inference; training warns (3 of 13) | portable inference; training warns (3 of 13) | portable inference; training warns (3 of 13) | CUDA, else PyTorch | warns, PyTorch |
| TriangleAttention (+ bidirectional, bias-only) | portable | portable | portable | CUDA, else PyTorch | warns, PyTorch |
| AdaptiveLayerNorm | warns (2 of 5 inference, 2 of 10 training) | same | same | CUDA on H100, PyTorch on B200 | warns, PyTorch |
| ConditionedTransition | warns (2 of 10 inference, 3 of 21 training) | warns (2 / 3 of 21) | warns (2 / 4 of 21) | CUDA on H100, PyTorch on B200 | warns, PyTorch |
| AugmentedAttention / AttentionPairBias | warns (2 of 9 inference, 3 of 20 training) | same | warns (2 of 9 / 2 of 20) | CUDA, else PyTorch | warns, PyTorch |
| OuterProductMean | warns (3 of 3 inference, 8 of 8 training) | same | same | CUDA, else PyTorch | warns, PyTorch |
| MSAPairWeightedAveraging | warns (3 inference, 7 training of 7) | same | same | CUDA, else PyTorch | warns, PyTorch |
| RMSNorm | portable | warns (1 inference, 2 training of 2) | same | CUDA | warns, PyTorch |
| LayerNorm | portable (forward and backward; asked per call: training and fp32 calls too) | same | same | CUDA (general kernel) | warns, PyTorch |
| SWA atom attention (QK norm + RoPE, gate) | warns (3 of 5 / 4 of 8) | warns (4 of 5 / 6 of 8) | warns (4 of 5 / 6 of 8) | CUDA, else PyTorch | warns, PyTorch |
| SWADiTBlock (and ESMFold2's `SWAAtomTransformer` fused block) | warns (3 / 9 of 11) | warns (4 / 11 of 11) | warns (4 / 11 of 11) | CUDA, else PyTorch | warns, PyTorch |
| RMSNorm + adaLN modulation (`ops.rms_norm_modulation`; the ESMFold2 atom block, SWADiTBlock's per-op path) | portable | warns (1 of 1 / 2 of 2) | same | Triton through the facade, PyTorch in the blocks | warns, PyTorch |
| `ops.layer_norm_linear` (fp32-statistics LayerNorm + projection) | portable | portable | portable | Triton (facade) | warns, PyTorch |
| `ops.swiglu_ffn`, `ops.conditioned_transition` (the SwiGLU tail), the ESMFold2 atom FFN | portable inference; training warns (1 of 2) | same | same | Triton (facade), PyTorch in the blocks | warns, PyTorch |
| `ops.gated_residual`, the bias-only DiT pair LayerNorm + bias projection (`bias_only_pair_bias`) | warns (registered 2026-10-31: no row on this card yet) | same | same | Triton | warns, PyTorch |
| ProteinMPNN `auto` (message reduction; edge MLP on sm_86) | warns (message: 3 of 3 inference, 4 of 4 training) | training portable; no-grad warns (the no-save fusion, registered 2026-10-31, has no row yet); edge MLP portable | warns (message 3 of 3 / 4 of 4, edge MLP 2 of 2) | Triton message reduction (as before) | warns, PyTorch |
| La-Proteina pair builders (`pair_embed_geom`, `pair_embed_ca`, `pair_embed_adaln`) | warns (2 inference, 5 training of 5 for the encoder / decoder pair; 1 / 4 of 4 for the initial pair) | same | same | CUDA on H100, PyTorch on B200 | warns, PyTorch |
| Token-pair initialisation (`token_pair_init`) | warns (1 inference, 4 training of 4) | same | same | CUDA | warns, PyTorch |
| smooth lDDT (`kernels.smooth_lddt`, the diffusion loss; called directly, not a module) | warns (no record) | warns | warns | CUDA | warns, PyTorch |
| Diffusion sampler glue (`kernels.sampler_ops`: `augment_noise`, `align_euler`, `pool_atoms`; called directly, not a module) | warns (no record) | warns | warns | CUDA | warns, PyTorch |
| LocalDiTBlock window attention (AF3 / Protenix / OpenFold3 / OpenDDE / ESMFold2 atom blocks) | warns (1 of 3 inference, 3 of 5 training) | same | same | CUDA (bf16 / fp32 inference), else PyTorch | warns, PyTorch |
| DiTVariantStack (Boltz-2 / RoseTTAFold3 / Chai-1 token stacks) | portable inference; training warns (1 of 7: the fp32 row of the split backward's reduce) | same | portable | CUDA (bf16 inference), else PyTorch | warns, PyTorch |
| AtomDiTVariantStack (their atom stacks) | warns (1 of 3 inference, 3 of 5 training) | same | same | CUDA (bf16 inference), else PyTorch | warns, PyTorch |

**Parts that route through other ops.** `BiasOnlyDiTBlock` builds its AdaptiveLayerNorm, ConditionedTransition and pair LayerNorm with the
block's implementation, so it has no row of its own: on a card it takes the portable kernels of those three rows when each is recorded
and warns per unrecorded part otherwise. Its attention core is the batched einsum, except in a training call: there the portable
family's feature-major core (`kernels.bias_only_attention.triton.core_t`, op `bias_only_core`: the flat sigmoid-gate kernels) and the one-pass
pair LayerNorm + bias projection (`triton.pair_bias`, op `bias_only_pair_bias`, also `pair_bias_all`'s portable leg) run where the card holds
their record (the pair-bias kernels got registry rows on 2026-10-31; no Ampere manifest has them yet). H100 / B200
run the CUDA block paths and the PyTorch leg of the parts (a training call the CUDA block declines still reaches those two portable kernels
there, as before). `DiTBlock(extra_pair_bias=)` rides the attention's bias, so it follows the `AugmentedAttention` row.
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

## The diffusion sampler glue

`kernels.sampler_ops.interface` (`augment_noise`, `align_euler` at fp32 coordinates; `pool_atoms` at bf16) routes itself with the same rule (`kernels/_pair_route.py`): the CUDA kernels on H100;
any other card with an `ok` row for every portable kernel of `OP_KERNELS["sampler_ops"]` at the dtype in use (fp32 for the coordinate ops, bf16 for the pooling) runs the portable kernels; otherwise
one `UnrecordedCardWarning` and the PyTorch reference. B200 has no CUDA kernel of the op and runs the reference silently; the A/B switch runs the portable kernels everywhere. The random draws are
inputs, so the three paths give the same result for the same draws. `serves(...)` / `pool_atoms_serves(...)` say whether a caller should hand the call over (any card but B200).

## The `ops.*` entry points

Every weights-as-args facade follows the same rule: `ops.transition`, `ops.gated_linear`, `ops.augmented_attention_pair_bias`,
`ops.swiglu_ffn`, `ops.layer_norm`, `ops.triangle_attention` (its attention core, and its QK norm under the `rmsnorm` record),
`ops.triangle_multiplicative_update` / `ops.bidirectional_triangle_multiplicative_update` (op `triangle_multiplication`),
`ops.conditioned_transition` (its SwiGLU tail: op `swiglu_ffn`), `ops.layer_norm_linear`, `ops.rms_norm_modulation` and `ops.gated_residual`
call `dispatch.portable_blocked(op, device, *tensors)` first (CUDA path, recorded portable, else the PyTorch statements and the same warning),
asking the rows at the call's precision (`dispatch.record_dtype`: fp32 for an fp32 activation, bf16 otherwise) and mode (training when a
tensor needs a gradient). The TriMul and triangle-attention cores are bf16 only: another dtype is a `KernelRefusedWarning` and the PyTorch
statements. `ops.gated_residual` (op `gated_residual`) got its registry rows on 2026-10-31; until a card's `run_all` records them it runs its
PyTorch statement there. H100
and B200 keep what they did (the facades run their kernels there, as before). The raw `kernels.*` functions are not routed: they are the
kernels, not an entry point.

## ProteinMPNN

`modules.mpnn` chooses its kernels by backend options, not by `implementation`. Their default `auto` puts the hidden-message reduction on
Triton wherever the call fits the kernel, and the edge MLP on sm_86 only; off H100 / B200 that choice now asks the card's record (ops
`mpnn_message`, `mpnn_edge_mlp`; the six MPNN families are all in `OP_KERNELS`): a card without it gets one `UnrecordedCardWarning` and the
PyTorch statements, the same numbers as `backend="pytorch"`. An explicit Triton backend (`triton`, `triton_compute`, `triton_memory`,
`bitpack`, the edge-tail / node-message / relative-position switches) is the caller's request and is not gated, as an explicit
`implementation` is not. H100 / B200 are unchanged.

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
* The native-CUDA test at build time is per op and shape-agnostic except for Transition (A100: d = 128, n = 4). A call the fused sm_80 kernel
  turns down (ragged rows, a dtype) asks the card's record before it reaches the portable kernels.
* The architecture blocks that call kernels directly obey the same rule: ESMFold2's `SWAAtomTransformer` asks `swa_dit` before its fused block
  and `rms_norm_modulation` / `swiglu_ffn` in each block; the standalone `LayerNorm` of `architecture.primitives` goes through `ops.layer_norm`.
* dtype: the module-level resolve checks bf16, the dtype the fused kernels run (`guard_dtype` sends the rest to PyTorch). An fp32 call asks
  for fp32 rows per call in Transition (`dispatch.dtype_gate`), AdaptiveLayerNorm, ConditionedTransition, AugmentedAttention
  (`dispatch.fp32_gate`), LayerNorm, RMSNorm, every `ops.*` facade, and -- through `portable_leg_allowed(..., dtype=)` -- `LocalDiTBlock`,
  the DiT variant stacks, SWADiTBlock and the SWA atom attention.
* A kernel that runs at one precision whatever the call's is asked at that precision (`records._RUNS_AT`): the split attention backward's
  `augmented_attention_bwd_reduce_triton` sums fp32 partials in a bf16 training call, so that call needs the card's fp32 row for it. A
  kernel that runs only without a recorded backward (`records._INFERENCE_ONLY`: the MPNN no-save message fusion) counts for inference calls
  only.
* LayerNorm and RMSNorm resolve the forward kernels when built and ask again per call (`portable_blocked`): a training call needs the
  backward rows, an fp32 call the fp32 rows. The LayerNorm backward choice (CUDA vs portable) stays kernel-internal.
* Four small Triton helpers run inside a gated op's path with no registry row of their own: `kernels/_cast.py` (a bit-identical dtype cast),
  the TF32 grid rounding of `kernels/_tf32.py` (exact), the MPNN bias-gradient zeroing (a fill) and Transition's split-K reduction (its
  numbers are checked under `transition_squeeze_residual_triton`'s row, whose check drives a split-K shape). They are covered by their op's
  gate. See `docs/records/triton_gate_audit_20261031.md` and `docs/records/triton_gate_rows_20261031.md`.
