# Changelog

Notable changes to the public API (`miniworld_engine.kernels`) are recorded
here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); versioning is [SemVer](https://semver.org/).

The public surface is enforced by `tests/cpu/compile/test_public_api.py`.

## [Unreleased]

### Added

- Portable kernels for the diffusion sampler glue `kernels/sampler_ops` (`augment_noise`, `align_euler`, `pool_atoms`): the contract and algorithm of the H100 CUDA kernels (the caller's random
  draws, the churn arithmetic on device scalars in IEEE operations, centres from fixed-order chunk partials, Kabsch rotation by Horn's quaternion method with 12 squarings in scalar registers, no SVD,
  no atomics, bitwise reproducible), for any card. `sampler_ops.interface` now routes by card like the pair ops (H100 CUDA unchanged; a recorded card or the A/B switch the portable kernels; else
  one `UnrecordedCardWarning` and the PyTorch reference; `serves()` / `pool_atoms_serves()` accept every card but B200). On H100, CUDA graph, A = 1 .. 48 samples, 4k .. 32k atoms: 3-16x faster than
  compiled PyTorch for `augment_noise` / `align_euler` and 1.4-6x for `pool_atoms`; within 0.8-1.8x of the CUDA kernels, and faster than them where the CUDA kernels
  are latency bound (small A, serial partial sum; the pooling at 2-3x). Registry rows `sampler_ops_{stats,augment,cov,align,pool}_triton`; record `docs/records/portable_sampler_ops_20261025.md`;
  page [gpus/h100/sampler_ops](gpus/h100/sampler_ops/sampler_ops.md). Tests: `tests/gpu/any/test_sampler_ops_portable_gpu.py`, `tests/gpu/sm90/test_sampler_ops_portable_vs_cuda_gpu.py`,
  `tests/cpu/compile/test_sampler_ops_routing.py`. No card other than H100 has been run.
- H100 pair-op CUDA kernels reworked after the portable kernels were measured faster (`docs/records/pair_ops_cuda_vs_portable_20261025.md`): `pair_embed_geom` / `pair_embed_ca` gather kernels with
  16 / 32 warps per CTA, a one-instruction bf16 unpack and a binary-search bucket (output and codes bit-identical to before), a pipelined LayerNorm backward (bit-identical), and wgmma
  table-gradient kernels for `pair_embed_geom` / `pair_embed_ca` (bit-identical to the previous kernel) and `pair_embed_adaln` (fp32 sums in a different fixed order). Against the portable
  kernels on H100 (CUDA graph, B = 16 .. 48, L = 128 .. 512): encoder forward 1.17-1.38x faster (was 0.84-0.97x), encoder training step 1.27-1.44x (was 0.74-0.87x), `pair_embed_adaln`
  training 1.8-2.2x (was 0.9-1.1x); the decoder training step 1.11-1.36x (was 0.57-0.76x). The kernel sources changed, so their prebuilt-bundle keys change.
  `MINIWORLD_PAIR_EMBED_BWD_IMPL` selects the table-gradient kernel (1 wgmma, 2 the previous `mma.sync`, 0 the shared-memory kernel) for A/B runs.
- The performance ladder (`miniworld-engine ladder {plan,run,report,gate}`, `benchmarks/runners/ladder.py`): for every op, mode, dtype and shape of its grid it times the three rungs
  of the maintainer's invariant -- hand CUDA < portable (Triton) < PyTorch (`torch.compile` + CUDA graph, eager reported next to it) -- through `bench.py`, three repeats each,
  and writes a table with a verdict per row (OK / VIOLATION(cuda_vs_portable) / VIOLATION(portable_vs_pytorch) / n.a.) and a gate that fails on a violation not on the allow list.
  `bench.py` gained `+seq_len_values`, `+repeats`, `+kernel_probe`, `+fp32_check_limit`, `+trimul_d_hidden`, `+opm_variant`, `+dit_extra_pair_bias`, `+local_dit_preset`, the `portable`
  rung for `outer_product` / `msa_pair_weighted_averaging` / `attention_pair_bias` / `bias_only_dit`, `BENCH_TRACEBACK=1`, and a `ladder_ops` module target for the ops without a module
  (pair builders, `token_pair_init`, `smooth_lddt`, sampler glue, LayerNorm / RMSNorm). Gate test: `MINIWORLD_LADDER=1 pytest -m ladder tests/gpu/sm90/test_performance_ladder_gpu.py`
  (never in the default run). First full run on an H100: `docs/records/performance_ladder_20261025.md`.
- Portable kernels for `pair_embed_geom` (+ `pair_embed_ca`, the decoder's CA-distance pair), `pair_embed_adaln` and `token_pair_init`: the same inputs, outputs, masks,
  `ln_eps` fusion and deterministic backward as the H100 CUDA kernels (bins in registers, a row gather instead of the one-hot Linear; the table / LayerNorm / gamma / beta
  gradients are fixed-order partial sums and a second reduce kernel, no atomics), for any card. `kernels/_pair_route.py` routes the three ops like the rest of the engine:
  the CUDA kernels where one exists (H100; `token_pair_init` also B200), the portable kernels on a card the engine holds a record for (and under the internal A/B switch, on every
  card), otherwise one `UnrecordedCardWarning` and the PyTorch reference; `refusal` / `refusal_ca` no longer require sm_90 (a restricted card without a CUDA kernel of the op still
  says so). H100 and B200 are unchanged. On H100 against the PyTorch reference, B = 16 .. 48, L = 128 .. 512, CUDA graph: 4-33x faster at 3-11x less memory, and the shapes the
  reference runs out of memory at (B = 48, L = 512) run; against the CUDA kernels the encoder / decoder pair embeddings are on par or 1.3-1.7x faster in training. Registry rows,
  drivers, checkers and config sets for the 14 kernels; record: `docs/records/portable_pair_ops_20261025.md`. Tests: `tests/gpu/any/test_pair_embed_geom_portable_gpu.py`,
  `test_pair_embed_adaln_portable_gpu.py`, `test_token_pair_init_portable_gpu.py`, `tests/gpu/sm90/test_pair_ops_portable_vs_cuda_gpu.py`, `tests/cpu/compile/test_pair_ops_routing.py`.
  No card other than H100 has been run: the Ampere manifests hold no rows for these kernels, so those cards warn and run PyTorch until `run_all` has recorded them.
- Smooth lDDT on every card: portable kernels for `kernels/smooth_lddt` (`smooth_lddt/triton/main.py`: `smooth_lddt_prep_triton`, `smooth_lddt_fwd_triton`, `smooth_lddt_bwd_triton`;
  fp32 or bf16 `x_pred`, fp32 compute, the same value, masks, nucleotide radii and analytic coordinate gradient as the H100 / B200 CUDA kernels). No N x N x augment tensor exists:
  the ground-truth pair set is summarised once per call in 32 x 32 tile flags and the kernels visit only the flagged tiles, recomputing each pair's ground-truth distance from the
  coordinates. Every reduction has a fixed order and there is no atomic, so value and gradient are bitwise reproducible. `smooth_lddt()` now routes by card: H100 / B200 run the
  CUDA kernels exactly as before; any other card runs the portable kernels once the engine holds a record for that exact card (op `smooth_lddt` in `autotune.records`), otherwise
  the PyTorch reference with the usual `UnrecordedCardWarning`; `settings.force_portable_kernels=True` (the A/B switch) runs the portable kernels on any card, H100 and B200 included.
  `refusal()` no longer rejects a card, only tensors no path can take (not CUDA, dtype, shapes). On an H100 at A = 48, N = 4096 (fwd + bwd, CUDA graph) the portable kernels take 1.7 ms
  (chain-ordered structure) to 5.8 ms (atoms in random order) against 45.7 ms for the fastest compiled PyTorch form, and 2 MiB of extra memory against 1.7 GiB; the PyTorch reference
  (`cdist`) takes 1.8 s in chunks of 8 augments and fails with an illegal memory access in one call at that shape. The CUDA kernels stay 2.7-4x faster than the portable ones. Tests:
  `tests/gpu/any/test_smooth_lddt_portable_gpu.py`, `tests/cpu/compile/test_smooth_lddt_routing.py`; page: [gpus/h100/smooth_lddt](gpus/h100/smooth_lddt/smooth_lddt.md).
- Module-level parity of the portable kernels with the H100 CUDA paths (every card but H100 / B200; the CUDA paths and their speed are untouched):
  - `BiasOnlyDiTBlock` builds its parts (AdaptiveLayerNorm, ConditionedTransition, the pair LayerNorm) with the block's implementation instead of
    pinning them to PyTorch: under `miniworld` a recorded card runs their portable kernels, forward and backward, bf16 and fp32 (TF32), for
    one or several augmentations, a pair of its own per sample, the hoisted bias and key masks; an unrecorded card warns per part and runs
    PyTorch; H100 / B200 take the CUDA block paths and the PyTorch leg of the parts. The attention core stays the batched einsum.
  - `DiTBlock(extra_pair_bias=)` (an `[L, L]` logit term for every head) is served on every path: the CUDA path where it exists (H100 bf16
    inference), otherwise `AugmentedAttentionPairBias(extra_pair_bias=)` adds it to the attention bias (portable attention kernels or the
    PyTorch statements, any dtype, differentiable). The `ValueError` is gone.
  - Transition fp32 on the portable residual path (TF32 tensor cores) wherever torch allows TF32 (the switch the H100 generic kernels follow;
    full-precision fp32 keeps PyTorch) and the card holds fp32 rows (`dispatch.dtype_gate`). Registry rows `transition_expand_swiglu_triton`,
    `transition_squeeze_residual_triton` and `transition_bwd_swiglu_recompute_triton` now declare fp32 with
    measured bands; the Transition kernels take their dot precision from `kernels/_dot_precision.py`; `OP_KERNELS["transition"]` also counts the
    input LayerNorm forward / backward kernels it launches.
  - TriangleAttention variants (gate / out-projection biases, untransposed / dual output) take fp32 master parameters under bf16 activations on
    the portable core: the projections are cast to the activation dtype inside, gradients come back in the master dtype.
  - TriMul with `d_hidden != d_pair` (Protenix / FoldForge template TriMul: pair 64, hidden 128), unidirectional and bidirectional, inference
    and training, on the portable front / back (the widths come from the weights). Served for pair <= 256 / hidden <= 512, both multiples of 16
    (`trimul_inproj.triton.contract.hidden_refusal`); outside that range a `KernelRefusedWarning` and PyTorch, as before.

### Changed

- H100 `sampler_ops` CUDA kernels (`augment_noise`, `align_euler`, `pool_atoms`) are now faster than the portable kernels in every measured cell (A = 1 .. 48, 4k .. 32k atoms, CUDA graph),
  restoring CUDA < portable < compiled PyTorch: 1.1-1.7x ahead of the portable kernels and 1.05-3.3x faster than the previous CUDA kernels (`augment_noise` 1.05-1.5x, `align_euler` 1.15-1.8x, `pool_atoms` 1.6-3.3x; `pool_atoms` 12.4-13.1 us -> 3.8-4.9 us at A = 1).
  Programmatic dependent launch between the kernels of an op, partials staged in shared memory and summed in chunk order, 256 / 512-atom chunks for small launches, a one-round-trip gather in
  `pool_atoms`. Results are bit-identical to before except where the chunk shrinks (small `augment_noise` / `align_euler` launches: fp32 sums associate differently, <= 1.8e-7 relative);
  `pool_atoms` is bit-identical. Record `docs/records/h100_sampler_ops_cuda_20261025.md`, page [gpus/h100/sampler_ops](gpus/h100/sampler_ops/sampler_ops.md).
- `AdaptiveLayerNorm`, `ConditionedTransition` and the augmented-attention core run their reference on a CPU tensor instead of reaching a
  GPU kernel when the module was built for the portable family on a CPU host.

## [2.3.12] - 2026-10-25

### Removed (BREAKING for code that used the removed names)

Triton is not something a user selects. `miniworld` routes per card automatically (`docs/dispatch.md`); the portable kernel family behind it is internal.
2.3.10 removed the public enum members; 2.3.12 removes every other public surface that still named it. No aliases, no shims:

| removed | use instead |
|---|---|
| `ImplementationType("triton")`, `("cuda")`, `implementation="triton"` / `"cuda"` (were read as `miniworld` with a `DeprecationWarning`) | `miniworld`. The strings now raise a `ValueError` that lists the four valid values (`pytorch`, `miniworld`, `cuequivariance`, `anthropic`). |
| `settings.force_triton_kernels` | `settings.force_portable_kernels` |
| `settings.triton_on_hopper_blackwell` | `settings.portable_on_hopper_blackwell` |
| `settings.b200_engine_triton` | `settings.b200_portable_kernels` |
| `settings.engine_backend` (`"auto"` / `"triton"`, read-only property and `configure(engine_backend=...)`) | `settings.force_portable_kernels` (`True` / `False`) |
| `settings.trimul_impl` values `"triton"`, `"cuda"` (and `KernelBackend` names) | `"portable"`, `"native"`, `"pytorch"` (also `"cuequivariance"`, `"anthropic"`); other values raise |
| `miniworld-engine build --backend {auto,native,triton}` (hidden alias) | `build --kernels {auto,native,portable}` |
| bench spec strings `triton`, `triton_pytorch_ln`, `triton_ln`, `triton_kernel_ln`, `triton_dispatch_ln` (and the `-` spellings) in `benchmarks/runners/bench.py` and the `bench.yaml` files | `portable`, `portable_pytorch_ln`, `portable_ln`, `portable_kernel_ln`, `portable_dispatch_ln` |
| `modules.dispatch.KernelBackend.TRITON` (value `"triton"`) | `KernelBackend.PORTABLE` (value `"portable"`) |
| `modules.dispatch.triton_forced`, `triton_leg_allowed`, `triton_blocked` | `portable_forced`, `portable_leg_allowed`, `portable_blocked` |
| dev build-unit label `triton` (`registry_module.csv` `impls`, `dev capture --impl triton`) and the `trimul_impl=triton` build option | `portable`, `trimul_impl=portable` |

`settings.configure(...)` with a removed or unknown name raises a `TypeError` that names the replacement (a removed name) or lists the unknown one;
an invalid `trimul_impl` raises a `ValueError`. Kernel module and directory names (`kernels/*/triton/`, `*_triton` kernel names) and the
registry's `backend` column describe source code and are unchanged, as are the MPNN module's own `*_backend` strings (a separate surface; not changed here).

### Changed

- `UnrecordedCardWarning`, `KernelRefusedWarning`, the refusal reasons and the README / `docs/dispatch.md` / `docs/gpus/` text no longer present Triton as a choice: the card is not
  recorded, run `python -m miniworld_engine.autotune.run_all` to record it, and pass `settings.force_portable_kernels=True` only to force the portable kernels (A/B work).
- Nothing about routing or performance changes; the derivation hash moved with `settings.py` / `modules/dispatch.py` and the plans were re-derived.

## [2.3.11] - 2026-10-25

### Added

- `pair_embed_geom` (H100, sm_90a, forward only): the backbone pair embedding of La-Proteina's autoencoder encoder (sequence separation, CA-to-N/CA/C/CB distances,
  five inter-residue orientation angles, chain product; 317 binned channels per pair, a bias-free Linear, masks) as a gather of up to ten rows of a bf16 table in shared memory.
  Distances and angles are computed in fp32. Frozen encoder forward at B = 16, L <= 512: 35.4 -> 17.8 ms. Tests: `tests/gpu/sm90/test_pair_embed_geom_gpu.py`.
- `pair_embed_geom` training path, `pair_embed_ca` and the LayerNorm fusion (H100, sm_90a): `pair_embed_geom_train` is the encoder's backbone pair embedding differentiable
  in the Linear weight (the binned features are inputs, so the only gradient is the table's: `dT[row] = sum of d pair over the pairs that use the row`, the chain row weighted
  by the chain product); `pair_embed_ca` is the autoencoder DECODER's pair embedding (sequence separation 127 + CA distance 30 bins, 157 rows) with the same forward and backward.
  The forward leaves 16 bytes per pair (the packed bins with a live bit, the five angle bins, the chain product). The backward is `dT = F^T dpair` on tensor cores: the bf16 one-hot
  A fragments of each feature group are built from the codes in registers (nothing is materialised), the dpair tiles come through a cp.async ring and `ldmatrix.trans`, fp32
  accumulation, one CTA per (64 channels, pair slice), the 33 slice partials added in a fixed order: bitwise deterministic, no atomics. L = 512, B = 16: 40.0 -> 5.3 ms against the
  earlier shared-memory kernel (kept, `MINIWORLD_PAIR_EMBED_BWD_IMPL=0`). The gradient is that of a bf16-autocast Linear (the weight is rounded to bf16 by an ordinary autograd op,
  the table gradient is rounded to bf16 once). `ln_eps=`: the kernel returns LayerNorm(pair) over the 256 channels (no affine) -- the input of the attention-bias projection --
  and never stores the pair; the backward recomputes it from the codes (`pair_embed_geom_ln_bwd`), which removes the separate LayerNorm kernels (fwd + bwd 8.8 ms of the 98 ms kernel
  time of an autoencoder step at B = 16). Autoencoder training step (4 x H100, bucketed, cap 512, B = 16 per GPU): kernel time per step 98 -> 53 ms with these, the CUDA graph
  and fused AdamW (0.129 -> 0.067 s/step). Decoder loss bit-identical to the dense builder, encoder loss within 2e-4 (the documented fp32-vs-autocast bin flips).
  Tests: `tests/gpu/sm90/test_pair_embed_geom_train_gpu.py` (values, weight gradient against the dense builder and against the kernel's own bins, both backward kernels,
  LayerNorm fusion, bitwise determinism).
- `pair_embed_adaln` (H100, sm_90a): La-Proteina's pair builder as one kernel pair -- the three binned one-hots (sequence separation, CA
  distances of x_t and of the self-conditioning input) as a row gather, the affine LayerNorm and the adaptive LayerNorm with the per-sample
  `gamma` / `beta` (the diffusion-time conditioning is constant over (i, j)), bf16 pair out; the backward is bitwise deterministic (per-CTA
  partials, fixed-order reduces; the batch is processed in chunks so the bf16 grad of the table sum stays under 512 MiB). 256 channels.
  La-Proteina denoiser, B = 8, L <= 512: pair builder forward 63.6 -> 5.1 ms, training step 212 -> 128 ms. Tests:
  `tests/gpu/sm90/test_pair_embed_adaln_gpu.py` (against the dense builder, all parameter gradients, determinism). The backward's table-gradient kernel keeps four
  pairs' loads in flight.
- H100 token DiT training over B independent proteins (`DiTBlock` with `single [1, B, L, d]`, `pair [B, L, L, dp]`, a key mask per protein), at
  La-Proteina's widths: d_cond 128 / 256 / 384 / 768, pair channels 128 / 256, transition x2 or x4, 12 x 64 heads. The fused bf16 block with an
  attention bias of its own per sample now takes these (`token_dit_train.serves_batched` / `block_batched`, `serves_psb` widened; the row
  kernels build per `(d_cond, n, d_pair)` with `-DTD_DC/-DTD_N/-DTD_DP`, the per-sample key mask in `train_keymask90`). Trunk of 14 blocks,
  B = 8, L = 384, bf16 stream: forward 56 -> 13 ms, forward + backward 113 -> 32 ms against the PyTorch block (3.5x), peak memory 18 -> 8 GiB.
  Tests: `tests/gpu/sm90/test_token_dit_batched_pairs_gpu.py`, the head-layout cases of `test_augmented_attention_psb_gpu.py`.
- Routing and the prebuilt bundle for the three La-Proteina items above: the pair builders are H100-only (`refusal` / `refusal_ca` say why a call is not served: not CUDA, not capability (9, 0),
  widths; the caller runs its PyTorch builder, no Triton family exists for them), and `DiTBlock` takes the batched-proteins block only on sm_90 under the existing `token_dit_train` setting; every other card
  and every call the kernels refuse keeps the module path. `build/bundle_coverage.py` gains `la_proteina_pair` (every pair-builder extension and backward variant) and `la_proteina_dit` (the block at token 768,
  12 x 64 heads, pair 256, d_cond 256 / 128, x4, inference and training, L = 128 / 256 / 384 / 512, B = 5 at 384 for the empty-uniform build) so a wheel carries them instead of JIT-compiling on a first call.

### Fixed

- Registry bands of the two window-attention Triton backward rows (`augmented_attention_bwd_window_triton`, `augmented_attention_bwd_reduce_window_triton`) set at 4x the H100
  `run_all` measurement (bf16); corrects the 2.3.10 bands.
- Documented hazard (no code change): the H100 token DiT training path rebuilds its weight packs only when a parameter's `_version` changes (`token_dit_train._pack`), and a *fused*
  optimizer (`torch.optim.Adam(W)(fused=True)`) updates parameters in place without bumping it -- eager steps then run on the previous step's weights (a fused AdamW
  autoencoder run learned visibly slower, and an eager step disagreed with a CUDA-graph replay of the same state). Call `torch.autograd.graph.increment_version(p)` for every
  parameter after a fused step (La-Proteina's `graph_step` does); inside a CUDA-graph capture the packs are rebuilt by the graph and need nothing.
- The sm_90 `attn_dq` kernel multiplied dQ by 1/sqrt(48) whatever the head width (`fwd_defs` never passed `RSQDV`): a 12 x 64 dQ was 15 % too large,
  a 24 x 32 one 18 % too small. Reached by the per-sample-bias backward and by the shared-bias backward when A is not a multiple of 3.
  `augmented_attention.cuda.dq_defs` carries the scale; the 16 x 48 build, and its key, are unchanged.

## [2.3.10] - 2026-10-24

### Added

- Portable Triton paths for the model variants that had no Triton kernel and ran the PyTorch statements on every card without a native CUDA
  path, routed like the standard ops (`miniworld` on a card with a record, `force_triton_kernels`; a call the kernels refuse warns once
  with `KernelRefusedWarning` and runs the PyTorch statements). Forward and backward, bf16, validated on H100 under the A/B switch only
  (no Ampere manifest holds these kernels yet: those cards keep warning and running PyTorch until `autotune.run_all` records them).
  Record: `docs/records/triton_gaps_a_20261024.md`.
  - `MSAPairWeightedAveraging` with a head narrower than a `tl.dot` tile -- AF3's `msa_value_dim` 8 (OpenFold3, Protenix, OpenDDE): the
    five kernels that carry the head width tile it as `next_pow2(max(16, d_hidden))` columns and mask the rest to zero on every load and
    store, so the padded columns contribute nothing and are never written; any `d_hidden` up to 128 with `n_head * d_hidden` a multiple
    of 16 is served (was: powers of two >= 16).
  - `OuterProductMean(projection_bias=True)` (RoseTTAFold3): the biases are added in the projection kernel before the mask; the backward
    kernel sums `db_left` / `db_right`. `OuterProductMean(groups=G)` (Chai-1): the projection kernel stores the hidden channels group-major,
    the outer product is one batched GEMM over the groups, and three new kernels (`outer_product_mean_epilogue_ln`,
    `outer_product_mean_bwd_epilogue_ln_dx`, `outer_product_mean_bwd_epilogue_ln_dw`) gather the pair's products, run Chai-1's product
    LayerNorm and give its gradients (the projection and `dz W_out` stay in cuBLAS). `kernels.outer_product_mean.reference` takes the
    variants' arguments.
  - `TriangleAttention` with `gate_bias` / `out_bias` (RoseTTAFold3), `untransposed_output` and `dual_output` (Chai-1): the variants'
    epilogue and orientation composed around the family's Triton attention core (`kernels/triangle_attention/triton/variants.py`), so
    no `[B, L, H, L, L]` logits tensor is built. Heads of 16..128 channels, any width the cuBLAS projections take.
- Registry rows, drivers (the projection kernels are driven at the base call and both variants; PWA at `d_hidden` 32 and 8) and checkers for
  the three new kernels; module rows for the variants in `registry_module.csv` (the derivation reaches them).
- Portable DiT paths on the engine's Triton kernels, behind `miniworld` on a card the engine holds a record for (`docs/dispatch.md`; H100 / B200 keep
  CUDA, else PyTorch, and run these under `settings.force_triton_kernels` for A/B work):
  - `kernels/augmented_attention/triton/window.py` (`window_attention`, `WindowAttentionFunction`): the AlphaFold-3 atom attention (32 queries x 128 / 192 keys per
    window, per-window pair bias, the slid or clipped key window, key mask, ESMFold2's rank window, Chai-1's same-conformer mask), forward and backward, no
    `[nwin, 32, keys]` fp32 intermediates and no atomics (the key / value gradients are summed in a fixed order, the bias gradient over fixed sample groups: bitwise
    reproducible). `LocalDiTBlock` takes it as its attention core (`_window_triton`; AF3 and ESMFold2 variants); registry rows
    `augmented_attention_{fwd,bwd,bwd_reduce}_window_triton`, op `local_dit` of `autotune.records`.
  - `kernels/dit_variants/triton/portable.py`: `DiTVariantStack` / `AtomDiTVariantStack` (Boltz-2, RoseTTAFold3, Chai-1) on those cores (token stacks on the
    token pair-bias attention), the pair's LayerNorm + projection in one pass for every block (`layernorm_linear_triton_fwd_t`), differentiable by autograd.
    Entry points: `portable.serves` / `portable.atom_serves` (the module's forward calls them after the H100 path).
  - `bench.py`: `+dit_variant=boltz2|rosettafold3|chai1` on the `dit` (token stack) and `local_dit` (atom stack) targets.

### Changed

- `OuterProductMean`'s variant branch tries the Triton leg before `_variant_reference` (H100 / B200: unchanged unless the A/B switch is on).
  The H100 CUDA variant paths (`composed`) step aside for `force_triton_kernels`.
- `TriangleAttention`'s variant branch (gate / out biases, Chai-1 output forms) tries the Triton core before `_biased_reference`.
- `_opm_layernorm_gemm_kernel` / `_opm_bwd_layernorm_gemm_kernel` take `K` and `HAS_PBIAS`; the five PWA kernels that carry the head width
  take `CP`: their autotune keys changed (cached configs of those kernels are re-tuned on first use).
- The OuterProductMean Triton backward reduces `db_out` straight from the bf16 gradient (`sum(dtype=float32)`, no fp32 copy of `dz`).
- Fixed: the Triton launchers of OuterProductMean / MSAPairWeightedAveraging passed the same empty tensor for every absent input, which
  `torch.compile` refuses in an `autograd.Function.apply` (a call with no mask and no residual).


### Changed

- **Triton is no longer a public backend.** `ImplementationType` has exactly `pytorch`, `miniworld`, `cuequivariance`, `anthropic`
  (`TRITON` and `CUDA` are gone). On a card other than H100 / B200, `miniworld` runs a CUDA path that serves the call; else the
  Triton kernel family **only if the engine holds a record for that exact card** (every Triton kernel of the op has an `ok` row in
  `autotune/manifests/<card>.csv`, written by `python -m miniworld_engine.autotune.run_all`); else it emits an
  `UnrecordedCardWarning` (once per op and card) and runs the PyTorch reference. H100 and B200 are unchanged (CUDA, else PyTorch, never
  Triton). Cards recorded today: A100 80GB PCIe, H100 80GB HBM3, RTX A5000, RTX A6000; their manifests predate several kernels, so most
  ops on the three Ampere cards warn and run PyTorch until the card is re-recorded (table: `docs/dispatch.md`). New:
  `autotune.records`, `modules.dispatch.route_for_card`, `UnrecordedCardWarning`.
  The record is mode-aware (an inference call needs the forward kernels only; a missing backward kernel blocks training calls), the
  `ops.*` entry points that sit on one Triton unit obey the same rule (`dispatch.triton_blocked`), and a Triton kernel that refuses a call
  (TriMul `d_hidden != d_pair`, OPM / PWA shape guards) is one `KernelRefusedWarning` and the PyTorch reference instead of a `ValueError`
  or a silent fallback.
- `settings.engine_backend: Literal["auto", "triton"]` became the internal A/B switch `settings.force_triton_kernels: bool`
  (the Triton family on every card, every `*_cuda` stage off). `engine_backend=` is accepted by `settings.configure` with a
  `DeprecationWarning`. `build --backend {auto,native,triton}` became `build --kernels {auto,native,portable}` (`--backend` is a hidden alias).
  `benchmarks/runners/bench.py` defaults to `implementations=[pytorch, miniworld]`; its `triton*` spec strings run `miniworld` under the switch.
- `OuterProductMean` / `MSAPairWeightedAveraging` no longer refuse a call the Triton kernels cannot serve: the `triton` request that did is gone.

### Deprecated

- `implementation="triton"` and `implementation="cuda"` are read as `miniworld` with a `DeprecationWarning` (on H100 / B200 that is the
  CUDA path; forcing Triton is only possible through `settings.force_triton_kernels`). **Removed in the next minor release**, together
  with the `settings.engine_backend` alias and `build --backend`.

### Fixed (release preparation)

- The Anthropic OPM path's epilogue extension (`integrations/anthropic_msa._ext()`, built as `miniworld_opm_epilogue`, no variant suffix)
  was not in the 2.3.9 wheel (only the `_native*` variants of `integrations/opm_train` were): the `anthropic` OPM path JIT-compiled it on its
  first call. `build/bundle_coverage.py` has a case for it (`anthropic_ext`), and a FoldForge test run with the Anthropic variables set and an
  empty cache compiles nothing.
- `anthropic_msa.opm_refusal` knows the module's `normalize_before_proj`: the epilogue divides by the pair count before the output projection and
  adds the bias undivided, so `OuterProductMean(normalize_before_proj=False)` (ESMFold2: the bias is divided too) is refused with a reason
  (`implementation="anthropic"` raises `PayloadUnavailable`; `miniworld` with a payload named falls through to its other legs) instead of being
  served with the wrong bias (relative error about 7).
- `integrations.opm_train.wanted` and `integrations.pwa_train.wanted` no longer answer to `ANTHROPIC`: `implementation="anthropic"` runs the
  release's path, not the engine's own H100 path first.
- `modules.dispatch.resolve` accepts `ANTHROPIC` for every op whose module implements it (`triangle_multiplication`, `triangle_attention`;
  `_ANTHROPIC_OPS`), not for TriMul only; the other ops still raise `ValueError`. (`Transition` has an adapter in `integrations.anthropic` but
  no branch in the module: it still rejects the option.)
- `architecture.exceptions.ImplementationType` has `ANTHROPIC` (`to_engine_impl` maps it to the engine's option; architecture layers that have
  no Anthropic path raise `InvalidImplementationError` at their forward, as for any unsupported value).
- `tests/cpu/compile/test_compile_wrap_coverage.py` passes again: the CUDA rows kernel of `kernels/dit_variants` is `gate_rows_inplace`
  (its `gate_rows` name collided with the Triton token-DiT `gate_rows`, whose callers the test follows by name).
- Prebuilt wheel size and stale artifacts: the 2.3.9 wheels packed 11 re-keyed old kernels (artifacts whose key no import of the release
  computes) because their bundle was grown with `--keep` over earlier releases' roots. 2.3.10's bundle is made from recipes only: the records of
  this release's coverage (engine cases, MiniWorld's production shapes, FoldForge's suite with and without the Anthropic variables, pdall's
  training step) plus the recipes of eight kernels earlier releases' drivers reached and this release's did not happen to
  (`atom_sm90` TF32 split-K backward cubins, `augattn_sm90` `DB_PER_SAMPLE1` / `NWG2`, `triattn_h100_lt4`, TriMul d64 `b7gate` / `gatepipe`),
  every one replayed against this tree's sources, so each key is the one the loader of this tree computes. New:
  `dev bundle-kernels --drop-stale` (and `kernel_bundle.stale_keys`) leaves out recorded artifacts whose code graph changed since they were
  built; a recipe recorded without CUTLASS (the stand-in include) now resolves from its sibling's artifact (the replay's parallel pool
  raced on it: `CutlassMissing`).

### Notes

- **New Triton paths are validated on H100 only.** The portable Triton paths of this release (Triton gaps A and B: PWA `value_dim` 8, OPM
  `projection_bias` / `groups`, TriangleAttention variants, `LocalDiTBlock` window attention, `DiTVariantStack` / `AtomDiTVariantStack`) were
  run under the A/B switch on an H100. A100, RTX A5000 and RTX A6000 have no manifest row for them: until someone runs
  `python -m miniworld_engine.autotune.run_all` on those cards and commits the manifest, `miniworld` warns (`UnrecordedCardWarning`) and runs
  PyTorch there for these calls.
- **`miniworld` still uses the Anthropic payloads automatically** when `OPT_CORE_DIR` (OPM / PWA) or `TRIMUL_NATIVE_BUILD_DIR` (TriMul) is set
  and a call fits them. This is unchanged and documented in `docs/dispatch.md` ("The Anthropic payload"); whether it should require
  `implementation="anthropic"` is open.

### Migration: `implementation="triton"` / `engine_backend="triton"`

`implementation="triton"` and `"cuda"` still parse (a `DeprecationWarning`), as `miniworld`; they are removed in the next minor release.
`settings.engine_backend` and `build --backend` are aliases of `force_triton_kernels` / `build --kernels` and go with them. Code that used
them to force the Triton family should call `settings.configure(force_triton_kernels=True)` (A/B work only) and pass `implementation="miniworld"`;
code that used `"cuda"` or `"triton"` to mean "the engine's kernels" should pass `"miniworld"` (on H100 / B200 that is the CUDA path).
MiniWorld scripts to migrate (not edited here): `check_trimul_f567_runtime`, `check_trimul_backward_runtime`, `benchmark_trimul_fusion_ab`
and `benchmark_trimul_backward_fusion` pass `implementation="triton"`; `build_phase1_runtime_cache` and `benchmark_miniworld_training_graph`
pass `engine_backend="triton"`.

## [2.3.9] - 2026-10-24

### Added

- `kernels/sampler_ops` (H100, fp32 coordinates): the diffusion sampler's glue as a few kernels instead of ~100 small torch ops -- the random
  rigid motion + re-noising (`augment_noise`), the Kabsch alignment (Horn's quaternion method, graph-capturable) + Euler update
  (`align_euler`) and the atom encoder's pooling (`pool_atoms`), with the churn arithmetic on device scalars. One extension, no shape keys
  (covered by the `sampler_ops` case of `build/bundle_coverage.py`).
- `DiTBlock(extra_pair_bias=)` / `integrations.token_dit.serves_extra_bias`: an `[L, L]` logit bias for every head on top of the pair's own
  (the structural-token family's relation matrix), folded into the Hopper BF16 CUDA path's hoisted bias; a call no path serves raises
  instead of dropping the term. The fp32 token path takes any L that is a multiple of 128.
- H100 PairWeightedAveraging inference up to N = 2048 (the forward pair kernel holds H rows of N logits in shared memory; training stays
  N <= 1024). Record: `docs/records/h100_pwa_long_inference_20261022.md`.

### Changed

- H100 TriangleAttention forward: a speculative softmax offset (no row maximum before the exponentials from S >= 512; an increment test
  after the tile, an exact second launch for the rare tile no raise repairs), a task order by head for S >= 2048, descriptors built once
  per CTA (`triattn_sm90.cu`); the fp16 core and the width-generic path below.
- Per-block caches (`kernels/_capture.BLOCK_CACHE_LIMIT = 64`) hold one entry per block of a 48-block model; a sample-expanded conditioning
  keeps its hoisted tables across calls (a token DiT's pair bias was 15 % of a 1,024-token sampler's captured step).
- H100 variants of 2.3.8, faster: the variant stacks' weight pack is no longer re-made in every CUDA-graph replay (token stacks 2.8-4.1x the
  compiled torch statements, from 1.3-2.9x), 128-wide rows warp-per-row and the atom edge windows' bias per pair (atom stacks 1.6x, from
  1.2-1.4x), RoseTTAFold3's OPM biases in the wide H100 prologue (`opm_prologue_wide(pbias=)`; 1.2-1.5x, from 0.9-1.0x). Record:
  `docs/records/model_variants_h100_speed_20261022.md`.
- H100 TriangleAttention inference on fp16 operands (q / k / v / pair bias; `h100_sources/triattn_h16_sm90.cu`), the AF3 widths and the
  width-generic path: core 1.04-1.09x Anthropic's sustained, module 1.13-1.51x (AF3 widths) and 1.03-1.08x (template widths); about a third
  of the bf16 core's error against fp32. Record: `docs/records/triattn_h100_fp16_core_20261022.md`.

### Fixed

- A CUDA graph captured over more than 64 bf16 Transitions (Boltz-2's trunk: 64 pairformer blocks plus its template, MSA and confidence
  stacks; Chai-1's; RoseTTAFold3's) faulted with an illegal memory access on a later replay. The fused sm90a Transition reads a cached
  `Ws^T` per weight (`kernels/transition/cuda/_weights.py`), a capture reads that eager entry instead of recording the copy, and the cache
  evicted its oldest entry past 64, freeing a transpose a captured graph still read. An entry now lives as long as its weight (dropped when
  the weight dies or is cached again at a newer version).
- H100 TriangleAttention with RoseTTAFold3's gate / out-projection biases added them after the bf16 rounding of the gate logits and of
  `pair + update`: a second rounding at the residual's scale, sqrt(2) the bf16 reference's error on the update at a trunk's pair scale
  (RoseTTAFold3's pLDDT moved by 1.5 at 256 tokens). The sm90 tail now takes both in fp32 before its one rounding (`tri_tail(bo=, gb=)`),
  the width-generic path the out bias in its out-projection GEMM's cublasLt epilogue (`lt_mm(..., bias)`; its gate bias is still a bf16
  add to the bf16 logits, an error at the logits' scale only).

## [2.3.8] - 2026-10-22

### Added

- Boltz-2, RoseTTAFold3 and Chai-1 modules on H100, inference (bf16): `modules.dit_variant` (`DiTVariantStack` / `AtomDiTVariantStack` with
  `BOLTZ2`, `ROSETTAFOLD3`, `CHAI1` and their `_ATOM` options: sigmoid or identity-centred AdaLN, kq LayerNorm, no gating query, no
  atom output projection, up-gate transition, parallel schedule, same-conformer atom mask); `TriangleAttention(gate_bias=, out_bias=,
  untransposed_output=, dual_output=)` and 64-channel heads (a mma.sync core, `kernels/triangle_attention/cuda/h100_dh.py`);
  `OuterProductMean(projection_bias=, groups=)`; `AttentionPairBias(gate_bias=)`; `PairformerBlock.from_components(parallel=True)`.
  New CUDA: `kernels/dit_variants/cuda/rows.cu`, `kernels/triangle_attention/cuda/h100_sources/triattn_dh_sm90.cu`,
  `kernels/outer_product_mean/cuda/gather.cu`. Against `torch.compile` of the same statements (CUDA graph): token DiT 1.3-2.9x,
  atom DiT 1.2-1.9x, Chai-1 trunk triangle attention 3.2-6.5x, grouped OPM 1.1-1.2x, biased OPM 0.9-1.0x. Modules without these
  options are unchanged. Page: `docs/gpus/h100/model_variants/model_variants.md`.

## [2.3.7] - 2026-10-21

### Fixed

- H100 SWA atom DiT (`SWADiTBlock`, `swa_dit_block`, ESMFold2's atom block) ran 5-9x slower than it did on 2026-10-14. Since 2026-10-18 every
  H100 call took the native stages (`_native_h100_fwd` / `_bwd`) instead of the composed hand-CUDA stages documented in
  `docs/gpus/h100/swa_atom_dit`, and inside the native stages the window attention was the untiled `native_attention` for every call whose
  samples share their conditioning (every diffusion sampling step) or that runs fp32: 1520 against 41 us at A 5, 8192 atoms. The tiled
  attention now serves every bf16-head call with S % 64 == 0 (`_tiled_serves`), and the stages are chosen by `native_h100_stages`: bf16, and
  fp32 under TF32, take the composed stages (all hand-CUDA, no Triton launch, A 1 - 48 x 256 - 16 384 atoms, ragged, train / infer);
  fp32 under "highest" and global attention keep the native ones. `native_h100_enabled` again only says whether H100 serves the call.
  The native stages' bf16 modulation runs on the tensor cores (it was an fp32 FFMA GEMM). A bf16 modulation reaching the native stages
  (global attention) is widened there, and the composed bf16 backward returns fp32 weight gradients as its contract says. One block,
  A 5, CUDA graph, bf16: 1024 / 8192 / 24 576 atoms 0.264 / 1.663 / 4.835 -> 0.051 / 0.196 / 0.545 ms (1.5 - 2.3x ahead of Anthropic's
  ESMFold2 kit); fp32 (TF32) 1.3 - 1.5x faster inference, 1.5 - 1.7x faster training than on the native stages. Record:
  `docs/records/swa_dit_h100_routing_20261021.md`.

## [2.3.6] - 2026-10-21

### Added

- H100 TriMul with a hidden width twice the pair width, one direction, at D64 / D128 (`TriangleMultiplication(d_pair, d_hidden=2 *
  d_pair)`: AF3 / Protenix template blocks, pair 64, hidden 128), inference and training, which ran the PyTorch reference on H100 (the
  dispatch required `d_hidden == d_pair`). Per token it has the bidirectional block's shapes (4 D planes, LN_out over 2 D), so the
  bidirectional D64 / D128 kernels serve it; only the cuBLAS contraction between them pairs the planes in the module's one direction
  over all 2 D channels (`direction` 1 / 2 of `h100_d64_b200gate` / `h100_training`, `trimul_h100._hidden_twice`). B samples (chunked
  with the bidirectional working set), token mask, row dropout, CUDA graphs; the hidden = D paths are bitwise unchanged. CUDA graph,
  against cuEquivariance 0.12: inference 2.0 - 3.5x, training fwd + bwd 2.0 - 3.7x (L128 - 768, B 1 / 4); against the PyTorch path it
  replaces 8 - 28x. Tests: `tests/gpu/sm90/test_trimul_h100_hidden2x_gpu.py`; page: `docs/gpus/h100/trimul/trimul.md`.
- `LocalDiTBlock(**modules.local_dit.ESMFOLD2)`: ESMFold2's atom block (FoldForge `spec.ESMFOLD2` / `ESMFOLD2_FAST`). New options
  `conditioning="rms_silu"` (`AdaptiveRMSNorm`: `(1 + W_s silu(c)) * rms(x) + W_b silu(c)`, float32 eps, no biases; raw output gates `W silu(c)`),
  `rope=True` (q / k rotated after the QK-norm with per-atom tables, `forward(..., rope=(cos, sin))`, `[B, N, 16]`), `keys=192` (AF3's slid
  subset at 192 keys, pair `[B, nwin, 32, 192, d_pair]`) and `window_half=64` (query `i` sees key `j` only when `|i - j| <= 64` by rank among
  the real atoms). Defaults are unchanged. Matches FoldForge's dense `DiffusionCrossAttTransformer(spec=ESMFOLD2)` to fp32 rounding (random and
  trained weights). The H100 kernels serve it for inference, bf16 and fp32/TF32: the row kernels built with `-DRMS=1`, `qknorm_rope_fwd`
  and `lattn90_esm.cu` (position-indexed pair bias, the +/- 64 band attention); 12-28x the composed PyTorch path and 3.4-5.5x its
  `torch.compile` at A = 5, N 1024-24 576 (docs/gpus/h100/local_dit). A call with grad takes the composed path. Tests
  `tests/cpu/compile/test_local_dit_esmfold2_variant.py`, `tests/gpu/sm90/test_h100_local_dit_esmfold2_gpu.py`. New sm90 artifacts (not in
  the 2.3.5 bundles): `atom_sm90/{cond,pre,post}90_*_rms` and `atom_local_sm90/lattn90_esm`; the AF3 block's forward row cubins
  (`atom_sm90/{cond,pre,post}90_*`, now including `rms90.cuh`) and `augattn_sm90_qknorm_local` change key too, so a 2.3.5 bundle no longer
  covers them: the next release's bundle must be rebuilt (the backward row cubins keep their keys).

## [2.3.5] - 2026-10-21

### Fixed

- The wheel's sm90 bundles carry the single-direction TriMul training backward for fp32 master parameters fed bf16
  activations (`trimul_h100/single_output/kernel.cu`, `single_b7/joint.cu` with `-DMASTER_FP32=1`, `h100_single`): pdall's
  training step compiled both on its first call with 2.3.4 (and failed without nvcc / CUTLASS). The engine coverage runs the
  modules in that precision too ("master"), and the release coverage records pdall's training step
  (docs/kernels/prebuilt-kernels.md). Kernels are otherwise the 2.3.4 ones.

## [2.3.4] - 2026-10-21

### Fixed

- The wheel's sm90 bundles also carry what FoldForge reaches: its families' single attention runs the H100 augmented
  attention at a 512-wide row (`augattn_sm90_attn_fwd ... ROW_WIDTH512 ... APB`), which no engine or MiniWorld driver
  reaches, so a FoldForge model compiled it on its first call with 2.3.3. The release coverage now also records FoldForge's
  suite (docs/kernels/prebuilt-kernels.md). Kernels are otherwise the 2.3.3 ones.

## [2.3.3] - 2026-10-21

### Added
- `LocalDiTBlock(kv_norm="parallel" | "chained", key_window="slide" | "pad")`: AlphaFold 3's atom attention. `kv_norm="parallel"` runs the K/V AdaLN on the
  raw atoms next to the first one (Protenix chains it, the default); `key_window="slide"` slides each 128-key window inside the real atoms
  (`[32 w - 48, 32 w + 80)` shifted back, as `AtomCrossAtt.compute_features`) instead of clipping and masking it. Defaults are unchanged. The H100
  mma kernels serve both for inference (`pre90_fwd(..., par)`; the first two and the last windows are recomputed in torch for the slide); a
  call with grad takes the composed path. The real atom count stays on the device (the edge windows are the first two and the last three real ones, recomputed in torch), so the block runs inside a CUDA graph capture and below 128 real atoms. Tests
  `tests/cpu/compile/test_local_dit_af3_variant.py`, `tests/gpu/sm90/test_h100_local_dit_af3_variant_gpu.py`.
- `TriangleAttention(transposed_bias=...)` and `kernels.triangle_attention.whole_op(transposed_bias=...)`: which pair the ending node projects its
  bias from. `True` (default, the behaviour so far) projects it from the transposed pair, the supplementary-information and OpenFold convention;
  `False` projects it from the pair as it is and reads it in the transposed frame, which is what AlphaFold 3's released code does (the two
  differ for an asymmetric pair). Composed legs (PyTorch, Triton, cuEquivariance) transpose the bias; the H100 hand-CUDA kernels serve the
  `False` orientation for inference by transposing the front's bias before the core (no kernel change; fused 128/4x32 and width-generic
  paths); a call with grad or training dropout, the B200 kernels and the Anthropic adapter do not serve it (the first two fall to the composed
  legs, the adapter refuses). Starting-node attention ignores it. Tests `tests/cpu/compile/test_triangle_attention_bias_direction.py`,
  `tests/gpu/sm90/test_triattn_bias_direction_gpu.py`.
- H100 token DiT training with a pair bias of its own per sample (`[1, A, L, d]` with pair `[A, L, L, d_pair]`, as the bias-only DiT got in
  2.3.2): the augmented-attention kernels are built with a per-sample bias (`PSB=1` variants of `attn_fwd` / `attn_dq` / `attn_dkv`) and
  `integrations/token_dit_pair_bias.py` (`pair_biases`, `serves_psb`) routes the call; d_cond 384 / 768. Tests
  `tests/gpu/sm90/test_augmented_attention_psb_gpu.py`, `tests/gpu/sm90/test_token_dit_psb_gpu.py`.
- H100 local atom DiT with QK-norm (`LocalDiTBlock(use_qk_norm=True)`) on the mma path: q / k are RMS-normalised per head by
  `kernels/augmented_attention/cuda/qknorm_local.cu` before the attention (the backward keeps the raw ones). Tests
  `tests/gpu/sm90/test_h100_local_dit_mma_gpu.py`.

### Fixed

- The wheel's sm90 bundles cover the bias-only token DiT's own paths, not only what MiniWorld's bf16-mixed runs reach: the
  engine coverage (`build/bundle_coverage.py`) now drives the block with a shared pair and a pair per sample, masked,
  hoisted, d_cond 384 / 768, in every precision. 2.3.2 compiled the fp32 rows and the per-sample / hoist-backward kernels on
  first call.
- Importing `triangle_attention/cuda/build_*.py` (a module walk: `dev audit`, the CPU suite) rebuilt those extensions and
  overwrote the committed `.so` / `*_manifest.json`; the walk skips the rebuild scripts. A manifest's failure detail no longer
  names an account or site path (`<path>`). `test_capture_saw_ops` no longer requires the per-op SWA path's Triton ops on
  H100 / B200, which run no Triton.
- H100 TriangleAttention forward core: the softmax offset is the row's running maximum raised lazily (threshold 8 log2 units, o and l rescaled) instead of the
  maximum of the first 32 keys fixed for the task, which overflowed to inf / NaN when a later logit exceeded it by ~126 log2 units (trained weights do). The reduction order
  is unchanged (deterministic). Test `tests/gpu/sm90/test_triattn_large_logits_gpu.py`.
- Pointer-keyed weight and conditioning caches (Transition weight re-layout, token / bias-only / local DiT pair-bias and conditioning tables, conditioned-transition
  packs) no longer serve a stale entry when a new tensor reuses a freed address at the same version: an entry keeps a weak reference to its key tensor and a hit
  needs that reference to be the tensor asked for. Tests `tests/cpu/kernels/test_transition_weight_cache_identity.py`, `tests/gpu/sm90/test_pointer_keyed_caches_gpu.py`.

## [2.3.2] - 2026-10-20

### Added

- H100 bias-only DiT: a pair bias of its own per augmentation (training, bf16 + TF32), d_cond 384 / 768
  (docs/records/h100_bias_only_dit_per_aug_bias_20261020.md).

### Fixed

- The wheel's sm90 bundles now carry MiniWorld's production kernels. 2.3.0 / 2.3.1 were covered by the engine's own
  drivers only, which do not reach the fused SWA atom DiT (`swa_dit`) or the bias-only token DiT (`bias_only_dit`), so a
  MiniWorld model on an H100 still compiled those on its first call. The release coverage now also records MiniWorld's
  `scripts/prebuild_engine.py --mode all` (every inference rung and training crop of the production configs).
- Extensions built with `extra_ldflags` (the SWA atom DiT, the token DiT's owned GEMM) were bundled under a key no
  process asks for: torch's `load` appends its link flags to that list in place, the recorded recipe kept the appended
  list, and its replay keyed the rebuild from it. The engine now hands torch a copy, and a replay links with the flags
  the key was computed from (`build.kernel_bundle.keyed_recipe`).

## [2.3.1] - 2026-10-20

### Fixed

- `pydantic` is a runtime dependency: `miniworld_engine.architecture` (the block and diffusion configs) imports it, and a
  plain `pip install miniworld-engine` (2.3.0) could not import `DiffusionTransformer` or any other block. It was only in
  the `bench` extra. The kernels and the prebuilt bundles are the same as 2.3.0.
- `miniworld-engine dev bundle-kernels --replay` compiles the tree it runs from and starts each bundle clean (records
  from another checkout are rebased, stale artifacts pruned).

## [2.3.0] - 2026-10-20

### Added

- Prebuilt CUDA kernels in the engine's own binary wheel (`miniworld_engine/_prebuilt/<arch>/`, docs/kernels/prebuilt-kernels.md):
  every torch extension and cubin is resolved by a content key from its include graph (`kernels/_codegraph.py`:
  normalized sources and the headers they reach, flags without machine paths, arch, torch, CUDA, Python ABI) -- shipped
  binary first, then the local cache, then a JIT build (`settings.jit_fallback`, `MINIWORLD_ENGINE_JIT_FALLBACK=0` makes a
  miss an error). The local JIT cache uses the same keys: an edit that does not reach a kernel (a comment, an unrelated
  header in the same directory) no longer rebuilds it, `TORCH_EXTENSIONS_DIR` is no longer used, and TriMul's
  `_source_digest` no longer keys cubins. `miniworld-engine dev bundle-kernels` (coverage drivers, recipe replay for other
  Python ABIs / torch / CUDA builds, incremental refresh), `dev build-wheel` (cpXY manylinux_2_28 wheels pinned to the
  torch minor, auditwheel-style binary check), `dev code-graph` (an artifact's graph and key, `--affected <file>`).
  CUTLASS / CUDA toolkit headers are needed only to build. License files for CUTLASS and FlashAttention-3 (BSD-3-Clause).
  The wheels (cp310, cp312, manylinux_2_28) carry the sm90 kernels for both CUDA builds of torch 2.13 (`sm90-cu129`,
  `sm90-cu130`; the running torch picks its own) and require `torch==2.13.*`.

### Fixed

- CUDA 13 builds: 15 sources that named the unversioned `PFN_cuTensorMapEncodeTiled` (removed in CUDA 13) resolve
  `cuTensorMapEncodeTiled` through `cudaGetDriverEntryPointByVersion(..., 12000, ...)` (OPM epilogue, PWA, `ln_vg`,
  `dgv_bwd`, SWA DiT sm90); same SASS under CUDA 12.9. The Hopper atom cubins (`atom_local_sm90`, `local_dit_sm90`) and
  the owned cuBLAS / cuBLASLt extensions no longer refuse CUDA newer than 12.9 (no reason was recorded; the four
  `atom_local_sm90` cubins in the bundle come out of 12.9 and 13.0 with identical SASS).

### Changed

- H100 TriangleAttention takes B samples in one call by design (`integrations.triattn_h100_sm90.max_batch(kind, length)` = 48 per launch
  for both paths, `kind_of`, `plan`): both CUDA paths already launched each kernel once over the B samples (per-sample key mask and
  dropout scale); a call with more samples, or whose working set would not fit, now runs in balanced chunks of a deterministic size
  (shapes, mode and the device's total memory; inference chunks write their output rows in place through new `out` arguments of
  `tri_tail` and the cublasLt out-projection, training chunks run in one autograd node, `_Chunked`, keeping or rerunning their
  forward, with the parameter gradients summed in fp32) instead of running out of memory (d_pair 128 and 64 training L768 B48, d_pair
  256 inference L768 B48, d_pair 512 inference L768 B16, d_pair 256 training L768 B16 did). Kernel scheduling for long launches: the
  attention core forward takes its tasks from a global counter (`TA_DYN`; the static round robin drifted apart: 940 vs 841 us per
  sample at L768 B48, same bits) and the query-side backward runs in one-wave launches of 96 // (heads x L / 128) samples
  (`h100_sm90.bwd_q`; one launch of 48 samples was 1.07-1.55x slower per sample); `triattn_bwd_q` gained a `slabs` override. y is
  bitwise equal to B one-sample calls in all 245 cells of the B 2-48 x L128 / 384 / 768 x seven-width grid (dx within 3e-5, parameter
  gradients within 2.8e-3 of the fp32 per-sample sum); B = 1 is unchanged in bits and the same or faster. CUDA graph: one call vs B
  calls 1.01-2.22x where the saved tensors fit, 1.51-4.53x over cuEquivariance and the compiled PyTorch statements. Tests
  `tests/gpu/sm90/test_triattn_h100_sm90_batched_gpu.py`, `tests/cpu/kernels/test_triattn_h100_batch_plan.py`; record
  `docs/records/triattn_h100_batched_20261020.md`.
- H100 TriMul takes B samples in one call at every width (`integrations.trimul_h100.max_batch(width)`, `MAX_BATCH_BY_WIDTH`, 48 at
  D64 / 128 / 256 / 384 / 512; `MAX_BATCH` keeps its D64 meaning): bidirectional and single direction, inference and training, tokens
  b-major, a token mask and row-dropout scale per sample. D128 on the native kernels (K1 / K3, `save_k3`, B1 with a separate
  `B1_BATCHED` build, B7, `h100_native.Front`, `single_b1` / `single_b7`), D256 / D384 / D512 on the B200-style port (K1 `rows`, K3 and
  `gate_bwd` read the dropout scale of the row's sample; wide inference at L % 128 != 0 is padded on the host). The contraction folds
  the samples into cuBLAS's batch dim. A call with more samples than one launch takes or than fit in memory runs in launches of a
  deterministic size (`trimul_h100._plan`: shapes, mode and the device's total memory; inference writes each chunk into the output in
  place via `h100_b200style.inference_into` / `h100_inference.inference_into`; training chunks run in one autograd node, `_Chunked`,
  writing their rows of y and dx in place, plain or checkpointed), so no batch falls back to the PyTorch statements or runs out of
  memory because of size while the caller's x / y / dy / dx fit (D256 L768 B48, D384 B26, D512 B20 training run). y and dx are bitwise
  equal to B calls of one sample at every width (B 16 / 32 / 48 at L128 / 384 also against the FP32 statement, 144 / 144 cells);
  B = 1 is unchanged (bitwise at D128, y / dx bitwise at D256+, time 0.984-1.012 of main). H100 80GB, CUDA graph: one call vs B calls
  1.13-2.25x at L128, inference 0.99-1.10x and D128 training 1.02-1.17x at L384 / 768; at the preserved D256+ training cells the faster
  of the port and per-sample preserved launches into y / dx is chosen per cell (`PER_SAMPLE_LEGACY_TRAINING`), 0.994-1.03x of B
  calls (1.02-1.16x of B calls that accumulate their gradients); checkpointed chunks 0.73-0.80x of a per-sample fwd + bwd loop that a
  model running B calls and one backward cannot use; 2.0-5.2x over cuEquivariance, 3.1-19.6x over the PyTorch statements H100 ran for B > 1 before. Tests `tests/gpu/sm90/test_h100_trimul_batch_wide_gpu.py`;
  record `docs/records/trimul_h100_batched_wide_20261019.md`.
- H100 TriMul beyond L1024 and at ragged L (`integrations.trimul_h100`): D64 / D128 (both modules, every direction, D64 with B <= 48)
  infer at any L >= 16 (`MIN_INFERENCE_LENGTH`; one sample pads to 16 inside K1 / K3, B samples at L % 16 != 0 are padded on the host)
  and train at any multiple of 128 up to 10240 (`TRAINING_GRANULE`, `MAX_TRAINING_LENGTH`, B200's bound; memory binds first); D256 /
  D384 / D512 take the B200-style port at multiples of 128 above L1024. The kernels were already length-generic; the length tuples
  L128 ... 1024 in the integration, `h100_b1.Plan`, `h100_single.Plan` and `h100_b200style.trimul` became `L % 128 == 0`. Fixed: one
  sample at L % 16 != 0 failed in `h100_inference` since the batched-samples change (`rows = Np` reached K1). MiniWorld's rungs
  L1536 / 2048 / 2560, D64 B = 4 and D128 bidirectional: 7.5-12.5x over the PyTorch statements H100 ran there before, 1.8-3.2x over
  cuEquivariance; the cells served before are unchanged (0.994-1.009). Tests `tests/gpu/sm90/test_trimul_h100_long_l_gpu.py`; record
  `docs/records/trimul_h100_long_l_20261019.md`.
- H100 TriMul D64 takes B samples in one call (`integrations.trimul_h100.MAX_BATCH = 48`, same meaning as the B200 one): bidirectional
  and single direction, inference and training. K1 / K3 / `save_k3_d64` stack the samples on the slow token axis (`rows = B L`), the
  token mask, pair mask and row-dropout scale are per sample, the contraction batches over channels x B, B1 / B2 run over every
  sample's rows. y and dx are bitwise equal to B calls of one sample; B = 1 is unchanged. MiniPairformer block (TriMul + Transition),
  B = 4: 1.13-2.21x over 4 calls, 4.1-11.5x over the PyTorch statements H100 ran before for B > 1.
- H100 AF3 local atom DiT (`LocalDiTBlock`, cross-attention): native fp32 (TF32) and bf16 paths for inference and training at any A and any N
  (`integrations/local_dit_h100.py`, `kernels/augmented_attention/cuda/sm90_atom/`, `sm100_atom_local/lattn90*.cu`). Warp-level tensor-core row
  kernels (wgmma with register A operands, 16-byte lane layout) and `mma.sync` window attention with the bias resident in shared memory.
  Harness (CUDA graph, 4096 atoms): bf16 inference A5 0.098 ms (Anthropic 0.322), fp32 A5 0.151 ms (0.468; the previous fp32 path was the
  module composition at 1.57 ms), bf16 training A48 3.6 ms (was 11.7; PyTorch eager composition 25.3, torch.compile 7.4), fp32 training A48 6.0 ms (was 21.8; PyTorch eager 26.2, torch.compile 12.0).
  fp32 gradients are within the TF32 module's error (2-5x better on the output). Switches `settings.local_dit_h100_mma{,_fp32,_bf16}`.
  A conditioning passed as an expanded view (stride 0 over the samples) is tabulated once per atom (A48 / 4096 atoms: bf16 training
  3.59 -> 3.09 ms, fp32 6.02 -> 4.79 ms; the backward reduces four samples per CTA group in shared memory and a single `cond_bwd_sh90`
  kernel replaces the sum + 13 small GEMM / elementwise launches). `bench.py target=local_dit` gained `+shared_cond=true`.
  Small row counts: one-launch weight pack (`pack90.cu`, was ~150 kernels / 230 us per step) and split launches of the row kernels (A1 N128
  fp32 training 0.538 -> 0.242 ms, now faster than `torch.compile` 0.277 and every other measured column in all 44 cells; a 336-cell matrix of A 1-48 x N 64-4096 x
  fp32 / bf16 x per-sample / shared conditioning x inference / training has ours fastest everywhere, smallest margin 1.09x).
  Tests `tests/gpu/sm90/test_h100_local_dit_mma_gpu.py`, `test_h100_local_dit_pack_gpu.py`; page `docs/gpus/h100/local_dit/local_dit.md`.
  `viz.measure_bars` draws a `PyTorch eager` column in the PyTorch colour.
### Added

- H100 (sm_90a) smooth lDDT: the B200 CUDA-core kernels (`kernels/smooth_lddt/cuda/sm100/smooth_lddt.cu`) now build for sm_90a and
  `smooth_lddt.refusal` serves compute capability (9, 0) as well as (10, 0). Forward + backward at A = 48: 0.37-0.98 ms
  (N = 4096-8192), 37-180x the fastest PyTorch form (`torch.compile`d dense). Page:
  [gpus/h100/smooth_lddt](gpus/h100/smooth_lddt/smooth_lddt.md).

### Changed

- H100 TriMul, single direction: D512 incoming training runs the CUDA path (it was on the Triton fallback: L768 24.0 -> 18.1 ms), the D512
  front reads a separately normalised operand and issues wgmma from shared memory (3.57 -> 3.18 ms; module L768 17.8 -> 17.1 ms), D256
  inference takes the tmn K1 / K3 tiles at every length but 384 (0.7-2.5 %), and the residual Triton pieces of the wide paths (weight
  pack, LN-affine fold, input LN) use the CUDA aux kernels in eager launches. Against PyTorch, cuEquivariance and DTv1 ours is the fastest in
  all 240 measured cells (D64-512, L128-1024, single direction and bidirectional, training and inference; smallest 1.10x) and against the
  Anthropic payload in all 48 cells it covers (thinnest 1.006x). Record:
  [records/trimul_h100_comparison_20261015.md](records/trimul_h100_comparison_20261015.md); page:
  [gpus/h100/trimul](gpus/h100/trimul/trimul.md).
- H100 AttentionPairBias training: 1.7-2.4x the fastest other implementation (L128-768, was 0.5x at L384 / L768). The natural-unit
  attention kernels divided by a float constant per score (attn_fwd / dq / dkv 59 / 62 / 84 -> 15 / 15 / 14 us at L768); the fp32
  pair-weight "shadow" gradient is now opt-in (`settings.apb_h100_exact_dwb`, default off: the fused gradient has the bf16
  PyTorch module's error, like every other gradient of the block); block-partial reduction 12 -> 3 us; fp32 master cast folded into
  `prep`, gate into the attention epilogue. The default build no longer needs `CUTLASS_PATH`. Page:
  [gpus/h100/attention_pair_bias](gpus/h100/attention_pair_bias/attention_pair_bias.md).
### Added

- H100 token DiT and bias-only DiT at every head layout (16 x 48, 24 x 32, 12 x 64 at d 768; 16 x 64 at d 1024), bf16 and fp32,
  inference and training: `attn_tf32.cu` / `attn_fwd.cu` / `attn_dq.cu` / `attn_dkv.cu` / `attn_dqb.cu` built per layout, `pvx32` templated
  on the head width, pack / row kernels at d 768 or 1024; `tests/gpu/sm90/test_h100_token_dit_layouts_gpu.py`.
- H100 bias-only token DiT: `pvx` / `pvx32` PV + gate cores, `dpb2` bias gradient, fp32 (TF32) inference and training served on
  sm90 ([page](gpus/h100/bias_only_dit/bias_only_dit.md)); `tests/gpu/sm90/test_h100_bias_only_dit_gpu.py`.

### Changed

- H100 fp32 token DiT inference: expand + SwiGLU as one TF32 GEMM at M >= 768 rows; `resgate_adaln_rows` can round its output to TF32
  (`tf32_round`); accuracy unchanged (2.92e-4), -16 us a block at A = 5, L768. Hopper fp32 inference no longer raises at head layouts
  other than 16 x 48 (it is served natively).
### Changed

- H100 bias-only DiT: the fused hoisted pair bias (`pair_bias_all`: one LayerNorm + one GEMM for every block, forward and
  backward) is served on sm_90 as well as sm_100; training step 1.02-1.04x (bf16) / 1.02-1.04x (fp32) at L384-768, A48.
- H100 token DiT: pair-bias cache under static inputs, one TF32 pass for the fp32 pair bias, bf16 dbias through a deterministic
  chunk-buffer `attn_dqb` (training L768 A48 8.47 -> 7.67 ms), no weight-gradient split on sm90.

### Removed

- The dense atom DiT (attention over all N atoms with the shared pair bias; the AF3 atom transformer is the block-local
  [local DiT](gpus/b200/local_dit/local_dit.md), which this block was never): `integrations/atom_dit.py` and its dispatch in
  `modules/dit`, the `sm100_atom` kernels only it used (`attn_fwd`, `attn_dkv`, `attn_dq`, `attn_dbias`, `attn_dd`, `pair_bias`), the
  `dit_atom` benchmark target and its A6000 tables, `tests/integrations/test_b200_atom_dit_gpu.py` and the B200 page
  `docs/gpus/b200/atom_dit/`. `modules.dit.DiTBlock` at atom widths runs the PyTorch / Triton path as before. Earlier entries below
  that name `integrations/atom_dit` describe the removed module.
- B200 fp32 SWA atom DiT backward (`kernels/swa_dit/cuda/sm100/tf32_bwd.py` and its `*_bwd_tf32.cu`, `attn_dq_tf32.cu`,
  `attn_dkv_tf32.cu`, `bwd_prep_tf32.cu`, `mod_bwd_tf32.cu`): identical training steps gave different gradients (a 2-row x 32-column
  dP fragment of `qkvg_bwd_tf32`, ~2e-2 on dq / dWqkv), and it faulted on poisoned memory. fp32 training takes the Triton fp32 path
  until a new backward lands; inference keeps the TF32 forward.
- B200 fp32 bias-only DiT `dpb_tf32.cu`: identical reruns gave different dbias at L 640 / 768. Replaced by `dpb32_sm100.cu` (below).

### Fixed

- B200 SWA atom DiT: the forward chain (modulation -> qkvg -> attention -> ffn, PDL) hung in CUDA-graph replays with one conditioning
  row per sample (A = 1, B = N) and ragged `seqused`. `ffn_fwd2_a1` (the A = 1 - 3 build of the last kernel) is built with `-DPDL_EARLY=0`
  (no early `griddepcontrol.launch_dependents`), and `qkvg_fwd2_a1` the same way (faster); the other builds are unchanged. A1 inference after the change: 0.031-0.043 ms for
  1024-4096 atoms (harness). Test: `tests/gpu/sm100/test_b200_swa_dit_graph_replay_gpu.py`; page:
  `docs/gpus/b200/swa_atom_dit/swa_atom_dit.md` ("Fixed (2026-10-11)").
### Changed

- H100 kernel and integration switches moved from `MINIWORLD_*` environment variables to typed fields of
  `miniworld_engine.settings.Settings`, read through `settings.current()` and set with `settings.configure(...)` (or a
  `Settings(...)` instance); the environment variables are no longer read, and a set old variable is reported through
  `_REMOVED_ENVIRONMENT` like the other migrated ones. Eleven new fields: `apb_h100_train` (`MINIWORLD_APB_H100_TRAIN`),
  `augattn_tf32_sm90` (`..._AUGATTN_TF32_SM90`), `bias_only_dit_h100` (`..._BIAS_ONLY_DIT_H100`), `local_dit_h100_train`
  (`..._LOCAL_DIT_H100_TRAIN`), `swa_dit_h100_native` (`..._SWA_H100_CUDA_ONLY`), `token_dit_gemm_swiglu_fp32`
  (`..._TOKEN_DIT_GEMM_SWIGLU_FP32`), `transition_d64_repl` (`..._TRANSITION_D64_REPL`), `transition_dw_repl`
  (`..._TRANSITION_DW_REPL`), `trimul_h100_b200style` (`..._TRIMUL_H100_B200STYLE`), `trimul_h100_native_buckets`
  (`..._TRIMUL_H100_NATIVE_BUCKETS`) and `trimul_h100_preserve_widths` (`..._TRIMUL_H100_PRESERVE_D128` / `_D256` / `_WIDE`).
  The remaining experimental H100 switches (the OPM, PWA, APB, token DiT, Transition, TriAttn and TriMul candidates) were baked in:
  the default behaviour is hard-wired, the kernels stay compiled and loadable, and the switch variables were deleted. The JIT cache
  root is read through `settings.engine_jit_root()` (field `engine_jit_root`) everywhere, including the H100 OPM, PWA, TriAttn,
  TriMul and conditioned-transition sources, which read `MINIWORLD_ENGINE_JIT_ROOT` directly before and so ignored it. The
  environment variables that stay are the build / driver / config-dir ones (`MINIWORLD_CONFIG_DIR`, `MINIWORLD_BUILD_SCOPE`,
  `MINIWORLD_BUILD_MODE`, `MINIWORLD_SMEM_LOG`, `MINIWORLD_DRIVER_*`, `MINIWORLD_SHAPE_MODE`, `MINIWORLD_ANTHROPIC_ROOT`,
  `MINIWORLD_MATHDX_HOME`). No change at the default settings.

- H100 OuterProductMean training: the persistent TMA prologue now serves every training cell at S 1024 / 2048 / 4096 (it was only
  (1024, 384) and (1024, 768)); the non-TMA prologue was 5-6x slower than the legacy Triton one. Training is up to 19 % faster (L128 S1024) and 6-10 % at S >= 2048; the two
  cells that already used it are unchanged. It now runs at 80-100 % of the H100 speed-of-light floor. The mask count comes from the bit mask with one new kernel (`opm_count_tiled`,
  exact) instead of the four-launch fp32 cuBLAS product (the cuBLAS count is gone; the switch that restored it was removed).
  Training and inference
  forwards are not bitwise equal at the cells whose inference path is the legacy Triton prologue. Page `docs/gpus/h100/opm/opm.md`,
  record `docs/records/h100_opm_sol_20261007.md`.

- H100 MSAPairWeightedAveraging: `ln_pair` has no beta any more (it cannot change the softmax over keys; its gradient is 0), so the
  H100 training step no longer recomputes that gradient in fp32 and is 4.5-5.6x faster (L768 S1024: 17.95 -> 3.31 ms). A checkpoint's
  `ln_pair.bias` is ignored when loading. dv runs on cuBLAS, and from N = 512 the contraction does too, followed by a new gate / out
  kernel (`pwa_gateout.cu`): inference L768 -18 %, training L768 -7 %. The forward output at N >= 512 is no longer bit-identical to the
  fused kernel's (o is rounded to bf16 before the gate); every gradient is. Removed the beta-shadow environment variables and
  `MINIWORLD_PWA_H100_CLUSTER`; the `pwa_train` and `pwa_infer` settings fields remain, and the former `MINIWORLD_PWA_H100_CUDA_AUX`
  switch is gone (the CUDA aux path is the only one). Record
  `docs/records/h100_pwa_20261008.md`.

- H100 MSAPairWeightedAveraging and bias-only DiT training keep the pair-bias logits in fp32 until the softmax (they were rounded to
  bf16 first). PWA: `pair3.cu` rounds the logits only under `-DMW_PWA_ROUND_LOGITS=1` (opt-in A/B), training and inference share the
  fp32 logits, and every fused gradient now equals the engine path (engine-vs-fused 0.0050 -> 0.0035). Bias-only DiT: `pair_bias_k`
  writes an fp32 logits buffer that `softmax_t` reads (fp32 logits are the only path; the former bf16 switch was removed); the
  `ln_pair.weight` gradient error against fp64 drops 0.0107 -> 0.0089 (L384) and 0.0106 -> 0.0093 (L768). sm_100a is unchanged.

### Added

- H100 (sm_90a) TriangleAttention is hand-written CUDA around the attention core for every registered width
  (`integrations/triattn_h100_sm90.py`, `kernels/triangle_attention/cuda/h100_sm90.py`, `h100_sm90_generic.py`, `h100_sources/`):
  d_pair 128 / 4 heads x 32 on fused front, tail and backward kernels (wgmma / TMA), d_pair 64-512 with heads of 16 or 32 channels
  (4 x 16, 2 / 4 / 8 / 12 / 16 x 32) on the width-generic path (LayerNorm, gate, residual, LayerNorm-backward kernels and the core's
  gate epilogue between cuBLAS projections). Inference and training, both nodes, any B, L a multiple of 16 (zero-padded to 128 with the
  pad keys masked), the module's broadcast dropout, bf16 or fp32 parameters. `module._h100_cuda = False` keeps the previous path;
  `bench.py +tri_h100_cuda=false` selects it. The kernel launches are `opaque` custom ops (the module stays one
  compiled graph; the benchmark harness' CUDA-graph capture of the compiled module failed on the traced weight packs before). Page: `docs/gpus/h100/triattn/triattn.md`.
- H100 TriangleAttention (continued, 2026-10-13): the generic path's inference residual is one cublasLt GEMM with D != C (torch.addmm copied the
  whole pair tensor into its output: 6 % of the call at d_pair 256), the GEMM algorithm is chosen by timing the heuristic's candidates
  once per shape, and a one-kernel front (LayerNorm + q | k | v | g + head-major scaled / masked bias, `front_fused3_sm90.cu`, needs
  `CUTLASS_PATH`) serves d_pair 64 at every L and d_pair 128 (8 x 16) / 256 / 384 / 512 at L128: 5-23 % faster inference.
  The ending node no longer transposes its pair tensor in and out (two copies, 29-39 % of the call at d_pair 128): the fused front, tail, gate
  backward, head backward and weight-gradient kernels read and write it through 4-D TMA tensor maps (c, j, i, b), the width-generic row
  kernels through a row index map; ending-node inference at d_pair 128 L768: 1722 -> 1345 us (starting node 1336 us).
  The fused d_pair 128 training step lost ten device copies and six cast kernels per call to the weight pack and the parameter-gradient
  slices: one `tri_pack` launch packs the six parameters (fp32 master or bf16), `tri_wgrad` writes the eight gradients into their own
  buffers, and the split-K gradient reduce is 8 x 32 threads instead of one -- L128 training step 193 -> 165 us device time (-15 %),
  L256 -6 %.
  Core experiments and what bounds the attention core: `docs/records/triattn_h100_core_experiments_20261013.md`.
- H100 TriangleAttention (2026-10-14 / 15): the weight packs replayed inside every captured CUDA graph are one launch each
  (`pack_generic_k` for the width-generic path, `tri_pack_k` for the fused inference path; twelve torch cast / pad / cat kernels before):
  (128, 256) inference replay 74 -> 60 us, (128, 128) 36 -> 29 us.  The re-measurement (all widths, L128-768, inference and training,
  ending node, key mask, B = 2, and against Anthropic's native cell) is in `docs/gpus/h100/triattn/triattn.md` and
  `docs/records/triattn_h100_remeasure_20261013_15.md`; backward / front experiments that did not pay (rounds 5-8) in
  `docs/records/triattn_h100_core_experiments_20261013.md`.
- H100 (sm_90a) OuterProductMean and MSAPairWeightedAveraging beyond MiniWorld's width, inference and training, bf16, CUDA + cuBLAS.
  OPM: d_msa 128 with d_pair 256 / 384 (Protenix-v2, ESMFold2, OpenDDE) and the ESMFold2 normalisation order
  (`normalize_before_proj=False`), any length / MSA depth (padded with masked rows); `csrc/opm_epilogue.cu` is built per
  (`OPM_CM`, `OPM_CZ`), with a new persistent TMA prologue `opm_prologue_tma_wide` (d_msa 64 / 128) that also replaces the Triton
  prologue at the former "CUDA + Triton" MiniWorld inference cells (1.6-5.5 % faster). PWA: `integrations/pwa_wide.py`
  (`csrc/pair_wide.cu`, `csrc/pwa_wide.cu`; d_msa 64 / 128, d_pair 128 / 256 / 384, 8 heads x 8 / 16, any depth, N <= 1024), bitwise
  deterministic, with faster `dm_bwd` and `pair_bwd`. `bench.py` gains `+msa_d_msa` / `+msa_d_hidden`. Tests:
  `tests/gpu/sm90/test_opm_wide_sm90a_gpu.py`, `test_pwa_wide_sm90a_gpu.py`; pages `docs/gpus/h100/opm/opm.md`, `pwa/pwa.md`; record
  `docs/records/h100_opm_pwa_wide_20261013.md`.
- B200 documentation: the CUDA-graph measurement tables of TriMul, TriAttn, Transition, OPM, PWA, AttentionPairBias, token DiT, the
  SWA atom DiT and the cross-attention rows of the local DiT page were re-measured on 2026-10-11 with fp32 master parameters (the new
  harness default; one run per cell, 131 + 82 harness runs on 6 cards). Training rows did not move; the first take of the inference
  rows was 10-50 % slower than with bf16 parameters because every call cast every fp32 weight, which `kernels._capture.param_forms` now
  caches (APB, TriMul, Transition, SWA entries); the tables are the second take. The no-CUDA-graph training tables, the bias-only DiT
  page and the kernel-level tables are unchanged.
- Inference with fp32 master parameters on B200 caches the bf16 forms of the weights (`kernels._capture.param_forms`, keyed on
  pointer / version / dtype, scoped to the CUDA-graph capture like the weight packs): APB at L128 0.035 -> 0.025 ms, SWA A1 at 1024
  atoms 0.045 -> 0.031 ms, TriMul / Transition 4-15 %. Test: `tests/gpu/sm100/test_b200_fp32_master_inference_cache_gpu.py`.
- Bias-only DiT bf16 attention core (`pv_gate_inf.cu`): the three-kernel inference step's core loads P before its PDL wait
  (`-DP_EARLY`, `PvGateCore(p_early=True)` from the runner only; v, g and the output stay after the wait). Safe because P is the
  runner's hoisted / per-call softmax, written before the step, and the step's first front is now launched without the
  programmatic attribute whenever the core takes P early (`FrontBF16(..., serial=True)`, `_LaunchPDL(serial=)`), so the whole PDL
  chain starts after P's writer is complete -- also on the hoisted-bias path that makes P right before the step. Never in training.
  `MINIWORLD_BIAS_ONLY_DIT_INF3_P_EARLY=0` keeps the plain PDL build. Step, A = 5, us (in-process A/B, on / off): L384 45.3 / 45.6,
  L512 55.5 / 55.9, L640 73.2 / 73.9, L768 77.0 / 77.3.
- Bias-only DiT bf16 core at training shapes: 256 keys per v tile at L768 for 16 x 48 with S >= 16 (was 192). A = 48, us (in-process
  A/B): forward pv 65.21 -> 58.28, standalone dV 55.15 -> 49.94. Other layouts / lengths unchanged.
  Tests: `tests/gpu/sm100/test_b200_bias_only_dit_gpu.py` (`test_pv_core_*`); page: `docs/gpus/b200/bias_only_dit/bias_only_dit.md`
  (I1, with the TRACE-build per-role timeline of the core, `-DTRACE` now %globaltimer into `g_trace`, and the schedule tried and
  not kept: P interleaved into sample 0, one-lane MMA issue, batched slot releases -- slower everywhere, e.g. step L768 76.8 -> 84.2).
- Bias-only DiT hoisted pair bias: `miniworld_engine.modules.bias_only_dit.pair_bias_all(blocks, pair)` makes every block's pair bias
  from ONE LayerNorm without affine and ONE GEMM against the gamma-folded `to_bias` weights (same parameters, the same result to
  rounding), and `BiasOnlyDiTBlock.forward(..., bias=b)` takes a block's bias precomputed ([B, H, L, L]; `pair` may then be None). On
  B200 (bf16; fp32 on TF32) `integrations/bias_only_dit_hoist.py`: LN0 rows + cuBLAS GEMMs, the blocks' bias gradients written into
  one [nb H, L^2] buffer, the LN0 backward in `bias_only_dit_hoist_rows.cu`; the fused training block on a bias skips `pair_bias` /
  `pair_bias_bwd` (`finalize` alone), inference on a bias makes P per call. `MINIWORLD_BIAS_ONLY_DIT_HOIST=0`: the PyTorch fold.
  The d LN0 backward rows also write the dW' GEMM's operand as a two-term bf16 split of LN0 (exact to ~2^-17), so ln_pair / to_bias
  gradients carry no LN0 rounding. 24 blocks of 16 x 48, A = 48, bf16 training step (CUDA graph, median of 9): L384 29.130 -> 28.051
  ms (-3.7 %), L768 57.596 -> 53.779 ms (-6.6 %); peak memory 13.90 -> 14.03 / 27.95 -> 28.52 GiB.
  Tests: `tests/gpu/sm100/test_b200_bias_only_dit_hoist_gpu.py`; page: `docs/gpus/b200/bias_only_dit/bias_only_dit.md` (T7);
  scripts: `bench_scripts/bo_hoist_bench.py`, `bench_scripts/bo_hoist_diag.py`.
- B200 bf16 bias-only DiT training backward fused from 23 launches per block to 11 (12 at L768), now the default
  (`MINIWORLD_BIAS_ONLY_DIT_BWD_FUSED=0` keeps the per-step launches; a failed build warns once and does the same):
  `bo_bwd_tail.cu` (res_c_bwd, the dh GEMM + SwiGLU backward, the dxt GEMM, res_adaln_b_bwd, the d(og) GEMM + gate backward as items
  over pairs of 128-row tiles on CTA pairs, GEMM items M 256 with `tcgen05.mma.cta_group::2`, three launches chained by programmatic
  dependent launch; `_BWD_TAIL=off` keeps the seven launches, `_BWD_TAIL_NST` = 3 / 4 / 5 operand stages, default 4),
  `bo_wgrad.cu` (the six weight gradients and the cond-LN unfold in one launch, one output tile per CTA over all rows), `bo_pvdpb.cu`
  (pv dV and the bias gradient in one launch where the bias gradient runs on single CTAs; `_BWD_PVDPB=0`; `_BWD_PVPDL=0` launches pv dV
  without PDL), and `pair_bias_bwd_fin` in `bias_only_dit_train_rows.cu` (pair_bias_bwd + finalize behind a grid barrier;
  `_BWD_PBFIN=0`). In-process graph node A/B, A = 48, 16 x 48, previous -> fused, us at L384 / L768: tail 270.3 -> 268.8 / 533.4 ->
  526.9, weight gradients 260.4 -> 236.4 / 483.8 -> 456.5, pv dV + dbias 37.6 -> 35.9 (L384), pair_bias_bwd + finalize +0.3 / +0.5
  (on: one launch fewer at the same speed). `bo_bwd_mid.cu` (the dxa / dcg GEMMs, adaln_a_bwd, the dchat GEMM + cond_bwd in three
  launches: 9 per block) is behind `_BWD_MID=1`, off: 141.5 -> 148.9 / 278.5 -> 300.7 us (TMA-intake bound GEMM items, an exposed
  cond-LN epilogue). The weight pack (`pack16`) gained scaled transposes and 48 segments for the kernels' K-major weights. Tests:
  `tests/gpu/sm100/test_b200_bias_only_dit_train_gpu.py` (`test_fused_backward_*`; the steady-memory test now warms up two steps);
  page: `docs/gpus/b200/bias_only_dit/bias_only_dit.md` (Training, T6); traces: `bench_scripts/bo_bwd_trace.py`.
- B200 bf16 bias-only DiT training step, three changes (now the defaults): the weight pack in one launch (`pack16`, bit-identical
  to the torch casts / cats it replaces; a captured training step repacks every replay: 49.7 / 57.2 -> 9.6 / 11.2 us at L384 / L768,
  `MINIWORLD_BIAS_ONLY_DIT_TRAIN_PACK1=0` keeps the torch pack); `dxt` / `dxa` written by cuBLAS straight into their d-shift columns of
  `dG` so the LayerNorm backward rows no longer copy them (the two rows 108.7 / 208.0 -> 101.8 / 193.6 us, `_BWD_DG=0`); the bias
  gradient on CTA pairs `dpbx2_sm100.cu` (`tcgen05.mma.cta_group::2`, 256 x 256 items, a third less TMA intake) at L768 (61.2 ->
  47.4 us, `_DPB_PAIR=0`; a failed build falls back to `dpb_sm100`). In-process per-node A/B, A = 48, 16 x 48; whole step -29.8 /
  -37.3 us. The fp32 step takes the TF32 build of `dpbx2_sm100.cu` (`-DTF32`) by the same rule (L768 148.2 -> 104.0 us;
  `MINIWORLD_BIAS_ONLY_DIT_TF32_DPB_PAIR=0` keeps `dpb32_sm100`). Fused GEMM epilogues
  for the backward (dh + SwiGLU backward, d(og) + gate backward) and the forward expand were measured and not kept (L2-bound, no
  gain; page: "What was tried and not kept"). Tests: `tests/gpu/sm100/test_b200_bias_only_dit_train_gpu.py`,
  `test_b200_bias_only_dit_tf32_gpu.py` (`test_dpb_pair_tf32_*`); page: `docs/gpus/b200/bias_only_dit/bias_only_dit.md` (Training, T3).
- B200 bf16 bias-only DiT inference as three kernels per block, now the default bf16 inference step: `bo_front_bf16` (LN + AdaLN +
  the v|g GEMM), `pv_gate_inf -DPDL_INF`, `bo_tail_bf16` (out GEMM, residual + gate, LN + AdaLN, a|b GEMM, SwiGLU, squeeze GEMM,
  residual + gate) or, where it saves a round, the pair tail `bo_tail2_bf16` (`tcgen05.mma.cta_group::2`, two row tiles x four column
  groups per cluster of 8), chained by programmatic dependent launch, behind bf16 conditioning tables hoisted once per conditioning
  tensor. bf16 operands (kind::f16, 64-column k-blocks), fp32 accumulators, LayerNorm and residual; the block input bf16 at block 0
  and fp32 after, the last tail writing bf16 (no cast kernel). `MINIWORLD_BIAS_ONLY_DIT_INF3_BF16=0` keeps the 12-launch bf16 step
  (also the fallback when the kernels fail to build); `_INF3_CL` / `_INF3_TAIL_CL` (8 / 6 / 4 = the pair tail) force a cluster size.
  One block, A = 5, 16 x 48 (whole-step graph replay): 47.4 / 59.6 / 69.3 / 86.2 / 97.7 / 104.2 -> 41.1 / 45.1 / 47.0 / 57.5 / 75.4 /
  78.7 us at L128-768; vs PyTorch compiled (bench.py, bf16-mixed, graphs) 1.66 / 1.92 / 2.24 / 2.39 / 2.36 / 2.65x. Tests: `tests/gpu/sm100/test_b200_bias_only_dit_gpu.py`
  (`test_inf3_bf16_*`, `test_front_bf16_*`, `test_tail_bf16_*`, `test_tail2_bf16_*`); page:
  `docs/gpus/b200/bias_only_dit/bias_only_dit.md` (I0).
- B200 fp32 bias-only DiT inference, pair tail: `bo_tail2_tf32` runs the tail as a cluster of 8 = two row tiles x four column
  groups, each column group's two CTAs one `tcgen05.mma.cta_group::2` pair (M = 256, each CTA loads half of each weight tile), so
  A = 5 L640 / L768 finish in one round of clusters instead of two. Selected only where it saves a round;
  `MINIWORLD_BIAS_ONLY_DIT_INF3_2CTA=0` keeps the CL 8 / CL 6 tail. Whole step at L640 / L768: 139.3 / 145.5 -> 122.9 / 127.4 us.
- B200 fp32 bias-only DiT inference as three kernels per block, now the default fp32 inference step: `bo_front_tf32` (LN + AdaLN +
  the v|g GEMM), `pv_gate_tf32 -DPDL_INF`, `bo_tail_tf32` (out GEMM, residual + gate, LN + AdaLN, a|b GEMM, SwiGLU, squeeze GEMM,
  residual + gate), chained by programmatic dependent launch, behind conditioning tables hoisted once per conditioning tensor.
  Front and tail run as CTA clusters (8 / 6 / 4 CTAs per 128-row tile) that all-reduce the LayerNorm statistics through DSMEM and
  exchange their GEMM operands through L2 scratch. `MINIWORLD_BIAS_ONLY_DIT_INF3=0` keeps the 12-launch cuBLAS + rows step (also the
  fallback when the kernels fail to build); `_INF3_CL` / `_INF3_TAIL_CL` force a cluster size. One block, A = 5, 16 x 48 (whole-step graph
  replay): 69.8 / 92.0 / 118.9 / 139.4 / 176.5 / 198.7 -> 62.0 / 69.2 / 75.9 / 100.7 / 138.7 / 144.8 us at L128-768
  (0.64-0.79x at L256-768). Tests: `tests/gpu/sm100/test_b200_bias_only_dit_tf32_gpu.py` (`test_inf3_*`,
  `test_front_kernel_*`, `test_tail_kernel_*`); page: `docs/gpus/b200/bias_only_dit/bias_only_dit.md` (F5).
- B200 fp32 bias-only DiT training, faster: `dpb32_sm100.cu` (the fp32 port of the bf16 `dpb_sm100.cu`) for the bias gradient; the
  pair bias and its backward on TF32 `mma.sync`; the dh GEMM with the SwiGLU backward in its epilogue (`gemm_glu_tf32.cu`); the six
  weight gradients as batched row chunks; one-launch weight pack; `res_adaln_b` at three blocks per SM; `softmax_t` 16 rows per block;
  dxt / dxa written into dG. Switches: `MINIWORLD_BIAS_ONLY_DIT_TF32_GLU(_BWD)`, `_PV_BATCH`, `_PAIR_EXACT`, `_WGRAD_SPLIT`,
  `_F32_RESB_MINB`. One block, A = 48, 16 x 48 heads, whole step (CUDA graph): 2161 -> 2023 us at L384, 4403 -> 4224 at L768,
  80% of the speed-of-light floor (every graph node: max(FLOP / 720 TF/s, bytes / 7 TB/s)); 1.23-1.25x PyTorch compiled fp32.

- B200 fp32 token DiT training, faster: the pair-bias projection and its gradients on TF32 tensor cores, narrow weight
  gradients split over rows. One block, A = 48, whole fused step: 1161 -> 1008 us at L128, 2898 -> 2675 at L384, 6599 -> 6369 at L768.
- B200 fp32 AF3 local atom DiT inference, faster: three fused kernels per hoisted inference call (`atom_pre_tf32`,
  attention, `atom_post_tf32`), the weight pack in 3 launches, the conditioning tables in 2, PDL. 3 blocks, 5 samples: 1.59-1.98x
  Anthropic's FastAtomStack fp32 per call, 1.14-1.39x with hoisted tables, at 1024-6144 atoms. Page: `docs/gpus/b200/local_dit/local_dit.md`.
- B200 AF3 local atom DiT training, faster (bf16 and fp32): all dense weight gradients of a block in one grouped split-K launch
  (`sm100_atom.atom_dw`, `dw_atom.cu`: tcgen05 with bf16 or TF32 operands, fp32 partials, a deterministic in-order reduce into each weight's own
  tensor; it replaces one cuBLAS GEMM per weight); the gated-residual GEMM with TMA-loaded residual and gate (`gemm_rg_tf32.cu`); the
  bias-gradient column sums in one pass; in cross-attention mode the K / V AdaLN and its backward inside `pre_fwd` / `pre_bwd` (`-DCROSS=1`
  builds of the same sources: they replace the separate K / V row kernels and the two K / V cuBLAS GEMMs, and add the K / V branch's dx1 before
  AdaLN 1's backward), half-warp 16-byte row kernels for the conditioning LayerNorm (`lcross.cu`), and the key mask passed as a uint8 view
  instead of a converted copy. One block, A = 48, 4096 atoms (CUDA graph): cross-attention bf16 2.52 -> 1.58 ms, fp32 4.33 -> 3.84 ms;
  shared-AdaLN bf16 1.38 ms (2.59 ms at 8192 atoms). Tests: `tests/gpu/sm100/test_b200_atom_dw_gpu.py`, `test_b200_local_dit_gpu.py`,
  `test_b200_local_dit_tf32_gpu.py`; page: `docs/gpus/b200/local_dit/local_dit.md`.
- B200 atom-block tile kernels (`sm100_atom`, the local DiT's row kernels): the TMA stores of `post_fwd`, `tr_bwd_gate`,
  `post_bwd`, `pre_bwd` and `cond_bwd` are issued and waited for by a lane of the producer warp instead of the compute leader (`post_fwd`
  162 -> 136 us at 196 608 rows), the two warpgroups of `cond_fwd` are staggered by half an iteration (125 -> 114 us) and the K / V AdaLN math
  uses `rsqrtf` / `__fdividef` (the IEEE division and square root were 2860 of 5200 clk per tile of the cross-attention `pre_fwd`: 130 -> 94 us).
  One block, A = 5, 4096 atoms, CUDA graph under `static_weights()` + `static_inputs()`: 54 us shared-AdaLN, 56 us cross-attention.
- B200 documentation: the B200 pages were re-verified through `benchmarks/runners/bench.py` on 2026-10-10 (TriMul, TriAttn, Transition, OPM, PWA,
  AttentionPairBias CUDA-graph tables; the token DiT layout tables), the fp32 token DiT head layouts were run and measured,
  and the SWA atom DiT page carries an engine harness cross-check. `AttentionPairBias`' PyTorch compiled / cuEquivariance rows are 20-25 % faster than
  on 2026-10-01 since LayerNorm / RMSNorm on B200 run as fused PyTorch ops.
- B200 fp32 SWA atom DiT forward, faster: the rounded weight forms in one kernel (`wprep_tf32.cu`), `mod_fwd_tf32` as a persistent
  store stream, and the per-row modulation / RoPE values of `qkvg_fwd_tf32` / `ffn_fwd_tf32` through TMA boxes. Inference, one block,
  5 samples: 1.37-1.86x Anthropic's fused block with TF32 GEMMs (modulation in the call; 1.43-2.03x precomputed) at 1024-6144 atoms,
  4x its accuracy. Page: `docs/gpus/b200/swa_atom_dit/swa_atom_dit.md`.
- B200 token DiT: fp32 (TF32) at every head layout (16 x 48, 24 x 32, 12 x 64, 16 x 64); `DiTBlock.forward` serves [A, B, L]
  masks with shared rows on the fused path (they used to take the per-op path in bf16 and fp32); the fused training op pads odd A
  and L % 128 != 0 instead of refusing. Tests: `tests/gpu/sm100/test_b200_token_dit_tf32_layouts_gpu.py`; page:
  `docs/gpus/b200/token_dit/token_dit.md`.
- B200 fp32 path of the bias-only token DiT (`kernels/bias_only_dit/cuda/tf32.py`, `pv_gate_tf32.cu`, `dpb_tf32.cu`,
  `bias_only_dit_f32_rows.cu`): inference and training, every head layout, TF32 MMA. Tests:
  `tests/gpu/sm100/test_b200_bias_only_dit_tf32_gpu.py`; page: `docs/gpus/b200/bias_only_dit/bias_only_dit.md`.
- B200 fp32 path of the AF3 local atom DiT (`integrations/local_dit.py` ops `local_dit_block_{fwd,bwd}_tf32`; kernels
  `sm100_atom_local/*_tf32.cu`, `sm100_atom/gemm_tf32.cu`, `rows_tf32.cu`): inference and training with fp32 activations and
  parameters on TF32 MMA (operands rounded to nearest); fp32 calls no longer fall back to the module path. Tests:
  `tests/gpu/sm100/test_b200_local_dit_tf32_gpu.py`; page: `docs/gpus/b200/local_dit/local_dit.md`.
- B200 fp32 SWA atom DiT on TF32 sm_100a kernels (`kernels/swa_dit/cuda/sm100/*_tf32.cu`, `tf32_fwd.py`, `tf32_bwd.py`): forward,
  backward and the hoisted modulation, fp32 activations with TF32 MMA on round-to-nearest operands (attention operands fp32, not
  bf16). Replaces the Triton fp32 path on B200 (`MINIWORLD_SWA_DIT_TF32=0` keeps it); compiles under `torch.compile(fullgraph=True)`.
  Tests: `tests/gpu/sm100/test_b200_swa_dit_tf32_gpu.py`, `test_b200_swa_dit_tf32_bwd_gpu.py`; page:
  `docs/gpus/b200/swa_atom_dit/swa_atom_dit.md`.
- `kernels._capture.static_weights()` / `MINIWORLD_STATIC_WEIGHTS=1`: declares that weights stay fixed between the replays of a CUDA graph
  (inference), so the weight-pack caches serve a capture like an eager call and a replay no longer re-runs the packing kernels. The
  capture scoping of the weight-pack caches (previous entry) repacks at every replay -- right when an optimizer step follows, but a token
  DiT inference block replayed in a graph went from 84 us to 200 us (58 kernels instead of 12), the bias-only DiT, AttentionPairBias and
  atom DiT inference likewise. Default unchanged (scoped); the benchmark harness sets it for `mode=inference`. Test:
  `tests/gpu/sm100/test_b200_pack_cache_capture_gpu.py` (`-k static`).
- `kernels._capture.static_inputs()` / `MINIWORLD_STATIC_CONDITIONING=1` and the AF3 (block-local) atom DiT hoist: in inference, the per-item
  conditioning tables (AdaLN modulation, cross-mode conditioning LayerNorm and K/V modulation GEMM) and the windowed pair bias are computed
  once per (tensor, in-place version, weights) and reused by later calls; under `static_inputs()` a CUDA-graph capture serves from the eager
  entries and a replay launches none of those kernels (inference call 11 -> 6 kernels). Entries hold a weak reference to the tensor, so a
  reused address never serves stale tables. Off by default and never applied when gradients are saved (training). Tests:
  `tests/integrations/test_b200_local_dit_gpu.py` (`-k hoist or static_inputs`).
- B200 (sm_100a) smooth lDDT of the diffusion loss, `kernels/smooth_lddt` (ops `smooth_lddt_prep` / `smooth_lddt_fwd` / `smooth_lddt_bwd`): AF3
  Algorithm 27 with no [N, N] float tensor per augment -- the ground-truth pair set (30 A nucleotide / 15 A radius) kept as bit masks built once
  for all augments, only the pairs inside the radius visited, distances from the coordinates, the four sigmoids sharing one exponential,
  an analytic deterministic backward (no atomics on floats) and CUDA-graph capture. A48 on real structures: 1.1 ms at N = 3000, 1.7 ms at
  N = 4500, 3.5 ms at N = 9000 forward + backward, against about 0.3 s per micro-step for the per-augment PyTorch loss. fp32 compute, fp32 / bf16 input. Page:
  `docs/gpus/b200/smooth_lddt/smooth_lddt.md`; tests: `tests/integrations/test_smooth_lddt_gpu.py`.
- fp32 master parameters (AMP `bf16-mixed`: fp32 parameters over bf16 activations) on every B200 training path, at the speed of bf16
  parameters: TriMul, Transition (D 64-512), AttentionPairBias, OuterProductMean, local and SWA atom DiT, token pair init (PWA,
  TriangleAttention and the token / bias-only DiT already were). The integrations cast the parameters for the kernels inside their
  autograd functions and return the kernels fp32 weight-gradient accumulators unrounded, in each parameters dtype; the SWA DiT and
  token pair init no longer refuse fp32 weights over bf16 activations (token pair init also takes bf16 `left` / `right`). Tests:
  `tests/gpu/sm100/test_b200_fp32_master_gpu.py`; page: `docs/gpus/b200/b200.md`.
- H100 (sm_90a) Transition D128 inference forward, hidden 2 D / 4 D: the LayerNorm is formed in the wgmma A-fragment layout and stays in registers as the
  first GEMM's operand (the hidden-chunk loop is unrolled: in the rolled loop ptxas reused the operand registers and the result was wrong by 5x the
  bf16 rounding error), the x tile's TMA is issued before the weights, and the two consumer warpgroups ping-pong (swiglu of one overlaps the GEMM of
  the other; the weight-slot refill no longer blocks either). fp64 error 1.66e-3 at every row count, bit-identical replays; 7 - 37 % ahead of the best
  anthropic row that loads (n = 2 3 072 rows 9.0 vs 9.7 us). `fused_generic_sm90a.prefers_generic` takes D128 inference rows <= 512 (n = 2) / 4 096 (n = 4).
  Test: `test_inference_matches_fp64_on_whole_persistent_tiles`; page `docs/gpus/h100/transition/transition.md`, record
  `docs/records/2026-10-15-h100-transition-vs-anthropic.md`.
- H100 (sm_90a) Transition beyond hidden 4 D / whole tiles / bf16: hidden 2 D at D64 / 128 / 256 through the tuned fused forward and backward
  kernels rebuilt for `HMULT=2` / `WIDE_HM=2` (`fused_sm90a`, `fused_wide_sm90a`; 1.35x - 2.2x the Triton residual path at L384 / L768, weight-role
  replicas measured: D128 16 x 4 slices, D64 28 x 2), and a generic hand-CUDA family `kernels/transition/cuda/fused_generic_sm90a.py`
  (`generic_sm90a/`: wgmma + TMA + mbarrier GEMM skeleton, forward and backward) for any D multiple of 64 up to 1024, any hidden width multiple of 128,
  any row count (ragged tail by TMA zero-fill / clipping, no PyTorch fallback) and fp32 activations on TF32 tensor cores (operands rounded to nearest
  TF32 so the error matches cuBLAS TF32). `Transition.forward` routes fp32 to it when TF32 is allowed. 1.1x - 6x `torch.compile` PyTorch on large inputs (ties it at 768 rows, D768);
  CUDA-graph timings vs PyTorch / Triton / Transformer Engine on `docs/gpus/h100/transition/transition.md`. `autotune.fused_config.transition_candidates`
  takes `slices` (default weight-role replicas now `~64 / slices`: unchanged 8 at n = 4, 16 at n = 2) and the native build driver tunes both hidden widths.
  Tests: `tests/numerics/test_transition_generic_sm90a_gpu.py`, `tests/autotune/test_v210_native.py`, `tests/compile/test_transition_fake_availability.py`.
- B200 TriMul with a hidden width twice the pair width, one direction, at D64 / D128 (`TriangleMultiplication(d_pair, d_hidden=2 *
  d_pair)`: AF3 / Protenix template blocks, pair 64, hidden 128): served by the D64 / D128 kernels unchanged -- per token it has the
  bidirectional block's shapes (4 D planes, LN_out over 2 D), only the cuBLAS contraction between the kernels pairs the planes in one
  direction over all 2 D channels. Inference and training, B <= 8 samples at D64, dropout, CUDA graphs (`b200_infer.hidden_ok`).
  Tests: `tests/gpu/sm100/test_trimul_b200_gpu.py` (`-k hidden`); page: `docs/gpus/b200/trimul/trimul.md`.
- B200 (sm_100a) block-local AF3 atom transformer block `modules/local_dit.LocalDiTBlock` (ops `local_dit_block_fwd` / `local_dit_block_bwd`):
  AF3 Alg. 23 with the 32 x 128 trunked attention (atom `i` sees the atoms `[32 (i // 32) - 48, 32 (i // 32) + 80)`, per-window pair bias
  from the trunked atom pair `[B, nwin, 32, 128, d_pair]`, `to_windows` makes it from a dense pair), hand-CUDA `mma.sync` attention and
  pair-bias kernels (`kernels/augmented_attention/cuda/sm100_atom_local`, forward and backward) on the atom DiT's row kernels. Inference and
  training, any N, optional key mask, eager / `torch.compile` / CUDA graph; A48 N4096 training 1.90 ms (7.4x the PyTorch path). Page:
  `docs/gpus/b200/local_dit/local_dit.md`; tests: `tests/gpu/sm100/test_b200_local_dit_gpu.py`, `tests/cpu/kernels/test_local_dit_reference.py`.
- B200 atom DiT block (`integrations/atom_dit`) is now two opaque ops under an autograd Function: it is served under `torch.compile` and
  in CUDA graphs (it used to refuse inside a compiled graph and take the module path); the weight pack is reused per parameter version and
  rebuilt inside a graph capture. Tests: `tests/integrations/test_b200_atom_dit_gpu.py`.
- B200 (sm_100a) token-pair initialisation `kernels/token_pair_init` (ops `token_pair_init_fwd` / `token_pair_init_bwd`): the input
  feature embedder's `left[i] + right[j] + Linear(relative-position one-hot) + Linear(bond one-hot)` as one kernel pair that never builds
  the 139-wide fp32 one-hot (84 MB at 384 tokens), in exact fp32, any B and ragged L, CUDA-graph capturable. 5.6-16.9x the dense
  reference forward and 4.0-6.2x forward + backward at L128-768. Page: `docs/gpus/b200/token_pair_init/token_pair_init.md`; tests:
  `tests/gpu/any/test_token_pair_init_gpu.py`.
- `modules.swa_atom_attention`: the FlashAttention-4 forward saves its output and lse and the backward reads them (`flash_window_fa4`,
  `settings.swa_flash_saves_lse`, default on) instead of re-running the flash forward and cleaning q / k / v / dq / dk / dv with about 11
  eager launches per block; same numbers (`tests/gpu/any/test_swa_fa4_attention_gpu.py`). The legacy op stays behind the setting.
- B200 SWA atom DiT block: a global-attention variant (`kernels/swa_dit.interface.is_global`, a window `< 0` or `>= 65536`) that runs
  FlashAttention-4 between the sm_100a stages (bf16, B200). Slower than the per-op path at A = 1, so callers opt in. Tests:
  `tests/gpu/sm100/test_b200_swa_dit_global_gpu.py`.
- B200 (sm_100a) AttentionPairBias (the Pairformer single track): `AttentionPairBias.forward` runs hand-written CUDA and
  cuBLAS only through `integrations/attention_pair_bias_b200.py` (`kernels/augmented_attention/cuda/apb/`: pair LayerNorm +
  projection and its backward on TMA rings, the sm_100a attention cores, CUDA row kernels) for implementation MINIWORLD, bf16,
  B = 1, no QK-norm, (heads, d_single) in 8 x 48 / 12 x 32 / 16 x 24 / 24 x 16 at 384 and 16 x 32 at 512, d_pair 128;
  inference at L % 16 == 0, training at L % 128 == 0. `bench.py` (+`apb_n_head`, `apb_anthropic_core`): inference 1.17-1.66x
  the fastest other row, training with CUDA graphs 1.38-1.90x. Page: `docs/gpus/b200/attention_pair_bias/attention_pair_bias.md`;
  tests: `tests/gpu/sm100/test_b200_apb_gpu.py`, `tests/gpu/sm100/test_apb_b200_gpu.py`.
- B200 token DiT head layouts: the sm_100a attention cores and the fused token DiT paths also serve 24 x 32 and 12 x 64 at
  d768 and 16 x 64 at d1024 in bf16 (16 x 48 stays the only fp32 layout); `bench.py +tdit_n_head`. Tests:
  `tests/gpu/sm100/test_b200_token_dit_layouts_gpu.py`, `tests/gpu/sm100/test_tdit_heads_sm100_gpu.py`.
- B200 (sm_100a) SWA atom DiT block: the fused block of `kernels/swa_dit` (what `SWADiTBlock` takes) runs every stage on
  hand-written kernels on B200 in bf16 (`kernels/swa_dit/cuda/sm100/`: qkvg forward / backward, window attention forward and
  dQ / dK dV, out-projection backward, out-projection + FFN forward, FFN backward; the weight gradients on cuBLAS, dWqkv | dWg
  as one GEMM), and the hoisted adaLN modulation on `mod_fwd` / `mod_bwd`. The atom count must be a multiple of 128 there
  (callers pad; `seqused` masks the padding) -- the block raises `ValueError` otherwise. `MINIWORLD_SWA_DIT_SM100=0` keeps the
  Triton path. Sources from the research capsule `experiments/swaatom_sm100`; tests:
  `tests/gpu/sm100/test_b200_swa_dit_gpu.py` (the Triton tests in `tests/gpu/any/test_swa_dit_fused_gpu.py` pin that path).
- B200 atom DiT (`integrations/atom_dit.py`): the sm_100a path now serves a [B, N] key mask and any atom count N, so MiniWorld's
  atom transformer (which always passes the structure's `atom_mask`) takes it. N is padded to a multiple of 128 inside the call
  (the padded keys masked, the pair tensor read in place); `pair_bias.cu` folds the mask and the padding into the pair bias
  (masked keys -1e4: zero softmax weight in any row with a valid key) and drops dbias on masked keys, so the attention kernels
  are unchanged. Tests: masked and padded training / inference / CUDA-graph cases in `tests/integrations/test_b200_atom_dit_gpu.py`.

- B200 (sm_100a) AF3-style atom DiT block: `DiTBlock` at atom widths (d_single = d_cond = 128, 4 heads x 32, d_pair 16,
  transition n = 2) runs the whole block, inference and training, on hand-written kernels (`kernels/augmented_attention/cuda/sm100_atom/`:
  conditioning projections, AdaLN + q / k / v / gate, the pair bias in both layouts, attention forward / dK dV / dQ / dbias, the
  post-attention + transition forward, and their backwards; the weight gradients on cuBLAS) through `integrations/atom_dit.py`.
  Gate: B200, bf16 inputs, B = 1, N a multiple of 128, no key mask, no QK-norm; anything else keeps the Triton path;
  `MINIWORLD_ATOM_DIT_SM100=0` turns it off. The sources are the research capsule's (`experiments/atomdit_sm100`), built on
  first use by the newest nvcc that knows sm_100a. Page: `docs/gpus/b200/atom_dit/atom_dit.md`; tests:
  `tests/integrations/test_b200_atom_dit_gpu.py`.
- B200 (sm_100a) TriangleAttention at d_pair 64-512 (heads of 16 or 32 channels) now serves training as well as inference
  (`b200_triattn.WideTrain`: the module's dropout, bf16 or fp32 parameters). The backward runs a gate backward and a
  projection-dgrad + LayerNorm-backward kernel on tcgen05; dq / dk / dv / dg / db land in one buffer, so the parameter
  gradients are one cuBLAS GEMM (K = L^2) and a finish kernel. The wide forward's projection kernel runs as 2-CTA clusters
  (`cta_group::2`, each SM streaming half of the weights) and the output kernel adds the residual through the MMA. Module
  step (CUDA graph) vs the fastest of PyTorch compiled / cuEquivariance / Anthropic: inference 1.23-1.86x, training
  1.84-3.05x; vs the repository's Triton path 1.30-2.04x / 1.44-2.23x. `viz.measure_bars` draws a "Triton path" column in
  the Triton colour. Page: `docs/gpus/b200/triattn/triattn.md` (measurement tables regrouped by head layout, with charts).
- `implementation="anthropic"` for TriangleMultiplication on B200 (sm_100). The release ships no sm_100 binary, so
  `miniworld-engine dev build-anthropic-sm100a <payload>/build` compiles its sm_80 member (unmodified sources) for sm_100a with the
  release's own builder and manifest; `integrations.anthropic_trimul` registers cc 10.0 as `sm_100a` in the release's loader and
  assembles the member as `sm80_ops.serve_sm80` does (K1 -> torch.bmm -> K3; the bidirectional module as one unit at twice the
  hidden width). Served: one direction D64-D384, bidirectional D64 / D128; everything else refuses with the reason. The harness's
  `anthropic` row runs it through `ANTHROPIC_TRIMUL_BUILD_DIR`; `tests/gpu/sm100/test_anthropic_trimul_b200_gpu.py` (23 tests)
  checks it against the fp32 reference.
- `miniworld_engine.viz.measure_bars`: a length sweep and a dimension sweep bar chart under every measurement table of a GPU page
  (the rule is in docs/gpus/README.md). Tables keyed `(Length, MSA depth)` get an MSA-depth sweep instead of the dimension one;
  `(Length, Dimension)` pages chart as before.
- docs/gpus/b200/opm/opm.md, docs/gpus/b200/pwa/pwa.md: kernel tables, flow figures and harness measurements of the B200
  OuterProductMean / MSAPairWeightedAveraging paths (inference L128-768 x S1024 / 2048 / 4096, training S1024), and their
  completion tables in b200.md.
- `modules/bias_only_dit`: `BiasOnlyDiTBlock` / `BiasOnlyAttention`, the token DiT block whose attention logits are the pair
  bias alone (no query, no key; the `bias_only_v` bench ablation made a module). On B200, bf16 inference with no autograd
  runs `integrations/bias_only_dit.py` -> `kernels/bias_only_dit` (hand CUDA + cuBLAS): the attention weights
  softmax(pair bias) are made once per pair, and the per-step attention is the sm_100a `pv_gate_inf` GEMM (P in tensor
  memory, sigmoid(g) in its epilogue). `bench.py target=bias_only_dit`: 1.52-2.22x PyTorch compiled (L128-768, per-sample
  conditioning), 1.58-2.55x with a shared conditioning. Training (bf16, autograd on) runs
  `integrations/bias_only_dit_train.py`: one opaque forward / backward per block (`dpb_sm100` for the bias gradient, the pair
  LayerNorm and bias on mma.sync in both directions, two-warp-per-row CUDA rows, the expand GEMM + SwiGLU keeping a | b),
  1.20-1.27x PyTorch compiled at A = 48 (L384 / L768), every gradient within 0.96-0.99x PyTorch bf16's error against fp32.
  Both paths serve 24 heads x 32 (`n_head=24`), 12 x 64 (`n_head=12`) and 16 x 64 (`BiasOnlyDiTBlock(d_head=64)`, 1024
  attention channels) as well as 16 x 48: training 1.15-1.30x, inference 1.46-2.63x (`bench.py target=bias_only_dit
  +n_head=24 | +n_head=12 | +d_head=64`).
  Page: `docs/gpus/b200/bias_only_dit/bias_only_dit.md`.
- B200 TriMul, D64 native path: B samples (B <= 8) in one call (tokens b-major, per-sample token mask and row-dropout scale, the plane bmm batched
  over d x B) instead of B calls; MiniWorld's template embedder runs its templates this way. 1.1-2.1x the B sequential calls at L128-384
  in inference, 1.2-2.6x in training (page: `docs/gpus/b200/trimul/trimul.md`); tests: `tests/gpu/sm100/test_b200_trimul_batch_gpu.py`.
  `integrations.trimul_b200.MAX_BATCH` (8) states the limit for callers that fold samples into the batch; on an engine without it a B > 1 call
  takes the slow path.

### Changed

- **Breaking -- every `x = x + f(x)` module returns `x + f(x)`.** `AugmentedAttentionPairBias`, `ConditionedTransition`,
  `TrianglePairAttention` and `BidirectionalTriangleAttention` now add their own input as the residual, like TriangleAttention,
  TriangleMultiplication, Transition, AttentionPairBias and MSAPairWeightedAveraging already did; `DiTBlock` only chains its two
  parts. A caller that wrote `x = x + module(x)` must write `x = module(x)` (the old form adds the residual twice, with no error).
  `AugmentedAttentionPairBias.delta` and `ConditionedTransition.delta` return the update alone through the same dispatch, for a
  caller that composes it some other way (a magnitude-preserving sum, a released checkpoint's own residual). Cross-tensor or
  externally gated updates are unchanged: OuterProduct / OuterProductMean (`residual=`), the SWA DiT's attention / FFN parts.
- B200 MSAPairWeightedAveraging: the contractions o = w · v and dv = wᵀ · do (and dw) run on cuBLAS `bmm` (5-15 % faster than
  the tcgen05 kernels at L256-1024, same bits), followed by one gate / out-projection / dropout / residual pass
  (`pwa_gate_out`, bf16x2 gate math); inference takes the same forward. Removed the unused sm_100a kernels (`pwa_ctr`,
  `pwa_plain`, `pwa_fwd`, `pwa_fwd2`, `pwa_glue`, `dgv_bwd`, `dgv_finish`, the ablation switches; `pwa_sm100.cu` ~3100 ->
  ~1950 lines) and the H100 `pair3` build on B200. The per-CTA partial sums of `pwa_glue2` / `dv_bwd` are combined by 8 split
  groups in a fixed order inside their finish kernels. `refusal` on B200 also requires S a multiple of 128 and N <= 1024.
  Harness speedup over the fastest other row: inference 2.00-2.44x, training (CUDA graph, dropout 0.15) 1.68-1.91x.
- B200 OuterProductMean: the prologue backward's column sums are finished by `opm_pbwd_finalize` (the ticket reduction
  `opm_pbwd_reduce` is removed). Harness speedup over the fastest other row: inference 1.16-1.57x, training (CUDA graph)
  1.24-1.57x.
- The forward fakes of the B200 OPM / PWA custom ops return the sm_100a saved-tensor shapes (compiled training on B200).
- `tests/gpu/any/test_{opm,pwa}_train_gpu.py` run on B200 as well; new: dropout in `pwa_glue2` matches a materialized
  gradient bit for bit, and the B200 PWA output, gradients and inference update stay within 1.15x of the module's own bf16
  statements' error against an fp32 copy (the out / dmsa tolerance against the Triton path is 8e-3 on B200).
- B200 TriMul D128 bidirectional training runs the shared D64 / D128 kernels (k1w -> k3g -> b1g at H = 256 -> b7g at
  eight plane chunks) instead of its own K1 / K3 / B1r / B7r extension (`b200_bidir.py`, removed). Its fp32
  LayerNorm-parameter gradients are now bit-identical across runs, and the 148-SM restriction is gone. The per-CTA
  LayerNorm-gradient rows of every D64 / D128 backward are summed by one fixed-order kernel (`lnpart_sum`) instead of two
  torch reductions. Harness training step (CUDA graph, dropout 0.25) 0.99-1.06x the retired path's 2026-09-29 numbers;
  a same-card A/B of the two paths varied from 0.93x to 1.08x between cards.
- SWA atom DiT (`kernels/swa_dit`), bf16 on H100: every stage is hand-CUDA now (was 3 CUDA + 6 Triton). New sources
  `cuda/swa_attn.cu` (window attention forward, and the backward's dq / dk / dv as ONE kernel; mma.sync, single-pass
  softmax over the <= 144 keys a warp's 16 queries reach) and `cuda/swa_slab.cu` (qkvg fwd / bwd, out-projection fwd /
  bwd, FFN fwd / bwd: 16-row slabs per 4-warp group, weights resident in smem, built for the small row counts the block
  mostly runs at). On the CUDA backward the attention side shares one [M, 4C] buffer, so the Triton path's dP copies of
  dv and dG are gone. New settings `swa_dit_attn_cuda`, `swa_dit_oproj_bwd_cuda`, `swa_dit_qkvg_bwd_cuda` (default on)
  and `swa_dit_rows_impl` ("auto" / "slab" / "wgmma": which CUDA kernels serve the qkvg fwd, out-projection + FFN fwd
  and FFN bwd; auto keeps team-gm's wgmma kernels for A % 8 == 0 at >= 5120 / 20480 rows). Measured on H100, 3 blocks
  fwd + bwd under a CUDA graph: team-gm's `SWAAtomTransformer` at N = 1, S = 4096 0.516 ms (Triton tuned 0.571,
  untuned 0.860, team-gm's CUDA stages 0.771); A = 48, S = 1024 1.860 ms (Triton tuned 2.325). Errors against the fp32
  reference equal the Triton path's. Registry rows `swa_dit_{softmax_fwd,softmax_bwd,output_bwd,inproj_bwd}_sm90_cuda`,
  `swa_dit_{inproj_fwd,output_fwd,swiglu_fwd,swiglu_bwd}_rows_sm90_cuda`.

- H100 TriMul inference K1 reads row-major `W_l, W_lg, W_r, W_rg` in place (`K1ParamsQ`, four
  TMA maps) instead of a per-call packed `w1`: no weight-pack kernels on bidirectional D64 and
  single-direction D64–384; output bitwise-identical. The column-major D128 bidirectional storage
  keeps the pack (a transposed-B K1 was slower and was not kept).
- H100 D128 training B7 (`B7_WT_MN`, mode bit 64) reads row-major front weights as MN-major
  tiles, so no transposed copies are made when the weights arrive row-major; the default
  column-major storage is unchanged.
- `docs/gpus/` is one folder per GPU (`<gpu>/<gpu>.md` module-level completion tables,
  `<gpu>/<module>/<module>.md` kernel-level tables and flow figures); `h100-dispatch.md` is now
  `h100/dispatch.md`. Page format: `docs/gpus/README.md`, `docs/gpus/template.md`.
- H100 TriMul K1 (inference, the bidirectional wide front and the D128 training front) and B7
  read the token mask [L] and form m[i] & m[j] themselves: no [L, L] pair mask is built per call
  (inference; bidirectional D128 training). The bidirectional wide inference front reads the
  projection weights in place (no pack). Outputs and gradients bitwise-identical.
- B200 LayerNorm / RMSNorm with implementation MINIWORLD resolve to PyTorch ops that `torch.compile` fuses (known-best table,
  compute capability 10.0); other cards are unchanged. `settings.b200_engine_triton` (default off) keeps the engine's Triton kernels on a
  B200. A strict `engine_backend="triton"` process still wins. The PyTorch `RMSNorm` that stands in for the kernel returns the
  input dtype like the kernel does (torch's `rms_norm` returns fp32 under autocast, which gave the attention's Triton kernels an
  fp32 q next to a bf16 k). Tests: `tests/cpu/compile/test_b200_norm_dispatch.py`.

### Fixed

- B200 bias-only DiT: saturated sigmoid gates gave NaN / inf. The fast sigmoid 1 / (1 + 2^(-g log2 e)) takes its reciprocal by Newton
  steps from a bit-trick seed that is valid only for a denominator below 2^126; at g <= -87.3 (inf from -88.7) the seed was garbage.
  `pv_gate_inf` (the bf16 attention core: training forward, 12-launch inference) turned a whole token row to NaN this way in a
  phase-2a bf16-mixed run (one gate at g = -88.5). The denominator is now clamped at 2^125 before the seed (NaN still propagates)
  in `pv_gate_inf.cu`, `cond_tables_sm100.cu` (opt-in; its f16 exponent overflowed already from g < -11) and the shared `rcp_nr`
  (`gemm_resln_sm100.cu`, opt-in). Supersedes the unpushed 5a46c2ef (pv_gate_inf only), whose two core tests are kept.
  `tests/gpu/sm100/test_b200_bias_only_dit_extreme_gates_gpu.py`: gates from -1e4 to +1e4 through the core, and blocks whose
  gate / scale pre-activations reach several hundred through every bf16 / fp32 inference and training path, finite and within
  bounds. Speed unchanged (bf16 inference L768 78.2 -> 78.4 us).
- Triton augmented attention, memory-efficient path (`compute_efficient=False`): a head dim below 16 padded its dot lanes to
  `next_power_of_2(D)` (8 for D = 8), under `tl.dot`'s K >= 16, so the kernels did not compile; it now pads to at least 16 as the
  compute-efficient path does (the padded lanes are masked loads of zero). `tests/gpu/any/test_augmented_attention_small_head_gpu.py`
  also picked a 32 x 32 tile that the B200 config set does not have (all 12 cases failed there); it now takes the smallest w4 / s2 tile
  of the active set (32 x 32 where there is one). 12 passed on B200.
- B200 token DiT attention cores could hang under concurrent GPU work (seen as a CUDA graph with two concurrent branches -- a sampling
  run beside a training step -- stuck at ~240 W): in `attn_inf_tf32`, `attn_fwd_tf32` and `attn_inf` (P double-buffered) the softmax
  warpgroup could hand over P(G) and P(G + 1) before the MMA warp tested P(G) (QK(G + 1) is issued before that wait), so the single
  `p_full` barrier completed two phases and the wait on its parity never returned. `p_full` is now one barrier per P buffer (G & 1,
  parity (G >> 1) & 1); completing one twice would need P(G + 2), whose S(G + 2) is only issued after the MMA warp passed P(G). Kernels
  with a single P buffer (`attn_fwd2`, the atom `attn_fwd`, SWA `attn_dkv` / `attn_dkvq`) wait for PV(G - 1) before writing P(G) and
  could not run ahead. Found with cuda-gdb (stuck warps' PCs mapped to source lines); a Protenix step graph with the rollout on a second
  stream, which hung in 6 of 7 runs, completed 4 of 4 runs of 300 replays. Same results (token DiT tests unchanged).
- B200 weight-pack caches were wrong inside CUDA graphs whose weights change between replays (an optimizer step, or a sampling
  run inside a training graph): the token DiT (inference runner and its pair-bias cache; training pack), bias-only DiT (inference runner
  and its attention-weight cache; training pack), TriAttn wide-width pack (inference and training), AttentionPairBias inference packs and
  the TriAttn modules' fused-inference weight concatenation reused, during a capture, an entry built by the eager warm-up -- so the graph
  recorded no packing kernel and every replay computed with the weights of capture time (training: gradients of stale weights). The
  caches are now scoped to the capture (`kernels/_capture.py`: the capture sequence id from the driver): an eager entry serves only eager
  calls; a capture packs once, recorded, so each replay repacks the current weights, and reuses that pack for its other calls; entries of
  finished captures are dropped. The atom DiT and SWA DiT caches, which bypassed the cache during a capture, now also pack once per
  capture instead of at every call. Updates that bypass the version counter (torch's fused optimizers) still need
  `torch.autograd.graph.increment_version(params)` after the step. Tests: `tests/gpu/sm100/test_b200_pack_cache_capture_gpu.py`
  (seven of its nine replay cases fail before the change).
- B200 SWA atom DiT weight-pack cache (`kernels/swa_dit/cuda/sm100._cached`) keyed on (data_ptr, version, shape) served the old
  tensor's packed copy to a new weight placed at a freed one's address (a bf16 cast, the next test's weights). It is keyed on the tensor
  objects now (weak references plus version), and bypassed while a CUDA graph is captured (a hit would record no pack kernel, so every
  replay would read the packed weights of the capture-time step) and for inference tensors. Tests:
  `tests/gpu/any/test_swa_dit_pack_cache.py`.
- B200 OuterProductMean training was not deterministic from run to run at L >= 384 (whole rows of one group): the prologue
  backward released an input stage to the next TMA load while its generic shared-memory reads could still be in flight. A
  `fence.proxy.async.shared::cta` before the release fixes it; the same fence now precedes the stage releases of the PWA
  `pair_fwd` / `pair_bwd`. Both paths give identical results across runs at every tested L; no speed change.

### Added

- SWA atom DiT on H100: the weight-gradient GEMMs and the hoisted modulation get hand-CUDA kernels (`kernels/swa_dit/cuda/swa_wgrad.cu`,
  `swa_mod.cu`; stages "wgrad" and "mod" of `cuda/loader.py`). `wgrad` / `wgrad_f32`: a block's dWqkv, dWg, dWo, dWu, dWd in one launch
  (clusters of 8 CTAs split K and sum in rank order through distributed shared memory; bf16 on mma.sync, fp32 on bf16 hi / lo planes:
  4.8e-6 against an fp64 reference where cuBLAS's TF32 GEMMs have 3.0e-4), served up to 16384 rows (bf16) / 8192 rows (fp32), cuBLAS
  above. `mod_fwd` / `mod_bwd` (bf16 c / Wmod): silu(c) Wmod^T and its backward with exact products (dmod split into three bf16
  planes); fp32 stays on torch's GEMMs. `dmod` is no longer zero-filled when the slab kernels store it plainly (A in 1, 2, 4, 8, 16).
  New settings `swa_dit_mod_cuda`, `swa_dit_wgrad_cuda` (default on). H100, 3 blocks fwd + bwd under a CUDA graph: bf16 A = 1,
  S = 4096 0.656 -> 0.444 ms (Triton fused 0.708), fp32 0.700 -> 0.652 ms (Triton fused 0.825, per-op 1.654); full matrix in
  `docs/gpus/h100/swa_atom_dit/swa_atom_dit.md`. Registry rows `swa_dit_gemm_{fwd,bwd}_sm90_cuda`, `swa_dit_dw_sm90_cuda`,
  `swa_dit_dw_fp32_sm90_cuda`, `swa_dit_transpose_sm90_cuda`. The same weight gradients also exist on Hopper's asynchronous units (`swa_wgrad_wg.cu` bf16,
  `swa_wgrad_wg_f32.cu` fp32: TMA producer warp, mbarrier ring, wgmma m64n128k16 MN-major, cluster-8 DSMEM reduction; stages "wgwg" /
  "wgwg32"): bf16 at every row count (74 us against cuBLAS 111 at 49152 rows), fp32 up to 65536 rows; the modulation forward on
  TMA + wgmma (`swa_mod_wg.cu`, stage "modwg"). Every kernel of the block chain after its heads launches with programmatic dependent
  launch (`cuda/pdl.cuh`). bf16 H100, 3 blocks fwd + bwd under a CUDA graph: A = 1, S = 4096 0.656 -> 0.426 ms, A = 48, S = 1024
  1.869 -> 1.653 ms; fp32 A = 1, S = 4096 0.700 -> 0.623 ms. The wgmma FFN kernels' four weight layouts
  (formerly four torch copies per block per step) come from one launch (`swa_pack.cu`, stage "pack"), kept for the backward of the same
  step (weakref + `_version`; not while a graph is captured).
- fp32 SWA atom DiT block in hand-written CUDA on H100 (`kernels/swa_dit/cuda/swa_fp32.cu`, stage "fp32" of
  `cuda/loader.py`): every stage of the fp32 fused block -- qkvg forward, window attention forward and backward,
  out-projection + gated residual + SwiGLU FFN forward, FFN backward, out-projection backward, qkvg backward, plus a
  per-call weight prep kernel -- as sm_90a tf32 wgmma kernels with cvt.rna-rounded operands (FFN weights split into
  two tf32 halves) and a bf16-operand mma.sync window attention. A 64-row tile runs on a pair of CTAs that split its
  weight stream; row tensors move by TMA; partial sums cross by `st.async` into the partner's shared memory. New
  setting `swa_dit_fp32_cuda` (default on); it serves while torch's fp32 matmul precision allows TF32 ("high" /
  "medium"), otherwise, off sm_90, with `engine_backend="triton"` or on a failed build the fp32 Triton kernels do.
  Same saved tensors and weight-gradient GEMMs as the Triton path. H100, 3 blocks fwd+bwd under a CUDA graph at
  "medium": N=1 S=4096 0.655 ms (fp32 Triton autotuned 0.780, per-op 1.78); A=48 S=1024 3.72 ms (5.70, 11.0).
  Registry rows `swa_dit_*_fp32_sm90_cuda` (8). `build/classify.py` recognises `mma.sync` and `__cluster_dims__`.
- SWA atom DiT fused block moved from team-gm 14f2c73 ("research(swa): preserve opt-in fused atom
  transformer work"; copied from team-gm 4fafa83, `swa_fused_triton.py` + `swa_cuda/`) into the new
  kernel family `kernels/swa_dit`: the ESMFold2 `SWAAtomBlock` (RMSNorm + adaLN-Zero, QKVG
  projections + q/k RMSNorm + 3D RoPE, window-128 attention with `seqused` masking, sigmoid gate +
  out-projection, SwiGLU FFN) as 9 autotuned Triton kernels (3 forward, 6 backward) and 3 hand-CUDA
  sm_90a bf16-wgmma stages (qkvg forward, out-projection + FFN forward, FFN backward), bf16,
  d_atom 128 / 4 heads / SwiGLU hidden 256. Kernel bodies and CUDA sources are unchanged; the
  autotune key is `atom_key(S, ...)` instead of team-gm's log2(rows), `configs/default` holds exactly
  the configs team-gm's decorators enumerated, the CUDA stages JIT-build under
  `MINIWORLD_ENGINE_JIT_ROOT` and fall back to Triton, and team-gm's `SWA_*` environment switches are
  `settings.swa_dit_*` (same defaults). Public: `kernels.swa_dit_block`,
  `kernels.swa_dit_hoist_modulation` (and `kernels.swa_dit.interface.refusal`). `SWADiTBlock`
  (implementation other than pytorch) now runs the fused block where `refusal` accepts the call
  (`settings.swa_dit_fused=False` keeps the per-op path), and `SWADiTBlock.forward_hoisted` takes the
  augment-invariant conditioning [B, S, d_cond] to compute the modulation once per batch element.
  Registry: 12 rows (`swa_dit_*`), drivers, checks, fp32 reference. Tests:
  `test_swa_dit_fused_gpu.py`, `test_swa_dit_fused_dispatch.py`.
- `swa_dit` keys every Triton launch on the augment count as well: `atom_key(S, A=N // B, C[, NHID])`,
  A = 1 included (MiniWorld's input feature embedder calls with num_aug=1, diffusion with 48 in
  training and 5 samples at evaluation); the drivers tune A = 1 / 48 (and 5 for the inference
  forward). The bf16 qkvg backward's config sets gained SP = 1 tiles for A = 1.
- fp32 fused SWA atom DiT block: `swa_dit_block` dispatches on the activation dtype; all-fp32 calls
  (MiniWorld v1.3's fp32 atom transformer) run 5 new Triton kernels (`swa_dit_*_fp32_triton`,
  `triton/forward_fp32.py`, `backward_fp32.py`; Triton only) around the shared bf16-operand window
  attention, as the per-op path's FlashAttention-4 does. Residual stream, norms, modulation, RoPE,
  gates and SwiGLU are fp32; projections tf32 and the FFN GEMMs tf32x3 (measured: Triton's tf32
  truncates, and only the FFN's truncation showed). Accuracy against the fp32 reference matches the
  per-op fp32 path; fwd+bwd of 3 blocks at N=1, S=4096 (H100, CUDA graph) 0.87 ms against 1.83 ms.
  `refusal` accepts all-bf16 or all-fp32 and refuses mixed dtypes.
- B200 (sm_100a) hand-CUDA TriMul for both modules (bidirectional and one direction), D64 / D128 / D256 / D384 /
  D512, inference and training (`kernels/trimul_inproj/cuda/b200_{infer,train,bidir}.py`, `b200_sources/`: k1w front ->
  cuBLAS contractions -> k3g (D <= 128) or k3w (D >= 256, LayerNorms folded into the output GEMMs); training backward
  b1s / b1g -> contraction grads -> b7m / b7g at D64 / D128 one direction, a cuBLAS + memory-kernel composite at D >= 256,
  and B1r / B7r for D128 bidirectional), dispatched by `integrations/trimul_b200.py`. LayerNorm-parameter gradients
  are fixed-order sums (bit-identical across runs). With a CUDA graph it is 1.3-3.1x the fastest of PyTorch compiled /
  cuEquivariance 0.12 in inference and 1.2-3.0x in training (`docs/gpus/b200/trimul/trimul.md`).
- B200 (sm_100): hand-CUDA fused Transition for D128/n=4 bf16 (`kernels/transition/cuda/fused_sm100a.py`,
  one forward kernel with 2-CTA tcgen05 products, one backward kernel + a partial reduction), dispatched
  from `modules.Transition` and `ops.transition`; `MINIWORLD_TRANSITION_FUSED_SM100A=0` opts out. Module
  training step 2.5x / 2.3x the Triton residual path at L384 / L768. The kernels are built into cubins
  by the newest nvcc that knows sm_100a (13.1 on the B200 box; 12.9's ptxas is ~8 % slower on the
  backward) and launched through the driver API. Page: `docs/gpus/b200/transition/transition.md`.
- B200 (sm_100): hand-CUDA Transition for D64/256/384/512, n=4, bf16 (`kernels/transition/cuda/fused_wide_sm100a.py`:
  D64 one fused forward + one fused backward; D256 fused forward, split backward; D384/512 LN -> expand+SwiGLU -> squeeze,
  split backward from the saved a / b), same dispatch and opt-out as D128. Module L384 vs torch.compile: inference
  1.3-3.6x, training 1.08-2.5x. The D128 kernels drop two cluster-scope releases per launch (D128 L128 inference
  15.1 -> 13.5 us kernel time, bit-identical outputs). Kernel sources under `sm100/` are generated from the research
  capsule (`experiments/transition_fused_sm100/export_engine.py`).
- B200 (sm_100): hand-CUDA Transition at n=2 for D64/128/256/384/512/768, bf16 (`fused_wide_sm100a`, same kernels
  built with `-DHID=2D`; D768 runs the D512 chain with the squeeze and d_xn GEMMs as two 384-column launches), for the
  pair, MSA and single streams (any whole number of 128-row tiles). D384 (n=2 and 4) is now one fused forward kernel
  (`widths/tfwd_d384.cu`) and one fused backward kernel for the gate, d_xn and the LayerNorm backward + a fixed-order
  dgamma / dbeta reduction (`widths/tbwd_d384.cu`; dW stays cuBLAS); single-stream D384 calls of at most 16 tiles take a
  cluster forward that splits the hidden units (`widths/tsmall_w.cu`) and the item-scheduled backward gate. Module
  (bf16 parameters, CUDA graph) vs the faster of Triton / torch.compile: pair L384 n=2 inference 1.8-2.8x, training
  1.2-1.9x; diffusion conditioning [48, L, 384] n=2 training 174.2 / 296.8 us at L384 / L768 (was 220.7 / 328.9 with the
  three-kernel D384 chain; torch.compile 228.1 / 370.5); single stream [1, L, 384] inference on par with Triton, training
  1.5-1.6x slower. Page: `docs/gpus/b200/transition/transition.md`.
- B200 (sm_100a) hand-CUDA TriangleAttention, the whole module (`integrations/triattn_b200.py`,
  `kernels/triangle_attention/cuda/b200_triattn.py`, `b200_sources/`): d_pair 128 / 4 heads fused for inference and
  training (L a multiple of 128), d_pair 64-512 inference. `module._b200_cuda = False` keeps the Triton path.
  Page: `docs/gpus/b200/triattn/triattn.md`.
- B200 (sm_100a) OuterProductMean / PWA training: `integrations/opm_train.py` and `pwa_train.py` accept compute
  capability (10, 0) with the H100 kernels' signatures on tcgen05 / TMEM (`integrations/csrc/sm100/`).
- B200 (sm_100a) token DiT, inference and training with hand CUDA + cuBLAS only (`integrations/token_dit.py`,
  new `integrations/token_dit_train.py`, `kernels/augmented_attention/cuda/sm100/`,
  `kernels/conditioned_transition/cuda/`). Page: `docs/gpus/b200/token_dit/token_dit.md`.
- `miniworld_engine.viz.kernel_flow`: kernel-flow SVG figures (one box per kernel, HBM reads and
  writes) from a JSON spec.
- H100 single-direction TriMul training in CUDA at D64 (`h100_uni_d64_training`, the
  bidirectional D64 kernels at hidden 64) and D256/384 (`h100_uni_wide_training`, the wide
  sources compiled for hidden D), L384/768. Measured on an H100 80GB HBM3 against the Triton path
  (CUDA graph, fwd+bwd, 2026-09-29):
  about 2x at D64, 1.27-1.33x at D256, 1.31-1.36x at D384; accuracy against an FP32 reference
  matches the Triton path. Tests: `test_trimul_uni_d64_training_gpu.py`,
  `test_trimul_uni_wide_training_gpu.py`.

## [2.2.0] - 2026-09-28

Two tiers per op: hand-written CUDA where it exists, a Triton fallback everywhere else, plus the
PyTorch reference and the comparison baselines (cuequivariance, Anthropic payload, dtv1).
GPU qualification of this release is pending.

### Runtime

- torch 2.13.0+cu129 (triton 3.7.1, CUDA toolkit 12.9 — the newest pair the cluster's 575
  driver runs), cuequivariance / cuequivariance-ops-torch-cu12 0.12.0, Transformer Engine 2.19.
  Tuned autotune caches are keyed on the torch version and need rebuilding.

### Removed

- `experiments/`: every capsule is either ported to `src/` (fastest variant only) or history.
  The tree is at tag `archive/experiments-20260928` (and tag `archive/wip-main-20260928` for the
  Sept 27–28 local-H100 research); see README "Research history". The Anthropic
  `NOTICE` and v5 source hashes moved to `licenses/`.
- Every CuTe DSL / nvidia-cutlass-dsl / quack kernel and the `cute` extra: `kernels/*/cute`
  (tm1, tm2, transition, layernorm, layernorm_linear, trimul_inproj), the `fused_ln_mask`
  family, `_quack_compat`, the autotune CuTe candidate spaces and compile paths, and the
  unused CUTLASS C++ trees `kernels/adaln/cutlass`, `kernels/conditioned_transition/cutlass`
  and `transition_b2b_sm100_kernel.cu`. Hand-CUDA kernels that include CUTLASS C++ headers
  (triangle-attention CUDA, transition b2b/expand-gate/gate-bwd) are kept.
- `ImplementationType.CUTE` / `KernelBackend.CUTE`, `modules.dispatch.trimul_out_layout`.
- Public kernels `cuda_transition` (never implemented), `cuda_transition_b2b`,
  `cute_transition_fused`; `layernorm_linear_fn` / `LayerNormLinearFn`; `tm2_cute`;
  `trimul_inproj_cute`; `adaln_inference_lnfold`.
- Intermediate module paths: TriMul CuTe training/inference and the `trimul_sm90_kernels`
  parity hooks; Transition's split / b2b-inference / CuTe routes (`_old_triton_forward`,
  `_inference_forward`, `_training_forward`).
- The previous wide/D64 H100 TriMul training port: `h100_width`, `h100_width_base`, `h100_gp`,
  the unused `h100_wide_forward` plan and their `h100_sources/wide`, `wide_base`, `wide_forward`
  sources.
- Settings: `trimul_out_layout`, `trimul_cute_dispatch`, `trimul_sm90_kernels`,
  `trimul_train_front_fused`, `lnl_ws`, `transition_force_split`,
  `transition_residual_fusion`, `transition_large_d_training`, `transition_cute_backward`,
  `transition_dab_lnbwd`.

### Changed

- Add portable Triton OuterProductMean and MSAPairWeightedAveraging (forward and
  backward; kernel families `outer_product_mean`, `pair_weighted_averaging`, 12
  autotuned kernels). `miniworld` now takes them on every GPU and width the
  native H100 paths do not serve, and under `engine_backend="triton"`; before,
  those calls ran the module's PyTorch statements. `implementation="triton"` on
  either module used to run the same statements silently and now runs the
  kernels or raises `NotImplementedError` naming the reason (CPU input, non-bf16,
  interchain masking, d_msa / d_hidden not a power of two, OPM d_pair not a
  multiple of 32). Their autotune grids ship at full ladder width; no cache is
  built yet, so a first call uses the bounded miss fallback.
- `build.launch_bind` reads a kernel's tuned axes from the shipped grid spec of
  its `configs_for("<op>")`, so axes not spelled `BLOCK_*` are recognised.
- A100 (sm_80): hand-CUDA fused Transition for D128/n=4 bf16 (`kernels/transition/cuda/fused_sm80.py`,
  forward + two-kernel backward, training step about 1.44x the Triton residual path on A100).
  The sm_80 CUDA TriMul and the other A100 research runners are not ported to `src/`; they are
  at tag `archive/a100-sm80-branch-20260928` (`experiments/a100_*`).
- Repository layout (no behaviour change): registry CSVs and evidence files in `kernels/registry/`;
  A/B config sets in `autotune/configs/ab/` (short names such as `blk16` still resolve); docs are
  `docs/gpus/` (per-GPU how-to and completion tables, judged by the maintainer) and `docs/kernels/`,
  plus `docs/CHANGELOG.md` and `docs/standards.md`; benchmark docs beside `benchmarks/`; agent rules in
  `.claude/CLAUDE.md`; `scripts/` folded into `miniworld-engine dev` commands (`tools/`) and
  `benchmarks/runners/`. Removed records, experiments and scripts are listed in README "Research history".
- Bidirectional TriMul H100 training at D256/384/512 (L384/768) is a new flattened hand-CUDA +
  cuBLASLt port of the qualified large-width research plans (`h100_wide_training`, 29 frozen
  kernels in `h100_sources/wide_train`): 1.44-1.62x the Triton path in paired CUDA-graph fwd+bwd
  replay on H100 (was 0.69-0.75x). Its retained activations are larger than the old port's
  (D512/L768 11.3 GiB) and equal to or below the Triton path's. cuBLASLt algorithms frozen under 12.8.4 are matched by configuration against the
  running cuBLASLt (`lt_selection.json`), falling back to the first heuristic with a warning.
  See docs/gpus/h100/dispatch.md.
- `trimul_h100_training_widths` defaults to `(128, 256, 384, 512)`; D64 bidirectional training
  runs on Triton (its CUDA port measured 0.61x of Triton in graph replay).
- The H100 training opaque ops are renamed (`trimul_h100_train_{fwd,bwd}_wide_port`,
  `trimul_h100_dropout_nograd_wide_port`) because the wide saved-tensor contract changed.
- `miniworld` TriMul resolves to the Triton family on every arch; the hand-CUDA H100 kernels
  are still tried first inside the modules. Both TriMul modules now run the Triton path plane
  by plane for B > 1 (the CuTe path used to be the only one that looped).
- `Transition` and `ops.transition` always take the residual path: hand-CUDA `fused_sm90a` /
  `fused_wide_sm90a` on sm_90, else Triton `transition_residual`.
- `layernorm_linear` is Triton on every arch; adaLN wide-d inference uses the fused
  GEMM+gate kernel on every arch; token DiT uses cuBLAS for all GEMMs.
- Triton TriMul front backward computes its weight gradient with cuBLAS directly.

### Fixed

- JIT build lock guard (`kernels._nvcc`): honour `load(build_directory=...)` when locating the
  lock, and do not trust an NFSv3 exclusive-create mtime (read back as 1981) when deciding a
  lock is stale -- use max(mtime, ctime) and never reclaim a pre-2000 stamp. Both hung
  multi-rank training jobs on the cluster's NFS home.

## [2.1.0] - 2026-09-27

- Follow production backend dispatch when building caches; force alternative
  implementations only with an explicit request.
- Ship compact per-kernel Triton defaults while preserving global config spaces,
  shared-memory prediction and opt-in multi-GPU global searches.
- Keep compatible measured global winners usable without global runtime searches.
- Restrict token training build lengths to 384/768 and preserve inference ladders.
- Expose native-only inference builds and register exact-shape tuning for fused
  Transition and packaged TriMul inference, including candidate output validation.
- Document fixed/manual CUDA schedules separately from integrated autotuners.
- GPU qualification of the new tuning integrations remains pending.

### Consolidated work since 2.0.0


- Fuse OPM/PWA training dropout and residual epilogues, retain live-weight graph
  replay, and record MSA comparisons with the standard depth of 1024.
- Reuse D128 TriMul preparation and expose optimizer-state layout migration for
  existing checkpoints; preserve source/measurement provenance and limitations.

- Reuse forward weight packing and FP32 masks in bidirectional TriMul backward
  at D64/256/384/512; remove duplicate D512 input normalization and fix wide
  training `x_n` metadata for compiled execution.

- Add native single-direction H100 TriMul training for D128/L384/768, both
  outgoing and incoming: Anthropic-derived K1/K3, fused output backward and
  producer-consumer input backward, including dropout/residual and graph replay.

- Connect packaged H100 TriMul inference and latest bidirectional CUDA training
  to default `auto` module dispatch. Training widths: D64/128/256/384/512,
  L384/768; width-specific performance tuning remains separate.
- Enable supported packaged OPM/PWA paths automatically and connect fused
  token DiT inference. Preserve live weights, saved-tensor ownership, dropout,
  residuals and `torch.compile` boundaries.
- Make explicit backend comparison policy consistent for Transition.
- Ship selected CUDA sources/includes and write compilation products only to
  the user cache. [Dispatch contracts](gpus/h100/dispatch.md).

## [2.0.0] - 2026-09-23

### Breaking changes and migration

- Release the accumulated public API removals listed below, including
  `kernels.triton_adaptive_layer_norm` and obsolete autotune budget settings.
- `SWADiTBlock` uses the ESMFold2 RMSNorm/adaLN-Zero contract; old AF3-style
  block checkpoints are incompatible. Recreate those blocks and retune caches.
- New H100 Transition dispatch is enabled for supported BF16 widths. Users
  requiring the Triton route can set `MINIWORLD_TRANSITION_FUSED_SM90A=0`.
  The new TriMul training implementation remains an explicit research entry.

### Anthropic-derived development

- Preserve upstream credit, licenses and provenance. MiniWorld inherits the
  stronger published inference implementation and extends it for training;
  see `licenses/THIRD_PARTY_NOTICES.md`.
- Integrate K1/K3 single/bidirectional TriMul inference, fused Transition
  forward/backward and its wide-width ports, MSA OPM/PWA training and
  inference integrations, and token DiT research histories.
- Preserve latest CUDA TriMul B1–B4 and single-launch B7–B12, saved input `x_n`,
  output-LN recomputation, dropout/residual/mask semantics, all eleven gradients,
  strict D128 LN-gradient fixes, and D64/128/256/384/512 validation records.
- Package a relocatable research capsule and its pinned benchmark engine so the
  latest work no longer depends on a private MiniWorld checkout. D128 is the
  performance winner; other widths remain slower than Triton and their further
  optimization is deferred. This is not a claim of SoL90 or complete cache coverage.
- Fix large packed TriMul gradient offset arithmetic using 64-bit K indices.
- Archive 27 stale runtime cache files unchanged with SHA-256 provenance; they
  require rebuilding and are not represented as valid 2.0.0 tuning records.
- Transition adds the parallel LN-gradient reduction and transposed dWs stores.

Release map, migration limits and evidence: 2.0.0 (`archive/records-20260928:docs/project/release-2.0.0.md`).


### Added

- `ops.gated_residual(x, gate, branch)`: fused linear residual gate with backward.
- Fused sm_90a Transition, forward and backward, for the AF3 pair width (`d_hidden`
  128, `n` 4, bf16, whole 128-row tiles). Two launches replace five: LayerNorm,
  expand-SwiGLU and squeeze-with-residual on the way forward, and the squeeze,
  SwiGLU and LayerNorm backward chain on the way back. Measured through the wired
  dispatch on an H100 SXM, `modules.Transition` forward plus backward goes from
  1074 to 559 us at L=384 (1.92x) and from 4025 to 2073 us at L=768 (1.94x).
  On by default where it applies, off with `transition_fused_sm90a=False` or
  `MINIWORLD_TRANSITION_FUSED_SM90A=0`; every other shape, dtype and architecture
  keeps the existing path. Development record in
  `experiments/transition_fused/`.

### Changed

- Merge the 24 active MiniWorld consumer patches: packed TriMul buffers, configured
  F567 forward and dual-dgrad/LayerNorm-residual backward fusions, strict Triton
  backend selection, and expanded resumable CuTe/CUDA tuning.
  See integration and validation (`archive/records-20260928:docs/records/local-patches-20260917.md`).
- Retire 37 incompatible H100/A5000/A6000 cache files from runtime selection;
  original measurements and checksums remain in the integration archive.

- FA2 sliding-window attention uses static-capacity packing in backward as well as
  forward, removing dynamic `nonzero`/host length reads from the AOT backward graph.
  SWA DiT compiled benchmarks now require `fullgraph=True`.

- `SWADiTBlock` now matches MiniWorld ESMFold2 adaLN-Zero. Engine backends use
  fused RMSNorm modulation, SwiGLU, and residual gates; the previous AF3-style
  block checkpoints and SWA DiT benchmark results are incompatible with this version.

### Fixed

- AdaLN GEMM alignment, large compute-efficient attention backward scratch usage,
  mixed-affine LayerNorm and compiled FA4 backward.
- Wide TriMul packed reduction keys (`KP=4096`), compact CuTe cache selection,
  and shape-only derivation of another architecture's external FlashAttention path.

### Removed
- **`kernels.triton_adaptive_layer_norm`, and twelve kernels no production path reached.** Two
  audits of the adaln and conditioned_transition families found that half their registry surface
  was unreachable from any module, and the build was tuning all of it.

  adaln dispatches to `adaln_train` / `adaln_inference` and to nothing else. `main.py`'s
  `TritonAdaptiveLayerNormFunction` and `fused3.py`'s `adaln_fused3` / `adaln_fused3_train` were
  reached only by the bench, the drivers and the checkers — six kernels between them. `main.py` is
  gone; `fused3.py` kept the one kernel `inference.py` imports and is now `ln_strided.py`, named
  for it, at 110 lines instead of 430.

  conditioned_transition had three training files and one live one. `train_12_345.py` could not run
  at all — neither of its Functions takes a `length`, so every inner launch hit the `shape_key=None`
  branch, which raises. `train_fused.py`'s fused backward was never selected, and its own H100
  measurement says why: it lost to cuBLAS+elementwise by 1.6-7.1x at every stage. Its forward pair
  survives as `fwd_saveact.py` because `training.py` calls it. Six more kernels gone, plus the
  duplicate `cond_transition_fwd_12_345`.

  `triton_adaptive_layer_norm` was the only one of these on the public kernel surface, so this is
  the semver-relevant part. The registry, `axes.csv`, the device manifests, the config sets, the
  shipped tuned caches and the drivers/checkers were pruned with them: 91 triton kernels -> 79.

### Removed
- **`--bench-budget` and `settings.bench_budget_*`.** The feature abandoned an autotune config
  once one timed launch exceeded `factor x` the best so far. Its safety argument — "a config
  that could still win runs FASTER than the current best and is therefore always inside the
  budget" — is true only if both numbers are the same quantity, and they were not.

  `best` comes from triton's `do_bench`, whose timed loop never synchronises, so the queue
  stays full and the event window measures device time alone. The probe drained the stream
  (warm launch + `synchronize`) and timed ONE launch, so its number carried the host launch
  latency as well. Measured here on an A6000, one kernel, BLOCK swept 256→16384 and warps
  1→16:

      probe (1 launch)   0.0256 – 0.0502 ms   (spread 0.025 — flat: it is not the kernel)
      do_bench median    0.0051 – 0.0645 ms   (spread 0.059 — it tracks the config)
      budget = best x 3  0.0154 ms            → 12 of 12 configs abandoned

  So the first config in grid order sets `best` under the 300 ms first-round cap and every
  other config is dropped. In the shipped cache this leaves a fingerprint: **349 of 1244
  entries across 51 ops have the first config in grid order as their winner**, 346 of them
  with a single ranked config, from grids of 750 to 15552. `adaln_gemm_gate_triton` records
  `{BLOCK_K:16, BLOCK_M1:32, BLOCK_N:32, GROUP_M:1} warps=1 stages=1` — grid position 1 of
  15552 — for three different shape keys.

  It breaks whenever `kernel time < launch floor / 3`, which is nearly every kernel here
  (0.005–0.03 ms). Removed rather than fixed: a corrected probe would have to bench a queued
  batch, which is a different implementation, and this one silently corrupts caches when
  enabled. The A5000/A6000 caches need rebuilding.

### Changed
- **Per-kernel numerical tolerance, derived from measurement.** `registry.csv` had an `rtol`
  column and 0 of 103 rows filled it, so every kernel was held to one global `5e-2` — 18x the
  median kernel's actual error, and 88 of 97 sat more than 10x inside it. Each row now declares
  a band computed as `4x` its worst measured relative error, from the two device manifests
  `run_all` writes. The margin is from the data, not chosen: the same kernel's error across the
  two cards varies by median 1.07x, p90 1.47x, **max 1.95x**, so 4x leaves two more doublings
  than anything observed. Calibration only tightens — the two kernels already measuring close to
  the global band (`adaln_fwd_saveact_triton` 2.77e-2, `triangle_attention_bwd_atomic_triton`
  1.44e-2) keep `5e-2` rather than being handed something looser. Median band is now `1.1e-2`,
  a 5x tightening. `tests/cpu/registry/test_declared_tolerance.py` fails if a band drops below what
  the kernel measured, or drifts far above it.
- **`settings.AutotuneKernel` is three names, and an unknown one now raises.** It declared
  seven; four of them (`tri_multi`, `layernorm`, `layer_norm_linear`, `augmented_attention`)
  had no call site, so naming one unlocked nothing and reported nothing. `autotune_kernels`
  is typed against the vocabulary and `configure` rejects a name outside it — previously
  `autotune_kernels={"triangle_attention"}`, the family's *current* name rather than this
  vocabulary's older `tri_attention`, was accepted and did nothing.

## [1.0.0] - 2026-08-25

### Breaking
- **The distribution and import name changed: `miniworld-kernels` →
  `miniworld-engine`, `import miniworld_kernels` → `import miniworld_engine`.**
  There is no compatibility alias: the old import raises `ModuleNotFoundError`.
  This is why the version is 1.0.0 rather than 0.2.0 — the previous release and
  this one are different packages, and `0.1.0` was published under both names,
  so no consumer could tell them apart by any declared field. Pin
  `miniworld-engine>=1.0.0` and rewrite the import; a submodule consumer must
  also update the path and URL.

### Added
- **`miniworld-engine dev audit`** — verifies the build system and, new, that every
  DECLARED `(op, dtype, shape-bucket)` is present in the shipped autotune cache.
  Declared means `registry.csv` crossed with each kernel's `level` and `dtypes`,
  so a hole is reported against the contract rather than against whatever the
  last build happened to measure. `build` already told users to run this; the
  subcommand did not exist and `build/audit.py` crashed on import.
- **`settings.autotune_miss_cap`** (default 24) — how many configs an autotune
  cache MISS may search before falling back to a heuristic subset.
- **`miniworld_engine.ops` — the whole-op consumer contract.** Complete,
  autograd-transparent model-layer ops (weights as arguments, backend dispatch +
  fwd/bwd inside), consumed as a single call: `triangle_multiplicative_update`,
  `triangle_attention`, `transition`, `conditioned_transition`,
  `augmented_attention_pair_bias`, `layer_norm_linear`, `layer_norm`. Lazy,
  side-effect-free import; pinned by `_OPS_CONTRACT` in `tests/cpu/compile/test_public_api.py`.
  Verified fwd+bwd vs the pytorch/cuequiv reference on B200 (≥0.9998).
- Public API contract test (`tests/cpu/compile/test_public_api.py`): freezes the
  `kernels` surface and asserts the package import is side-effect-free.
- Numerical correctness suite (`tests/gpu/any/test_numerical.py`): each op's fused
  MINIWORLD backend vs the PyTorch reference (forward + input gradient),
  GPU-gated, asserting the fused path is actually engaged (no silent
  dtype-degrade). Promotes the benchmark cosine checks into an enforced
  correctness gate.
- CI (`.github/workflows/ci.yml`): ruff + ty + the whole CPU suite on every
  push/PR (nested cutlass submodule skipped; CPU torch wheel installed so the
  type gate can see the stubs it is checking against). `ty`, not pyright, which
  cannot parse jaxtyping shape strings; the step gates, with no `|| true`. GPU
  numerical suite runs via `pixi run test-gpu` on an allocated node.

### Deprecated
- **`kernels.cuda_transition`** — it has never had an implementation. It deferred to a
  `transition/cuda` symbol that git has no record of, and calling it raises
  `NotImplementedError`; the module's `KernelBackend.CUDA` branch called it with a signature
  nothing here provides. Use `implementation='triton'` on `Transition`, or
  `kernels.cuda_transition_b2b` for the hand-CUDA LN-fused path. It stays in the surface and now
  emits `DeprecationWarning`; removal no earlier than two releases out, per the procedure in
  CONTRIBUTING.md.

### Changed
- **One vocabulary for benchmark targets, and a level to hold it.** `bench.py`'s
  targets lived in one flat dict, which forced the kernel-level ones to abbreviate
  around the module-level ones: `tri_attn`, `bias_attn`, `aug_attn`, `ln_mask`,
  `gate_bwd`, `gemm_epil`. `bench_kernel triangle_attention` — the family's own name —
  came back "unknown target". `BenchConfig.kernel` is now `target` + `level`
  (`kernel` | `module`), the two levels are separate namespaces, and every target is
  spelled the way the engine spells it: a kernel target names its family in
  `kernels/registry/registry.csv`, a module target names the module it constructs. So
  `triangle_attention` is now a legal name at both levels and means the right thing at
  each. Renamed: kernel `tri_attn`→`triangle_attention`, `bias_attn`→
  `bias_only_attention`, `aug_attn`→`augmented_attention`, `ln_mask`→`fused_ln_mask`,
  `gate_bwd`→`gemm_gate_bwd`, `gemm_epil[_bwd]`→`gemm_epilogue[_bwd]`,
  `dual_gemm_epil[_bwd]`→`dual_gemm_epilogue[_bwd]`, `cond_transition_tail`→
  `conditioned_transition_tail`; module `bias_only_attention`→`attention_pair_bias`
  (it benches `AttentionPairBias`, and the old name belongs to the kernel family);
  build cases `*_bidir`→`*_bidirectional`, `tm1_triton`/`tm2_triton`→`tm1`/`tm2`;
  implementation labels `triton_tri_attn*`/`triton_bias_attn`/`triton_aug_attn`/
  `aug_attn_memory_efficient` spelled out. The 120 committed result tables that carried
  an old name in their `run_name`/`target`/`implementation` columns were rewritten in
  place; **no measured value changed** (checked cell by cell), and the 40 plots whose
  drawn title named the old target were re-rendered from those same tables.
  `tests/cpu/layout/test_bench_target_vocabulary.py` now holds the four name spaces —
  bench.py's tables, the CLI's, `builder.CASE_NAMES`, and the directory tree — to
  each other.
- **Each bench target loads its own config.** `@hydra.main(config_path=...)` was the
  constant `../modules/triangle_multiplication/configs`, so every run — kernel, module,
  atom — loaded that one file and the other 25 `configs/bench.yaml` were read by
  nothing. They disagreed with what ran: `augmented_attention_atom` declares a 128–384
  ladder and was swept at 384–1024, while its own committed tables show 128/256/384.
  The path is now computed from the `target=`/`level=` overrides before hydra starts,
  every one of the 26 targets owns a `configs/bench.yaml` (the 17 kernel targets' are
  copies of the base they already loaded, so nothing they measure changed), and a
  target with no config is an error instead of a silent fall back to another target's
  ladders.
- **`bench_module all` means all of them.** The "all" group read a table that
  `triangle_multiplication_bidirectional` had never been added to, so it ran eight of
  the nine module targets. The bench-args table and the build-case table — keyed by the
  same names, maintained apart — are now one `MODULE_TARGETS`.
- **Coverage no longer guesses a target's directory.** `_report_coverage` rebuilt the
  path from `target in KERNEL_BUILD_CASES`, which was wrong for
  `augmented_attention_token`/`_atom`: they shared one directory named after neither, so
  the lookup missed and both targets' kernels were reported as never launched. Each
  target now owns exactly one directory and the path is derived from `level`.
- **`miniworld-engine build <typo>` fails immediately.** It used to resolve the config
  set and import every kernel — minutes of triton compilation — before saying "unknown
  case". `builder.CASE_NAMES` is a declared tuple (`test_case_names_are_declared` pins it
  to `cases()`), and the per-op name space is read straight from `registry.csv`, so both
  are checked before the first import.
- **Every model-level op is a folder.** `modules/__init__.py` has always opened with the
  rule; `attention_pair_bias.py`, `msa_pair_weighted_averaging.py` and
  `swa_atom_attention.py` were flat files. They are now packages like the other eight,
  and `modules/ops.py` — one level below `miniworld_engine.ops`, the public whole-op
  contract, and meaning the opposite thing — is `modules/functional.py`.
  `tests/cpu/layout/test_module_layout.py` holds the rule and the four shared modules that are
  legitimately flat.
- **The linter now checks what the code was already written against.** `select` was
  `["E", "F", "W"]` while the source carried ~700 `# noqa:` comments naming PLC0415,
  BLE001, ANN001, SLF001, S603, ARG005 and two dozen other rules that were not
  enabled — so none of them suppressed anything, and nothing said so. The set is now
  a deliberate one (import sorting, bugbear, pyupgrade, simplification, perf, pytest
  and logging idioms, RUF including RUF100) with every excluded family named and
  justified in `pyproject.toml`, and it runs clean. 264 dead directives were turned
  back into plain comments, keeping their reasons — ruff's own RUF100 fix deletes the
  whole comment, and the reason is the part worth having. Relative imports are banned
  outright (`ban-relative-imports = "all"`): 137 of them became absolute, which is the
  form that broke this session when a flat module became a package and `from .ops
  import sigmoid_gate` silently pointed somewhere else. `[tool.ruff] src = ["src"]`
  was missing, so ruff's own fix for those rewrote them to `src.miniworld_engine.*`.
  Two real defects surfaced: an `assert sig is None or True` that could never fail,
  and eleven late-bound loop variables captured by autotune lambdas in
  `autotune/build.py` (harmless today because each lambda is called in its own
  iteration, one refactor away from tuning every bucket against the last shape).
- **`tools/` is in both gates.** It was tracked code that `ruff check` and `ty check`
  never looked at, in the pixi tasks and in CI alike.
- **`ty` is a gate, not a report.** CI ran `ty check src tests || true` against a
  job that installed the package with `--no-deps`, so torch and triton were
  absent, every `import torch` was an unresolved import, and every torch
  attribute was `Unknown` — the step could not see the types it was checking. It
  now installs the CPU torch wheel plus `[dev]`, checks `src tests benchmarks`,
  and has no `|| true`. The `dev` extra names `ty` instead of `pyright`, which
  cannot parse jaxtyping shape strings and reported 144 parse errors and no real
  findings.
- **`pixi run test` had been dead since the checkout was renamed**
  (miniworld-kernels → miniworld-engine): the `pytest` launcher script carries an
  absolute shebang. The tasks use `python -m pytest`, and `pixi run ci` runs the
  type gate like the CI job does.
- Removed `autotune.elem_bucket_of`, a factory for the per-kernel `bucket_of`
  objects deleted in fcd3c7a. No caller, and the `_miniworld_keys` attribute it
  documented as "lets build.audit introspect the extractor" had no reader.
- **An autotune cache miss no longer sweeps the full grid.** Triton kernels now
  narrow a miss to `autotune_miss_cap` configs centred on `num_warps` in {4, 8}
  and `num_stages` in {2, 3, 4}; the shipped grid is 205,266 configs, so the old
  behaviour put a tuning sweep inside the first production forward on any GPU
  without a cache. A build (`run_autotune=True`) still gets the whole space.
  The warning text now names the fallback that actually happened instead of
  always claiming "the full autotune grid".
- **Autotune cache entries are keyed on the toolchain and the kernel source**,
  not only on the config grid: new `env_identity` (triton/torch/CUDA/ptxas) and
  `op_identity` (kernel source + `key=[...]`) fields. Entries written before
  these existed still read, so committed caches do not become permanent misses.
- **API framing**: `ops` is now the supported **consumer** surface; `kernels` is
  reframed as the **internal primitive** surface (per-GEMM/LN/gate/attention units)
  out of which the ops are built — still pinned (`_CONTRACT`) for internal stability
  but not intended for model code. `triangle_multiplicative_update` moved from
  `kernels` to `ops` accordingly.
- **Packaging**: runtime `[project.dependencies]` slimmed to the kernel core
  (`torch`, `triton`, `einops`, `jaxtyping`, `numpy`). Benchmark harness,
  comparison baselines, the CuTeDSL backend, and dev tooling moved to
  `[project.optional-dependencies]` extras (`bench`, `baselines`, `cute`,
  `dev`). Installing the package for `miniworld_engine.kernels` no longer pulls
  lightning / hydra / cuequivariance / scipy / cutlass.

- Moved 21 unreferenced dev/probe/experiment kernel files (perf probes,
  autotune/tile sweeps, micro-benchmarks, superseded variants) out of the
  importable `src/` package into `research/` (convention: `src/` ships only
  the canonical path). Verified 0 importers + contract & numerical suites green.

### Fixed
- **A `level=both` kernel keyed its cache on LENGTH, so its two sides shared a
  bucket.** A pair activation `(B, L, L, D)` at L=1024 and an atom activation
  `(B, A, D)` at A=1024 both have `shape[-2] == 1024`, so both landed in
  `shape_key=1024` — while the first launches 1,048,576 rows and the second
  launches 1,024. Visible in the shipped A6000 cache as two adjacent lines of one
  op: `shape_key=384 → 0.6103 ms` (pair, 147,456 rows) next to
  `shape_key=1024 → 0.0215 ms` (atom, 1,024 rows). Since the driver was corrected
  to build atom activations above L=512, the sweep measured only the atom side
  there and the module bench, which runs the pair side, was served those configs:
  transition on an A6000 at L=1024 in its committed baseline's exact configuration
  went from 5.498 ms to 9.504 ms while the pytorch reference reproduced within 3%.
  `both_key` now takes the ROW count (`shape_key.BOTH_ROWS`), a both-level kernel
  is two work lists with an explicit `side`, and `cache.KEY_SCHEME` invalidates the
  entries the change re-based — level-aware, so the 68 token/atom kernels whose
  buckets did not move keep theirs.
- **The `compiled` column of every benchmark CSV recorded the request, not what
  ran.** Four of the eight module benches guard `model.compile()` with
  `and conf.cudagraph == "disabled"`, so `compile=true cudagraph=manual` runs eager
  for those; `actual_compiled_flag` knew about `transition` by name and missed the
  other three. 330 committed tables say `compiled=True, cudagraph=manual`, and for
  triangle_multiplication, triangle_attention and bias_only_attention that is a
  measurement of eager code.
- **42 kernel-bench rows could not run at all** — the harness pre-flattened
  activations that seven entry points read their cache key off, so `triton_tm1`,
  `triton_tm2`, `triton_atomic/partial/persistent`, `triton_cond_transition` and
  `layernorm_linear_triton` came back `failed: shape ... is already flattened`.
- **The library needed an environment variable to work at all.** With
  `MINIWORLD_CONFIG_DIR` unset, every op registered an empty config list, triton
  substituted its own `Config({})`, and the first launch of every triton kernel
  died with `TypeError: dynamic_func() missing 2 required positional arguments`.
  Every sbatch script and bench entry point in the repo exports the variable, so
  the failure only ever showed up outside the scaffolding — measured on an A6000
  with it unset, `miniworld` was a `status=failed` row next to a healthy
  `pytorch` one; it is now 1.795 ms against pytorch's 7.084 ms.
  `autotune.configs.default_config_dir()` selects `grid` when nothing is set.
- **The `pre_hook` timer never ran.** `_install_launch_probes` assigned
  `Autotuner.pre_hook`, but triton sets that attribute per instance in
  `__init__` and the class carries no default, so the `hasattr` guard never
  opened and an instance would have shadowed the class attribute anyway. The
  build report printed `pre_hook 0 x -> 0s` for the whole life of the feature.
- **`bench_triangle_attention` crashed on the path that guards it.** Its
  unsupported-`old_triton` branch returned `BenchResult(status=..., error=...)`;
  `BenchResult` is a NamedTuple with neither field, so the guard raised
  `TypeError` instead of the NaN row every other bench returns.
- **A `str` appended to quack's `EXTRA_SOURCE_DIRS: list[Path]`.** The
  membership guard therefore never matched, and every import of
  `kernels/_quack_compat.py` added another copy of the same directory to the
  list the cute JIT cache key hashes.
- **`modules.primitives` imported scipy at module scope** for one weight
  initializer. scipy is in the `baselines` extra, not the core, so a core-only
  install could not import `miniworld_engine.modules.primitives` — or anything
  built on it.
- **`layernorm_linear_pytorch` declared `ln_bias: torch.Tensor`** while both of
  its callers pass `None` for a LayerNorm without beta, which `F.layer_norm`
  accepts.
- **`audit` failed the one kernel whose autotune key correctly carries no
  shape.** `transition_fold_triton` reads only the weights, so `N` and `K` are
  its whole shape and one cache bucket is right; the builder already knew and
  drove it at one length, while the audit reported `key is pinned` against a
  correct build. Both now answer through `builder._keys_on_shape`.
- **`kernels.cuda_transition` has never worked and now says so.** Its body
  deferred to `transition.cuda.cuda_transition`, a symbol with no history in
  this repo; the module binds only `cuda_transition_b2b` and
  `cuda_transition_expand_gate`. `Transition(implementation="cuda")` resolves to
  `KernelBackend.CUDA` and calls it in `forward`, so that path raised
  `ImportError` from four frames down. It now raises `NotImplementedError`
  naming what does exist. The name stays (the surface is frozen) but the
  CUDA backend for `Transition` should be considered unavailable.
- **`miniworld-engine build all` works as documented.** Its config-set argument
  defaulted to the string `"default"` and `configs/default` has never existed,
  so the command failed at argument parsing; `bench_module` hardcoded the same
  string. Both now default to the full search grid.
- **The per-op sweep never merged its shards.** `build all` ran to completion,
  wrote its shards, and left `data/` untouched — the build reported success and
  shipped nothing. It also refused to merge at all if any single unit failed,
  discarding every good measurement in the run.
- **`build all` never drove fp32.** The work list ignored `registry.csv`'s
  `dtypes` column and emitted bfloat16 for every unit, so the fp32 half of 66
  kernels was never tuned — and the coverage check counted `(op, bucket)` with
  no dtype axis, so it reported full coverage over a half-built cache.
- **`level=both` kernels built pair activations at atom shapes.** 18 kernels
  asked for `M = L*L` where production hands `M = A` — 67,108,864 rows at
  L=8192 — which is the whole explanation for the sweep's CUDA OOMs and one
  int32 offset overflow, and it poisoned those kernels' atom-bucket entries.
- **Shard writes are atomic.** A bare `write_text` let two workers interleave on
  one shard and produce unparseable JSON, which the merge then dropped silently,
  losing a whole unit's measurements with nothing said.
- Repo-wide int64 promotion of Triton `program_id` / loop-index pointer offsets
  (raw-indexing kernels), fixing an illegal-memory-access at large `L`
  (`augmented_attention` atom training, L≥768). `make_block_ptr` kernels keep
  int32 block offsets (Triton constraint); CUTE int32 (tmem addr, RNG seed)
  left as-is.

## [0.1.0]
- Initial consolidation of AF3-style op kernels (triangle multiplication,
  transition, triangle/bias/augmented attention, layernorm, adaLN) with
  Triton / CuTeDSL / CUDA backends.

### Checkpoint shape coverage and composite ops (2026-09-15)

- Add public `ops.gated_linear`, `ops.swiglu_ffn`, and `ops.rms_norm_modulation` over existing autograd kernels.
- Drive actual AF3, Protenix v1/v2, OpenDDE and ESMFold2 dimensions from the module build registry; include FP32 norm affine parameters, expansion ratios and projected attention head layouts.
- Handle absent LayerNorm affine tensors on the CUDA path without a mixed-dtype PyTorch fallback.
- Record constructor provenance and unsupported asymmetric TriMul/local attention in `docs/checkpoint-shapes-20260915.md`.
