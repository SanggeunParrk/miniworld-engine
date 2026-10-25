# Module-level parity of the portable kernels with the H100 CUDA paths (2026-10-25)

Five gaps between what the H100 CUDA paths serve and what any other card got through `miniworld` (a PyTorch fallback, a `ValueError`, or a
refusal), closed at the module / guard level. Nothing here touches an H100 / B200 CUDA path; the portable routes are reached on every other
card by the rule of [docs/dispatch.md](../dispatch.md) and, for the A/B runs below, on the H100 with `settings.force_portable_kernels`
(`tests/portable_helpers.py` is the one place the tests name the switch). Base: `origin/main` 195d968c (2.3.12 + the pair-embedding and smooth-lDDT
routes). **No card other than the H100 was available**: the Ampere manifests hold no rows for the kernels below, so by the rule those cards
still warn and run PyTorch until `python -m miniworld_engine.autotune.run_all` has been run there.

| # | gap | what was built |
|---|---|---|
| 1 | bias-only DiT: every part pinned to PyTorch off H100 / B200 | the block builds AdaptiveLayerNorm, ConditionedTransition and the pair LayerNorm with its own implementation: they route per card (portable kernels on a recorded card, PyTorch + one `UnrecordedCardWarning` otherwise, the PyTorch leg on H100 / B200). Forward and backward, bf16 and fp32 (TF32), one or several augmentations, a pair of its own per sample (psb), the hoisted `pair_bias_all` bias, key masks. The attention core stays the batched einsum (see the speed note) |
| 2 | `DiTBlock(extra_pair_bias=)` raised `ValueError` off H100 bf16 inference | `AugmentedAttentionPairBias(extra_pair_bias=)` adds the `[L, L]` term to the attention bias (every head, before the key mask, in the bias dtype, differentiable); the Hopper bf16 path still folds it into its hoisted bias, the fused training blocks are skipped when it is given |
| 3 | Transition fp32 went to PyTorch on every card | the portable residual path takes fp32 on TF32 tensor cores wherever torch allows TF32 (`set_float32_matmul_precision` not `"highest"` / `allow_tf32`, the switch the H100 generic kernels follow; full-precision fp32 keeps PyTorch). **Precision chosen: `tl.dot(input_precision="tf32")` for fp32 operands when torch allows TF32, `"ieee"` otherwise (`kernels/_dot_precision.py`; bf16 ignores it)**, so kernel callers that never turned TF32 on keep their IEEE results. A card needs fp32 rows for what the call launches (`dispatch.dtype_gate`) |
| 4 | TriAttn variants refused fp32 master parameters | `refusal()` takes bf16 or fp32 parameters under bf16 activations; the variant casts the projections to the activation dtype inside (autograd casts), gradients come back in the master dtype |
| 5 | TriMul `d_hidden != d_pair` refused | the portable front / back already took the widths from the weights; the guard and the `ValueError` were the only blockers. Served for pair <= 256, hidden <= 512, both multiples of 16 (`trimul_inproj.triton.contract.hidden_refusal`), unidirectional (outgoing / incoming) and bidirectional, inference and training, masks and a drop_row scale included; outside the box a `KernelRefusedWarning` and PyTorch, as before |

## Parity

Against the module's fp64 PyTorch statements (relative L2 error, worst case over the cases of a row). The bf16 engine error is held to the bf16
PyTorch statements' own (x 1.5, floor 3e-3); fp32 engine = TF32 tensor cores in the portable kernels, held to 6e-3 (PyTorch fp32 is full
precision, hence the 1e-7 column). `bwd` = the worst gradient (activations, conditioning, pair, every parameter). 165 cases in all (2 skipped), on the rebased
tree: `tests/gpu/any/test_{bias_only_dit,dit_extra_pair_bias,transition_fp32,triattn_variants_master,trimul_hidden_ne_pair}_portable_gpu.py`.

| op | dtype | mode | cases | fwd (engine / PyTorch) | bwd (engine / PyTorch) |
|---|---|---|---|---|---|
| bias-only DiT block (7 widths / layouts, A = 1..4, psb, hoisted bias) | bf16 | infer | 11 | 3.8e-03 / 4.3e-03 | - |
| | bf16 | train | 11 | 4.0e-03 / 4.3e-03 | 1.2e-02 / 1.1e-02 |
| | fp32 | infer | 11 | 1.4e-03 / 3.6e-07 | - |
| | fp32 | train | 11 | 9.5e-04 / 3.6e-07 | 2.3e-03 / 1.0e-06 |
| DiTBlock extra_pair_bias | bf16 | infer | 3 | 4.1e-03 / 4.7e-03 | - |
| | bf16 | train | 3 | 4.4e-03 / 4.7e-03 | 1.4e-02 / 1.8e-02 |
| | fp32 | infer | 3 | 1.5e-03 / 2.3e-07 | - |
| | fp32 | train | 3 | 9.6e-04 / 2.3e-07 | 2.3e-03 / 1.0e-06 |
| Transition fp32 (TF32; PyTorch column = its own TF32) | fp32 | infer | 5 | 6.3e-04 / 1.9e-04 | - |
| | fp32 | train | 5 | 6.3e-04 / 1.9e-04 | 1.9e-03 / 1.7e-03 |
| TriAttn variants, fp32 master under bf16 (PyTorch column = autocast) | bf16 act. | fwd + bwd | 12 | 2.9e-03 / 3.0e-03 | 9.8e-03 / 1.0e-02 |
| TriMul d_hidden != d_pair (uni out / in, bidir; 6 width pairs, corners 16..256 x 16..512) | bf16 | infer | 41 | 3.8e-03 / 4.3e-03 | - |
| | bf16 | train | 39 | 4.3e-03 / 4.8e-03 | 1.4e-02 / 1.9e-02 |

Also checked: the portable block against the **H100 CUDA block** (same call, both within the bf16 statements' distance of fp32), `extra_pair_bias`
against the **Hopper CUDA path**, the softmax shift property (a constant term changes nothing), the refusal outside the TriMul box, and
that fp32 Transition with TF32 off runs no portable kernel.

Registry rows now declaring fp32 (kernel bands = 4x the measured error against an fp32 reference run with TF32 off, H100, `MINIWORLD_DRIVER_DTYPE=fp32`):
`transition_expand_swiglu_triton` 1.65e-3 (band 7e-3), `transition_squeeze_residual_triton` 8.1e-4 (4e-3), `transition_bwd_swiglu_recompute_triton`
h 1.7e-3 / dAB 4.4e-3 (2e-2). `trimul_input_ln_residual_bwd_triton` (the input LayerNorm's backward with the residual gradient, level `token`) measured
4.1e-7 at fp32 but the registry allows bf16 only at the token level, so it stays bf16 and is **not** counted in the fp32 evidence: a card recorded
for fp32 vouches for four of the five kernels a Transition training call launches (`OP_KERNELS["transition"]` now also lists the two LayerNorm
kernels, so the bf16 denominator is 5 inference / 9 training kernels; the Ampere cards keep warning, they lacked `transition_squeeze_residual_triton`).

## Speed against PyTorch (H100, CUDA graphs, never eager)

PyTorch = the faster of (its ops in a graph, `torch.compile` in a graph); the portable column is the faster of (eager, compiled) in a graph. Times in
ms, forward (infer) or forward + backward (train), `bench` scripts in the session scratch (`parityC/scripts/bench*.py`). **These are Hopper
numbers: the portable kernels are deliberately not the H100 path (nothing is tuned for sm_90 and the CUDA paths exist), so they bound the composition, not
the Ampere cards the routes are for.**

| case | infer | train |
|---|---|---|
| TriMul pair 64 / hidden 128, outgoing, L 256 / 384 | 2.74x / 3.59x | 2.15x / 2.50x |
| TriMul pair 128 / hidden 256, outgoing, L 256 / 384 | 2.56x / 2.86x | 2.22x / 2.46x |
| TriMul bidirectional pair 64 / hidden 128, L 256 / 384 | 2.98x / 3.68x | 2.28x / 2.58x |
| TriMul bidirectional pair 128 / hidden 256, L 256 / 384 | 2.78x / 3.03x | 2.31x / 2.53x |
| Transition fp32 TF32, d128 n4, L 128 / 256 | 1.83x / 1.71x | 1.22x / 1.20x |
| Transition fp32 TF32, d64 n2 L256 / d256 n4 L128 | - | 1.29x / 1.11x |
| TriAttn variants, fp32 master (RF3 64/256, 128/128, Chai-1 256/256), L 128 / 256 | 1.5x - 1.9x / 1.8x - 2.4x | 1.3x - 1.5x / 1.5x - 1.8x |
| DiTBlock + extra_pair_bias fp32, A2 L256 | 1.43x | 1.13x |
| DiTBlock + extra_pair_bias bf16, A2 L256 / L512 | 0.95x / 1.05x | 0.97x / 0.98x |
| bias-only DiT block fp32, A4 L256 (A1 B4 psb train) | 1.61x | 1.05x (1.01x) |
| bias-only DiT block bf16, A4 L256 / L512 | 0.85x / 0.93x | 0.94x / 0.96x |
| bias-only DiT block bf16, psb (A1 B4) L256 | - | 1.03x |

**The bf16 DiT-block rows do not beat compiled PyTorch on the H100 (0.85x - 1.05x); everything else does.** A profile of the bias-only block (A4 L256, inference,
GPU time 137 us portable-compiled against 118 us PyTorch-compiled) shows why: the portable AdaLN and ConditionedTransition kernels cost 71 us there where
inductor + cuBLAS spend about 47 us (the portable expand / squeeze pair is 34 us against 13 us for two cuBLAS GEMMs), i.e. the loss is the untuned Hopper
build of two existing portable kernel families, not the composition (the attention core, the pair-bias path and the gating are at parity or better). Those families
are tuned on the Ampere cards (the committed autotune caches are the A5000 / A6000 ones), which could not be measured here.
The attention core itself stays the batched einsum on purpose: one softmax shared by every sample and a cuBLAS GEMM per head;
`kernels/bias_only_attention` (flash-style, built for the triangle rows) is recorded as losing to it on time on every committed card.

## H100 / B200 non-regression

The CUDA paths and routes of H100 / B200 are not edited. The module-level changes that run there are the `extra_pair_bias` argument (None by default),
the Transition fp32 branch (behind `x.dtype is float32 and backend == PORTABLE and not restricted`, so unreachable on H100 / B200), the bias-only block's
parts (built with `MINIWORLD`: on H100 / B200 they resolve to their PyTorch leg; the pair LayerNorm is the CUDA LayerNorm there, only in the composed
fallback, never in the fused block paths) and `AdaptiveLayerNorm` / `ConditionedTransition` taking their reference on a CPU tensor.

* official harness (`bench.py target=<op> level=module mode=training implementations=[miniworld]`, compiled + CUDA graph, D128 L384), two alternating runs each
  of `origin/main` and this branch: **Transition 0.528 / 0.533 ms against 0.527 / 0.520 ms; TriAttn 1.080 / 1.083 ms against 1.086 / 1.079 ms** (equal within the
  +-1 % run-to-run noise; the figures of the brief, 0.559 and 1.106, are other runs of the same cells).
* 705 passed / 37 skipped H100 tests, none failing, on the branch before the rebase and again after it (bias-only DiT fused + psb, token DiT layouts / psb /
  batched pairs, module wiring, Transition sm90a generic / fused / n2 / wide, TriAttn biases and variants, TriMul hidden 2x, determinism
  `test_determinism_gpu.py` (bitwise replay), conditioned-transition precision, the portable DiT variants).
* CPU suite: 4310 passed; the 2 failures + 2 errors that remain (`mpnn` DROPOUT one-sided flags, stale committed Ampere caches, the ladder test) fail
  identically on `origin/main`.

## Left

* fp32 Transition (and the bias-only DiT / `extra_pair_bias` parts) on the Ampere cards: needs `run_all` on the card (fp32 rows); until then the
  cards warn and run PyTorch.
* the bf16 portable DiT block against compiled PyTorch, measured on a card where the portable kernels are the tuned path (A100 / A6000).
* `trimul_input_ln_residual_bwd_triton` at fp32 (registry level `token` forbids it).
