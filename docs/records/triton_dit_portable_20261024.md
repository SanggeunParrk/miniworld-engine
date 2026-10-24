# Portable Triton paths for the diffusion transformers (2026-10-24)

Branch `feat/triton-gaps-b`. Two gaps of the cards that are not H100 / B200 (A100, A5000, A6000, ...), where `miniworld` has no CUDA path:

1. **`LocalDiTBlock`'s attention core** (AF3, OpenFold3, Protenix, OpenDDE atom blocks and the ESMFold2-style atom block): 32 queries x 128
   (192) keys per window with a per-window pair bias, per-window key gather, masks and an fp32 einsum. Only its AdaLN and conditioned
   transition were Triton.
2. **`DiTVariantStack` / `AtomDiTVariantStack`** (Boltz-2, RoseTTAFold3, Chai-1 token and atom diffusion transformers): CUDA-only
   (sm90 hard-coded in `serves`); everywhere else the caller's dense PyTorch path.

Everything below was run on **one H100 80GB HBM3** (torch 2.13 + cu129, triton 3.7) with the A/B switch `settings.force_triton_kernels`
(`implementation=triton` rows of `bench.py`). **No other card was run.** The kernels have no manifest rows for A100 / A5000 / A6000, so by the
routing rule (`docs/dispatch.md`) those cards still warn (`UnrecordedCardWarning`) and run PyTorch for the atom paths until
`python -m miniworld_engine.autotune.run_all` has been run on them; the token variant stacks route to Triton there at once, because the token
attention kernels and `layernorm_linear_fwd_triton` they launch already have `ok` rows in those manifests (old rows: nothing was run for this
change on those cards).

## What was built

| piece | file | notes |
|---|---|---|
| window attention, forward | `kernels/augmented_attention/triton/window.py` `_window_fwd` | one program per (window, batch x head, sample); keys in `BLOCK_N` chunks with an online softmax; `starts[b, w]` (first key of the window, any value) + key mask + optional `window_half` + optional `uid` decide liveness in-kernel, so the slid window (AF3), the clipped window (Protenix / OpenFold), ESMFold2's 192 keys with the +/-64 rank window and Chai-1's same-conformer mask are one kernel; a row with no live key gives 0 (the reference's `nan_to_num`); bias read through its strides (a permuted view of the projection works) |
| window attention, backward | `_window_bwd`, `_window_fold` | per window: `dq`, `dbias` (= dscores, summed over the A samples in strided groups of fixed order) and the window's partial `dk` / `dv`; then each key sums the partials of the windows that cover it in a fixed order. **No atomics: bitwise reproducible** (tested 3x). `fold_range` finds the covering windows from `starts` (also for the slid windows of a short system and the many windows that share the last start) |
| entry points | `window_attention`, `WindowAttentionFunction`, `plan(rule, n, nwin, keys, keep)` | differentiable in q, k, v and the bias |
| reference | `kernels/augmented_attention/reference.py` `window_attention_pytorch` / `window_logits_pytorch` | the fp32 definition the checkers and tests use |
| local DiT block | `modules/local_dit/module.py` | `attention_delta` takes the kernel when `_window_triton` says so (`serves_portable(single)` is the public question); AdaLN, projections, gates, rope / qk norm stay the module's; the `pytorch` core is kept as the `else` branch |
| inference pair bias | `kernels/layernorm_linear/triton/fused.py` `layernorm_linear_triton_fwd_t` | LayerNorm of the pair + the projection to the heads in one pass, stored `[heads, rows]` (the layout the attention kernels read). torch's `layer_norm` over 16-wide rows takes 1.2 ms for an 8k-atom pair, this 0.075 ms |
| variant stacks | `kernels/dit_variants/triton/portable.py` | the reference's block structure with the cores on the Triton kernels (token stacks: `augmented_attention/triton/main.py`, atom stacks: the window kernel); the pair LayerNorm is done once per stack, not per block (a call that needs no gradient: fused producer above, else one `layer_norm` + one GEMM for all blocks with the LN weight folded into the projection); differentiable by autograd |
| entry points | `modules.dit_variant.serves_portable(stack, *inputs)`, `portable.serves` / `portable.atom_serves` | the module's `forward` calls them after the H100 path |
| routing | `autotune/records.py` `OP_KERNELS`: `local_dit`, `dit_variant`, `dit_variant_atom`; `docs/dispatch.md` | the token attention kernels are now listed by name for `augmented_attention*` so the new window kernels are not held against the token ops |
| bench | `bench.py` `+dit_variant=boltz2 / rosettafold3 / chai1` on the `dit` and `local_dit` targets | |
| tests | `tests/gpu/any/test_window_attention_gpu.py`, `test_local_dit_triton_gpu.py`, `test_dit_variants_portable_gpu.py`; `tests/cpu/compile/test_dit_portable_routing.py` | |

Registry rows: `augmented_attention_{fwd,bwd,bwd_reduce}_window_triton` (drivers/checkers in `drivers|checks/augmented_attention.py`, configs
`default/` and `grid/`, `rtol` = 4x the error measured here, bf16 and fp32, aligned and ragged shapes).

## Accuracy (H100, against the fp32 / fp64 definitions)

| | bf16 | fp32 (TF32 tensor cores) |
|---|---|---|
| window kernel vs fp32 reference (max error / max value): out, dq, dk, dv, dbias | 2.3e-3 .. 5.4e-3 (aligned / ragged N) | 1.3e-3 .. 2.8e-3 |
| `LocalDiTBlock` (AF3, Protenix, ESMFold2, N = 384 .. 1024, masked / short systems) vs the fp64 module: output | 2.9e-3 (the PyTorch module's own bf16 run: 2.9e-3); ESMFold2 5.3e-3 (5.3e-3) | 1.4e-5 .. 1.1e-4 |
| same, input and parameter gradients | at most 1.15x the PyTorch module's own bf16 error (worst parameter 1.9e-2 against 1.6e-2, ESMFold2) | at most 2.3e-3 |
| variant stacks vs the fp64 reference, output and every input / parameter gradient | held to 2x the PyTorch stack's own error in the test; `bench.py` output error against the fp32 IEEE reference: 0.006 .. 0.014 | 6e-3 floor (TF32) |

The Triton window / token attention in fp32 runs on TF32 tensor cores (`tl.dot`), like the engine's other fp32 kernels.

## Speed on H100 (CUDA graph or compiled, `benchmarks/runners/bench.py`)

`pytorch` = the PyTorch reference, compiled; `triton` = the portable path under `force_triton_kernels`. bf16-mixed (fp32 master parameters, bf16
activations) unless noted; time per call of `blocks` blocks, `samples` = the harness' default (5 for inference, 48 for training); training =
forward + backward, `cudagraph=disabled` rows (both columns compiled; the `pytorch` row cannot pass the harness' repeatability check under a
manual graph), the graphed `triton` time is within 4% of it. Atom length = 8 x the token L of the column `L`.

| block | mode | atoms / tokens | blocks | pytorch ms | triton ms | speed-up |
|---|---|---|---|---|---|---|
| AF3 atom block (`local_dit`) | inference | 4,096 | 1 | 0.378 | 0.217 | 1.74x |
| | inference | 8,192 | 1 | 0.672 | 0.347 | 1.94x |
| | inference | 16,384 | 1 | 1.333 | 0.654 | 2.04x |
| | inference | 32,768 | 1 | 2.617 | 1.299 | 2.01x |
| | inference, fp32 | 8,192 | 1 | 1.022 | 0.704 | 1.45x |
| | inference, fp32 | 16,384 | 1 | 1.914 | 1.318 | 1.45x |
| | training | 4,096 | 1 | 8.80 | 5.74 | 1.53x |
| | training | 8,192 | 1 | 17.10 | 11.20 | 1.53x |
| | training | 16,384 | 1 | 34.43 | 22.21 | 1.55x |
| | training, fp32 | 8,192 | 1 | 24.59 | 20.76 | 1.18x |
| Boltz-2 token stack (`dit`, 768 wide, 16 heads) | inference | 384 | 4 | 1.107 | 0.748 | 1.48x |
| | inference | 768 | 4 | 2.519 | 1.328 | 1.90x |
| | training | 384 | 4 | 22.54 | 15.52 | 1.45x |
| | training | 768 | 4 | 65.51 | 38.02 | 1.72x |
| RoseTTAFold3 token stack | inference | 384 / 768 | 4 | 1.051 / 2.404 | 0.694 / 1.229 | 1.52x / 1.96x |
| | training | 384 / 768 | 4 | 21.12 / 63.45 | 14.38 / 35.90 | 1.47x / 1.77x |
| Chai-1 token stack | inference | 384 / 768 | 4 | 1.068 / 2.727 | 0.635 / 1.259 | 1.68x / 2.17x |
| | training | 384 / 768 | 4 | 20.09 / 61.72 | 12.51 / 32.88 | 1.61x / 1.88x |
| Boltz-2 atom stack | inference | 4,096 / 8,192 | 3 | 0.975 / 1.593 | 0.529 / 0.852 | 1.84x / 1.87x |
| | training | 4,096 / 8,192 | 3 | 28.16 / 54.58 | 13.47 / 25.47 | 2.09x / 2.14x |
| RoseTTAFold3 atom stack | inference | 4,096 / 8,192 | 3 | 0.988 / 1.602 | 0.517 / 0.824 | 1.91x / 1.94x |
| | training | 4,096 / 8,192 | 3 | 27.52 / 53.33 | 12.89 / 24.20 | 2.14x / 2.20x |
| Chai-1 atom stack | inference | 4,096 / 8,192 | 3 | 0.857 / 1.432 | 0.436 / 0.711 | 1.96x / 2.01x |
| | training | 4,096 / 8,192 | 3 | 24.25 / 47.44 | 11.05 / 20.97 | 2.19x / 2.26x |

ESMFold2's atom block (`LocalDiTBlock(**ESMFOLD2)`: RMS / SiLU conditioning, q / k RMSNorm + rotary, 192 keys, rank window) has no `bench.py`
target; a probe outside the checkout (bf16 parameters and inputs, one block, CUDA-graph timing, both columns `torch.compile`d) gives:

| mode | samples | atoms | pytorch (compiled) ms | triton core (compiled) ms | speed-up |
|---|---|---|---|---|---|
| inference | 1 | 4,096 | 0.201 | 0.123 | 1.63x |
| inference | 1 | 8,192 | 0.328 | 0.172 | 1.91x |
| inference | 1 | 16,384 | 0.578 | 0.278 | 2.08x |
| inference | 5 | 8,192 | 0.975 | 0.412 | 2.37x |
| training | 4 | 4,096 | 1.500 | 0.953 | 1.57x |
| training | 8 | 4,096 | 2.513 | 1.480 | 1.70x |

Without `torch.compile` around the block the Triton path is *not* faster than compiled PyTorch for ESMFold2 (0.30 ms against 0.20 ms inference,
3.3 ms against 1.5 ms training at 4,096 atoms): the block's RMS / SiLU / rotary rows are torch ops there and only fuse under the compiler.
Against the PyTorch path executed eagerly in a CUDA graph (what a caller without a compiler runs) it is 5x faster (1.50 ms).

The attention core alone (H = 4 heads x 32, 128 keys, slid window; CUDA graph; `torch` = the module's gather + fp32 einsum, `compiled` =
`torch.compile` of it), ms:

| dtype | samples | atoms | triton fwd | torch fwd | triton fwd+bwd | torch fwd+bwd | compiled fwd+bwd |
|---|---|---|---|---|---|---|---|
| bf16 | 1 | 2,048 | 0.030 | 0.124 | 0.061 | 0.333 | 0.104 |
| bf16 | 1 | 8,192 | 0.038 | 0.295 | 0.138 | 0.727 | 0.289 |
| bf16 | 1 | 32,768 | 0.077 | 1.122 | 0.438 | 2.558 | 1.026 |
| bf16 | 8 | 8,192 | 0.096 | 2.068 | 0.714 | 4.685 | 2.059 |
| bf16 | 24 | 4,096 | 0.124 | 3.001 | 0.788 | 6.838 | 2.938 |
| fp32 | 1 | 8,192 | 0.046 | 0.253 | 0.180 | 0.658 | 0.310 |
| fp32 | 1 | 32,768 | 0.109 | 0.941 | 0.562 | 2.273 | 1.107 |
| fp32 | 8 | 8,192 | 0.156 | 1.799 | 0.823 | 4.218 | 2.122 |
| fp32 | 24 | 4,096 | 0.182 | 2.610 | 1.139 | 6.153 | 3.018 |

Reading: the core is 4-24x faster forward and 2.3-3x faster forward + backward than the compiled PyTorch core; the block gains less (1.5-2.0x)
because the rest of an atom block (AdaLN, projections, the SwiGLU transition) is unchanged and was already about half of it. The fp32
training block gains least (1.18x): its dot products run on TF32 tensor cores, but the gradient partials are fp32 and the projections dominate.
The token variant stacks gain 1.5-2.2x from the attention core (no `[A, H, L, L]` fp32 scores) and the once-per-stack pair normalisation; their
GEMMs and rows are cuBLAS / torch as before. **None of this competes with the H100 CUDA paths** (`kernels/dit_variants/interface.py`: token
stacks 2.8-4.1x the compiled torch statements, atom stacks 1.6x; AF3 atom block 3.9x): Triton on H100 is an A/B switch, these numbers say how
much a card without CUDA paths gains over its PyTorch fallback.

## Entry points for the caller (FoldForge) on cards other than H100 / B200

Nothing new has to be constructed; the modules route themselves, and a `serve_*` seam only has to ask whether the portable path will run.

* AF3-style atom blocks (AF3, OpenFold3, Protenix, OpenDDE) and ESMFold2's: build `LocalDiTBlock(..., implementation=ImplementationType.MINIWORLD)` as
  before and call `block(single, cond, pair, mask[, rope=(cos, sin)])` (the trunked pair `[B, nwin, 32, keys, d_pair]`, inputs bf16 / fp16 / fp32;
  training works: the kernel is differentiable). `block.serves_portable(single)` is True when the attention core runs on the Triton kernel
  (recorded card or `settings.force_triton_kernels`; it may warn once).
* Boltz-2 / RoseTTAFold3 / Chai-1: `DiTVariantStack` / `AtomDiTVariantStack` as before; `stack(single, cond, pair, mask)` (token),
  `stack(single, cond, pair, mask, uid)` (atom) route H100 CUDA -> portable Triton -> reference themselves.
  `modules.dit_variant.serves_portable(stack, *those_inputs)` is the question to add to `variant_serves` after `interface.serves` /
  `interface.atom_serves` (which stay H100-only). The Triton path keeps no cache keyed on tensor versions (the H100 path does).
* Not recorded: a call on an unrecorded card runs the engine's PyTorch reference (the stack's `reference.py`, `LocalDiTBlock`'s torch core)
  after one `UnrecordedCardWarning`, which is the same function as FoldForge's dense path.
* To make the atom paths take Triton on A100 / A5000 / A6000: `python -m miniworld_engine.autotune.run_all` on the card and commit its manifest.

## Limits and what is left

* Validated on H100 only; the window kernels have `ok` rows nowhere (the H100 manifest was not edited: running `run_all` here wrote them
  as `ok`, with a dirty-tree provenance, and that file was restored). Tiles / stage counts of the default config set were not tuned on any card
  (`autotune/data` has no entry); on the H100 the autotuner picks among 8 configs on first use of a shape bucket.
* head width up to 64 and 256 keys (refusal: one `KernelRefusedWarning`, then the reference); N must be a multiple of 32 for the variant atom
  stacks (as before); the real atoms must come first when `mask` is given (as everywhere in the local DiT).
* The window backward keeps fp32 partial `dk` / `dv` of `[A, B, nwin, keys, H, D]` (4x the keys' size, fp32); a key-centric backward would not
  need them. The fp32 training block gains only 1.18x.
* ESMFold2 is faster than compiled PyTorch only if the block is compiled (see above); RMS / SiLU / rotary rows were not ported.
* Not done: Triton versions of the variant stacks' AdaLN rows / GLU (torch ops in the parameters' dtype, as the reference), a fused
  Triton pair-bias producer for the gradient case (it uses torch's `layer_norm`, one pass per stack); fp16 is accepted by the guards but
  was not run (bf16 / fp32 only).
