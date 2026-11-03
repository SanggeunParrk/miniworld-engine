# Released-checkpoint parity: options and the weight adapters

The engine owns the optimized modules AND their PyTorch reference path; a model repository (FoldForge, MiniWorld) holds model, loading and
IO code only. For the released AF3-family checkpoints (AlphaFold 3, Protenix, Boltz-1/2, Chai-1, RoseTTAFold3, ...) that needs the engine
modules to (1) compute the released equations, (2) take the released parameter layout. This page lists what exists. The oracle is
FoldForge's dense modules (`foldforge.modules.dense.*`, the xfold / AlphaFold 3 equations); `tests/cpu/kernels/test_released_parity.py` builds
both sides from the same random weights and compares (needs `FOLDFORGE_SRC`; the engine does not depend on FoldForge).

Rule for every option: the default is the old behaviour; the PYTORCH implementation computes the option exactly; the MINIWORLD
implementation either has a kernel for it or REFUSES it (one `KernelRefusedWarning` on a GPU, then the PyTorch statements for that call).
An option is never ignored. H100 / B200 keep their rule (no Triton fallback; the PyTorch statements run).

| Option | Module | Meaning | PYTORCH | MINIWORLD |
|---|---|---|---|---|
| `divide_by_length=True` | `TriangleMultiplication`, `TriangleMultiplicationReference`, `PairformerConfig.divide_by_length` | RoseTTAFold3: the contraction is multiplied by `1 / (unmasked residues of the sample)` (`L` without a mask) before `ln_out`. `modules/triangle_multiplication/reference.py::inverse_length` | exact | refused (all CUDA / Triton / cuEquivariance / Anthropic paths), PyTorch statements; trains |
| `count_offset=1e-3` | `OuterProductMean` | AlphaFold 3 / Protenix divisor `count + 1e-3` instead of `clamp(count, 1)`. AF3 = `count_offset=1e-3, normalize_before_proj=False` (`(W out + b) / (count + 1e-3)`); `normalize_before_proj=True` is the `bias_after_norm` form. Not with `groups > 1` (ValueError) | exact | H100 hand-CUDA (`opm_train`, `composed`; inference and training, bf16, batch 1, `opm_train.SHAPES`): the divisor is two kernel scalars `max(count, dmin) + doff`; elsewhere refused, PyTorch statements |
| `divide_by_length=True` | `OuterProductMean` | RoseTTAFold3: every pair is divided by the MSA depth S (`msa.shape[1]`, masked rows included) instead of its own count; excludes `count_offset`, not with `groups > 1` | exact | as `count_offset`: kernel scalars (S, 0) |
| `transposed_bias=False` | `TriangleAttention`, `PairformerConfig.transposed_bias` | AF3's ending node projects the pair bias from the pair as it is (already in 2.4.2) | exact | H100 sm90 kernels (inference), composed legs (training) |
| `gate_bias` / `out_bias` / `untransposed_output` / `dual_output` | `TriangleAttention`, `PairformerConfig.*` | RoseTTAFold3 gate / out-projection biases, Chai-1 output forms (already in 2.4.2) | exact | as before |
| `PairformerConfig(d_single=..., n_head_single, single_gate_bias, pair_transition_n, single_transition_n)` | `PairformerBlock`, `Pairformer` | the single track (AF3 Alg. 17: `AttentionPairBias` over the pair logits, single `Transition`); `forward(pair, mask, single)` returns `(pair, single)` | exact | each sub-module routes as before |
| `PairformerConfig(parallel=True)` | `PairformerBlock` | every update of a track reads the representation ENTERING the block, the deltas are summed in (the single's pair logits read the entering pair). A delta is `module(x) - x` (the modules own their residual): exact in fp32, one extra rounding of `x + delta` in bf16 | exact | as above |

Not carried by the block: the structural-token refiner's `extra_pair_bias` (needs an additive bias on `AttentionPairBias`).

## Weight adapters (`miniworld_engine.modules.released`)

`<module>.from_released(state_dict, prefix=..., dtype=None, norm_eps=1e-5, **options)` builds the engine module from a released module's
`state_dict` (the FoldForge dense names, listed at the top of `released.py`) and loads it STRICTLY; `released.<module>_from_released(...)`
returns just the engine-named tensors (pure reshapes, CPU-safe). Sizes, biases and the layouts follow from the weights; what they do not say
(`n_head`, `starting`, `outgoing`, the ending node's bias orientation, `bias_after_norm` / `clamped_norm`, `parallel`) is an argument.

* interleaved projection: `projection` / `gate` are one `[2 hidden, c]` matrix, row `2k` = branch a of channel k, row `2k+1` = branch b; the engine takes
  `to_left`/`to_right` (+ gates); an incoming module takes branch b as left (the released incoming contraction is `a[k, j] b[k, i]`).
* single `transition1`: `[2 n c, c]`, `chunk(2)` -> `expand_a`, `expand_b`; `n = transition2.in_features / c`.
* OPM `output_w` `[K, K, out]` (`[G, K, K, out]`) -> `to_out.weight` `[out, G K K]`; the AF3 divisor is the default of `OuterProductMean.from_released`
  (`bias_after_norm=False, clamped_norm=False`).
* Pairformer block: `PairformerBlock.from_released(state, config=PairformerConfig(...))`.
* Token / atom DiT variant stacks: `DiTVariantStack.from_released` / `AtomDiTVariantStack.from_released` (below).

## DiT variant stacks (AlphaFold 3, Boltz-2, RoseTTAFold3, Chai-1) and `serving`

`DiTVariantStack` / `AtomDiTVariantStack` with `ImplementationType.PYTORCH` run `kernels/dit_variants/reference.py` (differentiable torch
statements, any card, inference and training); `serves_portable` is the Triton-only question and is False for PYTORCH. `serving(stack, *inputs)`
(also `stack.serving(...)`) answers the question a caller that counts engine-served calls needs: `"cuda"`, `"portable"` or `"pytorch"`, and a
PYTORCH stack is always `"pytorch"` (served, not refused).

**AlphaFold 3's own stacks are configurations too**: `DiTVariantStack(n, **AF3)` (sequential, sigmoid AdaLN, gating query, plain SwiGLU transition,
d_cond 384, pair 128, ONE scale-only pair LayerNorm, one projection matrix per super block) and `AtomDiTVariantStack(n, **AF3_ATOM)`
(`key_window="slide"`, shared pair LayerNorm + `Linear(d_pair, 4 n_blocks)`) are the Boltz-2 rows with `up_gate` off; `from_released` takes
FoldForge's `DiffusionTransformer` / `DiffusionCrossAttTransformer` names unchanged (`super_block_size` splits the shared projection). They run
on the same CUDA rows / cores (inference bf16 and fp32, training bf16 and fp32); every flag is a runtime argument of the existing kernels,
nothing was added. Numerics are pinned against the frozen oracle `tests/oracles/af3_dit_oracle.pt` (made once by
`tests/oracles/make_af3_dit_oracle.py` from FoldForge's dense classes; the tests never import FoldForge), forward and the input gradients, in
`tests/cpu/kernels/test_af3_dit_oracle.py`; the H100 rows are in `tests/gpu/sm90/test_dit_variant_af3_h100_gpu.py`.
What the dense classes need beyond that is not a stack option: the unconditioned `AdaptiveLayerNorm` / `DiffusionTransition` of the diffusion
head's pair and single conditioning are a plain affine LayerNorm + SwiGLU (the engine's `Transition` / `swiglu_ffn`), and the dense atom stack
takes the queries / keys layout (`GatherInfo`) where the engine takes `[A, N, 128]` with the trunked pair (the caller's layout conversion).

Still open: ESMFold2's rotary / `extra_pair_bias` DiT and atom stack (RMS / SiLU AdaLN, raw gates, rank-based window, `LocalDiTBlock(**ESMFOLD2)`)
are not `DiTVariantStack` configurations; the structural-token refiner's `extra_pair_bias` has no slot either.
