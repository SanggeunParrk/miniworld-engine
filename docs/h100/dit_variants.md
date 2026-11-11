# DiT variants on H100 (sm90): Boltz-2, RoseTTAFold3, Chai-1

Scope: how the Boltz-2 / RoseTTAFold3 / Chai-1 DiT blocks reach the H100 kernels, and what is left of the `kernels/dit_variants`
family. There is no separate H100 variant algorithm any more:

- token blocks `TokenDiTBoltz2`, `TokenDiTRoseTTAFold3`, `TokenDiTChai1` (`modules/token_dit.py`, mixin `VariantTokenCuda`): the H100
  `"cuda"` backend is THE token DiT implementation, TokenDiTProtenix's `"cuda"` route with the model's options
  ([token_dit.md](token_dit.md#model-options-one-algorithm));
- atom blocks `AtomDiTBoltz2`, `AtomDiTRoseTTAFold3`, `AtomDiTChai1` (`modules/atom_dit.py`, `AtomDiT(model=...)` with `boltz2`,
  `rosettafold3`, `chai1`; mixin `VariantAtomCuda` in `modules/dit_variant_cuda.py`): on H100 `VariantAtomCuda._supports_cuda` /
  `_forward_cuda` call `_LocalAtomDiT._supports_cuda` / `_forward_cuda`, i.e. THE atom DiT implementation
  `kernels/integrations/local_dit_h100.py` (`run_block`) with the model's options, every key window. Its kernels are documented in
  [local_atom_dit.md](local_atom_dit.md); this page lists the options.

On H100 `kernels/dit_variants` contributes only `cuda/edge.cu` (the slid key window's edge windows, sm_90a, for every atom class). The
other files of the family serve A100 (below) or are host-side PyTorch / Triton code.

Backends: atom `AtomDiTBoltz2` `pytorch, anthropic, cuda, triton`; `AtomDiTRoseTTAFold3` / `AtomDiTChai1` `pytorch, cuda, triton`;
token `TokenDiTBoltz2` / `TokenDiTRoseTTAFold3` / `TokenDiTChai1` `pytorch, anthropic, cueq, cuda, triton`. Atom requirement rows:
`atom_single`, N 1024-8192, 128 wide x 4 heads, cond 128, pair 16, `key_window` `pad` (default) or `slide` (Chai-1's `slide` row also
`adaln_eps=1e-05`); dtypes `float32` and `master`; modes `eval` and `train`. Measurement and judging: [../measure.md](../measure.md);
conventions: [../modules.md](../modules.md), [../kernels.md](../kernels.md).

What differs from the AF3 blocks:

| class | AdaLN | attention | transition | schedule |
|---|---|---|---|---|
| Boltz-2 | sigmoid form | gated query; one AdaLN for q and k / v | up-gate `silu(W_a h) (W_b h) (W_u h)` | sequential |
| RoseTTAFold3 | sigmoid form | bias-free query, affine LayerNorm of the flat q / k over all heads | SwiGLU | parallel `a + dA(a) + dT(a)` |
| Chai-1 | identity `(1 + W_s s) LN(a) + W_b s`, eps 0.1 | no gating query, no out projection, same-conformer key mask (`uid`) | SwiGLU | parallel |

## Atom options of the one atom DiT implementation (`local_dit_h100.options`)

A model's differences are read off the block's own modules; its parameters are mapped onto the 27 canonical leaves
(`local_dit_h100._leaf`; a part the model lacks is a cached constant leaf). Build defines from `kernels/augmented_attention/cuda/sm90_atom`
(`_OPTION_KERNELS`):

| option | models | how |
|---|---|---|
| one AdaLN for q and k / v (`kvm` 2) | Boltz-2, RF3, Chai-1 | runtime mode of `pre90`; backward entry `pre_bwd90_sh`; the K / V AdaLN leaves are constant zeros |
| identity AdaLN (`IDENT`) | Chai-1 | conditioning tables of the raw c (ln_cond leaf 1, scale bias leaf 1), no sigmoid on the scales; `cond90`, `post_bwd90`, `pre_bwd90*`, `cond_bwd90*` builds |
| parallel block (`PARS`) | RF3, Chai-1 | the transition's AdaLN normalises the block input; `post90`, `post_bwd90` builds |
| no gating query, no out projection (`NOOUT`) | Chai-1 | gate leaf 0 (sigmoid = 0.5 exactly), out-projection leaf `2 I`: `u = o` exactly; `post90`, `post_bwd90` builds |
| flat q / k LayerNorm (`KQLN`) | RF3 | in `pre90`; backward `kqln_bwd90` (`augmented_attention_atom_kqln_bwd_sm90`) |
| bias-free query | RF3 | zero `to_query.bias` leaf |
| up-gate transition (`UP`) | Boltz-2 | `W_u` chunks beside `[W_a; W_b]` in `post90` / `tr_bwd90`, a 6-block dx2 in `post_bwd90` |
| conformer mask (`uid`) | Chai-1 | other-conformer keys `-inf` in the window pair bias and in the slid edge windows |
| AdaLN eps | Chai-1 (0.1) | the activations' AdaLN epsilon, one per block |

`options()` returns None (the call is refused) for mixed AdaLN forms or epsilons, a sigmoid-form conditioning LayerNorm eps != 1e-5,
a per-head q / k RMSNorm, an up-gate weight with a bias or not `[256, 128]`, flat q / k LayerNorms of different eps or without bias,
or a gating query without an out projection. The rest of the H100 range is `_LocalAtomDiT._supports_cuda`
([local_atom_dit.md](local_atom_dit.md#limits-_localatomdit_supports_cuda)).

## Edge kernels (`kernels/dit_variants/cuda/edge.cu`, extension `dit_variants_edge_sm90a`)

`key_window="slide"` / `"slide_qblock"`: the window attention reads the clipped window for every query window; windows 0, 1 and the
last three real ones are the only ones whose slid window differs. `local_dit_h100` picks those five on the device from the real atom
count (no host read, capturable) and redoes them over their own 128 keys with these kernels, inference and training, bf16 or fp32
operands, fp32 inside:

| kernel (`requirements_kernel.csv`) | role | what it does |
|---|---|---|
| `dit_variants_edge_bias_fwd_sm90` | forward | `LN(z_e) gamma . Wb` of the five windows, natural units, `-inf` off the valid keys (and other conformers); hoisted per pair in inference |
| `dit_variants_edge_attn_fwd_sm90` | forward | `softmax(q k^T / sqrt(32) + bias) v` of the five windows, overwriting the window kernel's rows |
| `dit_variants_edge_bwd_compute_sm90` | backward | before the window backward: the edge windows' dq / dk / dv into fp32 side buffers, their dbias, and their dO / D rows zeroed |
| `dit_variants_edge_bwd_apply_sm90` | backward | after it: the side buffers added into dq / dk / dv (one CTA per (sample, head, 8 columns), windows in order, deterministic) |
| `dit_variants_edge_bias_bwd_sm90` | backward | the edge windows' pair LayerNorm / projection backward into dz, d gamma_pair, d Wb |

The rows list all five atom classes (`AtomDiTProtenix` / `AtomDiTOpenFold3` take the same kernels for their slid windows).

## Other files of `kernels/dit_variants`

| file | serves |
|---|---|
| `cuda/train.py`, `cuda/train_rows.cu`, `cuda/sm80.py` | A100 only: every A100 call of the atom variants (and of OpenFold3's parallel K / V AdaLN, the slid windows and fp32 for the AF3 atom blocks), inference included, on the sm_80 builds (`dit_variants_*_sm80` rows) |
| `triton/portable.py` (+ `reference.py`, the PyTorch pieces it reuses) | a portable Triton path of the variant stacks; nothing else in `src/` imports `portable.py` |

The atom classes' `"triton"` backend is `_LocalAtomDiT`'s (the window core `kernels/augmented_attention/triton/window.py`), not
`triton/portable.py`.

## History

The H100 `kernels/dit_variants` row kernels (`interface.py` bf16 inference, `cuda/infer32.py` fp32 inference, `cuda/rows.cu`, the sm_90a
`train_rows.cu` build and the 17 `dit_variants_*_sm90` rows of those kernels) were deleted when every H100 atom block moved onto
`local_dit_h100` (commit 26e8277e0); the token host glue went with the one token DiT algorithm (ff36ec18f, f9c2c8d21). Old note:
`git show 65f37dc3c^:docs/gpus/h100/model_variants/model_variants.md` (its TriangleAttention, OuterProductMean, `AttentionPairBias(gate_bias)`
and parallel-Pairformer rows are not DiT modules). Historical (from the 2026-10-22 note, not re-measured under the current measurer;
measured on the deleted dit_variants path): token stacks 2.8-4.1x and atom stacks 1.6x against compiled torch (CUDA graph, L 384 / 768 and
8192 atoms).
