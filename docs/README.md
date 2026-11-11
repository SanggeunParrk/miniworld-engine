# Documentation

The engine in four parts. Every page describes the current code; where a page and the code disagree, the code is right.
Cards: H100 (sm_90a) and A100 (sm_80) are served, measured and judged. sm_100 (B200) sources exist in the tree but are not supported
(see [architecture.md](architecture.md#3-kernel-families-per-card)). Installation: [install.md](install.md).

| page | covers |
|---|---|
| [architecture.md](architecture.md) | the pipeline end to end, how a module reaches the card's kernels, the directory map |
| [requirements.md](requirements.md) | the two requirement tables, the derive rule (kernel range = its module rows), the accuracy rule |
| [modules.md](modules.md) | the `BackendModule` contract, the backends, selectors, class versions, master weights, `make_inputs` |
| [init.md](init.md) | parameter initialisation: the init vocabulary, presets (`engine`, `miniworld`, `af3`), the `init:` config block, `apply_init` / `describe_init` |
| [kernels.md](kernels.md) | kernel families, launch-config tables and shape keys, the prebuilder, the pretuner |
| [measure.md](measure.md) | the accuracy measurer (fp64 gold, lights) and the speed measurer (CUDA-graph timing) |
| [judge.md](judge.md) | versions, tokens, verdicts (carried / owed / pending), the Slurm measurement cycle |
| [ladder.md](ladder.md) | the performance ladder cuda < triton < pytorch: how it is checked, exceptions (LayerNorm, RMSNorm, L <= 256) |
| [../benchmarks/](../benchmarks/README.md) | speed figures (L and D sweeps) and tables per card, generated from the speed tokens by `python -m miniworld_engine.measure.plot` |

## H100 kernel notes

Design notes per kernel family (layout, tiling, saved tensors, numerics, limits). They describe the H100 (sm_90a) paths; where a family
also has an A100 path, the note names its sm_80 files in one line. Numbers in them are labelled historical: they were
not measured by the current measurer.

| module(s) | note |
|---|---|
| TriangleAttention (+ AF3 / RF3 / Chai1 / Chai1Confidence) | [h100/triangle_attention.md](h100/triangle_attention.md) |
| TriangleMultiplication, BidirectionalTriangleMultiplication | [h100/trimul.md](h100/trimul.md) |
| Transition, ConditionedTransition | [h100/transition.md](h100/transition.md) |
| OuterProductMean | [h100/outer_product_mean.md](h100/outer_product_mean.md) |
| MSAPairWeightedAveraging | [h100/pair_weighted_averaging.md](h100/pair_weighted_averaging.md) |
| LayerNorm, LayerNormLinear, RMSNorm | [h100/layernorm.md](h100/layernorm.md) |
| AttentionPairBias | [h100/attention_pair_bias.md](h100/attention_pair_bias.md) |
| TokenDiTProtenix / MiniWorld (and the Boltz-2 / RoseTTAFold3 / Chai-1 token blocks on the same algorithm) | [h100/token_dit.md](h100/token_dit.md) |
| TokenDiTMiniWorldBiasOnly | [h100/bias_only_dit.md](h100/bias_only_dit.md) |
| Boltz-2 / RoseTTAFold3 / Chai-1 DiT blocks: model options on the one token / atom algorithm, the slid-window edge kernels | [h100/dit_variants.md](h100/dit_variants.md) |
| AtomDiTProtenix / OpenFold3 / Boltz2 / RoseTTAFold3 / Chai1 (local atom attention, one implementation) | [h100/local_atom_dit.md](h100/local_atom_dit.md) |
| AtomDiTMiniWorld / AtomDiTESMFold2 / SWAAtomTransformer | [h100/swa_atom_dit.md](h100/swa_atom_dit.md) |
| AugmentNoise, AlignEuler, PoolAtoms | [h100/sampler_ops.md](h100/sampler_ops.md) |
| SmoothLddtLoss | [h100/smooth_lddt.md](h100/smooth_lddt.md) |
| PairEmbedAdaLN / PairEmbedGeom / PairEmbedCA, TokenPairInit | [h100/pair_embed.md](h100/pair_embed.md) |
| ProteinMPNN | [h100/mpnn.md](h100/mpnn.md) |

The earlier documentation (dispatch, autotune, registry, pass tokens, other GPUs, dated records) described systems that no longer
exist; it is in git history (`git show 65f37dc3c^:docs/...`, the parent of the commit that removed it).
