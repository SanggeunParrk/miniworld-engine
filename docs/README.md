# Documentation

Every page in the first two tables describes the current code; where a page and the code disagree, the code is right.
Cards: H100 (`h100_sm90`), A100 (`a100_sm80`), RTX 5090 (`5090_sm120`) and B200 (`b200_sm100`) are served and judged, each with a
prebuilt kernels wheel (B200: two). Other CUDA cards run the `triton` and `pytorch` backends.

## Using the engine

| page | covers |
|---|---|
| [install.md](install.md) | requirements, `pip install foldjet-engine`, prebuilt kernels vs JIT, CUTLASS / MathDx headers, environment variables, troubleshooting |
| [modules.md](modules.md) | the `BackendModule` contract, the backends, selectors, class versions, master weights, `make_inputs` |
| [init.md](init.md) | parameter initialisation: the init vocabulary, presets (`engine`, `miniworld`, `af3`), the `init:` config block, `apply_init` / `describe_init` |
| [fp8.md](fp8.md) | the opt-in FP8 (e4m3) inference routes: the user switch is in "The `engine-fp8` backend" and "The lower-level switch"; the rest is a long dated log of the H100 and RTX 5090 routes (B200: [b200/fp8.md](b200/fp8.md)) |
| [fp8_training.md](fp8_training.md) | the opt-in FP8 training routes (H100; all off by default), with their measurements (a dated log) |
| [bias_logits.md](bias_logits.md) | the bias-only token DiT's pair-bias logit precision: `fp32` (default, training) and the opt-in inference mode `bf16` |

## How the engine is built, measured and judged

| page | covers |
|---|---|
| [architecture.md](architecture.md) | **start here**: the pipeline end to end, how a module reaches the card's kernels, the directory map |
| [requirements.md](requirements.md) | the two requirement tables, the derive rule (kernel range = its module rows), the accuracy rule |
| [kernels.md](kernels.md) | kernel families, launch-config tables and shape keys, the prebuilder, the pretuner |
| [measure.md](measure.md) | the accuracy measurer (fp64 gold, lights) and the speed measurer (CUDA-graph timing) |
| [judge.md](judge.md) | versions, tokens, verdicts (carried / owed / pending), the Slurm measurement cycle, when to bump what |
| [ladder.md](ladder.md) | the performance ladder cuda < triton < pytorch: how it is checked, exceptions (LayerNorm, RMSNorm, L <= 256) |
| [release-notes/](release-notes/README.md) | one file per release (2.5.9 .. 3.0.12 as miniworld-engine, then foldjet-engine 1.0.0) and the release steps they record |

## Per-card kernel notes

### H100 design notes (`h100/`)

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

### B200 development log (`b200/`)

**History, not reference.** The dated per-module logs of the B200 (sm_100a) port, rounds 1-3, the S90 rounds and later agent rounds
with their evidence; rounds 2026-11-16 .. 11-21. [b200/README.md](b200/README.md) is the entry point (per-module notes, measured limits of the card, versions at
the tip). Kernel source comments cite these files by path (`docs/b200/<note>.md`, `docs/b200/rejected/*.diff`), so the folder stays
where it is. What it contains:

| path | what |
|---|---|
| `b200/{trimul,triattn,transition,atom-dit,token-dit,bias-only,msa,misc}.md` | per-family logs (newest pass at the end of each file) |
| `b200/fp8.md`, `b200/fp8_h100_*_digest.md` | the B200 FP8 routes; digests of the H100 FP8 designs written for the port |
| `b200/HANDOFF.md` | hand-off log of the agent rounds, sections 14-21 (2026-11-18 .. 11-21); names scratch trees outside the repository |
| `b200/rejected/` | diffs of variants that were built, measured and not adopted (not reachable from the code) |
| `b200/{s90_transition_inf,triattn_*,n2_tools,n7_tools}/`, `b200/*_sol*.{md,txt}` | per-pass evidence: probe drivers, tables, speed-of-light data |

The RTX 5090 and A100 have no separate note folders: the A100 paths are named in the H100 notes, the 5090 builds the A100 sources
([architecture.md](architecture.md#3-kernel-families-per-card)), and its FP8 routes are in [fp8.md](fp8.md).

## Elsewhere in the repository

| path | what |
|---|---|
| [../benchmarks/](../benchmarks/README.md) | generated speed figures and tables per card, the paper's data (E1-E6), experiment records |
| [../tools/](../tools/README.md) | the repository gate (`check.sh`), the Slurm judge cycle, benchmark helpers |
| [../src/foldjet_engine/requirements/README.md](../src/foldjet_engine/requirements/README.md) | the rules of the requirement tables |
| [../third_party/README.md](../third_party/README.md) | the external baselines (fetch scripts and pins; upstream trees not in git) |
| [../licenses/THIRD_PARTY_NOTICES.md](../licenses/THIRD_PARTY_NOTICES.md) | licences and per-file attributions |
| `../legacy/b200/` | B200 code removed by the 2.5.0 redesign, restored for reference; not imported |

The documentation from before the 2.5.0 redesign (dispatch, autotune, registry, pass tokens, other GPUs, dated records) described
systems that no longer exist; it is in git history (`git show 65f37dc3c^:docs/...`, the parent of the commit that removed it). Source
comments that still name such paths (`docs/gpus/...`, `docs/records/...`) point into that history.
