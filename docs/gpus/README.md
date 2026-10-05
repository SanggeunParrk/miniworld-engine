# GPUs

One page per GPU: how to run on it, then **which operation is finished for which shapes**.
The maintainer makes that call; the pages record it.

**Module benchmark protocol:** [모듈별 추론과 학습 성능 비교 기준](module-comparison-protocol.md)
records the agreed inference/training shapes, baselines, and compile/CUDA-graph exceptions.
Use it for subsequent comparisons, including A100; distinguish measured workloads from registry support ranges.

Latest protocol run: [A100 module comparison, 2026-10-05](../records/a100-module-comparison-20261005.md)
— full inference/training shape tables, compiled baselines, graph OFF/ON, and explicit unsupported/OOM cells.
SWA follow-up: [Anthropic fullgraph compile and six-shape remeasurement](../records/a100-swa-anthropic-compile-20261005.md)
supersedes the graph-only Anthropic SWA inference comparison in that run.

A100 inference review, 2026-10-05: the maintainer accepted OPM, PWA, Pair Transition,
SWA Atom DiT and Dense Atom DiT as the first completed review group. The `a100`
branch preserves the implementation and measurements at this checkpoint.
The next inference review covers TriMul, TriangleAttention and AttentionPairBias,
starting with per-shape speed-of-light measurements under the same shape protocol.
Results: [A100 TriMul, TriangleAttention and APB inference SOL](../records/a100-trimul-triattn-apb-inference-sol-20261005.md)
records all 120 shapes, measured latency, implementation-aware modeled floors and separate NCU counters.

| GPU | arch | cluster / partition | page | summary |
|---|---|---|---|---|
| H100 80GB HBM3 | sm90 | cssb, `h100` partition | [h100/h100.md](h100/h100.md) · [module dispatch](h100/dispatch.md) | hand CUDA for TriMul, Transition (n=4), TriAttn training, OPM/PWA, token DiT; Triton elsewhere |
| B200 | sm100 | lab-external server, no scheduler (see page) | [b200/b200.md](b200/b200.md) · [TriMul](b200/trimul/trimul.md) · [TriAttn](b200/triattn/triattn.md) · [Transition](b200/transition/transition.md) · [token DiT](b200/token_dit/token_dit.md) · [OPM](b200/opm/opm.md) · [PWA](b200/pwa/pwa.md) · [atom DiT](b200/atom_dit/atom_dit.md) · [SWA atom DiT](b200/swa_atom_dit/swa_atom_dit.md) | hand CUDA for TriMul (both modules, D64-D512, inference and training), TriAttn, Transition (n=4, D64-512), token DiT, OPM / PWA (inference and training); Triton elsewhere |
| A100 80GB PCIe | sm80 | cssb, `A100` partition | [a100/a100.md](a100/a100.md) · [TriMul](a100/trimul/trimul.md) · [TriAttn](a100/triattn/triattn.md) · [token DiT](a100/token_dit/token_dit.md) · [atom DiT](a100/atom_dit/atom_dit.md) · [SWA atom DiT](a100/swa_atom_dit/swa_atom_dit.md) · [bias-only DiT](a100/bias_only_dit/bias_only_dit.md) · [AttentionPairBias](a100/attention_pair_bias/attention_pair_bias.md) · [gated projection](a100/gated_projection/gated_projection.md) · [layernorm_linear](a100/layernorm_linear/layernorm_linear.md) · [OuterProductMean](a100/outer_product/outer_product.md) · [MSA PWA](a100/msa_pair_weighted_averaging/msa_pair_weighted_averaging.md) · [AdaLN](a100/adaptive_layernorm/adaptive_layernorm.md) · [ConditionedTransition](a100/conditioned_transition/conditioned_transition.md) · [Transition](a100/transition/transition.md) · [LayerNorm / RMSNorm / RoPE](a100/layernorm/layernorm.md) · [MPNN edge side](a100/mpnn/mpnn_edge.md) · [MPNN message side](a100/mpnn/mpnn_msg.md) | hand CUDA for TriMul (every registered width D64-D384, bf16 and fp32 (TF32), one direction and bidirectional, inference and training) and the gated projections, TriAttn (every registered width, d_pair 64-384, starting and ending node, inference and training; with the projected_attention token_pair leaf) and layernorm_linear (LayerNorm + projection to <= 16 outputs), OuterProductMean and MSAPairWeightedAveraging (inference and training), AdaLN and ConditionedTransition (atom and token streams, bf16 and fp32, inference and training), Transition (every registry width, n = 2 and 4, pair / single / MSA streams and the SwiGLU FFN, inference and training), LayerNorm / RMSNorm / QK-norm + RoPE / RMSNorm-modulation row kernels (inference and training), the token DiT block (16 heads x 48, inference and training: cuBLAS + CUDA rows + a hand-CUDA attention core), AugmentedAttentionPairBias at the atom and token widths (bf16 and fp32 / TF32, any A / B / L, shared or per-sample key masks, inference and training; the atom DiT block runs CUDA end to end with the AdaLN / ConditionedTransition paths), the SWA atom DiT block (window 64, 4 heads x 32: modulation, forward and backward stages in CUDA for every sample count, the weight gradients cuBLAS) with its CUDA output gate and the bias-only token DiT block (inference and training: cuBLAS + CUDA rows + hand-CUDA attention and bias-gradient cores) and the `bias_only_attention` kernel, and AttentionPairBias (the Pairformer single track, inference and training: cuBLAS + CUDA rows + a hand-CUDA pair bias and attention core), and the ProteinMPNN edge side (edge tail, edge MLP, edge LayerNorm backward, edge dropout mask: a fused chain with saved-activation and recompute policies) and message side (hidden message, encoder node message, relative-position backward), inference and training; Triton elsewhere |
| RTX A6000 / A5000 | sm86 | `cssb-master`, `gpu` partition | [ampere-workstation/ampere-workstation.md](ampere-workstation/ampere-workstation.md) | Triton only (no completion table) |

## Common to all


- Nothing runs on a login node (see `.claude/CLAUDE.md`). Always pass `--mem`.
- Install the pixi env on a CPU allocation of the cluster you will run on
  (`pixi install`, `pixi run fix-te-cu12`, `miniworld-engine dev install-flash --arch <sm>`).
  Conda prefixes are absolute paths: an env built in one checkout cannot be moved to another.
- Give each GPU/checkout its own JIT cache (`MINIWORLD_ENGINE_JIT_ROOT`); prebuilt
  TriangleAttention `.so` files are tied to the torch/python they were built with and the
  CUDA path silently falls back to Triton when they do not match
  (rebuild: see [h100/h100.md](h100/h100.md)).
- Autotune caches are keyed by GPU name **and** toolchain (`env_identity`): a cache built
  under another torch/triton is ignored. Check with `miniworld-engine dev cache-status --gpu <name>`.

## Layout and completion tables

```
docs/gpus/<gpu>/
  <gpu>.md              environment + module-level completion tables
  dispatch.md           module dispatch contracts (H100)
  <module>/
    <module>.md         kernel-level tables, kernel-flow figures, measurements
    figures/            <module>_<variant>.json spec -> *_<name>.svg -> *.png
```

Tables: columns are shapes (`(Length, Dimension)` for bf16-only TriMul / TriAttn, `(Length, MSA depth)` for the
fixed-width bf16 MSA modules, otherwise `(Length, Dimension, dtype)`); rows are **implementation** (agent: the backend default dispatch
runs, or 미구현), **성능 확인** (maintainer only: ✓ finished, △ the fastest measured but not yet complete,
✗ not confirmed) and **cache build** (agent, ✓ / ✗).
Kernel tables cover CUDA / Triton kernels; PyTorch and cuBLAS steps appear only in the figures.
Every measurement table is followed by two bar charts drawn from it (`python -m miniworld_engine.viz.measure_bars
<page>`): a length sweep at D128 and a dimension (or MSA-depth) sweep at L384 (`--length-d` / `--dim-l` for a page without them), one
bar per implementation, latency on a log axis; rerun it whenever a table changes.
Measurements compare PyTorch compiled / cuEquivariance / Anthropic / ours, CUDA-graph or compiled
timing only (`benchmarks/cautions.md`). Shapes come from the model shape registry
(`src/miniworld_engine/kernels/registry/registry_module.csv`). Full template:
[template.md](template.md).

H100 and B200 (TriMul, TriAttn, Transition, token DiT, OuterProductMean, PWA) and A100 (TriMul, TriAttn, Transition, token DiT, atom DiT, SWA atom DiT, bias-only DiT, AttentionPairBias, gated projection, layernorm_linear, OuterProductMean, PWA, AdaLN, ConditionedTransition) use this format. The rest of the A100 page and of the B200 page still carry the previous op-per-row table
(its **judgement** column is the maintainer's) until they are converted.
