# H100 attention dispatch audit — 2026-10-05

Source: `h100/master-weights`, commit `4ee68216232a2085f8fce7737948afb2d338d8d8`. The preceding comparison's 197 successful MiniWorld inference rows certify that those calls ran and passed the harness checks. They do not certify dedicated H100 CUDA coverage for every module. The table below describes hand-written native CUDA kernels; PyTorch and Triton also execute on the GPU.

| Module / workload | Current H100 route | Native CUDA coverage gap |
|---|---|---|
| Standalone AttentionPairBias, five head layouts | Triton augmented-attention core + separate normalization, projections, gates and output | Dedicated whole-module native path is B200-only. H100 standalone APB has no such route. |
| Token DiT, 16x48, L128/256/384/512/640/768 | H100 `attn_fwd.cu` gated core, CUDA row kernels, CUDA SwiGLU GEMM and cuBLAS | Native path exists. The full block remains slower than the named Anthropic composition in the preceding comparison; bottleneck attribution requires further measurement. |
| Token DiT, L200/450/700 (historical diagnostic only) | General composition in the diagnostic | User clarified mandatory bucketing. These lengths are not production inputs and are excluded from required shape coverage. |
| Dense Atom DiT, 4x32 / pair width16 | General module composition, Triton attention | Specialized atom block CUDA path is B200-only. General H100 attention CUDA core serves16x48, not this layout. |
| AF3 Local Atom DiT, 32x128 / cross_attention=True | PyTorch local attention + dispatched AdaLN/transition | Specialized local block CUDA path is B200-only. |
| Bias-only Token DiT | Reference block composition | Specialized fused CUDA attention core admits sm100 only. H100 inference falls back. |
| TriangleAttention, pair width128 / H4D32 | Qualified native attention forward at L384/L768 (also L1024 outside this protocol); other protocol lengths use Triton | Whole-module frontend/output are still composed; native forward has a narrow shape gate. |
| TriangleAttention, pair widths64/256/384/512 | Triton attention forward in this protocol | The qualified native forward gate is H4D32/C128. Wider production forward is not covered by it. |
| SWA DiT | FlashAttention4 CUDA core with surrounding block composition | This is an external attention CUDA core, not a dedicated complete H100 block fusion developed by this project. |

The existing `execution_path` strings are too coarse in two cases. Token DiT's string says Triton outside B200, but the live runner selects H100 native CUDA core and row kernels. TriangleAttention's string contains `triton_triangle_attention_pair_bias`, but that public entry internally takes qualified H100 native forward at the listed C128 lengths. Preserve the raw timing metadata, and use this audit to interpret it.

GPU job21539, one H100 under `normal_h100`, confirmed `attn_fwd_kernel<false,1>`, CUDA AdaLN/conditioning/residual row kernels and `gemm_swiglu_sm90` for Token L128; `_attn_fwd` and Triton row passes for Token L200; `training_fwd_stream<false>` for TriangleAttention C128/L384; and `_attn_fwd` for TriangleAttention C256/L128. Standalone APB's H100 gap was established from its source dispatch chain, which only calls the B200 native adapter before the generic Triton route. The graph profiler traces are representative and not used as exhaustive kernel coverage or performance attribution.

Source gates: `modules/attention_pair_bias/module.py`, `modules/augmented_attention/module.py`, `integrations/token_dit.py`, `kernels/conditioned_transition/triton/token_dit_runner.py`, `integrations/atom_dit.py`, `integrations/local_dit.py`, `kernels/bias_only_dit/cuda/__init__.py`, and `kernels/triangle_attention/cuda/training_forward.py`.

[GPU kernel observations](data/anthropic_comparison/dispatch_audit_20261005.json). No production source, dispatch or timing results were changed for this audit.

## Current user requirements

Token DiT always uses bucketing. All required execution paths must eventually have CUDA kernels with Triton removed. This is a future requirement, not the current implementation state. Broader CUDA implementation is deferred; the immediate focus is TriangleAttention optimization and comparison with the B200 algorithm. The 194 accepted inference configurations all ran successfully on the current H100 paths; no MiniWorld shape refusal was observed within this matrix.
