# NVIDIA BioNeMo Inference Runtime (BioIR) - inference-only fused kernels

Public on PyPI (`pip install bionemo-ir`, Apache-2.0; python 3.12-3.14, x86_64 / aarch64, driver >= 580 stated). The runtime is a full structure-prediction pipeline (ray, rdkit, ...);
E5 uses only its operator layer: `fetch.sh` unpacks the wheel (no dependencies) and the adapter imports `bionemo_ir._torch.*` under a stub package (the package `__init__` is not run).
Kernel support matrix (docs): triangle attention CuTeDSL CUBIN on SM80 / SM90 (ClaudeKit CUDA for large bf16 D=32 on SM90 when its launcher ships), cuEquivariance on SM100 / SM120; OPM CuTeDSL (SM80 / SM90 bf16),
Triton on SM120; pair-weighted averaging CuTeDSL; transition MLP (silu_gate) CuTeDSL bf16 SM80 / SM90 / SM100, none on SM120; dual GEMM, gated sigmoid, AdaLN also exist (not an E5 class).

| engine class | BioIR op | E5 |
|---|---|---|
| TriangleAttention | `triangle_attention` (HeuristicTriangleAttention) | yes, fp16 / bf16, inference |
| OuterProductMean | `outer_product_mean` (d_hidden 32, d_pair 128 / 256) | yes, inference |
| MSAPairWeightedAveraging | `pair_weighted_averaging` (8 heads, tuned (D, c_m)) | yes, inference |
| Transition | `transition_mlp` (silu_gate, bf16) | yes, inference |
| AttentionPairBias | pairwise attention (a different interface: not connected) | no |
| LayerNorm | fused LN only inside AdaLN / LN+proj kernels (no standalone LN op) | no |

Library fallbacks: a call outside a kernel envelope silently runs the library own PyTorch formula; the adapters check the envelope and report such calls `unsupported`.

Measured here (E5 ext_attn, see benchmarks/paper/E5_other_kernels/ext_attn_results.md): the wheel's native library `bionemo_ir/libs/_cutedsl_kernels.abi3.so` needs glibc >= 2.34 (the H100 cluster hosts have 2.28: import fails) and a CUDA driver >= 580 for its CUBINs (A100 hosts: 570: cuModuleLoadData fails),
so on H100 / A100 triangle attention and OPM are `unsupported` with that reason; pair-weighted averaging (python CUBIN launch) runs on all three cards. RTX 5090 (Ubuntu 24.04, driver 595): triangle attention (cuEquivariance route), OPM (Triton), PWA run; transition MLP has no sm_120 / (width 64 / 128, hidden 256 / 512) variant on any card. Inference only everywhere.
