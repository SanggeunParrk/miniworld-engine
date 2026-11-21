# OpenFold3 attention kernel backends

OpenFold3 (Apache-2.0) offers three attention backends (docs kernels page): cuEquivariance (`use_cueq_triangle_kernels`, the `cueq` backend of E1), DeepSpeed Evoformer
attention (`use_deepspeed_evo_attention` = `ds4sci` of E5) and the Triton backend (`use_triton_triangle_kernels`, the one the TurboPairFormer paper calls "OpenFold3 Triton").
The Triton backend is `openfold3/core/kernels/triton/evoformer.py` (`TritonEvoformer`, AMD authored, forward + backward, bf16 / fp16 / fp32), called from
`core/model/primitives/attention.py:_triton_evo_attn`. It imports only torch / triton, so the adapter loads that one file (`ext_of3tri.py`; no OpenFold3 install).
Mask convention: additive `[B, N, 1, 1, K]` in the q dtype; pair bias `[B, 1, H, Q, K]`. Covers triangle attention and AttentionPairBias.
`openfold3/core/kernels/triton/{fused_softmax,swiglu,triton_softmax}.py` (not used here: softmax / SwiGLU element-wise helpers, not an operator class of E5).
