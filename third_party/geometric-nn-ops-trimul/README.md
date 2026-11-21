# AMD geometric-nn-ops, triangle_multiplicative_update (E5 baseline `geonn`)

- Upstream: https://github.com/ROCm/geometric-nn-ops, branch `release/v1.0.0` @ c58e24cd (Apache-2.0; not on PyPI, installed from source per its docs; docs:
  https://rocm.docs.amd.com/projects/geometric-nn-ops/). Package `geometric_nn_ops_torch` (pure Python + Triton source; depends on torch + triton only).
- What it is: `triangle.triangle_multiplicative_update(x, direction, mask, norm_in_*, p_in_weight [a;b], g_in_weight [a;b], norm_out_*, p_out_*, g_out_*, eps)`:
  the cuEquivariance signature and function (no residual). Triton fused LayerNorm / linear kernels, forward only, fp32 / bf16 / fp16; a call that needs
  gradients takes the library's PyTorch reference path (so training is not a kernel result: the adapter records it unsupported).
- Validated by AMD on MI300X / MI325X / MI355X (ROCm 10, Triton 3.8); the Triton source has no ROCm-only code and runs on NVIDIA (see E5 records).
- Install: `third_party/geometric-nn-ops-trimul/fetch.sh [TARGET]` (raw files of the package, sha256 pinned; same commit as the attention entry of the third-party-attn branch). Patches: none.
- Adapter: `benchmarks/paper/E5_other_kernels/ext_geonn_trimul.py`.
