# AMD geometric-nn-ops (E5 baselines `geonn` = triangle_multiplicative_update, `gnops` = triangle_attention)

- Upstream: https://github.com/ROCm/geometric-nn-ops, branch `release/v1.0.0` @ c58e24cd (Apache-2.0; not on PyPI, installed from source per its docs; docs:
  https://rocm.docs.amd.com/projects/geometric-nn-ops/). Package `geometric_nn_ops_torch` (pure Python + Triton source; depends on torch + triton only).
- ONE entry serves both adapters (merged from the two branches that had a folder each): `ext_geonn_trimul.py` (TriMul) and `ext_gnops.py` (TriAttn,
  AttentionPairBias). Revision note: the attention records were measured from a checkout of the repository's `main` @ 232ed824 (the commit after the
  v1.0.0 release branch: it only pins Triton 3.8.0 in a requirements file); every file of the `geometric_nn_ops_torch` package, the LICENSE and
  `pyproject.toml` have the same sha256 at 232ed824 and at c58e24cd (checked file by file against the sha256 list of `UPSTREAM.json`), so the measured code is the
  pinned code.
- `triangle.triangle_multiplicative_update(x, direction, mask, norm_in_*, p_in_weight [a;b], g_in_weight [a;b], norm_out_*, p_out_*, g_out_*, eps)`:
  the cuEquivariance signature and function (no residual). Triton fused LayerNorm / linear kernels, forward only, fp32 / bf16 / fp16; a call that needs
  gradients takes the library's PyTorch reference path (so training is not a kernel result: the adapter records it unsupported).
- `triangle.triangle_attention(q, k, v, bias, mask, scale)`: the OpenFold3 Triton Evoformer-attention kernel in a cuEquivariance-style signature; forward +
  backward, bf16 / fp16 / fp32, bool key mask, pair bias `[B,1,H,N,N]`, scale inside. Its own `attention_pair_bias` is pure torch (no kernel yet): not used.
- Validated by AMD on MI300X / MI325X / MI355X (ROCm, Triton 3.8); the Triton source has no ROCm-only code and runs on NVIDIA (see the E5 records).
- Install: `third_party/geometric-nn-ops/fetch.sh [TARGET]` (raw files of the package, sha256 pinned). Where the adapters look: `$PAPER_EXT_PATH` (default
  `~/paper-ext`), `$PAPER_EXT_ATTN/geometric-nn-ops` (default `~/paper-ext-attn/geometric-nn-ops`), then `third_party/geometric-nn-ops/upstream`. Patches: none.
