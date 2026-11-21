"""E5 external backend ``gnops``: AMD ROCm ``geometric-nn-ops`` ``triangle_attention`` (Apache-2.0, Triton, forward + backward; runs on NVIDIA).

Upstream: github.com/ROCm/geometric-nn-ops (``geometric_nn_ops_torch.triangle.triangle_attention``, cuEquivariance-style signature on top of the
OpenFold3 Triton evoformer kernel). Serves the triangle-attention classes and AttentionPairBias (q / k / v ``[B, N, Q, H, d]``, bias ``[B, 1, H, Q, K]``,
bool key mask keep = True); bf16 / fp16 / fp32; scale ``d^-0.5`` inside. Its own ``attention_pair_bias`` is pure torch (no kernel): not used.
"""

from __future__ import annotations

import sys

import ext_attn_common as common
import torch

NAME = "gnops"


def _mod():
    root = common.upstream("geometric-nn-ops")
    if root not in sys.path:
        sys.path.insert(0, root)
    from geometric_nn_ops_torch.triangle import triangle_attention

    return triangle_attention


def evo(q, k, v, keep, bias):
    return _mod()(q, k, v, bias, keep, q.shape[-1] ** -0.5)


def reason(dtype, head_dim, length, grad):
    if dtype not in (torch.bfloat16, torch.float16, torch.float32):
        return f"dtype {dtype}: bf16 / fp16 / fp32 only"
    return None


def install() -> None:
    common.attach(NAME, evo, reason, _mod)
