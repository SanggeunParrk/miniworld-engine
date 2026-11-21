"""E5 external backend ``megafold``: MegaFold EvoFlash-3D Triton memory-efficient EvoAttention (MIT, forward + backward).

Upstream: github.com/Supercomputing-System-AI-Lab/MegaFold, ``megafold/model/FusedEvoAttention/evoattention.py`` (``TritonEvoformer``; loaded as one
source file, the megafold package and its DeepSpeed import are not needed). ``TritonEvoformer(Q, K, V, res_mask, pair_bias)``: Q / K / V
``[B, N, Q, H, d]``, ``res_mask`` an ADDITIVE ``[B, N, 1, 1, K]`` mask (0 / -1e9; the upstream example passes a bool, wrong), ``pair_bias`` ``[B, 1, H, Q, K]`` (fp32 in the upstream example); scale
``d^-0.5`` inside. Matches two of our classes: the triangle-attention classes (N = rows) and AttentionPairBias (N = 1, pair bias). Block sizes are fixed
(16 x 64, tuned upstream for H200 / L 384).
"""

from __future__ import annotations

import os

import ext_attn_common as common
import torch

NAME = "megafold"


def _mod():
    return common.load_file(
        "megafold_evoattention",
        os.path.join(common.upstream("megafold"), "megafold", "model", "FusedEvoAttention", "evoattention.py"),
    )


def evo(q, k, v, keep, bias):
    b, n = q.shape[:2]
    # the kernel ADDS ``res_mask`` to the logits (its __main__ example passes a bool tensor, which is wrong): additive 0 / -1e9 in the q dtype
    if keep is None:
        res = torch.zeros(b, n, 1, 1, q.shape[2], dtype=q.dtype, device=q.device)
    else:
        res = torch.zeros(keep.shape, dtype=q.dtype, device=q.device).masked_fill_(~keep, -1e9)
    return _mod().TritonEvoformer(q, k, v, res, bias.contiguous())


def reason(dtype, head_dim, length, grad):
    if dtype not in (torch.bfloat16, torch.float16, torch.float32):
        return f"dtype {dtype}: bf16 / fp16 / fp32 only"
    return None


def install() -> None:
    common.attach(NAME, evo, reason, _mod)
