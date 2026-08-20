"""trimul_inproj — INFERENCE forward (forward-only, saves NOTHING).

This is the maximally-fused, lowest-latency path: it computes y and discards every
intermediate (left/right, tri, preact, LN stats, gate, proj). Because nothing has
to be persisted for a backward, the back-half is the single fused triton kernel
``trimul_back_triton`` (LN_out + @Wp + gate-mul in one launch).

Contrast with the TRAINING forward (``training.py``): that one must SAVE the
tensors its backward consumes, and *what* it saves changes the backward algorithm
and speed (save-vs-recompute). Keep the two paths separate — do NOT use this for
training (no autograd is attached).

Pipeline (outgoing): triton LN_in(+mask fold) -> trimul_inproj front (one gated
GEMM, bdll) -> torch.bmm contraction -> triton fused back. bf16, B=1, D=128.
All weights are in x@W form (= nn.Linear weight .T).
"""

from __future__ import annotations

from miniworld_engine.kernels._compile import opaque

import torch

from miniworld_engine.kernels.layernorm.triton.main import (
    triton_layernorm, triton_layernorm_masked,
)
from miniworld_engine.kernels.trimul_inproj.cute.launch import trimul_inproj_cute_forward
from miniworld_engine.kernels.trimul_inproj.triton.back import trimul_back_triton


@opaque()
@torch.no_grad()
def trimul_inproj_inference(pair, WL, WLg, WR, WRg, Wg, Wp,
                            ln_in_w, ln_in_b, ln_out_w, ln_out_b, eps, b_lr, rmask=None):
    """Whole trimul (outgoing) forward, inference-only. Returns y [B,L,L,D].

    rmask: [M] AF pair-mask, folded into LN_in for free (proj(0)=0 -> left/right
    zeroed at masked positions, == AF's mask*projection at every valid position).
    None -> no mask. Saves nothing.
    """
    B, L, _, D = pair.shape
    xf = pair.reshape(B * L * L, D)
    if rmask is None:
        xn = triton_layernorm(xf, ln_in_w, ln_in_b, eps)
    else:
        xn = triton_layernorm_masked(xf, ln_in_w, ln_in_b, eps, rmask)
    xn = xn.view(B, L, L, D)
    left, right, _ = trimul_inproj_cute_forward(
        xn, WL, WLg, WR, WRg, None, bdll_direct=True, compute_gate=False, b_lr=b_lr)
    tri = torch.einsum("bdik,bdjk->bdij", left, right)        # (B,D,L,L)
    return trimul_back_triton(tri, xn, Wp, Wg, ln_out_w, ln_out_b, eps)
