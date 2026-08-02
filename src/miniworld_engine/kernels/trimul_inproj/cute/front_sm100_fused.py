"""SM100 (Blackwell / B200) trimul FRONT -> [B, 2D, L, L] "bdll" left/right.

BEATS both the Triton front (0.542 ms @ L=1024) and the H100 cute front (0.43 ms).
Measured (GPU bf16, B=1, D=128, do_bench warmup=25 rep=100, cos vs torch left/right):
    L=512 : 0.106 ms  (triton 0.163)  1.54x   cos 0.999999
    L=768 : 0.224 ms  (triton 0.323)  1.44x   cos 0.999999
    L=1024: 0.389 ms  (triton 0.542)  1.40x   cos 0.999999   (< 0.43 H100 bar)

Winning path: ONE single-pass gated GLU GEMM + a tuned Triton transpose
------------------------------------------------------------------------
1. A single quack sm_100 tcgen05 **gated-GLU** GEMM ``x @ B`` with gate/proj columns
   INTERLEAVED in the B-operand, so the GLU N-halving epilogue computes
   ``sigmoid(gate) * proj`` IN-KERNEL (the gate is never materialized to HBM, and x is
   read once): ~0.23 ms @ L=1024.  Output is N-contiguous ``blld`` (M, 2D), cols
   [:D]=left, [D:]=right (see `prepack_lr_operand_sm100` for the interleave).
2. A tuned Triton transpose ``blld (M, 2D) -> bdll (2D, M) = [B, 2D, L, L]`` at ~HBM
   peak (~6 TB/s, ~0.17 ms @ L=1024).  left/right are the two D-plane slices.

Why this beats the two-GEMM bdll-DIRECT variant (also implemented earlier, 0.448 ms):
the bdll-direct kernel writes the full gate ``[2D,M]`` (512 MB) and reloads it in the
second GEMM — a ~1 GB gate HBM round-trip that is half the traffic.  Folding the gate
into the GLU epilogue (gate never leaves registers) removes that round-trip; the only
remaining non-GEMM cost is the (BW-bound, near-peak) transpose, and GEMM(0.23)+
transpose(0.17) < the two-GEMM 0.448.

Why not a SINGLE fused gated GEMM writing bdll DIRECTLY (the ~0.12-0.2 ms ideal)?
---------------------------------------------------------------------------------
That needs the gate to pair two output CHANNELS adjacent in the register-contiguous mode
AND to store HALF the M (channel) extent.  Blocked on the sm_100 tcgen05 datapath
(verified): channel(M)-major t2r gives each thread 32 contiguous channels via a
NON-identity permutation (identity store is cos 1.0, but pairing adjacent registers as
gate/proj gives cos ~0), and the StMatrix m-major store atom delivers a fixed
32-channel/thread tile so the postact M extent cannot be halved to 16/thread the way
quack's column(N)-gated path halves N (which composes through t2r + smem + TMA together).
A true single-pass bdll gate would need a from-scratch dual-accumulator collective (two
MMAs, gate*proj in-register, m-major store) — research-grade; this GLU+transpose path
already clears the bar.  See project memory `trimul-sm100-front-fused-win`.

Public API
----------
    trimul_front_sm100_fused(x, WL, WLg, WR, WRg) -> (left_bdll, right_bdll)
        x: (B, L, L, D) bf16 contiguous, B==1.  Returns each (B, D, L, L) contiguous.
        left  = sigmoid(x@WLg) * (x@WL);  right = sigmoid(x@WRg) * (x@WR).
        W*:(in, out) = weight.T in the trimul convention (committed; do not change).
    prepack_lr_operand_sm100(WL,WLg,WR,WRg) -> B  # build the interleaved GLU B-operand once.

Constraints: B==1, square L, D=128, K=D=128, bf16.
"""
from __future__ import annotations

import torch


def prepack_lr_operand_sm100(WL, WLg, WR, WRg):
    """Pack the single fused GLU B-operand ONCE (per-call interleave avoided).

    Weight convention (committed, unchanged): W*:(in, out) is the trimul weight.T, so
    ``left = sigmoid(x@WLg) * (x@WL)`` with x:(M, in).  The single-pass GLU GEMM is
    ``A=x (M, in) @ B (in, 4*out)`` with gate/proj columns interleaved per half:

        B[:, 0:2out:2] = WLg   B[:, 1:2out:2] = WL    (left  half: glu(WLg, WL))
        B[:, 2out::2]  = WRg   B[:, 2out+1::2] = WR   (right half: glu(WRg, WR))

    quack's gated GLU epilogue halves N: out col j = sigmoid(preact[2j]) * preact[2j+1].
    So postact ``blld`` (M, 2out) has cols [:out]=left, [out:]=right.  Returns this
    (in, 4*out) bf16 B-operand (kept the legacy ``packed`` tuple/positional API).
    """
    Din, Dout = WL.shape  # (in, out)
    B = torch.empty(Din, 4 * Dout, device=WL.device, dtype=WL.dtype)
    B[:, 0:2 * Dout:2] = WLg
    B[:, 1:2 * Dout:2] = WL
    B[:, 2 * Dout::2] = WRg
    B[:, 2 * Dout + 1::2] = WR
    return B.contiguous()


def trimul_front_sm100_fused(
    x: torch.Tensor,
    WL=None, WLg=None, WR=None, WRg=None,
    *,
    packed=None,
    tile_M: int = 128, tile_N: int = 256,  # kept for back-compat; unused by GLU path
):
    """SM100 front. Returns (left_bdll, right_bdll), each (B, D, L, L) contiguous (B=1).

    left  = sigmoid(x@WLg) * (x@WL);  right = sigmoid(x@WRg) * (x@WR).
    x: (B, L, L, D) bf16 contiguous.

    Path (fastest measured, beats the H100 cute front 0.43ms@1024):
      1. ONE single-pass quack tcgen05 gated-GLU GEMM  x @ B(interleaved) -> blld (M, 2D)
         (cols [:D]=left, [D:]=right).  Gate is fused in-kernel (NOT materialized) and
         x is read once: ~0.23ms@1024, no gate HBM round-trip.
      2. A tuned Triton transpose blld (M,2D) -> bdll (2D, M) = [B,2D,L,L] at ~HBM peak.
    The single fused GEMM + cheap transpose beats the two-GEMM bdll-direct variant
    (which paid a 1 GB gate store+reload round-trip) — see `trimul-sm100-front-fused-win`.
    """
    from miniworld_engine.kernels._quack_compat import gemm_act
    from .front_sm100 import _transpose_blld_to_bdll

    assert x.dim() == 4 and x.is_cuda and x.is_contiguous()
    B, L, L2, D = x.shape
    assert B == 1 and L == L2
    M = L * L
    x_flat = x.reshape(M, D)
    if packed is None:
        packed = prepack_lr_operand_sm100(WL, WLg, WR, WRg)
    b_lr = packed  # (in, 4*out) interleaved GLU B-operand

    # 1. single-pass gated GLU GEMM -> N-contiguous blld (M, 2D); cols [:D]=left, [D:]=right.
    blld = torch.empty(M, 2 * D, device=x.device, dtype=x.dtype)
    gemm_act(A=x_flat, B=b_lr, activation="glu", postact_out=blld, store_preact=False)

    # 2. blld (M, 2D) -> bdll (2D, M) == [B, 2D, L, L]; left/right are the D-plane slices.
    lr = torch.empty(B, 2 * D, L, L, device=x.device, dtype=x.dtype)
    _transpose_blld_to_bdll(blld, lr.view(2 * D, M))
    return lr[:, :D], lr[:, D:]
