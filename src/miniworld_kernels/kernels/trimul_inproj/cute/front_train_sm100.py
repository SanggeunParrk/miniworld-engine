"""SM100 (B200) trimul FRONT for the v6-faithful TRAINING path — bdll-direct, 0 transposes.

Reproduces the H100 v6 front contract EXACTLY: one call returns
    (left_bdll, right_bdll, preact_bdll)
with ``left``/``right`` each ``[B, D, L, L]`` contiguous and ``preact`` ``[B, 4D, L, L]``
(the pre-GLU logits, channel-major, gate/proj interleaved per side, left then right — the
layout ``triton.back_fused.front_bwd_dW`` consumes). No ``.permute()``/transpose kernel.

WHY a separate sm100 front (vs the sm90 ``trimul_inproj_cute_forward(bdll_direct=True)``):
  The sm90 path hands quack's GATED-GLU epilogue an M-major (bdll) postact view so the GEMM
  writes ``[2D,L,L]`` directly. On quack's sm_100 gated epilogue that M-major store SILENTLY
  HALF-WRITES each tile (verified: left/right cos ~0.05) — the StMatrix m-major store atom
  delivers a fixed 32-channel/thread tile that the GLU N-halving cannot compose with (see
  ``front_sm100.py`` header + project memory ``trimul-sm100-front-fused-win``).

  BUT this is only the GATED epilogue. The finding does NOT apply to a NON-gated store: a plain
  GEMM store has a free layout (no reduction/no N-halving), so an M-major (bdll) postact is
  bit-correct on sm_100 (verified: preact cos 0.999999, left/right cos 0.999997 at L=128..1024).
  So the sm100 front = ONE non-gated m-major GEMM ``x @ b_lr -> preact[4D,L,L]`` + a Triton GLU
  pass ``left/right[d] = sigmoid(preact[gate_d]) * preact[proj_d]`` -> ``[2D,L,L]``. Same DATA as
  the sm90 gated launch (preact + postact), same layout contract, produced with 0 transposes.

  (This is exactly the store-layout freedom the round plan called out: a reduction-free GEMM
  store may pick col/row-major freely — unlike the dgrad LN-bwd epilogue whose reduction axis is
  coupled to the store. The GLU is done as a separate elementwise pass instead of fused into the
  GEMM epilogue; that is an implementation detail, not an algorithm/precision change.)

B=1, square L, bf16 in / fp32 acc / bf16 out.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl
from quack.gemm_interface import gemm_act


def _interleave(Wg: torch.Tensor, Wp: torch.Tensor) -> torch.Tensor:
    """(D,H),(D,H) -> (D,2H): GLU wants (gate, proj) adjacent so preact plane order is
    [g0,p0,g1,p1,...] per side (matches front_bwd_dW's 2d / 2d+1 indexing)."""
    H = Wg.shape[1]
    out = torch.empty(Wg.shape[0], 2 * H, device=Wg.device, dtype=Wg.dtype)
    out[:, 0::2] = Wg
    out[:, 1::2] = Wp
    return out


def prepack_lr_operand_sm100(WL, WLg, WR, WRg) -> torch.Tensor:
    """Build the fused (D, 4H) GLU B-operand ONCE. Cols: [il(WLg,WL) | il(WRg,WR)].
    Contract-identical to trimul_inproj.cute.launch.prepack_lr_operand (drop-in b_lr)."""
    return torch.cat([_interleave(WLg, WL), _interleave(WRg, WR)], dim=1).contiguous()


@triton.jit
def _glu_bdll_kernel(preact, lr, H: tl.constexpr, M, BLK: tl.constexpr):
    """preact (4H,M) channel-major -> lr (2H,M). Per side: even plane=gate, odd=proj.
    Grid over the (H,M) per-side positions; each program emits left plane d AND right plane d.
    int64 offsets (4*H*M can exceed int32 at L=1024, H=128)."""
    Mi = M.to(tl.int64)
    idx = tl.program_id(0).to(tl.int64) * BLK + tl.arange(0, BLK).to(tl.int64)
    HM = H * Mi
    mask = idx < HM
    d = idx // Mi
    m = idx - d * Mi
    D2 = 2 * H
    gL = tl.load(preact + (2 * d) * Mi + m, mask=mask, other=0.0).to(tl.float32)
    pL = tl.load(preact + (2 * d + 1) * Mi + m, mask=mask, other=0.0).to(tl.float32)
    gR = tl.load(preact + (D2 + 2 * d) * Mi + m, mask=mask, other=0.0).to(tl.float32)
    pR = tl.load(preact + (D2 + 2 * d + 1) * Mi + m, mask=mask, other=0.0).to(tl.float32)
    et = lr.dtype.element_ty
    tl.store(lr + idx, (tl.sigmoid(gL) * pL).to(et), mask=mask)        # left plane d
    tl.store(lr + HM + idx, (tl.sigmoid(gR) * pR).to(et), mask=mask)   # right plane d


def trimul_front_sm100_train(x_n: torch.Tensor, b_lr: torch.Tensor, H: int):
    """SM100 v6-faithful front. Returns (left_bdll, right_bdll, preact_bdll):
      left/right : (B, H, L, L) contiguous  (== left.reshape(H,L,L) is a free view)
      preact     : (B, 4H, L, L)            ([g,p] interleaved per side, left|right)
    x_n : (B, L, L, D) contiguous bf16 ; b_lr : (D, 4H) from prepack_lr_operand_sm100.
    """
    assert x_n.dim() == 4 and x_n.is_cuda and x_n.is_contiguous()
    B, L, L2, D = x_n.shape
    assert B == 1 and L == L2
    M = L * L
    x_flat = x_n.reshape(M, D)
    # (1) non-gated m-major GEMM: preact[4H,L,L] written straight into the bdll buffer (no permute)
    preact = torch.empty(B, 4 * H, L, L, device=x_n.device, dtype=x_n.dtype)
    preact_view = preact.view(4 * H, M).T  # (M,4H) strides (1,M) -> m-major store
    gemm_act(A=x_flat, B=b_lr, activation=None, store_preact=False, postact_out=preact_view)
    # (2) GLU -> left/right [2H,L,L] bdll (contiguous)
    lr = torch.empty(B, 2 * H, L, L, device=x_n.device, dtype=x_n.dtype)
    grid = lambda meta: (triton.cdiv(H * M, meta["BLK"]),)  # noqa: E731
    _glu_bdll_kernel[grid](preact.view(4 * H, M), lr.view(2 * H, M), H=H, M=M,
                           BLK=4096, num_warps=8)
    return lr[:, :H], lr[:, H:], preact
