"""The token DiT block's conditioning projections on the sm_80 path (A100; the RTX 5090 runs the sm_120a build of the same sources): inference, bf16 activations.

A sampler step calls every token block with the SAME conditioning rows for all of its A samples (``cond`` is the per-token single conditioning, expanded over the samples: stride 0), and each block
projects it six times: the attention AdaLN's scale and bias (on LN(cond) w_a), the attention output gate, the transition AdaLN's scale and bias (on LN(cond) w_t) and the transition gate (on cond).
Run as separate cuBLAS Linears over the A x L staged rows (and, for the attention AdaLN, as PyTorch ops under bf16 autocast, with the fp32 master weights cast at every call) these are about 70 us of
a block's ~180 us on the 5I28 n = 1 ESMFold2 step; the projections are tiny GEMMs (M = 128 distinct rows) that need no more than two launches:

    hat   = LN(cond)                                            one pass over the P distinct rows (``cond_ln`` with unit weights)
    [S_a | B_a | S_t | B_t] = hat [w_a Ws_a; w_a Wb_a; w_t Ws_t; w_t Wb_t]^T         one GEMM, the LayerNorm weights folded into the (cached, bf16) weight pack
    [g_a | g_t] = cond [Wg_a; Wg_t]^T + [bg_a; bg_t]                                    one batched GEMM (the bias is in the accumulator's rounding, as the Linear's)

and the rows kernels read the tables with their period (row r of the A x L rows takes table row r % P: the epilogue ``y = sigmoid(S + sb) LN(x) + B``, and the gate kernel
``out = res + sigmoid(g) z``). ``augattn_sm80._fwd`` takes the attention gate as ``g2_table``; the transition's tail is the existing swiglu / GEMM / gate kernels.

Numerics: the LayerNorm weight is multiplied into the projection weight before the bf16 rounding (instead of into the activation before it): the same bf16 rounding noise, a different rounding
(inference only; ``MINIWORLD_TOKEN_COND_PACK=0`` turns the path off for an A / B).
"""

from __future__ import annotations

import os

import torch

from foldjet_engine.kernels._compile import opaque
from foldjet_engine.kernels._device import capability as _devcap
from foldjet_engine.kernels.integrations import adaln_sm80 as _ad
from foldjet_engine.kernels.integrations import augattn_sm80 as _aa
from foldjet_engine.kernels.integrations import conditioned_transition_sm80 as _ct

BF = torch.bfloat16


def serves(block, single: torch.Tensor, cond: torch.Tensor, grad: bool) -> bool:
    """Whether this call of ``TokenDiTProtenix._forward_a100`` takes the pack: inference, bf16 activations, one fold, L a multiple of 128 (no row padding: the tables' period is L), both AdaLNs
    the sigmoid form of one LayerNorm epsilon."""
    if grad or os.environ.get("MINIWORLD_TOKEN_COND_PACK", "1") == "0":
        return False
    if _devcap(single.device) != (8, 0):  # measured and tested on the A100 only; the RTX 5090 keeps its unpacked path (cuda:5090_sm120 unchanged)
        return False
    if single.dtype is not BF or cond.dtype is not BF or single.ndim != 4 or single.shape[1] != 1 or cond.ndim != 4:
        return False
    d, dc = single.shape[-1], cond.shape[-1]
    if single.shape[2] % 128 or _aa._pad(single.shape[2]) != single.shape[2] or d not in _ad.WIDTHS or dc not in _ad.WIDTHS:
        return False
    ada_a, ada_t = block.attention.ada_ln_in, block.transition.ada_ln_in
    if _ad.is_identity(ada_a) or _ad.is_identity(ada_t) or float(ada_a.ln_cond.eps) != float(ada_t.ln_cond.eps):
        return False
    return _ad._period(single, cond, False) is not None


def _front_fake(x, cs, a_lnw, a_ws, a_sb, a_wb, a_gw, a_gb, t_lnw, t_ws, t_sb, t_wb, t_gw, t_gb, eps_x, eps_c):
    p, d = cs.shape[0], x.shape[-1]
    return [x.new_empty(x.shape), x.new_empty((p, 4 * d)), x.new_empty((2, p, d))]


@opaque(fake=_front_fake, name="token_cond_front_sm80")
def _front(
    x: torch.Tensor,
    cs: torch.Tensor,
    a_lnw: torch.Tensor,
    a_ws: torch.Tensor,
    a_sb: torch.Tensor,
    a_wb: torch.Tensor,
    a_gw: torch.Tensor,
    a_gb: torch.Tensor,
    t_lnw: torch.Tensor,
    t_ws: torch.Tensor,
    t_sb: torch.Tensor,
    t_wb: torch.Tensor,
    t_gw: torch.Tensor,
    t_gb: torch.Tensor,
    eps_x: float,
    eps_c: float,
) -> list[torch.Tensor]:
    """x [M, d] (the block's input rows), cs [P, dc] (the distinct conditioning rows, P | M) -> [attention AdaLN output [M, d], the table [P, 4 d] = S_a | B_a | S_t | B_t, the gate logits [2, P, d]]."""
    ka = _ct._adaln()
    d = x.shape[-1]
    params = (a_lnw, a_ws, a_sb, a_wb, a_gw, a_gb, t_lnw, t_ws, t_sb, t_wb, t_gw, t_gb)

    def build():
        wa = torch.cat([a_ws.float() * a_lnw.float()[None], a_wb.float() * a_lnw.float()[None], t_ws.float() * t_lnw.float()[None], t_wb.float() * t_lnw.float()[None]]).to(BF)
        wg = torch.stack([a_gw, t_gw]).to(BF).contiguous()
        bg = torch.stack([a_gb, t_gb]).to(BF)[:, None, :].contiguous()
        return wa, wg, bg, a_sb.to(BF).contiguous(), torch.ones_like(a_lnw, dtype=torch.float32)

    with torch.cuda.device(x.device):
        wa, wg, bg, sb_a, ones = _ad.cached("token_cond_front", params, build)
        hat, _ = ka.cond_ln(cs, ones, BF, eps_c)
        table = torch.mm(hat, wa.t())
        gate = torch.baddbmm(bg, cs.unsqueeze(0).expand(2, -1, -1), wg.transpose(1, 2))
        xa, _ = ka.epilogue(x, table[:, : 2 * d], sb_a, eps_x)
    return [xa, table, gate]


def _tail_fake(x, table, gate, t_sb, wa, wb, ws, eps_x):
    return torch.empty_like(x)


@opaque(fake=_tail_fake, name="token_cond_tail_sm80")
def _tail(
    x: torch.Tensor,
    table: torch.Tensor,
    gate: torch.Tensor,
    t_sb: torch.Tensor,
    wa: torch.Tensor,
    wb: torch.Tensor,
    ws: torch.Tensor,
    eps_x: float,
) -> torch.Tensor:
    """The transition on the table: x [M, d] -> x + sigmoid(g_t) (swiglu(AdaLN(x) [Wa; Wb]^T) Ws^T)."""
    ka, kt = _ct._adaln(), _ct._tail()
    d = x.shape[-1]
    with torch.cuda.device(x.device):
        wa, wb, ws, sb = _ad.cast_params((wa, wb, ws, t_sb), BF, True)
        wab = _ad.cached("token_cond_wab", (wa, wb), lambda: torch.cat([wa, wb]))
        xb, _ = ka.epilogue(x, table[:, 2 * d :], sb, eps_x)
        h = kt.swiglu_fwd(torch.mm(xb, wab.t()))
        z = torch.mm(h, ws.t())
        return kt.gate_res_fwd(x, z, gate[1])


def forward(block, single: torch.Tensor, cond: torch.Tensor, pair: torch.Tensor, mask, extra=None) -> torch.Tensor:
    """``TokenDiTProtenix._forward_a100`` for a call ``serves`` accepted (the attention and transition ranges are the caller's)."""
    att, tr = block.attention, block.transition
    ada_a, ada_t = att.ada_ln_in, tr.ada_ln_in
    a, _, length, d = single.shape
    m = a * length
    p = _ad._period(single, cond, False)
    cs = (cond if p == m else cond[0]).reshape(-1, cond.shape[-1]).contiguous()
    res = single.contiguous()
    xa, table, gate = _front(
        res.view(m, d),
        cs,
        ada_a.ln_cond.weight,
        ada_a.to_scale.weight,
        ada_a.to_scale.bias,
        ada_a.to_bias.weight,
        att.to_scale.weight,
        att.to_scale.bias,
        ada_t.ln_cond.weight,
        ada_t.to_scale.weight,
        ada_t.to_scale.bias,
        ada_t.to_bias.weight,
        tr.to_scale.weight,
        tr.to_scale.bias,
        float(ada_a.ln_in.eps),
        float(ada_a.ln_cond.eps),
    )
    mid = _aa._fused_update(att, xa.view(a, 1, length, d), cond, pair, mask, res, extra, gate[0])
    out = _tail(
        mid.reshape(m, d),
        table,
        gate,
        ada_t.to_scale.bias,
        tr.expand_a.weight,
        tr.expand_b.weight,
        tr.squeeze.weight,
        float(ada_t.ln_in.eps),
    )
    return out.view(a, 1, length, d)
