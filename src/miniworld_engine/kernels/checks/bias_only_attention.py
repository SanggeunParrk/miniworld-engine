"""Accuracy checks for the ``bias_only_attention`` family.

The three attention families were one module (``checks_attn.py``). What the references have to
get right -- the ``(m, n)``-only bias indexing, the key mask, the log2-space ``m`` the forward
checkers also return, and why the backward references are fp32 autograd on a dense ``dy`` -- is
written out in ``checks/triangle_attention.py``; the helpers all three use are in
``checks/__init__.py``.
"""
from __future__ import annotations

import torch
import triton

from miniworld_engine.kernels.checks import (
    Pair,
    _fp32_matmul,
    _fwd_saved,
    _grads,
    _lse2,
    _rowsum,
)
from miniworld_engine.kernels.drivers import BF16, TensorKw, dev
from miniworld_engine.kernels.drivers.bias_only_attention import DH, DP, _bias_only_vb
from miniworld_engine.kernels.drivers.triangle_attention import D, H, L

def _bias_only_ref(v, bias) -> torch.Tensor:
    """No q/k: p = softmax(bias) is one ``[B,H,L,L]`` matrix, reused by every row of v."""
    return torch.einsum("bhmn,bhind->bhimd", bias.softmax(-1), v)


# ── bias_only_attention / triton / main.py ──────────────────────────────────────────────────


def bias_only_attention_fwd_triton() -> dict[str, Pair]:
    from miniworld_engine.kernels.bias_only_attention.triton.main import TritonBiasOnlyAttentionFunction as Fn
    _fp32_matmul()
    v, bias = _bias_only_vb()
    # save order: (v, bias, m, out) -> m at 2.
    out, m = _fwd_saved(Fn.apply, (v, bias), 2)
    vf, bf = v.detach().float(), bias.detach().float()
    # There is no q/k, so the logits ARE the bias and m does not depend on the row axis at all:
    # the kernel's bias base offset carries no off_t, yet it still stores m per row into a
    # [B,H,L,L] buffer indexed [b, h, row i, query j]. So the reference is one [B,H,L] log-sum-exp
    # over the key axis, broadcast back over the L rows -- if the kernel ever let a row leak into
    # its softmax, this is where it would show.
    m_ref = _lse2(bf).unsqueeze(2).expand(-1, -1, L, -1)
    return {"out": (out, _bias_only_ref(vf, bf)), "m": (m, m_ref)}


def bias_only_attention_bwd_pre_triton() -> Pair:
    from miniworld_engine.autotune.shape_key import token_key

    from miniworld_engine.kernels.bias_only_attention.triton.main import _attn_bwd_preprocess
    B, HL = 1, H * L
    # Contiguous addressing, no strides -- same contract as atomic.py's, and the reason its
    # backward calls grad_output.contiguous() before this launch.
    o = torch.randn(B, HL, L, D, device=dev(), dtype=BF16)
    do = torch.randn_like(o)
    delta = torch.empty(B, HL, L, device=dev(), dtype=torch.float32)
    grid = lambda META: [triton.cdiv(L, META["BLOCK_M1"]), B * HL, 1]
    _attn_bwd_preprocess[grid](
        o, do, delta, B, L, D,
        shape_key=token_key(L), HEAD_DIM_PAD=triton.next_power_of_2(D),
    )
    return delta, _rowsum(o, do)


def bias_only_attention_bwd_triton() -> dict[str, Pair]:
    from miniworld_engine.kernels.bias_only_attention.triton.main import TritonBiasOnlyAttentionFunction as Fn
    # Only two grads exist on this path: there is no q/k to differentiate.
    return _grads(Fn.apply, _bias_only_vb(), _bias_only_ref, ("dv", "dbias"))


# ── bias_only_attention / triton / gate_out.py ──────────────────────────────────────────────


def gated_projection_gate_gemm_triton() -> Pair:
    from miniworld_engine.kernels.bias_only_attention.triton.gate_out import _fwd
    kw: TensorKw = {"device": dev(), "dtype": BF16}
    gate = torch.randn(L * L, DH, **kw)
    out_r = torch.randn(L * L, DH, **kw)
    wo = torch.randn(DP, DH, **kw)          # to_out.weight [N, DH]; out = A @ wo.T
    # The fused kernel builds its A-tile as sigmoid(gate)*out_r in the GEMM prologue, so `gated`
    # never reaches HBM. The reference forms it and does the GEMM in fp32.
    ref = (torch.sigmoid(gate.float()) * out_r.float()) @ wo.float().t()
    return _fwd(gate, out_r, wo), ref


def gated_projection_bwd_dx_triton() -> dict[str, Pair]:
    from miniworld_engine.kernels.bias_only_attention.triton.gate_out import _dgrad_epilogue
    kw: TensorKw = {"device": dev(), "dtype": BF16}
    do2 = torch.randn(L * L, DP, **kw)      # grad wrt [M, N], N == d_pair
    wo = torch.randn(DP, DH, **kw)
    g2 = torch.randn(L * L, DH, **kw)
    r2 = torch.randn(L * L, DH, **kw)
    dr, dg, a = _dgrad_epilogue(do2, wo, g2, r2)
    # out = (sigmoid(gate) * out_r) @ wo.T, so with da = do @ wo:
    #   d_out_r = s*da,  d_gate = da*r*s*(1-s),  and `gated` = s*r is handed to the d_wo GEMM.
    da = do2.float() @ wo.float()
    s, r = torch.sigmoid(g2.float()), r2.float()
    return {
        "d_out_r": (dr, s * da),
        "d_gate": (dg, da * r * s * (1.0 - s)),
        "gated": (a, s * r),
    }
