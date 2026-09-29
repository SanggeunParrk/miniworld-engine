"""The fused SWA atom DiT block's host side: which implementation serves each stage, and its launches.

Moved from team-gm ``swa_fused_triton.block_fwd`` / ``block_bwd`` (commit 14f2c73). The launch sequence, the buffers and
the arguments are team-gm's; what differs is the engine plumbing:

* the CUDA-vs-Triton choice reads ``settings`` instead of team-gm's environment switches
  (``SWA_QKVG_FWD`` / ``SWA_FFN_FWD`` / ``SWA_FFN_BWD`` -> ``swa_dit_{qkvg_fwd,ffn_fwd,ffn_bwd}_cuda``,
  ``SWA_FFN_DW`` -> ``swa_dit_ffn_dw``, ``SWA_DQ1`` -> ``swa_dit_dq1``; same defaults), and
  ``engine_backend="triton"`` turns every CUDA stage off, as it does for the other families;
* the CUDA kernels build through ``cuda/loader.py`` (``MINIWORLD_ENGINE_JIT_ROOT``) and any failure falls back to Triton;
* each Triton launch passes ``shape_key=atom_key(S, A=..., C=C[, NHID=NHID])`` where team-gm passed ``SB = log2(rows)``:
  the atom count S bucketed like every other atom-level kernel, and the AUGMENT COUNT A = N // B exactly, always -- the
  input feature embedder calls with A = 1 and diffusion with its num_augment (48 in training), and the row tiles of the
  modulation-reducing backward kernels (SP augments x AT atoms) want different shapes for the two;
* the forward and the backward are each ONE opaque op (``kernels._compile.opaque``): the per-stage choice is as
  untraceable as the launches, so it sits inside the same op. ``autograd.SWADiTBlockFunction`` owns the saved tensors.

bf16 stages. Forward: qkvg (hand-CUDA on sm_90, else ``_swa_qkvg_fwd_kernel``) -> window attention (Triton) ->
out-projection + gated residual + FFN (hand-CUDA on sm_90, else ``_swa_oproj_ffn_fwd_kernel``). Backward: FFN (hand-CUDA
on sm_90 with materialised dW operands, else ``_swa_ffn_bwd_kernel`` [+ ``_swa_ffn_dw_kernel`` when
``swa_dit_ffn_dw="fused"``]) -> out-projection -> attention dq, dk/dv -> qkvg, then the weight gradients as cuBLAS GEMMs
over the saved operands.

fp32 stages (q fp32; Triton only -- the hand-CUDA kernels are bf16 wgmma code): ``triton/forward_fp32.py`` and
``triton/backward_fp32.py`` for qkvg, out-projection + FFN and their backwards, around the SAME window-attention kernels on
bf16 operands (what FlashAttention-4 runs in the per-op fp32 path). ``swa_dit_ffn_dw`` and ``swa_dit_dq1`` are bf16 knobs;
the fp32 backward always materialises the dW operands and keeps dq1 in fp32.
"""
from __future__ import annotations

import functools
from typing import Any

import torch
import triton

from miniworld_engine import settings
from miniworld_engine.autotune.shape_key import atom_key
from miniworld_engine.kernels._compile import opaque
from miniworld_engine.kernels.swa_dit.interface import (
    D_ATOM,
    FP32_EPS,
    N_HEAD,
    N_HIDDEN,
)
from miniworld_engine.kernels.swa_dit.triton.backward import (
    _swa_attn_bwd_dkv_kernel,
    _swa_attn_bwd_dq_kernel,
    _swa_ffn_bwd_kernel,
    _swa_ffn_dw_kernel,
    _swa_oproj_bwd_kernel,
    _swa_qkvg_bwd_kernel,
)
from miniworld_engine.kernels.swa_dit.triton.backward_fp32 import (
    _swa_ffn_bwd_fp32_kernel,
    _swa_oproj_bwd_fp32_kernel,
    _swa_qkvg_bwd_fp32_kernel,
)
from miniworld_engine.kernels.swa_dit.triton.forward import (
    _swa_attn_fwd_kernel,
    _swa_oproj_ffn_fwd_kernel,
    _swa_qkvg_fwd_kernel,
)
from miniworld_engine.kernels.swa_dit.triton.forward_fp32 import (
    _swa_oproj_ffn_fwd_fp32_kernel,
    _swa_qkvg_fwd_fp32_kernel,
)

#: ``pack`` keys an axis only below 4096; an augment count at or above it shares the top key.
_A_KEY_MAX = 4095


def _augments(N: int, B: int) -> int:
    """The augment count as the cache key carries it: A = N // B, exactly (1 included), capped to what ``pack`` keys."""
    return min(N // B, _A_KEY_MAX)


def _cuda(which: str, wanted: bool, device: torch.device) -> Any:
    """The hand-CUDA extension for stage ``which``, or None -> the Triton kernel serves the stage."""
    if not wanted or settings.current().engine_backend == "triton":
        return None
    from miniworld_engine.kernels.swa_dit.cuda.loader import extension, is_sm90

    if not is_sm90(device):
        return None
    return extension(which)


@functools.lru_cache(maxsize=16)
def _sm_count(index: int) -> int:
    return torch.cuda.get_device_properties(index).multi_processor_count


def _pack_ffn64(wu: torch.Tensor) -> torch.Tensor:
    """rows per 64-wide hidden chunk j: [Wu[64j..64j+63] ; Wu[NHID + 64j..]] (the forward kernel's a|b tile)."""
    NH = wu.shape[0] // 2
    C = wu.shape[1]
    return wu.view(2, NH // 64, 64, C).permute(1, 0, 2, 3).reshape(2 * NH, C).contiguous()


def _pack_ffn(wu: torch.Tensor, wd: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """wab: per 32-wide hidden chunk j rows [Wu[32j..] ; Wu[NHID + 32j..]]; wdt = Wd^T; wabt = wab^T."""
    NH = wd.shape[1]
    C = wd.shape[0]
    wab = wu.view(2, NH // 32, 32, C).permute(1, 0, 2, 3).reshape(2 * NH, C).contiguous()
    return wab, wd.t().contiguous(), wab.t().contiguous()


def _swa_dit_block_fwd_fake(q, mod, cos, sin, seqused, wqkv, wg, wo, wu, wd, B, half_window, eps, save):
    """Shapes of the forward's outputs: [out [N, S, C] in q's dtype] alone, or with ``save`` the 13 tensors the backward
    reads -- Q / K / V head-major [N, 4, S, C/4] bf16 (the attention operands in both precisions), G [N*S, C] in q's dtype,
    O [N*S, C] bf16 (the attention output), lse [N, 4, S] fp32, then q1, x, pq, pk, att, y, ffn [N*S, C] in q's dtype.
    The structure depends on ``save`` alone."""
    N, S, C = q.shape
    H = N_HEAD
    M = N * S
    outputs = [q.new_empty((N, S, C))]
    if save:
        heads = [q.new_empty((N, H, S, C // H), dtype=torch.bfloat16) for _ in range(3)]
        saved = [q.new_empty((M, C)) for _ in range(7)]
        outputs += [*heads, q.new_empty((M, C)), q.new_empty((M, C), dtype=torch.bfloat16),
                    q.new_empty((N, H, S), dtype=torch.float32), *saved]
    return outputs


@opaque(fake=_swa_dit_block_fwd_fake, name="swa_dit_block_fwd")
def swa_dit_block_fwd(q: torch.Tensor, mod: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, seqused: torch.Tensor,
                      wqkv: torch.Tensor, wg: torch.Tensor, wo: torch.Tensor, wu: torch.Tensor, wd: torch.Tensor, B: int,
                      half_window: int, eps: float, save: bool) -> list[torch.Tensor]:
    """The block forward. q [N, S, C] bf16 or fp32 (the weights in the same dtype); mod [B*S, 6C] fp32 (hoisted adaLN
    modulation of this block); cos / sin [B*S, D/2] fp32; seqused [N] int32. Returns [out] or, with ``save``,
    [out, Qh, Kh, Vh, G, O, lse, q1, X, PQ, PK, Att, Y, FF] (the tensors the backward reads)."""
    if q.dtype == torch.float32:
        return _swa_dit_fwd_fp32_launch(q, mod, cos, sin, seqused, wqkv, wg, wo, wu, wd, B, half_window, eps, save)
    N, S, C = q.shape
    H = N_HEAD
    D = C // H
    M = N * S
    A = _augments(N, B)
    NHID = wd.shape[1]
    policy = settings.current()
    qf = q.reshape(M, C)
    ext = _cuda("qkvg", policy.swa_dit_qkvg_fwd_cuda and C == D_ATOM and H == N_HEAD and q.dtype == torch.bfloat16, q.device)
    if ext is not None:
        Qh, Kh, Vh, G, Xs, PQs, PKs = ext.qkvg_fwd(qf, mod, cos, sin, torch.cat([wqkv, wg]), N // B, B, S, eps, FP32_EPS, save)
    else:
        Qh = torch.empty(N, H, S, D, device=q.device, dtype=q.dtype)
        Kh = torch.empty_like(Qh)
        Vh = torch.empty_like(Qh)
        G = torch.empty_like(qf)
        r1 = torch.empty(M, device=q.device, dtype=torch.float32)
        if save:
            Xs = torch.empty_like(qf)
            PQs = torch.empty_like(qf)
            PKs = torch.empty_like(qf)
        else:
            Xs = PQs = PKs = G
        _swa_qkvg_fwd_kernel[lambda m: (triton.cdiv(M, m["BR"]),)](
            qf, mod, cos, sin, wqkv, wg, Qh, Kh, Vh, G, r1, Xs, PQs, PKs, M, S, B, eps, FP32_EPS,
            shape_key=atom_key(S, A=A, C=C), C=C, H=H, D=D, MODW=6 * C, SAVE=save)
    O = torch.empty_like(qf)
    lse = torch.empty(N, H, S, device=q.device, dtype=torch.float32)
    _swa_attn_fwd_kernel[lambda m: (triton.cdiv(S, m["BM"]), N * H)](
        Qh, Kh, Vh, seqused, O, lse, S, D ** -0.5, shape_key=atom_key(S, A=A, C=C), C=C, H=H, D=D, HW=half_window)
    ext = _cuda("fwd", policy.swa_dit_ffn_fwd_cuda and C == D_ATOM and NHID == N_HIDDEN and q.dtype == torch.bfloat16, q.device)
    if ext is not None:
        q2, q1, Att, Ys, FFs = ext.ffn_fwd(qf, G, O, mod, wo, _pack_ffn64(wu), wd, N // B, B, S, eps, save)
    else:
        q2 = torch.empty_like(qf)
        q1 = torch.empty_like(qf) if save else q2
        r2 = torch.empty(M, device=q.device, dtype=torch.float32)
        if save:
            Att = torch.empty_like(qf)
            Ys = torch.empty_like(qf)
            ABs = Ys
            FFs = torch.empty_like(qf)
        else:
            Att = Ys = ABs = FFs = q2
        _swa_oproj_ffn_fwd_kernel[lambda m: (triton.cdiv(M, m["BR"]),)](
            qf, O, G, mod, wo, wu, wd, q1, q2, r2, Att, Ys, ABs, FFs, M, S, B, eps,
            shape_key=atom_key(S, A=A, C=C, NHID=NHID), C=C, NHID=NHID, MODW=6 * C, SAVE=save)
    outputs = [q2.view(N, S, C)]
    if save:
        outputs += [Qh, Kh, Vh, G, O, lse, q1, Xs, PQs, PKs, Att, Ys, FFs]
    return outputs


def _swa_dit_fwd_fp32_launch(q, mod, cos, sin, seqused, wqkv, wg, wo, wu, wd, B, half_window, eps, save):
    """The fp32 forward: ``forward_fp32`` qkvg -> window attention on bf16 operands -> ``forward_fp32`` out-proj + FFN.
    Same outputs as the bf16 forward; everything but the attention operands (Q / K / V, O) is fp32."""
    N, S, C = q.shape
    H = N_HEAD
    D = C // H
    M = N * S
    A = _augments(N, B)
    NHID = wd.shape[1]
    dev = q.device
    qf = q.reshape(M, C)
    Qh = torch.empty(N, H, S, D, device=dev, dtype=torch.bfloat16)
    Kh = torch.empty_like(Qh)
    Vh = torch.empty_like(Qh)
    G = torch.empty_like(qf)
    r1 = torch.empty(M, device=dev, dtype=torch.float32)
    if save:
        Xs = torch.empty_like(qf)
        PQs = torch.empty_like(qf)
        PKs = torch.empty_like(qf)
    else:
        Xs = PQs = PKs = G
    _swa_qkvg_fwd_fp32_kernel[lambda m: (triton.cdiv(M, m["BR"]),)](
        qf, mod, cos, sin, wqkv, wg, Qh, Kh, Vh, G, r1, Xs, PQs, PKs, M, S, B, eps, FP32_EPS,
        shape_key=atom_key(S, A=A, C=C), C=C, H=H, D=D, MODW=6 * C, SAVE=save)
    O = torch.empty(M, C, device=dev, dtype=torch.bfloat16)
    lse = torch.empty(N, H, S, device=dev, dtype=torch.float32)
    _swa_attn_fwd_kernel[lambda m: (triton.cdiv(S, m["BM"]), N * H)](
        Qh, Kh, Vh, seqused, O, lse, S, D ** -0.5, shape_key=atom_key(S, A=A, C=C), C=C, H=H, D=D, HW=half_window)
    q2 = torch.empty_like(qf)
    q1 = torch.empty_like(qf) if save else q2
    r2 = torch.empty(M, device=dev, dtype=torch.float32)
    if save:
        Att = torch.empty_like(qf)
        Ys = torch.empty_like(qf)
        FFs = torch.empty_like(qf)
    else:
        Att = Ys = FFs = q2
    _swa_oproj_ffn_fwd_fp32_kernel[lambda m: (triton.cdiv(M, m["BR"]),)](
        qf, O, G, mod, wo, wu, wd, q1, q2, r2, Att, Ys, FFs, M, S, B, eps,
        shape_key=atom_key(S, A=A, C=C, NHID=NHID), C=C, NHID=NHID, MODW=6 * C, SAVE=save)
    outputs = [q2.view(N, S, C)]
    if save:
        outputs += [Qh, Kh, Vh, G, O, lse, q1, Xs, PQs, PKs, Att, Ys, FFs]
    return outputs


def _swa_dit_block_bwd_fake(dy, q, mod, cos, sin, seqused, wqkv, wg, wo, wu, wd, qh, kh, vh, g, o, lse, q1, x, pq, pk,
                            att, y, ffn, B, half_window, eps):
    """Shapes of the backward's outputs: dq like q ([N, S, C], q's dtype), dmod like mod (fp32 [B*S, 6C]), and each
    weight gradient in its weight's shape and dtype."""
    return [torch.empty_like(q), torch.empty_like(mod), torch.empty_like(wqkv), torch.empty_like(wg), torch.empty_like(wo),
            torch.empty_like(wu), torch.empty_like(wd)]


@opaque(fake=_swa_dit_block_bwd_fake, name="swa_dit_block_bwd")
def swa_dit_block_bwd(dy: torch.Tensor, q: torch.Tensor, mod: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                      seqused: torch.Tensor, wqkv: torch.Tensor, wg: torch.Tensor, wo: torch.Tensor, wu: torch.Tensor,
                      wd: torch.Tensor, qh: torch.Tensor, kh: torch.Tensor, vh: torch.Tensor, g: torch.Tensor, o: torch.Tensor,
                      lse: torch.Tensor, q1: torch.Tensor, x: torch.Tensor, pq: torch.Tensor, pk: torch.Tensor,
                      att: torch.Tensor, y: torch.Tensor, ffn: torch.Tensor, B: int, half_window: int,
                      eps: float) -> list[torch.Tensor]:
    """Backward of :func:`swa_dit_block_fwd` from its saved tensors. dy [N, S, C] in q's dtype. Returns [dq [N, S, C],
    dmod [B*S, 6C] fp32, dWqkv, dWg, dWo, dWu, dWd], each in its input's dtype."""
    if q.dtype == torch.float32:
        return _swa_dit_bwd_fp32_launch(dy, q, mod, cos, sin, seqused, wqkv, wg, wo, wu, wd, qh, kh, vh, g, o, lse, q1, x,
                                        pq, pk, att, y, ffn, B, half_window, eps)
    N, S, C = q.shape
    H = N_HEAD
    D = C // H
    M = N * S
    A = N // B
    Ak = _augments(N, B)
    NHID = wd.shape[1]
    dev = q.device
    policy = settings.current()
    fused_dw = policy.swa_dit_ffn_dw == "fused"
    dq1_dtype = torch.float32 if policy.swa_dit_dq1 == "fp32" else torch.bfloat16
    dmod = torch.zeros(B * S, 6 * C, device=dev, dtype=torch.float32)
    dq1 = torch.empty(M, C, device=dev, dtype=dq1_dtype)
    dO = torch.empty(M, C, device=dev, dtype=torch.bfloat16)
    dG = torch.empty_like(dO)
    Dv = torch.empty(N, H, S, device=dev, dtype=torch.float32)
    if fused_dw:
        dab = hh = dffn = dO
    else:
        dab = torch.empty(M, 2 * NHID, device=dev, dtype=torch.bfloat16)
        hh = torch.empty(M, NHID, device=dev, dtype=torch.bfloat16)
        dffn = torch.empty_like(dO)
    datt = torch.empty_like(dO)
    gated = torch.empty_like(dO)
    grid_t = lambda m: (triton.cdiv(S, m["AT"]), triton.cdiv(A, m["SP"]), B)
    dy2 = dy.reshape(M, C)
    dwu: torch.Tensor | None = None
    dwd: torch.Tensor | None = None
    wanted = policy.swa_dit_ffn_bwd_cuda and not fused_dw and C == D_ATOM and NHID == N_HIDDEN and q.dtype == torch.bfloat16
    ext = _cuda("bwd", wanted, dev)
    if ext is not None:
        dq1, dffn, hh, dab = ext.ffn_bwd(dy2, q1, y, ffn, mod, dmod, *_pack_ffn(wu, wd), A, B, S, eps, 8 if A % 8 == 0 else 0)
    else:
        _swa_ffn_bwd_kernel[grid_t](
            dy2, q1, mod, wu, wd, y, ffn, dq1, dab, hh, dffn, dmod, S, A, B, eps,
            shape_key=atom_key(S, A=Ak, C=C, NHID=NHID), C=C, NHID=NHID, MODW=6 * C, DWOPS=not fused_dw)
        if fused_dw:
            dwu32 = torch.zeros(2 * NHID, C, device=dev, dtype=torch.float32)
            dwd32 = torch.zeros(C, NHID, device=dev, dtype=torch.float32)
            sms = _sm_count(dev.index if dev.index is not None else torch.cuda.current_device())
            ns = lambda m: max(1, min(triton.cdiv(M, m["BR"]), 2 * sms // (NHID // m["HS"])))
            _swa_ffn_dw_kernel[lambda m: (ns(m), NHID // m["HS"])](
                dy2, mod, wu, wd, y, dwu32, dwd32, M, S, B,
                shape_key=atom_key(S, A=Ak, C=C, NHID=NHID), NSPLIT=0, C=C, NHID=NHID, MODW=6 * C)
            dwu = dwu32.to(wu.dtype)
            dwd = dwd32.to(wd.dtype)
    _swa_oproj_bwd_kernel[grid_t](
        dq1, o, g, mod, wo, att, dO, dG, Dv, datt, gated, dmod, S, A, B,
        shape_key=atom_key(S, A=Ak, C=C), C=C, H=H, D=D, MODW=6 * C)
    dQh = torch.empty(N, H, S, D, device=dev, dtype=torch.bfloat16)
    dKh = torch.empty_like(dQh)
    dVh = torch.empty_like(dQh)
    _swa_attn_bwd_dq_kernel[lambda m: (triton.cdiv(S, m["BM"]), N * H)](
        qh, kh, vh, dO, lse, Dv, seqused, dQh, S, D ** -0.5, shape_key=atom_key(S, A=Ak, C=C), C=C, H=H, D=D, HW=half_window)
    _swa_attn_bwd_dkv_kernel[lambda m: (triton.cdiv(S, m["BN"]), N * H)](
        qh, kh, vh, dO, lse, Dv, seqused, dKh, dVh, S, D ** -0.5, shape_key=atom_key(S, A=Ak, C=C), C=C, H=H, D=D,
        HW=half_window)
    dq = torch.empty(M, C, device=dev, dtype=torch.bfloat16)
    dP = torch.empty(M, 4 * C, device=dev, dtype=torch.bfloat16)
    _swa_qkvg_bwd_kernel[grid_t](
        q.reshape(M, C), mod, cos, sin, wqkv, wg, pq, pk, dQh, dKh, dVh, dG, dq1, dq, dP, dmod, S, A, B, eps, FP32_EPS,
        shape_key=atom_key(S, A=Ak, C=C), C=C, H=H, D=D, MODW=6 * C)
    dwqkv = dP[:, :3 * C].t() @ x
    dwg = dP[:, 3 * C:].t() @ x
    dwo = datt.t() @ gated
    if dwu is None or dwd is None:
        dwu = dab.t() @ y
        dwd = dffn.t() @ hh
    return [dq.view(N, S, C), dmod, dwqkv, dwg, dwo, dwu, dwd]


def _swa_dit_bwd_fp32_launch(dy, q, mod, cos, sin, seqused, wqkv, wg, wo, wu, wd, qh, kh, vh, g, o, lse, q1, x, pq, pk,
                             att, y, ffn, B, half_window, eps):
    """The fp32 backward: ``backward_fp32`` FFN -> out-projection -> the attention backward on bf16 operands (dO, and
    dQ / dK / dV, in bf16 as FlashAttention-4 has them) -> ``backward_fp32`` qkvg; fp32 weight gradients from torch GEMMs
    over the fp32 operands."""
    N, S, C = q.shape
    H = N_HEAD
    D = C // H
    M = N * S
    A = N // B
    Ak = _augments(N, B)
    NHID = wd.shape[1]
    dev = q.device
    f32 = torch.float32
    dmod = torch.zeros(B * S, 6 * C, device=dev, dtype=f32)
    dq1 = torch.empty(M, C, device=dev, dtype=f32)
    dO = torch.empty(M, C, device=dev, dtype=torch.bfloat16)
    dG = torch.empty(M, C, device=dev, dtype=f32)
    Dv = torch.empty(N, H, S, device=dev, dtype=f32)
    dab = torch.empty(M, 2 * NHID, device=dev, dtype=f32)
    hh = torch.empty(M, NHID, device=dev, dtype=f32)
    dffn = torch.empty(M, C, device=dev, dtype=f32)
    datt = torch.empty(M, C, device=dev, dtype=f32)
    gated = torch.empty(M, C, device=dev, dtype=f32)
    grid_t = lambda m: (triton.cdiv(S, m["AT"]), triton.cdiv(A, m["SP"]), B)
    _swa_ffn_bwd_fp32_kernel[grid_t](
        dy.reshape(M, C), q1, mod, wu, wd, y, ffn, dq1, dab, hh, dffn, dmod, S, A, B, eps,
        shape_key=atom_key(S, A=Ak, C=C, NHID=NHID), C=C, NHID=NHID, MODW=6 * C)
    _swa_oproj_bwd_fp32_kernel[grid_t](
        dq1, o, g, mod, wo, att, dO, dG, Dv, datt, gated, dmod, S, A, B,
        shape_key=atom_key(S, A=Ak, C=C), C=C, H=H, D=D, MODW=6 * C)
    dQh = torch.empty(N, H, S, D, device=dev, dtype=torch.bfloat16)
    dKh = torch.empty_like(dQh)
    dVh = torch.empty_like(dQh)
    _swa_attn_bwd_dq_kernel[lambda m: (triton.cdiv(S, m["BM"]), N * H)](
        qh, kh, vh, dO, lse, Dv, seqused, dQh, S, D ** -0.5, shape_key=atom_key(S, A=Ak, C=C), C=C, H=H, D=D, HW=half_window)
    _swa_attn_bwd_dkv_kernel[lambda m: (triton.cdiv(S, m["BN"]), N * H)](
        qh, kh, vh, dO, lse, Dv, seqused, dKh, dVh, S, D ** -0.5, shape_key=atom_key(S, A=Ak, C=C), C=C, H=H, D=D,
        HW=half_window)
    dq = torch.empty(M, C, device=dev, dtype=f32)
    dP = torch.empty(M, 4 * C, device=dev, dtype=f32)
    _swa_qkvg_bwd_fp32_kernel[grid_t](
        q.reshape(M, C), mod, cos, sin, wqkv, wg, pq, pk, dQh, dKh, dVh, dG, dq1, dq, dP, dmod, S, A, B, eps, FP32_EPS,
        shape_key=atom_key(S, A=Ak, C=C), C=C, H=H, D=D, MODW=6 * C)
    dwqkv = dP[:, :3 * C].t() @ x
    dwg = dP[:, 3 * C:].t() @ x
    dwo = datt.t() @ gated
    dwu = dab.t() @ y
    dwd = dffn.t() @ hh
    return [dq.view(N, S, C), dmod, dwqkv, dwg, dwo, dwu, dwd]
