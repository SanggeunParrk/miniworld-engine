"""A100 (sm_80) kernels of the fused SWA atom DiT block (``kernels/swa_dit``), hand-written CUDA (``mma.sync`` / ``ldmatrix`` / ``cp.async``).

The block is ``interface.swa_dit_block`` -- the ESMFold2 SWA atom block over the flattened [N = A*B, S] atom sequence -- and each stage here is the twin
of a Triton stage of ``dispatch.py`` with the same operands, layouts and rounding points, so the stages mix freely with the Triton ones:

    modulation  silu(c) Wmod^T (``mod_fwd``) and its backward (``mod_bwd``: dc, dWmod), behind ``interface.swa_dit_hoist_modulation``
    forward     qkvg (norm + modulate + the 4 projections + head norm + RoPE) -> window attention -> out-projection + gated residual + FFN   (``block_fwd``)
    backward    FFN (gate, then dy + the adaLN backward) -> out-projection -> window attention (dQ, then dK / dV) -> qkvg, each writing its columns of the modulation
                gradient once (``bwd_mode``: one modulation row per sample, or one shared by a multiple of 16 samples); the weight gradients stay cuBLAS GEMMs

Written for sm_80 (``mma.sync.m16n8k16`` bf16, ``ldmatrix``, ``cp.async``), d_atom 128, 4 heads x 32, SwiGLU hidden 256, half window 64.  Built on first use (``load_extension``), never at import.
"""
from __future__ import annotations

import functools
import hashlib
import os
from pathlib import Path

import torch

from miniworld_engine.kernels._nvcc import ensure_cuda_home, host_flags, load_extension

_dir = Path(__file__).parent

C, H, D, NHID, HW = 128, 4, 32, 256, 64


@functools.lru_cache(maxsize=1)
def _ext():
    ensure_cuda_home()
    # MINIWORLD_SWA_DIT_SM80_FLAGS: extra -D / nvcc flags for A/B experiments (their own build)
    extra = os.environ.get("MINIWORLD_SWA_DIT_SM80_FLAGS", "").split()
    tag = "" if not extra else "_" + hashlib.sha1(" ".join(extra).encode()).hexdigest()[:8]
    return load_extension(
        name=f"swa_dit_sm80{tag}",
        sources=[str(_dir / "ops.cu")],
        extra_cuda_cflags=[*host_flags(), "-std=c++17", "-O3", "-gencode=arch=compute_80,code=sm_80", f"-I{_dir}", *extra],
        extra_cflags=["-std=c++17", "-O3"], verbose=False,
    )


def attn_fwd(qh: torch.Tensor, kh: torch.Tensor, vh: torch.Tensor, seqused: torch.Tensor):
    """The window attention forward on head-major Q / K / V [N, 4, S, 32] bf16: returns ``(O [N S, 128] bf16, lse [N, 4, S] fp32)``."""
    n, _, s, _ = qh.shape
    o = torch.empty(n * s, C, device=qh.device, dtype=torch.bfloat16)
    lse = torch.empty(n, H, s, device=qh.device, dtype=torch.float32)
    _ext().swa_attn_fwd(qh.transpose(1, 2), kh.transpose(1, 2), vh.transpose(1, 2), seqused, o, lse)
    return o, lse


def attn_bwd(qh: torch.Tensor, kh: torch.Tensor, vh: torch.Tensor, d_o: torch.Tensor, lse: torch.Tensor, dvv: torch.Tensor, seqused: torch.Tensor):
    """The window attention backward on head-major Q / K / V [N, 4, S, 32] bf16 with ``d_o`` [N S, 128] bf16 (the gradient of the attention output), ``lse`` (the forward's) and ``dvv`` (D = rowsum(dO o), from the
    out-projection backward) [N, 4, S] fp32: returns ``(dQh, dKh, dVh)`` head-major bf16."""
    dq, dk, dv = (torch.empty_like(qh) for _ in range(3))
    _ext().swa_attn_bwd(qh.transpose(1, 2), kh.transpose(1, 2), vh.transpose(1, 2), d_o, lse, dvv, seqused, dq.transpose(1, 2), dk.transpose(1, 2), dv.transpose(1, 2))
    return dq, dk, dv


def window_supported(q: torch.Tensor, n_heads: int, half_window: int) -> bool:
    """Whether the window attention of ``modules/swa_atom_attention`` ([N, S, H, D] q) runs on these kernels: an A100, 4 heads x 32, half window 64 (``MINIWORLD_SWA_DIT_SM80=0`` keeps flash)."""
    if os.environ.get("MINIWORLD_SWA_DIT_SM80", "1") == "0":
        return False
    if not q.is_cuda or q.dim() != 4 or q.shape[2] != H or q.shape[3] != D or n_heads != H or half_window != HW or not q.dtype.is_floating_point:
        return False
    return _is_a100(q.device.index if q.device.index is not None else torch.cuda.current_device())


def window_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, seqused: torch.Tensor):
    """The window attention forward on q, k, v [N, S, 4, 32] bf16 views with any strides (unit channel stride, 16-byte aligned rows: the module's slices of the fused qkv projection are):
    returns ``(out [N, S, 4, 32] bf16, lse [N, 4, S] fp32)``; rows >= ``seqused`` are 0."""
    n, s = q.shape[:2]
    out = torch.empty(n, s, H, D, device=q.device, dtype=torch.bfloat16)
    lse = torch.empty(n, H, s, device=q.device, dtype=torch.float32)
    _ext().swa_attn_fwd(q, k, v, seqused, out, lse)
    return out, lse


def window_bwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, out: torch.Tensor, d_out: torch.Tensor, lse: torch.Tensor, seqused: torch.Tensor):
    """The backward of :func:`window_fwd`: ``out`` / ``d_out`` [N, S, 4, 32] bf16 contiguous (the forward's output and its gradient), ``lse`` the forward's: returns ``(dq, dk, dv)`` [N, S, 4, 32] bf16
    contiguous (rows >= ``seqused`` are 0).  The attention's row term D = rowsum(d_out out) is one small pass here (the fused block gets it from the out-projection backward)."""
    n, s = q.shape[:2]
    dvv = torch.empty(n, H, s, device=q.device, dtype=torch.float32)
    _ext().swa_attn_delta(d_out, out, dvv, s)
    dq, dk, dv = (torch.empty(n, s, H, D, device=q.device, dtype=torch.bfloat16) for _ in range(3))
    _ext().swa_attn_bwd(q, k, v, d_out, lse, dvv, seqused, dq, dk, dv)
    return dq, dk, dv


def qkvg_fwd(x: torch.Tensor, mod: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, wqkv: torch.Tensor, wg: torch.Tensor, b: int, s: int, eps: float,
             qk_eps: float, save: bool, cfg: int = 1):
    """The qkvg stage: ``x`` [M, 128] bf16 (M = N S, N = A b), ``mod`` [b S, 768] fp32, ``cos`` / ``sin`` [b S, 16] fp32.  Returns ``(Qh, Kh, Vh, G, X, PQ, PK)``: head-major
    [N, 4, S, 32] and row-major [M, 128] bf16; X, PQ, PK are the saves of the backward (G itself, unused, when ``save`` is False)."""
    m = x.shape[0]
    n = m // s
    dev = x.device
    qh, kh, vh = (torch.empty(n, H, s, D, device=dev, dtype=torch.bfloat16) for _ in range(3))
    g = torch.empty(m, C, device=dev, dtype=torch.bfloat16)
    xs, pqs, pks = (torch.empty(m, C, device=dev, dtype=torch.bfloat16) for _ in range(3)) if save else (g, g, g)
    _ext().swa_qkvg_fwd(x, mod, cos, sin, wqkv.contiguous(), wg.contiguous(), qh, kh, vh, g, xs, pqs, pks, s, b, eps, qk_eps, save, cfg)
    return qh, kh, vh, g, xs, pqs, pks


def ffn_fwd(qi: torch.Tensor, g: torch.Tensor, o: torch.Tensor, mod: torch.Tensor, wo: torch.Tensor, wu: torch.Tensor, wd: torch.Tensor, b: int, s: int, eps: float, save: bool,
            cfg: int = 0, prof: torch.Tensor | None = None):
    """The last stage: ``qi`` / ``g`` / ``o`` [M, 128] bf16 (the stage's residual input, the gate projection, the attention output), ``mod`` [b S, 768] fp32.  Returns
    ``(out, q1, Att, Y, FF)``: ``out`` [M, 128] bf16 and the saves of the backward (``out`` itself four times when ``save`` is False)."""
    m = qi.shape[0]
    out = torch.empty_like(qi)
    q1s, atts, ys, ffs = (torch.empty_like(qi) for _ in range(4)) if save else (out, out, out, out)
    _ext().swa_ffn_fwd(qi, g, o, mod, wo.contiguous(), wu.contiguous(), wd.contiguous(), out, q1s, atts, ys, ffs, s, b, eps, save, cfg, torch.empty(0, device=qi.device, dtype=torch.int64) if prof is None else prof)
    return out, q1s, atts, ys, ffs


#: columns of the adaLN modulation (shift / scale / gate of the attention and of the FFN, 128 each); P of the partial sums of dWmod; rows from which the wide tiles pay off
MOD_N = 6 * C
MOD_DW_PARTS = 36
_MOD_WIDE_ROWS = 40960


def mod_fwd(c: torch.Tensor, wmod: torch.Tensor, save: bool = False, cfg: int | None = None) -> list[torch.Tensor]:
    """The adaLN modulation ``rn(silu(c)) @ Wmod.T`` [R, 768] fp32: ``c`` [R, 128] bf16 contiguous, ``wmod`` [768, 128] bf16 (one hand kernel; the products of bf16 numbers are exact
    in fp32).  Returns ``[mod]``, or with ``save`` ``[mod, a]`` with ``a = rn(silu(c))`` [R, 128] bf16, the operand of :func:`mod_bwd`.  ``cfg`` (None: by the row count) 0 / 1: the
    row-stationary kernel with tiles of 128 / 64 rows, 2: the weight-stationary one."""
    rows = c.shape[0]
    out = torch.empty(rows, MOD_N, device=c.device, dtype=torch.float32)
    a = torch.empty(rows, C, device=c.device, dtype=torch.bfloat16) if save else c
    if cfg is None:
        cfg = 0 if rows >= _MOD_WIDE_ROWS else 2          # row-stationary 128-row tiles from ~40 K rows, else the weight-stationary kernel
    _ext().swa_mod_fwd(c, wmod, out, a, save, cfg)
    return [out, a] if save else [out]


def mod_bwd(g: torch.Tensor, c: torch.Tensor, a: torch.Tensor, wmod: torch.Tensor, dc_cfg: int = 4, dw_terms: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    """``(dc [R, 128], dWmod [768, 128])`` bf16 of :func:`mod_fwd` from ``g`` = d mod [R, 768] fp32 (contiguous) and ``a`` = rn(silu(c)): dc through the framework's silu backward on
    the bf16 gradient, dWmod from fixed-order partial sums (deterministic).  g reaches the tensor cores as two bf16 terms (hi, lo: 16 significant bits) for dc and, with ``dw_terms`` = 2, for
    dWmod (1: its rounding to bf16, which the sum over the rows averages out but which shows at 2e-3 on random data); ``dc_cfg`` selects the dc kernel (see ``ops.cu``)."""
    dc = torch.empty_like(c)
    dw = torch.empty_like(wmod)
    part = torch.empty(MOD_DW_PARTS, MOD_N, C, device=c.device, dtype=torch.float32)
    _ext().swa_mod_bwd(g, c, a, wmod, dc, part, dw, dc_cfg, dw_terms)
    return dc, dw


#: how the row-tiled backward stages tile a call and where the modulation gradient goes (``bwd_rows_sm80.cuh``): MODE_SINGLE -- every row has its own modulation row (N = b, the conditioning per sample):
#: ``dmod`` [b S, 768]; MODE_HOIST -- the conditioning is shared by the A = N / b samples of a batch element, A a multiple of 16: ``dmod`` [A / 16, b S, 768] partial buffers that the caller adds
MODE_SINGLE, MODE_HOIST = 0, 1


def bwd_mode(n: int, b: int) -> int | None:
    """The tiling of the backward stages for N = n sequences with b modulation rows per atom, or None when they do not serve it (A = n / b neither 1 nor a multiple of 16)."""
    if n % b:
        return None
    a = n // b
    if a == 1:
        return MODE_SINGLE
    return MODE_HOIST if a % 16 == 0 else None


def dmod_buffer(mode: int, n: int, b: int, s: int, device: torch.device) -> torch.Tensor:
    """The (uninitialised: every element is written once) modulation-gradient buffer of the backward stages: [b S, 768], or [A / 16, b S, 768] fp32 in MODE_HOIST."""
    shape = (b * s, MOD_N) if mode == MODE_SINGLE else (n // b // 16, b * s, MOD_N)
    return torch.empty(*shape, device=device, dtype=torch.float32)


def oproj_bwd(dq1: torch.Tensor, o: torch.Tensor, g: torch.Tensor, att: torch.Tensor, mod: torch.Tensor, wo: torch.Tensor, dmod: torch.Tensor, b: int, s: int, mode: int, cfg: int = 0):
    """The out-projection backward: ``dq1`` / ``o`` / ``g`` / ``att`` [M, 128] bf16 (the gradient of q1, the attention output, the gate projection, the saved ``att``), ``mod`` [b S, 768] fp32,
    ``wo`` [128, 128].  Returns ``(dO, dG, datt, gated, dv)``: [M, 128] bf16 and the attention backward's D, [N, 4, S] fp32; d gate_a goes into columns 256 .. 383 of ``dmod`` (see :func:`dmod_buffer`)."""
    m = dq1.shape[0]
    dev = dq1.device
    d_o, d_g, datt, gated = (torch.empty(m, C, device=dev, dtype=torch.bfloat16) for _ in range(4))
    dv = torch.empty(m // s, H, s, device=dev, dtype=torch.float32)
    _ext().swa_oproj_bwd(dq1, o, g, att, mod, wo.t().contiguous(), d_o, d_g, datt, gated, dv, dmod, s, b, mode, cfg)
    return d_o, d_g, datt, gated, dv


def ffn_bwd(dy: torch.Tensor, q1: torch.Tensor, y: torch.Tensor, ffn: torch.Tensor, mod: torch.Tensor, wu: torch.Tensor, wd: torch.Tensor, dmod: torch.Tensor, b: int, s: int, eps: float,
            mode: int, cfg: tuple[int, int] = (2, 1)):
    """The FFN backward: ``dy`` (= dq2, the gradient of the block output) / ``q1`` / ``y`` / ``ffn`` [M, 128] bf16 (``y``, ``ffn`` and ``q1`` saved by the forward), ``mod`` [b S, 768] fp32,
    ``wu`` [512, 128], ``wd`` [128, 256].  Returns ``(dq1, dffn, hh, dab)`` bf16 -- [M, 128], [M, 128], [M, 256], [M, 512] (da | db): the gradient of q1 and the dW operands; d shift_f | d scale_f | d gate_f go
    into columns 384 .. 767 of ``dmod`` (see :func:`dmod_buffer`)."""
    m = dy.shape[0]
    dev = dy.device
    dq1, dffn = (torch.empty(m, C, device=dev, dtype=torch.bfloat16) for _ in range(2))
    hh = torch.empty(m, NHID, device=dev, dtype=torch.bfloat16)
    dab = torch.empty(m, 2 * NHID, device=dev, dtype=torch.bfloat16)
    _ext().swa_ffn_bwd(dy, q1, y, ffn, mod, wu.contiguous(), wd.t().contiguous(), wu.t().contiguous(), dq1, dffn, hh, dab, dmod, s, b, eps, mode, cfg[0], cfg[1])
    return dq1, dffn, hh, dab


def qkvg_bwd(qi: torch.Tensor, pq: torch.Tensor, pk: torch.Tensor, dqh: torch.Tensor, dkh: torch.Tensor, dvh: torch.Tensor, dg: torch.Tensor, dq1: torch.Tensor, mod: torch.Tensor, cos: torch.Tensor,
             sin: torch.Tensor, wqkv: torch.Tensor, wg: torch.Tensor, dmod: torch.Tensor, b: int, s: int, eps: float, qk_eps: float, mode: int, cfg: int = 0):
    """The qkvg backward: ``qi`` (the block input q) / ``pq`` / ``pk`` (the saved rounded pre-norm projections) / ``dg`` / ``dq1`` [M, 128] bf16, ``dqh`` / ``dkh`` / ``dvh`` head-major [N, 4, S, 32] bf16 (the
    attention backward's outputs), ``mod`` [b S, 768] fp32, ``cos`` / ``sin`` [b S, 16] fp32, ``wqkv`` [384, 128], ``wg`` [128, 128].  Returns ``(dq [M, 128], dP [M, 512])`` bf16 (dP = dpq | dpk | dpv | dpg, the
    dW operand); d shift_a | d scale_a go into columns 0 .. 255 of ``dmod`` (see :func:`dmod_buffer`)."""
    m = qi.shape[0]
    dev = qi.device
    dq = torch.empty(m, C, device=dev, dtype=torch.bfloat16)
    dp = torch.empty(m, 4 * C, device=dev, dtype=torch.bfloat16)
    _ext().swa_qkvg_bwd(qi, pq, pk, dqh, dkh, dvh, dg, dq1, mod, cos, sin, torch.cat([wqkv, wg]).t().contiguous(), dq, dp, dmod, s, b, eps, qk_eps, mode, cfg)
    return dq, dp


def _is_a100(index: int) -> bool:
    return _capability(index) == (8, 0)


@functools.lru_cache(maxsize=8)
def _capability(index: int) -> tuple[int, int]:
    return torch.cuda.get_device_capability(index)


def supported(q: torch.Tensor, nhid: int, half_window: int, eps: float) -> bool:
    """Whether these kernels serve a forward / backward call: an A100 (cc 8.0), a bf16 ``q`` [N, S, 128], SwiGLU hidden 256, half window 64 (the fp32 eps is the contract of the
    block).  ``MINIWORLD_SWA_DIT_SM80=0`` keeps the Triton stages."""
    if os.environ.get("MINIWORLD_SWA_DIT_SM80", "1") == "0":
        return False
    if not q.is_cuda or q.dtype != torch.bfloat16 or q.dim() != 3 or q.shape[-1] != C or nhid != NHID or half_window != HW:
        return False
    return _is_a100(q.device.index if q.device.index is not None else torch.cuda.current_device())


def block_fwd(q: torch.Tensor, mod: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, seqused: torch.Tensor, wqkv: torch.Tensor, wg: torch.Tensor, wo: torch.Tensor,
              wu: torch.Tensor, wd: torch.Tensor, b: int, eps: float, qk_eps: float, save: bool) -> list[torch.Tensor]:
    """The forward of ``dispatch.swa_dit_block_fwd`` on the sm_80 stages: ``q`` [N, S, 128] bf16, ``mod`` [b S, 768] fp32, ``cos`` / ``sin`` [b, S, 16] fp32.  Returns ``[out]`` or, with
    ``save``, ``[out, Qh, Kh, Vh, G, O, lse, q1, X, PQ, PK, Att, Y, FF]`` (the tensors the backward reads)."""
    n, s, _ = q.shape
    qf = q.reshape(n * s, C)
    cos2, sin2 = cos.reshape(-1, D // 2).contiguous(), sin.reshape(-1, D // 2).contiguous()
    qh, kh, vh, g, xs, pqs, pks = qkvg_fwd(qf, mod, cos2, sin2, wqkv, wg, b, s, eps, qk_eps, save)
    o, lse = attn_fwd(qh, kh, vh, seqused)
    out, q1s, atts, ys, ffs = ffn_fwd(qf, g, o, mod, wo, wu, wd, b, s, eps, save)
    outputs = [out.view(n, s, C)]
    if save:
        outputs += [qh, kh, vh, g, o, lse, q1s, xs, pqs, pks, atts, ys, ffs]
    return outputs
