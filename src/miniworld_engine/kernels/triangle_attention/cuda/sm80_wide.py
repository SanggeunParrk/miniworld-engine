"""A100 (sm_80) hand-CUDA triangle attention at every registered width: d_pair 64 .. 512 (a multiple of 64), H heads of 16 or 32 channels, inference and training.

``sm80.py`` is the fused d_pair-128 / 4 x 32 path.  This module serves every other geometry with one structure and two engines for its front and back (the attention core is the
same ``sm80/attn_*_sm80.cuh`` kernels, head dim 32, any head count, reading q / k / v straight out of the front's output):

  * ``fused`` (d_pair 64 or 128 with 2 or 4 heads of 32, i.e. hidden 64 / 128; ``sm80/fused_sm80.cuh``): the d_pair-128 kernels with the widths as template parameters.  The front
    (LayerNorm + q | k | v | g + bias projections in one pass), the back (gate + output projection + residual) and their backward are persistent kernels with the weights resident in
    shared memory.  ``MINIWORLD_TRIATTN_SM80_ROWS=1`` takes the other engine at these widths (A/B).
  * ``rows`` (every other width, the weights of d_pair 256 are 528 KiB): hand-CUDA row kernels (``sm80/rows_sm80.cuh``) around cuBLAS GEMMs.
        front   ``ln_rows`` (the input LayerNorm -> the normalised input ``xh`` [T, D + 8]; a constant column pair carries the folded shift), ONE GEMM against the packed
                ``[W diag(gamma) | b_hi | b_lo]`` -> q | k | v | g | bias heads [T, 4 H 32 + Hpad], ``bias_planes`` (the head planes, masked keys = bf16 min)
        back    ``gate_rows`` (a = sigmoid(g) o), one GEMM ``a Wo^T``, ``out_rows`` (the broadcast dropout scale and the residual, written at the transposed positions for the ending node)

A 16-channel head is zero-padded to 32 inside the packed weights (its padded q / k / v / g channels are 0 and meet zero columns of Wo; the softmax scale stays 1/sqrt(16)), so every
width runs the head-dim-32 core, as the B200 path does.  Training is one autograd function over two opaque ops: the forward saves ``xh``, the LayerNorm statistics, the front's output,
the bias planes, the attention output and the log-sum-exp; the backward is the back's backward (``dy``, the gate's gradients and the attention backward's row term), the core's backward
(dq | dk | dv into one buffer, the bias gradient), the front's backward (the projections' input gradient, the bias heads' term, the LayerNorm backward and the residual) and the weight
gradients from two GEMMs over the tokens (``G = D^T [xh | 1 ...]``) and one finalizer -- the same stages in both engines.

Ampere-only and shape-gated by construction: ``supports()`` is the whole gate; everything it rejects keeps the module's Triton path.
"""

import functools
import math
import os
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

import torch
from torch.autograd.function import once_differentiable

from ..._compile import opaque
from ..._nvcc import ensure_cuda_home, host_flags, load_extension
from . import sm80 as _core
from . import sm80_core2 as _core2

_dir = Path(__file__).parent / "sm80"
BF = torch.bfloat16
F32 = torch.float32
#: the head dim of the core (a 16-channel head is zero-padded to this)
HD_CORE = 32
MAX_D, MAX_H = 512, 16


class Cfg(NamedTuple):
    """The geometry of one module: ``d`` = d_pair, ``h`` heads of ``hd`` (16 or 32) real channels, ``fused`` = the fused engine (else the row kernels + cuBLAS)."""

    d: int
    h: int
    hd: int
    fused: bool = False

    @property
    def native(self) -> bool:    # 16-channel heads kept as they are (the fused kernels and the core are instantiated for them at d_pair 64, 4 heads): no padding to 32
        return self.fused and self.hd == 16 and self.d == 64 and self.h == 4

    @property
    def lhd(self) -> int:        # the head dim of the layout the front writes and the core reads
        return 16 if self.native else HD_CORE

    @property
    def dhp(self) -> int:        # the hidden width of that layout: h heads of lhd channels (a padded 16-channel head takes 32)
        return self.h * self.lhd

    @property
    def hpad(self) -> int:       # the bias heads, padded to the mma's 8
        return -(-self.h // 8) * 8

    @property
    def n(self) -> int:          # the row stride of the front's output: q | k | v | g, and (rows engine) the bias heads
        return 4 * self.dhp + (0 if self.fused else self.hpad)


def cfg_for(d_pair: int, d_hidden: int, n_head: int) -> Cfg | None:
    """The geometry when this path serves it (d_pair a multiple of 64 in 64 .. 512, 1 .. 16 heads of 16 or 32 channels), else None."""
    if n_head < 1 or n_head > MAX_H or d_hidden % n_head:
        return None
    hd = d_hidden // n_head
    if hd not in (16, 32) or d_pair % 64 or not 64 <= d_pair <= MAX_D:
        return None
    fused = d_pair in (64, 128) and n_head in (2, 4) and os.environ.get("MINIWORLD_TRIATTN_SM80_ROWS", "0") != "1"
    return Cfg(d_pair, n_head, hd, fused)


@functools.lru_cache(maxsize=1)
def _rows():
    ensure_cuda_home()
    return load_extension(
        name="triattn_sm80_rows",
        sources=[str(_dir / "rows_ops.cu")],
        extra_cuda_cflags=[*host_flags(), "-std=c++17", "-O3", "-gencode=arch=compute_80,code=sm_80", f"-I{_dir}"],
        extra_cflags=["-std=c++17", "-O3"], verbose=False,
    )


_BUILD_FAILED = False


@torch.compiler.assume_constant_result
def _loads() -> bool:
    """Builds (first call) or loads the two extensions (the row / fused kernels and the attention core); False, with one warning, when the toolchain fails (the Triton path then serves).

    A process-level constant, so ``torch.compile`` evaluates it once at trace time instead of tracing the nvcc lookup / JIT build into the graph."""
    global _BUILD_FAILED
    if _BUILD_FAILED:
        return False
    try:
        _rows()
        _core._ext()
    except Exception as exc:  # noqa: BLE001 -- any build failure means "use the other path"
        _BUILD_FAILED = True
        warnings.warn(f"sm80 triangle attention (all widths) unavailable, keeping the existing path: {exc!r}", RuntimeWarning, stacklevel=2)
        return False
    return True


@torch.compiler.assume_constant_result
def _loads_native() -> bool:
    """``_loads()`` for the native 16-channel-head geometry: the row / fused kernels and the generalised core (``sm80_core2``) instead of the d_pair-128 core."""
    global _BUILD_FAILED
    if _BUILD_FAILED:
        return False
    try:
        _rows()
        _core2.ext()
    except Exception as exc:  # noqa: BLE001 -- any build failure means "use the other path"
        _BUILD_FAILED = True
        warnings.warn(f"sm80 triangle attention (16-channel heads) unavailable, keeping the existing path: {exc!r}", RuntimeWarning, stacklevel=2)
        return False
    return True


def loads(cfg: Cfg) -> bool:
    """The extensions of ``cfg``'s kernels are built (first call) / loadable."""
    return _loads_native() if cfg.native else _loads()


def enabled() -> bool:
    return os.environ.get("MINIWORLD_TRIATTN_SM80", "1") != "0"


def supports(pair: torch.Tensor, cfg: Cfg | None, weights: Sequence[torch.Tensor], wo: torch.Tensor, ln_w: torch.Tensor, ln_b: torch.Tensor,
             mask: torch.Tensor | None = None, *, train: bool = False) -> bool:
    """The path's requirements: sm_80, a bf16 contiguous ``[B, L, L, D]`` pair stack with L a multiple of 128, bf16 weights (q / k / v / g ``[H hd, D]``, bias ``[H, D]``, ``wo``
    ``[D, H hd]``), LayerNorm parameters ``[D]`` in fp32 or bf16, a ``[B, L]`` bool key mask if any; for ``train`` the attention backward's bias-gradient partials must fit
    ``sm80_core2.MAX_DBP_BYTES`` (they are cubic in L)."""
    if cfg is None or not enabled():
        return False
    if not pair.is_cuda or pair.dtype is not BF or pair.ndim != 4 or not pair.is_contiguous():
        return False
    b, length, length2, d = pair.shape
    if d != cfg.d or length != length2 or length % 128 or b * length * length >= 2**31:
        return False
    dh = cfg.h * cfg.hd
    shapes = [(dh, d)] * 4 + [(cfg.h, d)]
    if len(weights) != 5 or any(w.dtype is not BF or w.device != pair.device or tuple(w.shape) != s for w, s in zip(weights, shapes, strict=True)):
        return False
    if wo.dtype is not BF or wo.device != pair.device or tuple(wo.shape) != (d, dh):
        return False
    if any(t.shape != (d,) or t.device != pair.device or t.dtype not in (F32, BF) for t in (ln_w, ln_b)) or ln_w.dtype != ln_b.dtype:
        return False
    if mask is not None and (mask.dtype is not torch.bool or mask.shape != (b, length) or mask.device != pair.device):
        return False
    if train and not _core2.dbp_fits(length, cfg.h, b):
        return False
    return _core._is_ampere(pair.device.index if pair.device.index is not None else torch.cuda.current_device())


def _mask_u8(mask: torch.Tensor | None, device) -> torch.Tensor:
    return _core._mask_u8(mask, device)


# --------------------------------------------------------------------------------------------------------------------------------- weights
def _wo_padded(wo: torch.Tensor, cfg: Cfg) -> torch.Tensor:
    """``Wo`` ``[D, H hd]`` as ``[D, H 32]`` (zero columns for the padded channels of a 16-channel head); the weight itself at hd 32."""
    wo = wo.contiguous()
    if cfg.hd == HD_CORE or cfg.native:
        return wo
    out = torch.empty((cfg.d, cfg.dhp), dtype=BF, device=wo.device)
    _rows().wo_pad(wo, out, cfg.d, cfg.h, cfg.hd)
    return out


def _pack_folded(weights, ln_w: torch.Tensor, ln_b: torch.Tensor, cfg: Cfg) -> torch.Tensor:
    """Rows engine: the front GEMM's weights ``[N, D + 8]``: row r of q | k | v | g (head h, channel c, padded channels 0) is ``bf16(W diag(gamma))`` | ``b_hi`` | ``b_lo`` | 0 ... with
    ``b = W beta`` split into two bf16; then the bias heads' rows (padded to 8).  Never cached: a captured CUDA graph must repack after an optimizer step."""
    wp = torch.empty((4 * cfg.dhp + cfg.hpad, cfg.d + 8), dtype=BF, device=ln_w.device)
    _rows().w_pack(*(w.contiguous() for w in weights), ln_w.detach().float().contiguous(), ln_b.detach().float().contiguous(), wp, cfg.d, cfg.h, cfg.hd, 1)
    return wp


def _pack_fused(weights, ln_w: torch.Tensor, ln_b: torch.Tensor, cfg: Cfg) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused engine: ``(wp [4 DH + 8, D], bvec [4 DH + 8] fp32)``: W' = bf16(W diag(gamma)) in the kernel's row order and the shift b = W beta (``fused_sm80.cuh``)."""
    rows = 4 * cfg.dhp + 8
    wp = torch.empty((rows, cfg.d), dtype=BF, device=ln_w.device)
    bvec = torch.empty((rows,), dtype=F32, device=ln_w.device)
    _rows().fused_pack(*(w.contiguous() for w in weights), ln_w.detach().float().contiguous(), ln_b.detach().float().contiguous(), wp, bvec, cfg.d, cfg.dhp, cfg.h, cfg.hd)
    return wp, bvec


def _pack_plain(weights, cfg: Cfg) -> torch.Tensor:
    """Rows engine: ``[Wq; Wk; Wv; Wg]`` as ``[4 H 32, D]`` (padded rows 0): the backward's ``dxn = [dq | dk | dv | dg] W`` operand."""
    wcat = torch.empty((4 * cfg.dhp, cfg.d), dtype=BF, device=weights[0].device)
    _rows().w_pack(*(w.contiguous() for w in weights), weights[0], weights[0], wcat, cfg.d, cfg.h, cfg.hd, 0)
    return wcat


def _pack_bwd_fused(weights, wo: torch.Tensor, ln_w: torch.Tensor, cfg: Cfg):
    """Fused engine: ``(wt [4, D, DH], wbt [D, H], wot [DH, D], gamma32 [D])`` (the layouts of ``fused_back_bwd`` / ``fused_front_bwd``)."""
    dev = wo.device
    wt = torch.empty((4, cfg.d, cfg.dhp), dtype=BF, device=dev)
    wbt = torch.empty((cfg.d, cfg.h), dtype=BF, device=dev)
    wot = torch.empty((cfg.dhp, cfg.d), dtype=BF, device=dev)
    gamma32 = torch.empty((cfg.d,), dtype=F32, device=dev)
    _rows().fused_bwd_pack(*(w.contiguous() for w in weights), wo.contiguous(), ln_w.detach().contiguous(), wt, wbt, wot, gamma32, cfg.d, cfg.dhp, cfg.h, cfg.hd)
    return wt, wbt, wot, gamma32


# ----------------------------------------------------------------------------------------------------------------------------------- forward
def _forward_ops_fake(x, weights, wo, ln_w, ln_b, mask, ds, eps, transposed, n_head, hd, fused, save):
    """[out] and, when ``save``, [xh, stats, y, bias, o, lse]: fresh tensors."""
    cfg = Cfg(x.shape[-1], n_head, hd, fused)
    b, length = x.shape[0], x.shape[1]
    t = b * length * length
    out = [x.new_empty(x.shape)]
    if save:
        out += [x.new_empty((t, cfg.d + 8)), x.new_empty((t, 2), dtype=F32), x.new_empty((t, cfg.n)), x.new_empty((b, cfg.h, length, length)),
                x.new_empty((t, cfg.dhp)), x.new_empty((b, cfg.h, length, length), dtype=F32)]
    return out


@opaque(fake=_forward_ops_fake, name="triangle_attention_sm80w_forward")
def forward_ops(x: torch.Tensor, weights: list[torch.Tensor], wo: torch.Tensor, ln_w: torch.Tensor, ln_b: torch.Tensor, mask: torch.Tensor, ds: torch.Tensor, eps: float,
                transposed: bool, n_head: int, hd: int, fused: bool, save: bool) -> list[torch.Tensor]:
    """``pair + drop(TriangleAttention(pair))`` at any supported width: ``x`` the contiguous pair stack ``[B, L, L, D]``, ``weights`` = (q, k, v, g, bias), ``mask`` ``[B, L]`` uint8 or
    empty, ``ds`` the dropout scale ``[B, L, D]`` (indexed by the token's second index) or empty.  Returns [out] and, with ``save``, [xh, stats, y, bias, o, lse] for the backward
    (``xh`` the normalised input followed by the constant column(s), ``y`` the front's output [T, cfg.n])."""
    cfg = Cfg(x.shape[-1], n_head, hd, fused)
    rows, core = _rows(), (None if cfg.native else _core._ext())
    b, length = x.shape[0], x.shape[1]
    t, d, dhp, n = b * length * length, cfg.d, cfg.dhp, cfg.n
    x2 = x.view(t, d)
    dev = x.device
    bias = torch.empty((b, cfg.h, length, length), dtype=BF, device=dev)
    stats = torch.empty((t, 2), dtype=F32, device=dev) if save else x.new_empty((0,), dtype=F32)
    if fused:
        wp, bvec = _pack_fused(weights, ln_w, ln_b, cfg)
        y = torch.empty((t, n), dtype=BF, device=dev)
        xh = torch.empty((t, d + 8), dtype=BF, device=dev) if save else x.new_empty((0,))
        rows.fused_front(x2, wp, bvec, mask, y, bias, stats, xh, length, int(transposed), float(eps), d, dhp, cfg.lhd)
    else:
        wp = _pack_folded(weights, ln_w, ln_b, cfg)
        xh = torch.empty((t, d + 8), dtype=BF, device=dev)
        rows.ln_rows(x2, xh, stats, d, length, int(transposed), float(eps))
        y = torch.mm(xh, wp.t())                                                         # q | k | v | g | bias heads
        rows.bias_planes(y, mask, bias, 4 * dhp, cfg.h, length)
    lse = torch.empty((b, cfg.h, length, length), dtype=F32, device=dev) if save else x.new_empty((0,), dtype=F32)
    scl = _core._L2E / math.sqrt(hd)
    q, k, v, g = y[:, :dhp], y[:, dhp:2 * dhp], y[:, 2 * dhp:3 * dhp], y[:, 3 * dhp:4 * dhp]
    if fused:
        o = torch.empty((t, dhp), dtype=BF, device=dev)
        if cfg.native:                                                                   # 16-channel heads, read in place by the generalised core
            lay, lay_o = _core2.token_major(length, n, 16), _core2.token_major(length, dhp, 16)
            _core2.forward(q, k, v, bias, x.new_empty((0,), dtype=torch.uint8), o, lse, (lay, lay, lay, lay_o), length, cfg.h, b, 16, 1.0 / math.sqrt(hd))
        else:
            core.attn_fwd(q, k, v, bias, o, lse, length, cfg.h, b, n, n, n, dhp, scl, _core.ROWS, _core.BN, _core.MINB)
        out = torch.empty_like(x)
        rows.fused_back(o, g, _wo_padded(wo, cfg), x2, out.view(t, d), ds, length, int(transposed), d, dhp, cfg.lhd)
    else:
        # the gate is fused into the core's epilogue: a = bf16(sigmoid(g) o) is the output projection's operand, the plain output only when the backward needs it
        o = torch.empty((t, dhp), dtype=BF, device=dev) if save else x.new_empty((0,))
        a = torch.empty((t, dhp), dtype=BF, device=dev)
        core.attn_fwd_gate(q, k, v, bias, o, lse, g, a, length, cfg.h, b, n, n, n, dhp, n, dhp, scl)
        yo = torch.mm(a, _wo_padded(wo, cfg).t())                                        # [T, D]
        out = torch.empty_like(x)
        rows.out_rows(yo, x2, ds, out.view(t, d), length, int(transposed))
    return [out, xh, stats, y, bias, o, lse] if save else [out]


def _dropout_scale(ds: torch.Tensor | None, like: torch.Tensor) -> torch.Tensor:
    return ds.contiguous() if ds is not None else like.new_empty((0,))


def forward(pair: torch.Tensor, cfg: Cfg, weights: Sequence[torch.Tensor], wo: torch.Tensor, ln_w: torch.Tensor, ln_b: torch.Tensor, eps: float,
            mask: torch.Tensor | None = None, *, transposed: bool = False, ds: torch.Tensor | None = None) -> torch.Tensor:
    """``pair + drop(TriangleAttention(pair))`` without autograd (inference): ``weights`` = (q, k, v, g, bias).  Call ``supports()`` first."""
    return forward_ops(pair, list(weights), wo, ln_w, ln_b, _mask_u8(mask, pair.device), _dropout_scale(ds, pair), float(eps), bool(transposed), cfg.h, cfg.hd, cfg.fused, False)[0]


# ---------------------------------------------------------------------------------------------------------------------------------- backward
def _attention_backward(y, bias, lse, delta, dov, dqkvg, cfg: Cfg, b: int, length: int) -> torch.Tensor:
    """The core's backward on the front's output ``y`` (q | k | v columns): dq | dk | dv go into the first 3 H 32 columns of ``dqkvg`` ``[T, 4 H 32]``; returns the bias gradient
    ``[B, H, L, L]`` bf16 (the sum over the pair rows of dS, from bf16 partials over groups of ``DQ_ROWS`` rows and a fixed-order reduction)."""
    dhp, n = cfg.dhp, cfg.n
    sm_scale = 1.0 / math.sqrt(cfg.hd)
    if cfg.native:                                                                       # 16-channel heads: the generalised core, in place
        q, k, v = y[:, :dhp], y[:, dhp:2 * dhp], y[:, 2 * dhp:3 * dhp]
        lay, lay_h, lay_d = _core2.token_major(length, n, 16), _core2.token_major(length, dhp, 16), _core2.token_major(length, 4 * dhp, 16)
        return _core2.backward(q, k, v, dov, bias, y.new_empty((0,), dtype=torch.uint8), lse, delta, dqkvg[:, :dhp], dqkvg[:, dhp:2 * dhp], dqkvg[:, 2 * dhp:3 * dhp],
                               (lay, lay, lay, lay_h, lay_d, lay_d, lay_d), length, cfg.h, b, 16, sm_scale)
    core = _core._ext()
    scl = sm_scale * _core._L2E
    groups = length // _core.DQ_ROWS
    dbp = torch.empty((groups, b * cfg.h, length, length), dtype=BF, device=y.device)
    q, k, v = y[:, :dhp], y[:, dhp:2 * dhp], y[:, 2 * dhp:3 * dhp]
    core.attn_bwd_dq(q, k, v, dov, bias, lse, delta, dqkvg[:, :dhp], dbp, length, cfg.h, b, n, n, n, dhp, 4 * dhp, scl, sm_scale,
                     _core.DQ_ROWS, _core.DQ_NKV, _core.DQ_BN, _core.DQ_MINB)
    db = torch.empty((b, cfg.h, length, length), dtype=BF, device=y.device)
    core.db_reduce(dbp, db, groups)
    del dbp
    bias_t = torch.empty_like(bias)
    core.bias_transpose(bias, bias_t, length)
    core.attn_bwd_dkv(q, k, v, dov, bias_t, lse, delta, dqkvg[:, dhp:2 * dhp], dqkvg[:, 2 * dhp:3 * dhp], length, cfg.h, b, n, n, n, dhp, 4 * dhp, 4 * dhp,
                      scl, sm_scale, _core.DKV_NST, 32, _core.DKV_MINB)
    return db


def _backward_ops_fake(x, weights, wo, ln_w, ln_b, mask, ds, saved, dout, transposed, n_head, hd, fused):
    """[dpair, dwq, dwk, dwv, dwg, dwb, dwo, dln_w, dln_b] in the leaves' shapes and dtypes."""
    return [x.new_empty(x.shape), *(w.new_empty(w.shape) for w in weights), wo.new_empty(wo.shape), ln_w.new_empty(ln_w.shape), ln_b.new_empty(ln_b.shape)]


@opaque(fake=_backward_ops_fake, name="triangle_attention_sm80w_backward")
def backward_ops(x: torch.Tensor, weights: list[torch.Tensor], wo: torch.Tensor, ln_w: torch.Tensor, ln_b: torch.Tensor, mask: torch.Tensor, ds: torch.Tensor,
                 saved: list[torch.Tensor], dout: torch.Tensor, transposed: bool, n_head: int, hd: int, fused: bool) -> list[torch.Tensor]:
    """The gradients of ``forward_ops`` (``saved`` = its [xh, stats, y, bias, o, lse]): [dpair, dwq, dwk, dwv, dwg, dwb, dwo, dln_w, dln_b], the weights' and the LayerNorm
    parameters' gradients in their dtypes."""
    rows = _rows()
    xh, stats, y, bias, o, lse = saved
    cfg = Cfg(x.shape[-1], n_head, hd, fused)
    b, length = x.shape[0], x.shape[1]
    t, d, dhp = b * length * length, cfg.d, cfg.dhp
    dev = x.device
    dout2 = dout.contiguous().view(t, d)
    dqkvg = torch.empty((t, 4 * dhp), dtype=BF, device=dev)
    dov = torch.empty((t, dhp), dtype=BF, device=dev)
    a = torch.empty((t, dhp), dtype=BF, device=dev)
    delta = torch.empty((b, cfg.h, length, length), dtype=F32, device=dev)
    if fused:
        wt, wbt, wot, gamma32 = _pack_bwd_fused(weights, wo, ln_w, cfg)
        dy = torch.empty((t, d), dtype=BF, device=dev)
        rows.fused_back_bwd(dout2, ds, o, y[:, 3 * dhp:4 * dhp], wot, dqkvg[:, 3 * dhp:], dov, dy, a, delta, length, int(transposed), d, dhp, cfg.lhd)
    else:
        if transposed or ds.numel():                                                      # dy in the starting frame after the dropout scale
            dy = torch.empty((t, d), dtype=BF, device=dev)
            rows.dy_rows(dout2, ds, dy, length, int(transposed))
        else:
            dy = dout2
        da = torch.mm(dy, _wo_padded(wo, cfg))                                           # [T, H 32] = bf16(dy Wo)
        rows.gate_bwd(da, o, y[:, 3 * dhp:4 * dhp], dqkvg[:, 3 * dhp:], dov, a, delta, length, cfg.h)
        del da
    go = torch.mm(dy.t(), a, out_dtype=F32)                                              # dWo [D, H 32]
    del a
    db = _attention_backward(y, bias, lse, delta, dov, dqkvg, cfg, b, length)
    del dov, delta
    dpair = torch.empty_like(x)
    if fused:
        rows.fused_front_bwd(dqkvg, db, x.view(t, d), stats, dout2, dpair.view(t, d), wt, wbt, gamma32, length, int(transposed), d, dhp, cfg.lhd)
    else:
        dxn = torch.mm(dqkvg, _pack_plain(weights, cfg))                                 # [T, D]
        rows.ln_bwd(dxn, db, weights[4].contiguous(), x.view(t, d), stats, dout2, ln_w.detach().float().contiguous(), dpair.view(t, d), length, cfg.h, int(transposed))
        del dxn
    g = torch.cat([torch.mm(dqkvg.t(), xh, out_dtype=F32), torch.mm(db.permute(1, 0, 2, 3).reshape(cfg.h, t), xh, out_dtype=F32)])      # [4 H 32 + H, D + 8]
    dws = [torch.empty_like(w) for w in weights]
    dwo = torch.empty_like(wo)
    dg, dbeta = torch.empty_like(ln_w), torch.empty_like(ln_b)
    rows.wgrad_w(g, go, *(w.contiguous() for w in weights), *dws, dwo, ln_w.detach().contiguous(), ln_b.detach().contiguous(), dg, dbeta, d, cfg.h, cfg.hd)
    return [dpair, *dws, dwo, dg, dbeta]


class _Training(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, wq, wk, wv, wg, wb, wo, ln_w, ln_b, mask, ds, eps, transposed, n_head, hd, fused):
        weights = [wq, wk, wv, wg, wb]
        out, *saved = forward_ops(x, weights, wo, ln_w, ln_b, mask, ds, eps, transposed, n_head, hd, fused, True)
        ctx.save_for_backward(x, wq, wk, wv, wg, wb, wo, ln_w, ln_b, mask, ds, *saved)
        ctx.cfg = (transposed, n_head, hd, fused)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, dout):
        x, wq, wk, wv, wg, wb, wo, ln_w, ln_b, mask, ds, *saved = ctx.saved_tensors
        grads = backward_ops(x, [wq, wk, wv, wg, wb], wo, ln_w, ln_b, mask, ds, saved, dout.contiguous(), *ctx.cfg)
        return (*grads, None, None, None, None, None, None, None)


def trainable(pair: torch.Tensor, cfg: Cfg, weights: Sequence[torch.Tensor], wo: torch.Tensor, ln_w: torch.Tensor, ln_b: torch.Tensor, eps: float,
              mask: torch.Tensor | None = None, *, transposed: bool = False, ds: torch.Tensor | None = None) -> torch.Tensor:
    """``pair + drop(TriangleAttention(pair))`` with autograd: ``weights`` = (q, k, v, g, bias), ``ds`` the dropout scale ``[B, L, D]`` (indexed by the token's second index) or None.
    Call ``supports()`` first."""
    leaves = (pair, *weights, wo, ln_w, ln_b)
    mk = _mask_u8(mask, pair.device)
    d = _dropout_scale(ds, pair)
    if not (torch.is_grad_enabled() and any(t.requires_grad for t in leaves)):
        return forward_ops(pair, list(weights), wo, ln_w, ln_b, mk, d, float(eps), bool(transposed), cfg.h, cfg.hd, cfg.fused, False)[0]
    return _Training.apply(*leaves, mk, d, float(eps), bool(transposed), cfg.h, cfg.hd, cfg.fused)
