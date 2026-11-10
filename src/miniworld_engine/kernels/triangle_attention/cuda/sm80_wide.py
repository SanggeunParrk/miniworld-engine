"""A100 (sm_80) hand-CUDA triangle attention at every registered width: d_pair 64 .. 512 (a multiple of 64), H heads of 16 or 32 channels, inference and training.

``sm80.py`` is the fused d_pair-128 / 4 x 32 path.  This module serves every other geometry with one structure and two engines for its front and back (the attention core is the
same ``sm80/attn_*_sm80.cuh`` kernels, head dim 32, any head count, reading q / k / v straight out of the front's output):

  * ``fused`` (d_pair 64 or 128 with 2 or 4 heads of 32, i.e. hidden 64 / 128; ``sm80/fused_sm80.cuh``): the d_pair-128 kernels with the widths as template parameters.  The front
    (LayerNorm + q | k | v | g + bias projections in one pass), the back (gate + output projection + residual) and their backward are persistent kernels with the weights resident in
    shared memory.
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
    def native(
        self,
    ) -> bool:  # 16-channel heads kept as they are (the fused kernels and the core are instantiated for them at d_pair 64, 4 heads): no padding to 32
        return self.fused and self.hd == 16 and self.d == 64 and self.h == 4

    @property
    def lhd(
        self,
    ) -> int:  # the head dim of the layout the front writes and the core reads
        return 16 if self.native else HD_CORE

    @property
    def dhp(
        self,
    ) -> int:  # the hidden width of that layout: h heads of lhd channels (a padded 16-channel head takes 32)
        return self.h * self.lhd

    @property
    def hpad(self) -> int:  # the bias heads, padded to the mma's 8
        return -(-self.h // 8) * 8

    @property
    def n(
        self,
    ) -> int:  # the row stride of the front's output: q | k | v | g, and (rows engine) the bias heads
        return 4 * self.dhp + (0 if self.fused else self.hpad)


def cfg_for(d_pair: int, d_hidden: int, n_head: int) -> Cfg | None:
    """The geometry when this path serves it (d_pair a multiple of 64 in 64 .. 512, 1 .. 16 heads of 16 or 32 channels), else None."""
    if n_head < 1 or n_head > MAX_H or d_hidden % n_head:
        return None
    hd = d_hidden // n_head
    if hd not in (16, 32) or d_pair % 64 or not 64 <= d_pair <= MAX_D:
        return None
    fused = d_pair in (64, 128) and n_head in (2, 4)
    return Cfg(d_pair, n_head, hd, fused)


@functools.lru_cache(maxsize=1)
def _rows():
    ensure_cuda_home()
    return load_extension(
        name="triattn_sm80_rows",
        sources=[str(_dir / "rows_ops.cu")],
        extra_cuda_cflags=[
            *host_flags(),
            "-std=c++17",
            "-O3",
            "-gencode=arch=compute_80,code=sm_80",
            f"-I{_dir}",
        ],
        extra_cflags=["-std=c++17", "-O3"],
        extra_ldflags=["-lcublasLt"],  # the output projection with the residual in the epilogue
        verbose=False,
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
        warnings.warn(
            f"sm80 triangle attention (all widths) unavailable, keeping the existing path: {exc!r}",
            RuntimeWarning,
            stacklevel=2,
        )
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
        warnings.warn(
            f"sm80 triangle attention (16-channel heads) unavailable, keeping the existing path: {exc!r}",
            RuntimeWarning,
            stacklevel=2,
        )
        return False
    return True


def loads(cfg: Cfg) -> bool:
    """The extensions of ``cfg``'s kernels are built (first call) / loadable."""
    return _loads_native() if cfg.native else _loads()


def supports(
    pair: torch.Tensor,
    cfg: Cfg | None,
    weights: Sequence[torch.Tensor],
    wo: torch.Tensor,
    ln_w: torch.Tensor,
    ln_b: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    train: bool = False,
) -> bool:
    """The path's requirements: sm_80, a bf16 contiguous ``[B, L, L, D]`` pair stack with L a multiple of 128, bf16 weights (q / k / v / g ``[H hd, D]``, bias ``[H, D]``, ``wo``
    ``[D, H hd]``), LayerNorm parameters ``[D]`` in fp32 or bf16, a ``[B, L]`` bool key mask if any; for ``train`` the attention backward's bias-gradient partials must fit
    ``sm80_core2.MAX_DBP_BYTES`` (they are cubic in L)."""
    if cfg is None:
        return False
    if (
        not pair.is_cuda
        or pair.dtype is not BF
        or pair.ndim != 4
        or not pair.is_contiguous()
    ):
        return False
    b, length, length2, d = pair.shape
    if d != cfg.d or length != length2 or length % 128 or b * length * length >= 2**31:
        return False
    dh = cfg.h * cfg.hd
    shapes = [(dh, d)] * 4 + [(cfg.h, d)]
    if len(weights) != 5 or any(
        w.dtype is not BF or w.device != pair.device or tuple(w.shape) != s
        for w, s in zip(weights, shapes, strict=True)
    ):
        return False
    if wo.dtype is not BF or wo.device != pair.device or tuple(wo.shape) != (d, dh):
        return False
    if (
        any(
            t.shape != (d,) or t.device != pair.device or t.dtype not in (F32, BF)
            for t in (ln_w, ln_b)
        )
        or ln_w.dtype != ln_b.dtype
    ):
        return False
    if mask is not None and (
        mask.dtype is not torch.bool
        or mask.shape != (b, length)
        or mask.device != pair.device
    ):
        return False
    if train and not _core2.dbp_fits(length, cfg.h, b):
        return False
    return _core._is_ampere(
        pair.device.index
        if pair.device.index is not None
        else torch.cuda.current_device()
    )


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


def _pack_folded(
    weights,
    ln_w: torch.Tensor,
    ln_b: torch.Tensor,
    cfg: Cfg,
    padding: int = 8,
    columns: int | None = None,
) -> torch.Tensor:
    """Rows engine: the front GEMM's weights ``[N, D + padding]`` (padding 8 or 32): row r of q | k | v | g (head h, channel c, padded channels 0) is ``bf16(W diag(gamma))`` | ``b_hi`` | ``b_lo`` | 0 ... with
    ``b = W beta`` split into two bf16; then the bias heads' rows (padded to 8).  Never cached: a captured CUDA graph must repack after an optimizer step."""
    count = 4 * cfg.dhp + cfg.hpad
    wp = torch.empty((columns or count, cfg.d + padding), dtype=BF, device=ln_w.device)
    _rows().w_pack(
        *(w.contiguous() for w in weights),
        ln_w.detach().float().contiguous(),
        ln_b.detach().float().contiguous(),
        wp,
        cfg.d,
        cfg.h,
        cfg.hd,
        1,
    )
    return wp


def _pack_fused(
    weights, ln_w: torch.Tensor, ln_b: torch.Tensor, cfg: Cfg
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused engine: ``(wp [4 DH + 8, D], bvec [4 DH + 8] fp32)``: W' = bf16(W diag(gamma)) in the kernel's row order and the shift b = W beta (``fused_sm80.cuh``)."""
    rows = 4 * cfg.dhp + 8
    wp = torch.empty((rows, cfg.d), dtype=BF, device=ln_w.device)
    bvec = torch.empty((rows,), dtype=F32, device=ln_w.device)
    _rows().fused_pack(
        *(w.contiguous() for w in weights),
        ln_w.detach().float().contiguous(),
        ln_b.detach().float().contiguous(),
        wp,
        bvec,
        cfg.d,
        cfg.dhp,
        cfg.h,
        cfg.hd,
    )
    return wp, bvec


def _pack_plain(weights, cfg: Cfg) -> torch.Tensor:
    """Rows engine: ``[Wq; Wk; Wv; Wg]`` as ``[4 H 32, D]`` (padded rows 0): the backward's ``dxn = [dq | dk | dv | dg] W`` operand."""
    wcat = torch.empty((4 * cfg.dhp, cfg.d), dtype=BF, device=weights[0].device)
    _rows().w_pack(
        *(w.contiguous() for w in weights),
        weights[0],
        weights[0],
        wcat,
        cfg.d,
        cfg.h,
        cfg.hd,
        0,
    )
    return wcat


def _pack_bwd_fused(weights, wo: torch.Tensor, ln_w: torch.Tensor, cfg: Cfg):
    """Fused engine: ``(wt [4, D, DH], wbt [D, H], wot [DH, D], gamma32 [D])`` (the layouts of ``fused_back_bwd`` / ``fused_front_bwd``)."""
    dev = wo.device
    wt = torch.empty((4, cfg.d, cfg.dhp), dtype=BF, device=dev)
    wbt = torch.empty((cfg.d, cfg.h), dtype=BF, device=dev)
    wot = torch.empty((cfg.dhp, cfg.d), dtype=BF, device=dev)
    gamma32 = torch.empty((cfg.d,), dtype=F32, device=dev)
    _rows().fused_bwd_pack(
        *(w.contiguous() for w in weights),
        wo.contiguous(),
        ln_w.detach().contiguous(),
        wt,
        wbt,
        wot,
        gamma32,
        cfg.d,
        cfg.dhp,
        cfg.h,
        cfg.hd,
    )
    return wt, wbt, wot, gamma32


# ----------------------------------------------------------------------------------------------------------------------------------- forward
def _original_forward_ops_fake(
    x, weights, wo, ln_w, ln_b, mask, ds, eps, transposed, n_head, hd, fused, save, flip=False
):
    """[out] and, when ``save``, [xh, stats, y, bias, o, lse]: fresh tensors."""
    cfg = Cfg(x.shape[-1], n_head, hd, fused)
    b, length = x.shape[0], x.shape[1]
    t = b * length * length
    out = [x.new_empty(x.shape)]
    if save:
        out += [
            x.new_empty((t, cfg.d + 8)),
            x.new_empty((t, 2), dtype=F32),
            x.new_empty((t, cfg.n)),
            x.new_empty((b, cfg.h, length, length)),
            x.new_empty((t, cfg.dhp)),
            x.new_empty((b, cfg.h, length, length), dtype=F32),
        ]
    return out


@opaque(fake=_original_forward_ops_fake, name="triangle_attention_sm80w_forward")
def _original_forward_ops(
    x: torch.Tensor,
    weights: list[torch.Tensor],
    wo: torch.Tensor,
    ln_w: torch.Tensor,
    ln_b: torch.Tensor,
    mask: torch.Tensor,
    ds: torch.Tensor,
    eps: float,
    transposed: bool,
    n_head: int,
    hd: int,
    fused: bool,
    save: bool,
    flip: bool = False,
) -> list[torch.Tensor]:
    """``pair + drop(TriangleAttention(pair))`` at any supported width (``flip``: AlphaFold 3's ending-node bias, written transposed by the front): ``x`` the contiguous pair stack ``[B, L, L, D]``, ``weights`` = (q, k, v, g, bias), ``mask`` ``[B, L]`` uint8 or
    empty, ``ds`` the dropout scale ``[B, L, D]`` (indexed by the token's second index) or empty.  Returns [out] and, with ``save``, [xh, stats, y, bias, o, lse] for the backward
    (``xh`` the normalised input followed by the constant column(s), ``y`` the front's output [T, cfg.n])."""
    cfg = Cfg(x.shape[-1], n_head, hd, fused)
    rows, core = _rows(), (None if cfg.native else _core._ext())
    b, length = x.shape[0], x.shape[1]
    t, d, dhp, n = b * length * length, cfg.d, cfg.dhp, cfg.n
    x2 = x.view(t, d)
    dev = x.device
    bias = torch.empty((b, cfg.h, length, length), dtype=BF, device=dev)
    stats = (
        torch.empty((t, 2), dtype=F32, device=dev)
        if save
        else x.new_empty((0,), dtype=F32)
    )
    if fused:
        wp, bvec = _core.eval_form(("triattn_sm80w_fused", cfg), (*weights, ln_w, ln_b), lambda: _pack_fused(weights, ln_w, ln_b, cfg), save)
        y = torch.empty((t, n), dtype=BF, device=dev)
        xh = (
            torch.empty((t, d + 8), dtype=BF, device=dev) if save else x.new_empty((0,))
        )
        rows.fused_front(
            x2,
            wp,
            bvec,
            mask,
            y,
            bias,
            stats,
            xh,
            length,
            int(transposed) | (int(flip) << 1),
            float(eps),
            d,
            dhp,
            cfg.lhd,
        )
    else:
        padding = (
            32
            if (
                not save
                and (
                    (d == 256 and length >= 512)
                    or (d in (384, 512) and length >= 384)
                    or ((d, cfg.h, cfg.hd, length) == (256, 8, 32, 256))
                    or ((d, cfg.h, cfg.hd, length) == (256, 8, 32, 384))
                )
            )
            else 8
        )
        if padding == 32:
            # Align each Q/K/V row to 64 bytes as well as the GEMM reduction
            # dimension. The eight bias columns otherwise leave a 16-byte offset.
            n = -(-n // 32) * 32
        wp = _core.eval_form(("triattn_sm80w_folded", cfg, padding, n), (*weights, ln_w, ln_b), lambda: _pack_folded(weights, ln_w, ln_b, cfg, padding, n), save)
        xh = torch.empty((t, d + padding), dtype=BF, device=dev)
        rows.ln_rows(x2, xh, stats, d, length, int(transposed), float(eps))
        y = torch.mm(xh, wp.t())  # q | k | v | g | bias heads
        rows.bias_planes(y, mask, bias, 4 * dhp, cfg.h, length, int(flip))
    lse = (
        torch.empty((b, cfg.h, length, length), dtype=F32, device=dev)
        if save
        else x.new_empty((0,), dtype=F32)
    )
    scl = _core._L2E / math.sqrt(hd)
    wo_p = _core.eval_form(("triattn_sm80w_wo", cfg), (wo,), lambda: _wo_padded(wo, cfg), save)
    q, k, v, g = (
        y[:, :dhp],
        y[:, dhp : 2 * dhp],
        y[:, 2 * dhp : 3 * dhp],
        y[:, 3 * dhp : 4 * dhp],
    )
    if fused:
        o = torch.empty((t, dhp), dtype=BF, device=dev)
        if cfg.native:  # 16-channel heads, read in place by the generalised core
            lay, lay_o = (
                _core2.token_major(length, n, 16),
                _core2.token_major(length, dhp, 16),
            )
            _core2.forward(
                q,
                k,
                v,
                bias,
                x.new_empty((0,), dtype=torch.uint8),
                o,
                lse,
                (lay, lay, lay, lay_o),
                length,
                cfg.h,
                b,
                16,
                1.0 / math.sqrt(hd),
                fixed=True,
            )
        else:
            fwd = _core._forward_entry(core, save)
            fwd(
                q,
                k,
                v,
                bias,
                o,
                lse,
                length,
                cfg.h,
                b,
                n,
                n,
                n,
                dhp,
                scl,
                _core.ROWS,
                _core.BN,
                _core.MINB,
            )
        out = torch.empty_like(x)
        rows.fused_back(
            o,
            g,
            wo_p,
            x2,
            out.view(t, d),
            ds,
            length,
            int(transposed),
            d,
            dhp,
            cfg.lhd,
        )
    else:
        # the gate is fused into the core's epilogue: a = bf16(sigmoid(g) o) is the output projection's operand, the plain output only when the backward needs it
        o = torch.empty((t, dhp), dtype=BF, device=dev) if save else x.new_empty((0,))
        a = torch.empty((t, dhp), dtype=BF, device=dev)
        if (
            not save
            and mask.numel()
            and length >= 384
            and length % 384 == 0
        ):
            indices = torch.empty((b, length), dtype=torch.int32, device=dev)
            counts = torch.empty((b,), dtype=torch.int32, device=dev)
            compact_bias = torch.empty_like(bias)
            shape = (b, length, length, dhp)
            core.compact_attention_gate_fast(
                q.view(shape),
                k.view(shape),
                v.view(shape),
                bias,
                mask,
                a.view(shape),
                indices,
                counts,
                compact_bias,
                scl,
                g.view(shape),
            )
        else:
            fwd = _core._forward_entry(core, save, gate=True)
            fwd(
                q, k, v, bias, o, lse, g, a, length, cfg.h, b, n, n, n, dhp, n, dhp, scl
            )
        out = torch.empty_like(x)
        if (
            not save
            and (d, cfg.h, hd) == (256, 8, 32)
            and length == 128
        ):
            from . import native_out_fused_sm80

            out_cfg = 4 if transposed else 14
            native_out_fused_sm80.run(
                a,
                wo_p,
                x2,
                ds,
                out.view(t, d),
                length,
                transposed,
                out_cfg,
            )
        elif (
            not save
            and not ds.numel()
            and rows.out_gemm_residual(
                a, wo_p, x2, out.view(t, d), length, int(transposed)
            )
        ):
            pass  # out = bf16(x + a Wo^T) written by the GEMM's epilogue
        else:
            yo = torch.mm(a, wo_p.t())  # [T, D]
            rows.out_rows(yo, x2, ds, out.view(t, d), length, int(transposed))
    return [out, xh, stats, y, bias, o, lse] if save else [out]


from . import affine_training_sm80 as _accurate_front


def _accurate_training_forward_ops_fake(
    x, weights, wo, ln_w, ln_b, mask, ds, eps, transposed, n_head, hd, fused, save
):
    """Same outputs as the original forward: [out] and, when ``save``, [xh, stats, y, bias, o, lse]."""
    return _original_forward_ops_fake(
        x, weights, wo, ln_w, ln_b, mask, ds, eps, transposed, n_head, hd, fused, save
    )


@opaque(
    fake=_accurate_training_forward_ops_fake,
    name="triangle_attention_sm80w_accurate_training_a100_forward",
)
def _accurate_training_forward_ops(
    x: torch.Tensor,
    weights: list[torch.Tensor],
    wo: torch.Tensor,
    ln_w: torch.Tensor,
    ln_b: torch.Tensor,
    mask: torch.Tensor,
    ds: torch.Tensor,
    eps: float,
    transposed: bool,
    n_head: int,
    hd: int,
    fused: bool,
    save: bool,
) -> list[torch.Tensor]:
    """``pair + drop(TriangleAttention(pair))`` at any supported width: ``x`` the contiguous pair stack ``[B, L, L, D]``, ``weights`` = (q, k, v, g, bias), ``mask`` ``[B, L]`` uint8 or
    empty, ``ds`` the dropout scale ``[B, L, D]`` (indexed by the token's second index) or empty.  Returns [out] and, with ``save``, [xh, stats, y, bias, o, lse] for the backward
    (``xh`` the normalised input followed by the constant column(s), ``y`` the front's output [T, cfg.n])."""
    cfg = Cfg(x.shape[-1], n_head, hd, fused)
    rows, core = (_rows(), None if cfg.native else _core._ext())
    b, length = (x.shape[0], x.shape[1])
    t, d, dhp, n = (b * length * length, cfg.d, cfg.dhp, cfg.n)
    x2 = x.view(t, d)
    dev = x.device
    bias = torch.empty((b, cfg.h, length, length), dtype=BF, device=dev)
    stats = (
        torch.empty((t, 2), dtype=F32, device=dev)
        if save
        else x.new_empty((0,), dtype=F32)
    )
    if fused:
        y = torch.empty((t, n), dtype=BF, device=dev)
        xh = (
            torch.empty((t, d + 8), dtype=BF, device=dev) if save else x.new_empty((0,))
        )
        y = _accurate_front.front(
            x2,
            weights,
            ln_w,
            ln_b,
            mask,
            y,
            bias,
            stats,
            xh,
            length,
            int(transposed),
            float(eps),
            d,
            dhp,
            cfg.lhd,
        )
    else:
        padding = 32 if not save and d == 256 and length >= 512 else 8
        if padding == 32:
            n = -(-n // 32) * 32
        wp = _pack_folded(weights, ln_w, ln_b, cfg, padding, n)
        xh = torch.empty((t, d + padding), dtype=BF, device=dev)
        rows.ln_rows(x2, xh, stats, d, length, int(transposed), float(eps))
        y = torch.mm(xh, wp.t())
        rows.bias_planes(y, mask, bias, 4 * dhp, cfg.h, length, 0)
    lse = (
        torch.empty((b, cfg.h, length, length), dtype=F32, device=dev)
        if save
        else x.new_empty((0,), dtype=F32)
    )
    scl = _core._L2E / math.sqrt(hd)
    q, k, v, g = (
        y[:, :dhp],
        y[:, dhp : 2 * dhp],
        y[:, 2 * dhp : 3 * dhp],
        y[:, 3 * dhp : 4 * dhp],
    )
    if fused:
        o = torch.empty((t, dhp), dtype=BF, device=dev)
        if cfg.native:
            lay, lay_o = (
                _core2.token_major(length, n, 16),
                _core2.token_major(length, dhp, 16),
            )
            _core2.forward(
                q,
                k,
                v,
                bias,
                x.new_empty((0,), dtype=torch.uint8),
                o,
                lse,
                (lay, lay, lay, lay_o),
                length,
                cfg.h,
                b,
                16,
                1.0 / math.sqrt(hd),
                fixed=True,
            )
        else:
            fwd = _core._forward_entry(core, save)
            fwd(
                q,
                k,
                v,
                bias,
                o,
                lse,
                length,
                cfg.h,
                b,
                n,
                n,
                n,
                dhp,
                scl,
                _core.ROWS,
                _core.BN,
                _core.MINB,
            )
        out = torch.empty_like(x)
        rows.fused_back(
            o,
            g,
            _wo_padded(wo, cfg),
            x2,
            out.view(t, d),
            ds,
            length,
            int(transposed),
            d,
            dhp,
            cfg.lhd,
        )
    else:
        o = torch.empty((t, dhp), dtype=BF, device=dev) if save else x.new_empty((0,))
        a = torch.empty((t, dhp), dtype=BF, device=dev)
        if (
            not save
            and mask.numel()
            and (length >= 384)
            and (length % 384 == 0)
        ):
            indices = torch.empty((b, length), dtype=torch.int32, device=dev)
            counts = torch.empty((b,), dtype=torch.int32, device=dev)
            compact_bias = torch.empty_like(bias)
            shape = (b, length, length, dhp)
            core.compact_attention_gate_fast(
                q.view(shape),
                k.view(shape),
                v.view(shape),
                bias,
                mask,
                a.view(shape),
                indices,
                counts,
                compact_bias,
                scl,
                g.view(shape),
            )
        else:
            fwd = _core._forward_entry(core, save, gate=True)
            fwd(
                q, k, v, bias, o, lse, g, a, length, cfg.h, b, n, n, n, dhp, n, dhp, scl
            )
        out = torch.empty_like(x)
        if (
            not save
            and (d, cfg.h, hd) == (256, 8, 32)
            and length == 128
        ):
            from . import native_out_fused_sm80

            out_cfg = 4 if transposed else 14
            native_out_fused_sm80.run(
                a,
                _wo_padded(wo, cfg),
                x2,
                ds,
                out.view(t, d),
                length,
                transposed,
                out_cfg,
            )
        else:
            yo = torch.mm(a, _wo_padded(wo, cfg).t())
            rows.out_rows(yo, x2, ds, out.view(t, d), length, int(transposed))
    return [out, xh, stats, y, bias, o, lse] if save else [out]


def forward_ops(
    x, weights, wo, ln_w, ln_b, mask, ds, eps, transposed, n_head, hd, fused, save, flip=False
):
    if save and fused:
        cfg = Cfg(x.shape[-1], n_head, hd, fused)
        if _accurate_front.eligible(
            x, weights, ln_w, ln_b, cfg.d, cfg.dhp, cfg.lhd, x.shape[1]
        ):
            return _accurate_training_forward_ops(
                x,
                weights,
                wo,
                ln_w,
                ln_b,
                mask,
                ds,
                eps,
                transposed,
                n_head,
                hd,
                fused,
                save,
            )
    return _original_forward_ops(
        x, weights, wo, ln_w, ln_b, mask, ds, eps, transposed, n_head, hd, fused, save, flip
    )


def _dropout_scale(ds: torch.Tensor | None, like: torch.Tensor) -> torch.Tensor:
    return ds.contiguous() if ds is not None else like.new_empty((0,))


def forward(
    pair: torch.Tensor,
    cfg: Cfg,
    weights: Sequence[torch.Tensor],
    wo: torch.Tensor,
    ln_w: torch.Tensor,
    ln_b: torch.Tensor,
    eps: float,
    mask: torch.Tensor | None = None,
    *,
    transposed: bool = False,
    ds: torch.Tensor | None = None,
    flip: bool = False,
) -> torch.Tensor:
    """``pair + drop(TriangleAttention(pair))`` without autograd (inference): ``weights`` = (q, k, v, g, bias); ``flip``: AlphaFold 3's ending-node bias
    orientation (the variant path's ``flip``, written by the front).  Call ``supports()`` first."""
    return forward_ops(
        pair,
        list(weights),
        wo,
        ln_w,
        ln_b,
        _mask_u8(mask, pair.device),
        _dropout_scale(ds, pair),
        float(eps),
        bool(transposed),
        cfg.h,
        cfg.hd,
        cfg.fused,
        False,
        bool(flip),
    )[0]


# ---------------------------------------------------------------------------------------------------------------------------------- backward
def _attention_backward(
    y, bias, lse, delta, dov, dqkvg, cfg: Cfg, b: int, length: int
) -> torch.Tensor:
    """The core's backward on the front's output ``y`` (q | k | v columns): dq | dk | dv go into the first 3 H 32 columns of ``dqkvg`` ``[T, 4 H 32]``; returns the bias gradient
    ``[B, H, L, L]`` bf16 (the sum over the pair rows of dS, from bf16 partials over groups of ``DQ_ROWS`` rows and a fixed-order reduction)."""
    dhp, n = cfg.dhp, cfg.n
    sm_scale = 1.0 / math.sqrt(cfg.hd)
    if cfg.native:  # 16-channel heads: the generalised core, in place
        q, k, v = y[:, :dhp], y[:, dhp : 2 * dhp], y[:, 2 * dhp : 3 * dhp]
        lay, lay_h, lay_d = (
            _core2.token_major(length, n, 16),
            _core2.token_major(length, dhp, 16),
            _core2.token_major(length, 4 * dhp, 16),
        )
        return _core2.backward(
            q,
            k,
            v,
            dov,
            bias,
            y.new_empty((0,), dtype=torch.uint8),
            lse,
            delta,
            dqkvg[:, :dhp],
            dqkvg[:, dhp : 2 * dhp],
            dqkvg[:, 2 * dhp : 3 * dhp],
            (lay, lay, lay, lay_h, lay_d, lay_d, lay_d),
            length,
            cfg.h,
            b,
            16,
            sm_scale,
        )
    core = _core._ext()
    scl = sm_scale * _core._L2E
    groups = length // _core.DQ_ROWS
    dbp = torch.empty((groups, b * cfg.h, length, length), dtype=BF, device=y.device)
    q, k, v = y[:, :dhp], y[:, dhp : 2 * dhp], y[:, 2 * dhp : 3 * dhp]
    core.attn_bwd_dq(
        q,
        k,
        v,
        dov,
        bias,
        lse,
        delta,
        dqkvg[:, :dhp],
        dbp,
        length,
        cfg.h,
        b,
        n,
        n,
        n,
        dhp,
        4 * dhp,
        scl,
        sm_scale,
        _core.DQ_ROWS,
        _core.DQ_NKV,
        _core.DQ_BN,
        _core.DQ_MINB,
    )
    db = torch.empty((b, cfg.h, length, length), dtype=BF, device=y.device)
    core.db_reduce(dbp, db, groups)
    del dbp
    bias_t = torch.empty_like(bias)
    core.bias_transpose(bias, bias_t, length)
    core.attn_bwd_dkv(
        q,
        k,
        v,
        dov,
        bias_t,
        lse,
        delta,
        dqkvg[:, dhp : 2 * dhp],
        dqkvg[:, 2 * dhp : 3 * dhp],
        length,
        cfg.h,
        b,
        n,
        n,
        n,
        dhp,
        4 * dhp,
        4 * dhp,
        scl,
        sm_scale,
        _core.DKV_NST,
        32,
        _core.DKV_MINB,
    )
    return db


def _backward_ops_fake(
    x, weights, wo, ln_w, ln_b, mask, ds, saved, dout, transposed, n_head, hd, fused
):
    """[dpair, dwq, dwk, dwv, dwg, dwb, dwo, dln_w, dln_b] in the leaves' shapes and dtypes."""
    return [
        x.new_empty(x.shape),
        *(w.new_empty(w.shape) for w in weights),
        wo.new_empty(wo.shape),
        ln_w.new_empty(ln_w.shape),
        ln_b.new_empty(ln_b.shape),
    ]


@opaque(fake=_backward_ops_fake, name="triangle_attention_sm80w_backward")
def backward_ops(
    x: torch.Tensor,
    weights: list[torch.Tensor],
    wo: torch.Tensor,
    ln_w: torch.Tensor,
    ln_b: torch.Tensor,
    mask: torch.Tensor,
    ds: torch.Tensor,
    saved: list[torch.Tensor],
    dout: torch.Tensor,
    transposed: bool,
    n_head: int,
    hd: int,
    fused: bool,
) -> list[torch.Tensor]:
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
        rows.fused_back_bwd(
            dout2,
            ds,
            o,
            y[:, 3 * dhp : 4 * dhp],
            wot,
            dqkvg[:, 3 * dhp :],
            dov,
            dy,
            a,
            delta,
            length,
            int(transposed),
            d,
            dhp,
            cfg.lhd,
        )
    else:
        if transposed or ds.numel():  # dy in the starting frame after the dropout scale
            dy = torch.empty((t, d), dtype=BF, device=dev)
            rows.dy_rows(dout2, ds, dy, length, int(transposed))
        else:
            dy = dout2
        da = torch.mm(dy, _wo_padded(wo, cfg))  # [T, H 32] = bf16(dy Wo)
        rows.gate_bwd(
            da,
            o,
            y[:, 3 * dhp : 4 * dhp],
            dqkvg[:, 3 * dhp :],
            dov,
            a,
            delta,
            length,
            cfg.h,
        )
        del da
    go = torch.mm(dy.t(), a, out_dtype=F32)  # dWo [D, H 32]
    del a
    db = _attention_backward(y, bias, lse, delta, dov, dqkvg, cfg, b, length)
    del dov, delta
    dpair = torch.empty_like(x)
    if fused:
        rows.fused_front_bwd(
            dqkvg,
            db,
            x.view(t, d),
            stats,
            dout2,
            dpair.view(t, d),
            wt,
            wbt,
            gamma32,
            length,
            int(transposed),
            d,
            dhp,
            cfg.lhd,
        )
    else:
        dxn = torch.mm(dqkvg, _pack_plain(weights, cfg))  # [T, D]
        rows.ln_bwd(
            dxn,
            db,
            weights[4].contiguous(),
            x.view(t, d),
            stats,
            dout2,
            ln_w.detach().float().contiguous(),
            dpair.view(t, d),
            length,
            cfg.h,
            int(transposed),
        )
        del dxn
    g = torch.cat(
        [
            torch.mm(dqkvg.t(), xh, out_dtype=F32),
            torch.mm(db.permute(1, 0, 2, 3).reshape(cfg.h, t), xh, out_dtype=F32),
        ]
    )  # [4 H 32 + H, D + 8]
    dws = [torch.empty_like(w) for w in weights]
    dwo = torch.empty_like(wo)
    dg, dbeta = torch.empty_like(ln_w), torch.empty_like(ln_b)
    rows.wgrad_w(
        g,
        go,
        *(w.contiguous() for w in weights),
        *dws,
        dwo,
        ln_w.detach().contiguous(),
        ln_b.detach().contiguous(),
        dg,
        dbeta,
        d,
        cfg.h,
        cfg.hd,
    )
    return [dpair, *dws, dwo, dg, dbeta]


class _Training(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        wq,
        wk,
        wv,
        wg,
        wb,
        wo,
        ln_w,
        ln_b,
        mask,
        ds,
        eps,
        transposed,
        n_head,
        hd,
        fused,
    ):
        weights = [wq, wk, wv, wg, wb]
        out, *saved = forward_ops(
            x,
            weights,
            wo,
            ln_w,
            ln_b,
            mask,
            ds,
            eps,
            transposed,
            n_head,
            hd,
            fused,
            True,
        )
        ctx.save_for_backward(x, wq, wk, wv, wg, wb, wo, ln_w, ln_b, mask, ds, *saved)
        ctx.cfg = (transposed, n_head, hd, fused)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, dout):
        x, wq, wk, wv, wg, wb, wo, ln_w, ln_b, mask, ds, *saved = ctx.saved_tensors
        grads = backward_ops(
            x,
            [wq, wk, wv, wg, wb],
            wo,
            ln_w,
            ln_b,
            mask,
            ds,
            saved,
            dout.contiguous(),
            *ctx.cfg,
        )
        return (*grads, None, None, None, None, None, None, None)


def trainable(
    pair: torch.Tensor,
    cfg: Cfg,
    weights: Sequence[torch.Tensor],
    wo: torch.Tensor,
    ln_w: torch.Tensor,
    ln_b: torch.Tensor,
    eps: float,
    mask: torch.Tensor | None = None,
    *,
    transposed: bool = False,
    ds: torch.Tensor | None = None,
) -> torch.Tensor:
    """``pair + drop(TriangleAttention(pair))`` with autograd: ``weights`` = (q, k, v, g, bias), ``ds`` the dropout scale ``[B, L, D]`` (indexed by the token's second index) or None.
    Call ``supports()`` first."""
    leaves = (pair, *weights, wo, ln_w, ln_b)
    mk = _mask_u8(mask, pair.device)
    d = _dropout_scale(ds, pair)
    if not (torch.is_grad_enabled() and any(t.requires_grad for t in leaves)):
        return forward_ops(
            pair,
            list(weights),
            wo,
            ln_w,
            ln_b,
            mk,
            d,
            float(eps),
            bool(transposed),
            cfg.h,
            cfg.hd,
            cfg.fused,
            False,
        )[0]
    return _Training.apply(
        *leaves, mk, d, float(eps), bool(transposed), cfg.h, cfg.hd, cfg.fused
    )


# ------------------------------------------------------------------------------------------------------------------------------ the variants
# The module variants the H100 width-generic path serves in its glue (``h100_sm90_generic._forward`` / ``train_backward``), on the rows engine: AlphaFold 3's ending-node bias
# orientation (``flip``: the bias planes of the pair as it is, read in the transposed frame), RoseTTAFold3's gate / out-projection biases, Chai-1's untransposed (ending node) and
# dual (confidence head) outputs, and heads of 64 channels.  The front is ``ln_rows`` and one GEMM (the gate bias in its epilogue), the core the generalised core (``sm80_core2``:
# head dim 32 -- a 16-channel head zero-padded as above -- or 64), the back ``gate_rows`` and the out-projection GEMM(s) (the out bias in the epilogue) with the residual.
# A 64-channel head packs as two 32-channel heads (q | k | v | g rows in their natural order; the bias heads' weight padded with zero rows), so every row kernel and the
# weight-gradient finalizer run unchanged; only the core and the row term (summed over the two halves) see the real heads.
STANDARD, UNTRANSPOSED, DUAL = 0, 1, 2


class VCfg(NamedTuple):
    """A variant geometry: ``d`` = d_pair, ``h`` heads of ``hd`` (16 / 32 / 64) channels; ``pack`` the rows-engine geometry of the packed weights, ``ch`` the core's head dim."""

    d: int
    h: int
    hd: int

    @property
    def pack(self) -> Cfg:
        return Cfg(self.d, 2 * self.h, 32) if self.hd == 64 else Cfg(self.d, self.h, self.hd)

    @property
    def ch(self) -> int:
        return 64 if self.hd == 64 else HD_CORE


def variant_cfg(d_pair: int, d_hidden: int, n_head: int) -> VCfg | None:
    """The variant geometry when the rows engine serves it (d_pair a multiple of 64 in 64 .. 512, heads of 16 / 32 channels up to 16, of 64 channels up to 8), else None."""
    if n_head < 1 or d_hidden % n_head or d_pair % 64 or not 64 <= d_pair <= MAX_D:
        return None
    hd = d_hidden // n_head
    if hd not in (16, 32, 64) or n_head * (2 if hd == 64 else 1) > MAX_H:
        return None
    return VCfg(d_pair, n_head, hd)


def variant_supports(pair, vc: VCfg | None, weights, wo, ln_w, ln_b, mask=None, *, gate_bias=None, out_bias=None, wo_t=None, train=False) -> bool:
    """``supports()`` for the variant path: the same pair / weight / LayerNorm / mask contract, the variants' parameters bf16 (gate bias ``[H hd]``, out bias and ``wo_t`` as
    ``to_out``'s), and for ``train`` the bias-gradient partials within ``sm80_core2.MAX_DBP_BYTES``."""
    if vc is None:
        return False
    if not supports(pair, Cfg(vc.d, vc.h, vc.hd), weights, wo, ln_w, ln_b, mask):  # (the train budget below, at the core's rows per partial)
        return False
    dh = vc.h * vc.hd
    for t, shape in ((gate_bias, (dh,)), (out_bias, (vc.d,)), (wo_t, (vc.d, dh))):
        if t is not None and (t.dtype is not BF or t.device != pair.device or tuple(t.shape) != shape):
            return False
    if train and not _core2.dbp_fits(pair.shape[1], vc.h, pair.shape[0], vc.ch):
        return False
    return _core2.loads() and _loads()


def _v_weights(weights, vc: VCfg) -> list[torch.Tensor]:
    """The packed geometry's (q, k, v, g, bias) weights: a 64-channel head's bias weight padded with zero rows to the two halves' head count."""
    ws = [w.contiguous() for w in weights]
    if vc.hd == 64:
        ws[4] = torch.cat([ws[4], ws[4].new_zeros(ws[4].shape)])
    return ws


def _v_drop_view(ds: torch.Tensor, b: int, length: int, d: int, transposed: bool) -> torch.Tensor:
    """The module's dropout scale ``[B, L, C]`` in the pair's own frame: starting ``[B, 1, L, C]``, ending ``[B, L, 1, C]``."""
    return ds.view(b, length, 1, d) if transposed else ds.view(b, 1, length, d)


def _v_lays(vc: VCfg, length: int, n: int):
    tm = _core2.token_major
    return tm(length, n, vc.ch), tm(length, vc.h * vc.ch, vc.ch), tm(length, 4 * vc.h * vc.ch, vc.ch)


def _variant_forward_fake(x, weights, wo, ln_w, ln_b, mask, ds, gb, bo, wt, eps, transposed, n_head, hd, flip, out_mode, save):
    vc = VCfg(x.shape[-1], n_head, hd)
    pc = vc.pack
    b, length = x.shape[0], x.shape[1]
    t = b * length * length
    out = [x.new_empty(x.shape)]
    if save:
        out += [
            x.new_empty((t, pc.d + 8)),
            x.new_empty((t, 2), dtype=F32),
            x.new_empty((t, pc.n)),
            x.new_empty((b, vc.h, length, length)),
            x.new_empty((t, pc.dhp)),
            x.new_empty((b, vc.h, length, length), dtype=F32),
        ]
    return out


@opaque(fake=_variant_forward_fake, name="triangle_attention_sm80w_variant_forward")
def variant_forward_ops(
    x: torch.Tensor,
    weights: list[torch.Tensor],
    wo: torch.Tensor,
    ln_w: torch.Tensor,
    ln_b: torch.Tensor,
    mask: torch.Tensor,
    ds: torch.Tensor,
    gb: torch.Tensor,
    bo: torch.Tensor,
    wt: torch.Tensor,
    eps: float,
    transposed: bool,
    n_head: int,
    hd: int,
    flip: bool,
    out_mode: int,
    save: bool,
) -> list[torch.Tensor]:
    """``pair + drop(update(pair))`` of a variant: ``mask`` ``[B, L]`` uint8 or empty, ``ds`` ``[B, L, D]`` or empty, ``gb`` / ``bo`` / ``wt`` the gate bias, the out bias and the
    second out-projection (empty when absent).  Returns [out] and, with ``save``, [xh, stats, y, bias, o, lse]."""
    vc = VCfg(x.shape[-1], n_head, hd)
    pc, ch, h = vc.pack, vc.ch, n_head
    rows = _rows()
    b, length = x.shape[0], x.shape[1]
    t, d, dhp, n = b * length * length, vc.d, pc.dhp, pc.n
    x2 = x.view(t, d)
    dev = x.device
    wp = _core.eval_form(("triattn_sm80w_variant", vc), (*weights, ln_w, ln_b), lambda: _pack_folded(_v_weights(weights, vc), ln_w, ln_b, pc, 8, n), save)
    xh = torch.empty((t, d + 8), dtype=BF, device=dev)
    stats = torch.empty((t, 2), dtype=F32, device=dev) if save else x.new_empty((0,), dtype=F32)
    rows.ln_rows(x2, xh, stats, d, length, int(transposed), float(eps))
    if gb.numel():  # RoseTTAFold3's gate bias in the GEMM's epilogue (one rounding, as the reference's biased projection)
        bvec = torch.zeros((n,), dtype=BF, device=dev)
        bvec[3 * dhp : 4 * dhp] = torch.nn.functional.pad(gb.view(h, hd), (0, ch - hd)).reshape(-1)
        y = torch.addmm(bvec, xh, wp.t())
    else:
        y = torch.mm(xh, wp.t())
    bias = torch.empty((b, h, length, length), dtype=BF, device=dev)
    # flip: the bias planes of the pair as it is, read in the transposed frame -- the kernel writes the planes transposed with the key mask on
    # the transposed axis (bit-identical to the unmasked planes, a transposing copy and a masked_fill, which were three more kernels)
    rows.bias_planes(y, mask, bias, 4 * dhp, h, length, int(flip))
    lay_y, lay_o, _ = _v_lays(vc, length, n)
    o = torch.empty((t, dhp), dtype=BF, device=dev)
    lse = torch.empty((b, h, length, length), dtype=F32, device=dev) if save else x.new_empty((0,), dtype=F32)
    q, k, v, g = (y[:, i * dhp : (i + 1) * dhp] for i in range(4))
    _core2.forward(q, k, v, bias, x.new_empty((0,), dtype=torch.uint8), o, lse, (lay_y, lay_y, lay_y, lay_o), length, h, b, ch, 1.0 / math.sqrt(hd))
    a = torch.empty((t, dhp), dtype=BF, device=dev)
    rows.gate_rows(o, g, a)
    wo_p = _core.eval_form(("triattn_sm80w_variant_wo", pc), (wo,), lambda: _wo_padded(wo, pc), save)
    bo_ = bo if bo.numel() else None

    def proj(w):
        return torch.mm(a, w.t()) if bo_ is None else torch.addmm(bo_, a, w.t())

    out = torch.empty_like(x)
    if out_mode == STANDARD:
        rows.out_rows(proj(wo_p), x2, ds, out.view(t, d), length, int(transposed))
    elif out_mode == UNTRANSPOSED:  # Chai-1's ending node: the update at the attended pair's own positions
        yo = proj(wo_p)
        if ds.numel():
            out = x + yo.view(x.shape) * _v_drop_view(ds, b, length, d, transposed)
        else:
            rows.out_rows(yo, x2, ds, out.view(t, d), length, 0)
    else:  # Chai-1's confidence head: one projection lands untransposed, the other transposed (starting: W_o u + (W_t u)^T; ending: (W_o u)^T + W_t u)
        wt_p = _wo_padded(wt, pc)
        first, second = (wt_p, wo_p) if transposed else (wo_p, wt_p)
        upd = proj(first).view(x.shape) + torch.mm(a, second.t()).view(x.shape).transpose(1, 2)
        if ds.numel():
            upd = upd * _v_drop_view(ds, b, length, d, transposed)
        out = x + upd
    return [out, xh, stats, y, bias, o, lse] if save else [out]


def _variant_backward_fake(x, weights, wo, ln_w, ln_b, mask, ds, gb, bo, wt, saved, dout, transposed, n_head, hd, flip, out_mode):
    return [x.new_empty(x.shape), *(w.new_empty(w.shape) for w in weights), wo.new_empty(wo.shape), ln_w.new_empty(ln_w.shape), ln_b.new_empty(ln_b.shape),
            gb.new_empty(gb.shape), bo.new_empty(bo.shape), wt.new_empty(wt.shape)]


@opaque(fake=_variant_backward_fake, name="triangle_attention_sm80w_variant_backward")
def variant_backward_ops(
    x: torch.Tensor,
    weights: list[torch.Tensor],
    wo: torch.Tensor,
    ln_w: torch.Tensor,
    ln_b: torch.Tensor,
    mask: torch.Tensor,
    ds: torch.Tensor,
    gb: torch.Tensor,
    bo: torch.Tensor,
    wt: torch.Tensor,
    saved: list[torch.Tensor],
    dout: torch.Tensor,
    transposed: bool,
    n_head: int,
    hd: int,
    flip: bool,
    out_mode: int,
) -> list[torch.Tensor]:
    """The gradients of ``variant_forward_ops``: [dpair, dwq, dwk, dwv, dwg, dwb, dwo, dln_w, dln_b, dgb, dbo, dwt] (an absent parameter's gradient empty)."""
    rows = _rows()
    xh, stats, y, bias, o, lse = saved
    vc = VCfg(x.shape[-1], n_head, hd)
    pc, ch, h = vc.pack, vc.ch, n_head
    b, length = x.shape[0], x.shape[1]
    t, d, dhp, n = b * length * length, vc.d, pc.dhp, pc.n
    dev = x.device
    dout2 = dout.contiguous().view(t, d)
    wo_p = _wo_padded(wo, pc)
    wt_p = _wo_padded(wt, pc) if out_mode == DUAL else None
    # the update's gradient in the attended frame, for each out-projection
    if out_mode == STANDARD:
        if transposed or ds.numel():
            dy = torch.empty((t, d), dtype=BF, device=dev)
            rows.dy_rows(dout2, ds, dy, length, int(transposed))
        else:
            dy = dout2
        da, dy_t = torch.mm(dy, wo_p), None
    else:
        dt = dout2 if not ds.numel() else (dout2.view(x.shape) * _v_drop_view(ds, b, length, d, transposed)).view(t, d)
        if out_mode == UNTRANSPOSED:
            dy, dy_t = dt, None
            da = torch.mm(dy, wo_p)
        else:
            dtt = dt.view(x.shape).transpose(1, 2).contiguous().view(t, d)
            dy, dy_t = (dtt, dt) if transposed else (dt, dtt)  # (for Wo, for Wo_t)
            da = torch.addmm(torch.mm(dy, wo_p), dy_t, wt_p)
    dqkvg = torch.empty((t, 4 * dhp), dtype=BF, device=dev)
    dov = torch.empty((t, dhp), dtype=BF, device=dev)
    a = torch.empty((t, dhp), dtype=BF, device=dev)
    delta = torch.empty((b, pc.h, length, length), dtype=F32, device=dev)
    rows.gate_bwd(da, o, y[:, 3 * dhp : 4 * dhp], dqkvg[:, 3 * dhp :], dov, a, delta, length, pc.h)
    del da
    if vc.hd == 64:  # the row term of a 64-channel head: its two 32-channel halves
        delta = delta.view(b, h, 2, length, length).sum(2)
    go = torch.mm(dy.t(), a, out_dtype=F32)  # dWo [D, H ch]
    dwt = wt.new_empty((0,))
    if dy_t is not None:
        gt = torch.mm(dy_t.t(), a, out_dtype=F32)
        dwt = (gt if vc.hd != 16 else gt.view(d, h, HD_CORE)[:, :, :16].reshape(d, h * 16)).to(wt.dtype)
    dbo = bo.new_empty((0,)) if not bo.numel() else torch.sum(dy, 0, dtype=F32).to(bo.dtype)
    del a
    lay_y, lay_o, lay_d = _v_lays(vc, length, n)
    q, k, v = (y[:, i * dhp : (i + 1) * dhp] for i in range(3))
    dq, dk, dv = (dqkvg[:, i * dhp : (i + 1) * dhp] for i in range(3))
    db = _core2.backward(q, k, v, dov, bias, x.new_empty((0,), dtype=torch.uint8), lse, delta, dq, dk, dv,
                         (lay_y, lay_y, lay_y, lay_o, lay_d, lay_d, lay_d), length, h, b, ch, 1.0 / math.sqrt(hd))
    del dov, delta
    if flip:  # the core's bias was the transpose of the projected planes
        db = db.transpose(-1, -2).contiguous()
    dgb = gb.new_empty((0,))
    if gb.numel():
        dgb = torch.sum(dqkvg[:, 3 * dhp : 4 * dhp], 0, dtype=F32).view(h, ch)[:, :hd].reshape(-1).to(gb.dtype)
    ws = _v_weights(weights, vc)
    dpair = torch.empty_like(x)
    dxn = torch.mm(dqkvg, _pack_plain(ws, pc))
    rows.ln_bwd(dxn, db, weights[4].contiguous(), x.view(t, d), stats, dout2, ln_w.detach().float().contiguous(), dpair.view(t, d), length, h, int(transposed))
    del dxn
    parts = [torch.mm(dqkvg.t(), xh, out_dtype=F32), torch.mm(db.permute(1, 0, 2, 3).reshape(h, t), xh, out_dtype=F32)]
    if pc.h != h:
        parts.append(xh.new_zeros((pc.h - h, d + 8), dtype=F32))
    g = torch.cat(parts)  # [4 DHp + H', D + 8]
    dws = [torch.empty_like(w) for w in ws]
    dwo = torch.empty_like(wo)
    dg, dbeta = torch.empty_like(ln_w), torch.empty_like(ln_b)
    rows.wgrad_w(g, go, *ws, *dws, dwo, ln_w.detach().contiguous(), ln_b.detach().contiguous(), dg, dbeta, d, pc.h, pc.hd)
    dws[4] = dws[4][:h].clone() if pc.h != h else dws[4]
    return [dpair, *dws, dwo, dg, dbeta, dgb, dbo, dwt]


class _VariantTraining(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, wq, wk, wv, wg, wb, wo, ln_w, ln_b, gb, bo, wt, mask, ds, eps, transposed, n_head, hd, flip, out_mode):
        weights = [wq, wk, wv, wg, wb]
        out, *saved = variant_forward_ops(x, weights, wo, ln_w, ln_b, mask, ds, gb, bo, wt, eps, transposed, n_head, hd, flip, out_mode, True)
        ctx.save_for_backward(x, wq, wk, wv, wg, wb, wo, ln_w, ln_b, gb, bo, wt, mask, ds, *saved)
        ctx.cfg = (transposed, n_head, hd, flip, out_mode)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, dout):
        x, wq, wk, wv, wg, wb, wo, ln_w, ln_b, gb, bo, wt, mask, ds, *saved = ctx.saved_tensors
        grads = variant_backward_ops(x, [wq, wk, wv, wg, wb], wo, ln_w, ln_b, mask, ds, gb, bo, wt, saved, dout.contiguous(), *ctx.cfg)
        grads = [gr if gr.numel() else None for gr in grads]
        return (*grads, None, None, None, None, None, None, None, None)


def variant(
    pair: torch.Tensor,
    vc: VCfg,
    weights: Sequence[torch.Tensor],
    wo: torch.Tensor,
    ln_w: torch.Tensor,
    ln_b: torch.Tensor,
    eps: float,
    mask: torch.Tensor | None = None,
    *,
    transposed: bool = False,
    flip: bool = False,
    out_mode: int = STANDARD,
    gate_bias: torch.Tensor | None = None,
    out_bias: torch.Tensor | None = None,
    wo_t: torch.Tensor | None = None,
    ds: torch.Tensor | None = None,
) -> torch.Tensor:
    """``pair + drop(update(pair))`` of a variant, with autograd when a leaf wants a gradient: ``weights`` = (q, k, v, g, bias), ``ds`` the dropout scale ``[B, L, D]`` or None.
    Call ``variant_supports()`` first."""
    empty = pair.new_empty((0,))
    gb, bo, wt = (t if t is not None else empty for t in (gate_bias, out_bias, wo_t))
    leaves = (pair, *weights, wo, ln_w, ln_b, gb, bo, wt)
    mk = _mask_u8(mask, pair.device)
    d = _dropout_scale(ds, pair)
    args = (float(eps), bool(transposed), vc.h, vc.hd, bool(flip), int(out_mode))
    if not (torch.is_grad_enabled() and any(t.requires_grad for t in leaves)):
        return variant_forward_ops(pair, list(weights), wo, ln_w, ln_b, mk, d, gb, bo, wt, *args, False)[0]
    return _VariantTraining.apply(*leaves, mk, d, *args)
