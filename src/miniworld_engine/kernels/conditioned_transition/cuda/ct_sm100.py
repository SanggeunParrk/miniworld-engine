"""B200 (sm_100a) ConditionedTransition tail, inference: two tcgen05 GEMMs of the generic sm_100a Transition
(``kernels/transition/cuda/sm100/generic/tg_gemm.cu``, launched through ``fused_generic_sm100a``)::

    h = silu(x Wa^T) * (x Wb^T)                              SWIGLU: both expand products in tensor memory, SwiGLU in the epilogue
    y = res + sigmoid(cond Wsc^T + bsc) * (h Ws^T)            CTGATE: the squeeze and the gate projection as the two phases of one
                                                              output tile (two accumulators), gate and residual in the epilogue

``x`` is the output of the module's AdaLN (the tail is post-AdaLN, as on every other path). bf16 on the bf16 tensor cores, fp32 on the
TF32 ones (where torch allows TF32); any row count. The weights are cast (fp32: rounded to the TF32 grid) once per parameter version
(``_capture.param_forms``). The H100 tail is ``ct_inference`` (sm_90a wgmma); the training tail on both cards is ``ct_train``.
"""

from __future__ import annotations

import functools
from pathlib import Path

import torch

from miniworld_engine.kernels import _capture
from miniworld_engine.kernels._compile import opaque


def supported(x: torch.Tensor, cond: torch.Tensor, nd: int, d: int) -> bool:
    """sm_100, x / cond of one dtype (bf16, or fp32 with TF32 allowed), D a multiple of 64, the hidden width a multiple of 128, the
    conditioning width a multiple of one 128-byte K-block (64 bf16 / 32 fp32)."""
    from miniworld_engine.kernels.transition.cuda import fused_generic_sm100a as g

    if not (x.device.type == "cuda" and cond.device.type == "cuda" and x.dtype == cond.dtype and x.dtype in (torch.bfloat16, torch.float32)):
        return False
    if x.dtype is torch.float32 and not g._tf32_allowed():
        return False
    idx = x.device.index if x.device.index is not None else torch.cuda.current_device()
    if not g._is_b200(idx):
        return False
    kq = 32 if x.dtype is torch.float32 else 64
    return d % 64 == 0 and nd % 128 == 0 and cond.shape[-1] % kq == 0 and x.shape[-1] == d and x.numel() > 0


def _weights(wa, wb, ws, wsc, bsc, f32: bool):
    """(wa, wb, ws, wsc) in the activation dtype (fp32: TF32 grid) and the gate bias in fp32 (rounded to bf16 first for bf16)."""
    from miniworld_engine.kernels._tf32 import tf32_grid
    from miniworld_engine.kernels.transition.cuda.fused_generic_sm100a import _prep

    dt = torch.float32 if f32 else torch.bfloat16
    src = wa.dtype
    oa, ob, os_, _, _ = _prep(wa, wb.to(src), ws.to(src), f32, False)
    wsc_ = wsc.to(dt).contiguous()
    if f32:
        wsc_ = tf32_grid(wsc_)
    return oa, ob, os_, wsc_, bsc.to(dt).float().contiguous()


def _tail_fake(xn, cond, wa, wb, ws, wsc, bsc, res):
    """Output structure of ``_tail``: y [M, D] in xn's dtype."""
    return torch.empty((xn.shape[0], ws.shape[0]), dtype=xn.dtype, device=xn.device)


@opaque(fake=_tail_fake, name="conditioned_transition_tail_sm100a")
def _tail(xn: torch.Tensor, cond: torch.Tensor, wa: torch.Tensor, wb: torch.Tensor, ws: torch.Tensor, wsc: torch.Tensor, bsc: torch.Tensor,
          res: torch.Tensor | None) -> torch.Tensor:
    """xn [M, D], cond [M, DC], the weights from ``_weights``; returns y [M, D] (res + the gated update, or the update alone)."""
    from miniworld_engine.kernels.transition.cuda.fused_generic_sm100a import _K

    k = _K(xn)
    m, d = xn.shape
    nd = wa.shape[0]
    h = torch.empty((m, nd), dtype=xn.dtype, device=xn.device)
    k.gemm("swiglu", m, nd, xn, wa, h, b1=wb, k0=d)
    y = torch.empty((m, d), dtype=xn.dtype, device=xn.device)
    k.gemm("ctgate", m, d, h, ws, y, a1=cond, b1=wsc, bias=bsc, res=res, k0=nd, k1=cond.shape[1], ldres=d if res is not None else 0)
    return y


#: fp32 (TF32) tail: "cublas" (the expand, squeeze and gate GEMMs on cuBLAS TF32 between the CUDA SwiGLU and gate row kernels of ct_rows.cu,
#: the weights packed once per parameter version) or "tcgen05" (the two generic sm_100a GEMMs). Measured (ladder, A = 5, us): d768 c384
#: L128 / 384 / 768 the tcgen05 tail's CT 61.2 / 90.0 / 114.5 against the Triton family's cuBLAS route 51.0 / 73.5 / 106.3 -- the TF32
#: tcgen05 GEMMs of the generic Transition kernel are below cuBLAS at these thin K.
F32_ROUTE = "cublas"
#: the widths each dtype takes the cuBLAS tail at: d_hidden above ``CUBLAS_MIN_D[dtype]`` (fp32: the token width only -- at the atom width
#: d128 the tcgen05 tail is the faster, ladder L4096 / 8192 59.4 / 84.0 against 69.5 / 108.4 us); bf16 also only up to ``BF16_CUBLAS_MAX_ROWS``
#: rows (module inference in a CUDA graph, A = 5, d768, us, tcgen05 / cuBLAS tail: L384 73.6 / 65.3, L768 91.8 / 88.0, L1024 100.0 / 104.3)
CUBLAS_MIN_D = {"fp32": 128, "bf16": 128}
BF16_CUBLAS_MAX_ROWS = 4096


def _f32_weights(wa, wb, ws, wsc, bsc, dt=torch.float32):
    """[Wa; Wb], Ws, Wsc in ``dt`` and bsc (contiguous; fp32: cuBLAS rounds TF32 operands itself; bf16: the bias in bf16 too, as addmm takes it)."""
    f = lambda w: w.to(dt).contiguous()  # noqa: E731
    return torch.cat([wa, wb]).to(dt).contiguous(), f(ws), f(wsc), f(bsc)


def _tail_cublas_fake(xn, cond, wab, ws, wsc, bsc, res):
    """Output structure of ``_tail_cublas``: y [M, D] like xn."""
    return torch.empty((xn.shape[0], ws.shape[0]), dtype=xn.dtype, device=xn.device)


@opaque(fake=_tail_cublas_fake, name="conditioned_transition_tail_f32_cublas_sm100a")
def _tail_cublas(xn: torch.Tensor, cond: torch.Tensor, wab: torch.Tensor, ws: torch.Tensor, wsc: torch.Tensor, bsc: torch.Tensor,
                 res: torch.Tensor | None) -> torch.Tensor:
    """ab = xn [Wa; Wb]^T, h = silu(a) b (swiglu_fwd), sp = cond Wsc^T + bsc, y = res + sigmoid(sp) * (h Ws^T) (sigmul_fwd)."""
    from miniworld_engine.kernels._tf32 import tf32_matmul

    from .ct_train import ext as rows_ext

    e = rows_ext()
    with tf32_matmul():
        ab = torch.mm(xn, wab.t())
        h = e.swiglu_fwd(ab)
        out = torch.mm(h, ws.t())
        sp = torch.addmm(bsc, cond, wsc.t())
    return e.sigmul_fwd(out, sp, res)


#: bf16 inference on the cuBLAS tail (d_hidden above ``CUBLAS_MIN_D``, rows up to ``BF16_CUBLAS_MAX_ROWS``): the module's AdaLN and the tail's
#: gate projection sp = cond Wsc^T + bsc as ONE tcgen05 GEMM over the raw cond rows (``ct_head_sm100.cu``: LN(cond) folded into the AdaLN
#: weights, three accumulators on one A operand) after ``head_stats`` (the row statistics, its programmatic primary), replacing
#: ln_cond_xstats -> adaln_gemm_sm100 and the cuBLAS addmm; then cuBLAS expand -> swiglu_fwd -> cuBLAS squeeze -> sigmul_fwd.
#: False: the AdaLN entry and ``ct_inference_sm100`` (A/B).
HEAD_FUSED = True
#: the head kernel's output tile widths, chosen per call by ``_head_bn`` (BN 128: three accumulators single-buffered; 64: double-buffered)
_HEAD_BNS = (128, 64)


def head_supported(x2: torch.Tensor, cond2: torch.Tensor, nd: int, d: int) -> bool:
    """The calls ``head_inference`` serves: bf16 rows on the cuBLAS tail, widths the head kernel and its row pass tile."""
    floor = CUBLAS_MIN_D["bf16"]
    dc = cond2.shape[-1]
    return (HEAD_FUSED and x2.dtype is torch.bfloat16 and cond2.dtype is torch.bfloat16 and floor is not None and d > floor
            and x2.shape[0] <= BF16_CUBLAS_MAX_ROWS and d % 128 == 0 and d <= 1024 and dc % 64 == 0 and dc <= 1024)


@functools.lru_cache(maxsize=4)
def _head_cubin(bn: int):
    from miniworld_engine.kernels import _nvcc
    from miniworld_engine.kernels.transition.cuda import fused_sm100a

    here = Path(__file__).parent
    inc = here.parent.parent / "transition" / "cuda" / "sm100"
    tag = f"bn{bn}"
    path = _nvcc.cubin(f"ct_sm100a/head_{tag}", ["-std=c++17", "-O3", "-arch=sm_100a", "-cubin", f"-I{inc}", f"-DBN={bn}"],
                       source=here / "ct_head_sm100.cu", toolchain="sm100")
    ext = fused_sm100a._ext()
    ext.load_cubin(f"ct_sm100a_head_{tag}", str(path))
    return ext, f"ct_sm100a_head_{tag}"


#: the split build (``ct_head_split_sm100.cu``: xn and sp tiles on different CTAs, each streaming 128 + 256 rows a K-block) where its grid
#: fits one wave; None turns it off (A/B)
HEAD_SPLIT = True


@functools.lru_cache(maxsize=1)
def _head_split_cubin():
    from miniworld_engine.kernels import _nvcc
    from miniworld_engine.kernels.transition.cuda import fused_sm100a

    here = Path(__file__).parent
    inc = here.parent.parent / "transition" / "cuda" / "sm100"
    path = _nvcc.cubin("ct_sm100a/head_split", ["-std=c++17", "-O3", "-arch=sm_100a", "-cubin", f"-I{inc}"],
                       source=here / "ct_head_split_sm100.cu", toolchain="sm100")
    ext = fused_sm100a._ext()
    ext.load_cubin("ct_sm100a_head_split", str(path))
    return ext, "ct_sm100a_head_split"


def _head_bn(m: int, n: int, nsm: int) -> int:
    """The tile width with the least (waves x per-tile operand intake: 128 cond rows and 3 BN weight rows per K-block). Measured (module
    inference, kernel alone, [3840, 768] c384): BN 64 16.8 us, BN 128 19.6."""
    best = None
    for bn in _HEAD_BNS:
        tiles = -(-m // 128) * (n // bn)
        cost = -(-tiles // nsm) * (128 + 3 * bn)
        if best is None or cost < best[0]:
            best = (cost, bn)
    return best[1]


def _head_fake(x2, c2, wbw, wsw, wg, bs, bg, ub, us, eps_x, eps_c):
    """(xn, sp) [M, D] like x2."""
    return torch.empty_like(x2), torch.empty_like(x2)


@opaque(fake=_head_fake, name="conditioned_transition_b200_bf16_head")
def _head(x2: torch.Tensor, c2: torch.Tensor, wbw: torch.Tensor, wsw: torch.Tensor, wg: torch.Tensor, bs: torch.Tensor, bg: torch.Tensor,
          ub: torch.Tensor, us: torch.Tensor, eps_x: float, eps_c: float) -> tuple[torch.Tensor, torch.Tensor]:
    """xn (the AdaLN of x2 by c2) and sp = c2 Wg^T + bg: x2 [M, D], c2 [M, DC] bf16 contiguous, the folded weights from ``_head_weights``."""
    from miniworld_engine.kernels._device import num_sms

    from .ct_train import ext as rows_ext

    m, d = x2.shape
    dc = c2.shape[1]
    nsm = num_sms(x2.device)
    st = rows_ext().head_stats(x2, c2, float(eps_x), float(eps_c))
    xn, sp = torch.empty_like(x2), torch.empty_like(x2)
    mt = -(-m // 128)
    if HEAD_SPLIT and d % 256 == 0 and mt * (d // 128 + d // 256) <= nsm:   # one wave of balanced xn / sp tiles
        ext, name = _head_split_cubin()
        t = ext.tmap
        ext.launch_kernel(name, "ct_head_split_sm100", mt * (d // 128 + d // 256), 320, 1,
                          [t(c2, dc, m, 64, 128), t(wbw, dc, d, 64, 128), t(wsw, dc, d, 64, 128), t(wg, dc, d, 64, 256), t(xn, d, m, 64, 32),
                           t(sp, d, m, 64, 32), x2, st, bs, bg, ub, us, m, d, dc], pdl=True)
        return xn, sp
    bn = _head_bn(m, d, nsm)
    ext, name = _head_cubin(bn)
    t = ext.tmap
    tiles = mt * (d // bn)
    ext.launch_kernel(name, "ct_head_sm100", max(1, min(nsm, tiles)), 320, 1,
                      [t(c2, dc, m, 64, 128), t(wbw, dc, d, 64, bn), t(wsw, dc, d, 64, bn), t(wg, dc, d, 64, bn), t(xn, d, m, 64, 32),
                       t(sp, d, m, 64, 32), x2, st, bs, bg, ub, us, m, d, dc], pdl=True)
    return xn, sp


def _head_weights(wc, ada_ws, ada_bs, ada_wb, wsc, bsc):
    """(Wb wc, Ws wc, Wg) bf16 [D, DC] and (bs, bg, ub, us) fp32 [D]: the AdaLN weights with the cond LayerNorm weight folded in, their row sums
    taken over the rounded values, and the gate biases rounded to bf16 first (as the unfused bf16 paths add them)."""
    bf = torch.bfloat16
    wcf = wc.float()
    wbw = (ada_wb.float() * wcf).to(bf).contiguous()
    wsw = (ada_ws.float() * wcf).to(bf).contiguous()
    d = wbw.shape[0]
    bs = ada_bs.to(bf).float().contiguous() if ada_bs is not None else torch.zeros(d, dtype=torch.float32, device=wbw.device)
    return (wbw, wsw, wsc.to(bf).contiguous(), bs, bsc.to(bf).float().contiguous(), wbw.float().sum(1).contiguous(),
            wsw.float().sum(1).contiguous())


def _tail_sp_fake(xn, wab, ws, sp, res):
    """y [M, D] like xn."""
    return torch.empty((xn.shape[0], ws.shape[0]), dtype=xn.dtype, device=xn.device)


@opaque(fake=_tail_sp_fake, name="conditioned_transition_tail_bf16_cublas_sp_sm100a")
def _tail_cublas_sp(xn: torch.Tensor, wab: torch.Tensor, ws: torch.Tensor, sp: torch.Tensor, res: torch.Tensor | None) -> torch.Tensor:
    """``_tail_cublas`` with the gate pre-activation sp handed in: ab = xn [Wa; Wb]^T, h = silu(a) b, y = res + sigmoid(sp) (h Ws^T)."""
    from .ct_train import ext as rows_ext

    e = rows_ext()
    h = e.swiglu_fwd(torch.mm(xn, wab.t()))
    return e.sigmul_fwd(torch.mm(h, ws.t()), sp, res)


def head_inference(x2, cond2, ln_cond_w, ada_ws, ada_bs, ada_wb, eps_x, eps_c, wa, wb, ws, wsc, bsc, residual=None):
    """The module's no-backward call (AdaLN + tail) for ``head_supported`` rows: x2 [M, D], cond2 [M, DC] bf16, the weights in any dtype
    (prepared once per parameter version), ``residual`` [M, D] bf16 or None. Returns y [M, D]."""
    hp = (ln_cond_w, ada_ws, ada_wb, wsc, bsc) + ((ada_bs,) if ada_bs is not None else ())
    hw = _capture.param_forms("ct_sm100a_head_bf16", hp, lambda: _head_weights(ln_cond_w, ada_ws, ada_bs, ada_wb, wsc, bsc))
    wab, wsq, _, _ = _capture.param_forms(f"ct_sm100a_cublas_{x2.dtype}", (wa, wb, ws, wsc, bsc), lambda: _f32_weights(wa, wb, ws, wsc, bsc, x2.dtype))
    xn, sp = _head(x2.contiguous(), cond2.contiguous(), *hw, float(eps_x), float(eps_c))
    return _tail_cublas_sp(xn, wab, wsq, sp, residual.contiguous() if residual is not None else None)


def ct_inference_sm100(xn, cond, wa, wb, ws, wsc, bsc, residual=None):
    """``residual + sigmoid(cond Wsc^T + bsc) * (silu(xn Wa^T) * (xn Wb^T)) Ws^T`` without a backward: ``xn`` [M, D], ``cond`` [M, DC] of
    one dtype, ``residual`` [M, D] in that dtype or None (then the update alone); the weights in any dtype."""
    f32 = xn.dtype is torch.float32
    floor = CUBLAS_MIN_D["fp32" if f32 else "bf16"]
    if (F32_ROUTE == "cublas" or not f32) and floor is not None and xn.shape[1] > floor and (f32 or xn.shape[0] <= BF16_CUBLAS_MAX_ROWS):
        ww = _capture.param_forms(f"ct_sm100a_cublas_{xn.dtype}", (wa, wb, ws, wsc, bsc), lambda: _f32_weights(wa, wb, ws, wsc, bsc, xn.dtype))
        return _tail_cublas(xn.contiguous(), cond.contiguous(), *ww, residual.contiguous() if residual is not None else None)
    ww = _capture.param_forms(f"ct_sm100a_{xn.dtype}", (wa, wb, ws, wsc, bsc), lambda: _weights(wa, wb, ws, wsc, bsc, f32))
    res = residual.contiguous() if residual is not None else None
    return _tail(xn.contiguous(), cond.contiguous(), *ww, res)


__all__ = ["ct_inference_sm100", "head_inference", "head_supported", "supported"]
