"""Hand-CUDA sm_100a (B200) Transition for every call the tuned ``fused_sm100a`` / ``fused_wide_sm100a`` builds do not take.

``y = x + Ws (silu(Wa LN(x)) * (Wb LN(x)))`` with

* any channel width ``D`` that is a multiple of 64 up to 1024, any hidden width that is a multiple of 128 (the expansion factor is read
  off the weights),
* any row count (TMA zero-fills the tail of the last 128-row tile on the way in and clips it on the way out: no padding copy),
* bf16 activations on the bf16 tensor cores (``kind::f16``), or fp32 activations on the TF32 tensor cores (``kind::tf32``; only where
  torch allows TF32, the switch cuBLAS follows -- strict fp32 keeps the PyTorch reference); fp32 accumulation either way.

The kernels (``sm100/generic/``): ``tg_gemm.cu`` is one persistent warp-specialised tcgen05 GEMM (TMA producer warp, MMA warp,
four epilogue warps; accumulators double-buffered in tensor memory) built per epilogue, and ``tg_rows.cu`` the row passes.

    forward    tg_ln_fwd (xn, mean, rstd) -> GEMM SWIGLU (h = silu(xn Wa^T) (xn Wb^T)) -> GEMM RESID (y = h Ws^T + x)
    backward   GEMM GATE (dh = dy Ws and a, b recomputed from xn in one tile: h, [dA | dB]) -> cuBLAS dWs = dy^T h, [dWa; dWb] = [dA | dB]^T xn
               (fp32 out) -> GEMM PLAIN (d_xn = [dA | dB] [Wa; Wb], fp32) -> tg_ln_bwd (dx with the residual, dgamma / dbeta partials)
               -> tg_colsum

At a few row tiles (a single track [B, L, D]: the GEMM grids would leave most SMs idle, each CTA bound by its operand intake) the output
tile widths come from ``_plan``, the backward's d_xn GEMM splits K into fp32 slabs that the LayerNorm backward adds in order (tg_ln_bwds),
the squeeze runs on cuBLAS ``addmm`` (it splits K inside one launch), and a training forward keeps a, b (SWIGLU_AB) so the backward's gate
reads them (GATE_AB, one operand phase) instead of recomputing them.

Elsewhere the forward keeps nothing but xn and the statistics (a and b are recomputed by the backward's GATE tile, which already streams xn). The
parameters keep their dtype: the kernels get activation-dtype copies made outside autograd (``tg_prep``; fp32 on the TF32 grid), and the
parameters get fp32 gradients in their own dtype. Every reduction has one order: a replay is bit-identical.

``supported()`` is the whole gate. Every launch is an opaque op with a fake, so the module traces under ``torch.compile``.
"""

from __future__ import annotations

import functools

import torch

from miniworld_engine.kernels._device import num_sms

from ..._compile import opaque
from .fused_sm100a import _ext, _is_b200, build_cubins

MAX_D = 1024
#: Output-tile widths (columns of N per tile) each GEMM epilogue is built at, widest first: SWIGLU / CTGATE keep two accumulators, GATE
#: three, so their double-buffered accumulators fit the 512 tensor-memory columns. A bf16 output is staged in 64-column (128-B) groups, so
#: 32-column tiles exist only where the output is fp32. ``_pick_bn`` takes the widest that still gives every SM a tile (small M: a
#: single-track [1, L, D] activation is a few row tiles).
_BNS = {"swiglu": (128, 64), "resid": (128, 64), "plain": (128, 64, 32), "gate": (64,), "ctgate": (128, 64), "swiglu_ab": (128, 64),
        "gate_ab": (128, 64)}
_BNS_F32 = {"swiglu": (128, 64, 32), "resid": (128, 64, 32), "plain": (128, 64, 32), "gate": (64, 32), "ctgate": (128, 64, 32),
            "swiglu_ab": (128, 64, 32), "gate_ab": (128, 64, 32)}
_MODE = {"swiglu": 0, "resid": 1, "plain": 2, "gate": 3, "ctgate": 4, "swiglu_ab": 5, "gate_ab": 6}


def _tf32_allowed() -> bool:
    return bool(torch.backends.cuda.matmul.allow_tf32) or torch.get_float32_matmul_precision() != "highest"


@functools.lru_cache(maxsize=4)
def _load(f32: bool):
    """The cubins of one activation dtype, registered with the sm_100a extension (loaded on the current device: a cubin the driver
    rejects fails here, inside ``available()``)."""
    ext = _ext()
    tag = "f32" if f32 else "bf16"
    specs = [("rows", "generic/tg_rows.cu", (f"-DF32={int(f32)}",))]
    for name, mode in _MODE.items():
        for bn in (_BNS_F32 if f32 else _BNS)[name]:
            specs.append((f"{name}{bn}", "generic/tg_gemm.cu", (f"-DMODE={mode}", f"-DF32={int(f32)}", f"-DBN={bn}")))
    for name, path in build_cubins(f"generic_{tag}", tuple(specs)).items():
        ext.load_cubin(f"generic_{tag}_{name}", path)
    return ext


def supported(x: torch.Tensor, wa: torch.Tensor, ws: torch.Tensor) -> bool:
    """sm_100 (B200), bf16 (or fp32 with TF32 allowed), D a multiple of 64 up to 1024, hidden a multiple of 128, weights bf16 or fp32
    (cast to the activation dtype inside; the gradients come back in the weights' own dtype). Any row count."""
    from miniworld_engine import settings

    if not settings.current().transition_fused_sm100a:
        return False
    if x.device.type != "cuda" or x.dtype not in (torch.bfloat16, torch.float32) or x.numel() == 0:
        return False
    if any(w.dtype not in (torch.bfloat16, torch.float32) for w in (wa, ws)):
        return False
    if x.dtype is torch.float32 and not _tf32_allowed():
        return False
    if not _is_b200(x.device.index if x.device.index is not None else torch.cuda.current_device()):
        return False
    d = x.shape[-1]
    if d % 64 or d > MAX_D:
        return False
    hid = wa.shape[0]
    return hid % 128 == 0 and tuple(wa.shape) == (hid, d) and tuple(ws.shape) == (d, hid)


_BUILD_FAILED = False


def _is_fake(*tensors) -> bool:
    from torch._subclasses.fake_tensor import FakeTensor

    return any(isinstance(t, FakeTensor) for t in tensors)


def available(x: torch.Tensor, wa: torch.Tensor, ws: torch.Tensor) -> bool:
    """``supported()`` plus a successful (cached) build; a build failure raises (B200 has no Triton / silent PyTorch fallback)."""
    if _BUILD_FAILED or not supported(x, wa, ws):
        return False
    if torch.compiler.is_compiling() or _is_fake(x, wa, ws):
        return True
    try:
        _load(x.dtype is torch.float32)
    except Exception as exc:
        raise RuntimeError(f"generic sm100a Transition kernels failed to build or load on B200: {exc!r}") from exc
    return True


# ---------------------------------------------------------------------------------------------------------------- launches
def _ln_variant(d: int, f32: bool) -> tuple[str, int]:
    """(suffix of the tg_ln_fwd / tg_ln_bwd instantiation for width ``d``, rows per warp): 16-B vectors and as many lanes per row as
    divide the row (see tg_rows.cu)."""
    if f32:
        vec, tpr = (4, 16) if d == 64 else ((4, 32) if d % 128 == 0 else (2, 32))
    elif d == 64:
        vec, tpr = 8, 8
    elif d == 128:
        vec, tpr = 8, 16
    else:
        vec, tpr = (8 if d % 256 == 0 else 4 if d % 128 == 0 else 2), 32
    return f"v{vec}t{tpr}n{d // tpr}", 32 // tpr


def _pick_bn(widths: tuple[int, ...], m: int, n: int, nsm: int) -> int:
    """The widest tile with at least one tile per SM, else the narrowest."""
    for bn in widths:
        if -(-m // 128) * -(-n // bn) >= nsm:
            return bn
    return widths[-1]


#: Per-SM TMA intake the small-M plan below assumes (bytes per microsecond; ~52 B / clock), and the rate a split-K GEMM's fp32 partials
#: are written and read back at (bytes per microsecond, the whole GPU)
_INTAKE = 1.0e5
_PARTIAL_RATE = 4.0e6
#: A/B override of the forward squeeze at small M: None (plan), "split", "resid", "cublas"
SQUEEZE_ROUTE = None
#: A/B override of the backward d_xn GEMM: None (plan), "nosplit", "cublas"
DXN_ROUTE = None
#: K splits a PLAIN GEMM may take at small M
_SPLITS = (1, 2, 3, 4, 6, 8, 12, 16)


def _plan(name: str, m: int, n: int, ks: tuple[int, ...], f32: bool, nsm: int, split: bool) -> tuple[int, int]:
    """(output tile width, K splits) of one GEMM. ``ks``: the K of each operand phase (GATE: two). Where the widest tile already gives every
    SM a tile, the widest (no split). Below that (a single track: a few row tiles) each CTA's time is its operand intake -- the A rows and
    the B columns of every K-block it walks, per tile it takes -- so the plan minimises waves x per-tile intake over the tile widths and,
    with ``split`` (PLAIN), over K splits (whose fp32 partials cost a write and a read: the consumer adds them). Measured (fp32 TF32,
    [1, 768, 384] n4, kernels alone): GATE 32 -> 64 columns 20.1 -> 13.5 us (one wave instead of two); d_xn split 4 ways 9.9 us against
    cuBLAS 10.8 (the K-serial GEMM: 19.6 at D512 n2). A model, not a tuning: the ratios hold, the absolute times are ~2x it (latency)."""
    widths = (_BNS_F32 if f32 else _BNS)[name]
    mt = -(-m // 128)
    if mt * -(-n // widths[0]) >= nsm:
        return widths[0], 1
    esz, bke = (4, 32) if f32 else (2, 64)
    nb = {"swiglu": (2,), "resid": (1,), "plain": (1,), "gate": (1, 2), "ctgate": (1, 1), "swiglu_ab": (2,), "gate_ab": (1,)}[name]
    best = None
    for bn in widths:
        tiles = mt * -(-n // bn)
        for s in (_SPLITS if split else (1,)):
            if any((k // bke) % s or k // bke < 2 * s for k in ks):
                continue
            per_tile = sum((128 + b * bn) * (k // s) * esz for b, k in zip(nb, ks, strict=True))
            if name == "gate_ab":   # the saved a, b rows of the tile
                per_tile += 2 * 128 * bn * esz
            cost = -(-tiles * s // nsm) * per_tile / _INTAKE
            if s > 1:
                cost += 2 * s * (mt * 128) * n * 4 / _PARTIAL_RATE
            if best is None or cost < best[0] - 1e-9:
                best = (cost, bn, s)
    return best[1], best[2]


def _ab(m: int, hid: int, nsm: int) -> bool:
    """Whether a training call keeps a, b from the forward (SWIGLU_AB -> GATE_AB) instead of recomputing them in the backward's GATE: at
    a few row tiles (the gate's 64-column output tiles under one wave), where the GATE grid is bound by its per-CTA operand intake and the
    recompute doubles it, while the stores are small. Measured (module training step, CUDA graph): fp32 [1, 154, 768] n2 104.0 -> 96.6 us,
    bf16 [1, 500, 1024] n2 114.8 -> 109.0, [1, 300, 768] n4 110.7 -> 107.3; [1, 768, 384] n4 fp32 even (88.4 / 89.1)."""
    return AB_FORCE if AB_FORCE is not None else -(-m // 128) * -(-hid // 64) < nsm


#: A/B override of ``_ab`` (None: automatic)
AB_FORCE = None


class _K:
    """The kernels of one activation dtype on the current device."""

    def __init__(self, x: torch.Tensor):
        self.f32 = x.dtype is torch.float32
        self.tag = "f32" if self.f32 else "bf16"
        self.ext = _load(self.f32)
        self.nsm = num_sms(x.device)
        self.bke = 32 if self.f32 else 64     # elements in a 128-byte K-block row

    def rows(self, kernel, grid, args, block=256):
        self.ext.launch_kernel(f"generic_{self.tag}_rows", kernel, grid, block, 1, list(args))

    def gemm(self, name, m, n, a0, b0, o0, *, a1=None, b1=None, b2=None, o1=None, res=None, bias=None, k0=0, k1=0, ldres=0, hoff=0,
             bn=None, ks=1, mstride=0):
        """C[m, n] by the ``name`` epilogue. a* [m, K] and b* [n, K] (K-major), o* the outputs; every map is built here. ``bn``: the
        output tile width (None: ``_pick_bn``); ``ks`` > 1: split K, slice s storing its fp32 partial at rows s ``mstride`` + m of o0."""
        if bn is None:
            bn = _pick_bn((_BNS_F32 if self.f32 else _BNS)[name], m, n, self.nsm)
        ocg = 32 if (self.f32 or name == "plain") else 64
        t = self.ext.tmap
        ma0 = t(a0, k0, m, self.bke, 128)
        ma1 = t(a1, k1, m, self.bke, 128) if a1 is not None else ma0
        mb0 = t(b0, b0.shape[1], b0.shape[0], self.bke, bn)
        mb1 = t(b1, b1.shape[1], b1.shape[0], self.bke, bn) if b1 is not None else mb0
        mb2 = t(b2, b2.shape[1], b2.shape[0], self.bke, bn) if b2 is not None else mb0
        mo0 = t(o0, o0.shape[1], o0.shape[0], ocg, 32)
        mo1 = t(o1, o1.shape[1], o1.shape[0], ocg, 32) if o1 is not None else mo0
        tiles = -(-m // 128) * -(-n // bn) * ks
        self.ext.launch_kernel(
            f"generic_{self.tag}_{name}{bn}", "tg_gemm_sm100", max(1, min(self.nsm, tiles)), 320, 1,
            [ma0, ma1, mb0, mb1, mb2, mo0, mo1, res if res is not None else o0, bias if bias is not None else o0,
             m, n, k0, k1, ldres, hoff, ks, mstride],
        )


def _prep_fake(wa, wb, ws, f32, bwd):
    """Output structure of ``_prep``: (wa, wb, ws) and, with ``bwd``, Ws^T and [Wa; Wb]^T (1 x 1 placeholders without)."""
    h, d = wa.shape
    dtype = torch.float32 if f32 else torch.bfloat16
    e = lambda *s: wa.new_empty(s, dtype=dtype)  # noqa: E731
    return (e(h, d), e(h, d), e(d, h)) + ((e(h, d), e(d, 2 * h)) if bwd else (e(1, 1), e(1, 1)))


@opaque(fake=_prep_fake, name="transition_generic_prep_sm100a")
def _prep(wa: torch.Tensor, wb: torch.Tensor, ws: torch.Tensor, f32: bool, bwd: bool
          ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """(wa, wb, ws) in fp32 on the TF32 grid (``f32``) or bf16 and, with ``bwd``, Ws^T [H, D] and [Wa; Wb]^T [D, 2H]; one launch."""
    if _is_fake(wa, wb, ws):
        return _prep_fake(wa, wb, ws, f32, bwd)
    dtype = torch.float32 if f32 else torch.bfloat16
    h, d = wa.shape
    src = wa.dtype
    wa, wb, ws = (w.to(src).contiguous() for w in (wa, wb, ws))
    e = lambda *s: torch.empty(s, dtype=dtype, device=wa.device)  # noqa: E731
    oa, ob, os_ = e(h, d), e(h, d), e(d, h)
    ost, oabt = (e(h, d), e(d, 2 * h)) if bwd else (e(1, 1), e(1, 1))
    k = _K(oa)
    k.rows("tg_prep", 2 * (h // 32) * (d // 32) if bwd else min(-(-h * d // 256), k.nsm * 8),
           (wa, wb, ws, oa, ob, os_, ost, oabt, h, d, int(src is torch.float32), int(bwd)))
    return oa, ob, os_, ost, oabt


def _fwd_launch_fake(x, gamma, beta, wa, wb, ws, eps, save):
    """Output structure of ``_fwd_launch``: (out, xn, mean, rstd, ab); mean / rstd are 1-element placeholders unless ``save``, ab the
    forward's [a | b] [M, 2H] where the backward reads it (``save`` and ``_ab``), else a 1 x 1 placeholder."""
    m, d = x.shape
    f32 = dict(dtype=torch.float32, device=x.device)
    rows = m if save else 1
    ab = x.new_empty((m, 2 * wa.shape[0])) if save and _ab(m, wa.shape[0], num_sms(x.device)) else x.new_empty((1, 1))
    return torch.empty_like(x), torch.empty_like(x), torch.empty((rows,), **f32), torch.empty((rows,), **f32), ab


@opaque(fake=_fwd_launch_fake, name="transition_generic_fwd_sm100a")
def _fwd_launch(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, wa: torch.Tensor, wb: torch.Tensor, ws: torch.Tensor, eps: float,
                save: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """x [M, D]; the weights in x's dtype (fp32: TF32 grid). Returns (out, xn, mean, rstd, ab), out = transition(x) + x; see the fake."""
    if _is_fake(x, wa):
        return _fwd_launch_fake(x, gamma, beta, wa, wb, ws, eps, save)
    k = _K(x)
    m, d = x.shape
    hid = wa.shape[0]
    f32 = dict(dtype=torch.float32, device=x.device)
    xn = torch.empty_like(x)
    mean = torch.empty((m if save else 1,), **f32)
    rstd = torch.empty_like(mean)
    sfx, rpw = _ln_variant(d, k.f32)
    k.rows(f"tg_ln_fwd_{sfx}", max(1, min(-(-m // (8 * rpw)), k.nsm * 8)), (x, gamma, beta, xn, mean, rstd, m, d, float(eps), int(save)))
    h = torch.empty((m, hid), dtype=x.dtype, device=x.device)
    if save and _ab(m, hid, k.nsm):   # a few row tiles: keep a, b for the backward's GATE_AB
        ab = torch.empty((m, 2 * hid), dtype=x.dtype, device=x.device)
        k.gemm("swiglu_ab", m, hid, xn, wa, h, b1=wb, o1=ab, k0=d, hoff=hid,
               bn=_plan("swiglu_ab", m, hid, (d,), k.f32, k.nsm, False)[0])
    else:
        ab = x.new_empty((1, 1))
        k.gemm("swiglu", m, hid, xn, wa, h, b1=wb, k0=d, bn=_plan("swiglu", m, hid, (d,), k.f32, k.nsm, False)[0])
    # the squeeze + residual: at a few row tiles (where the plan would split K) cuBLAS addmm, which splits K inside one launch --
    # measured faster than the split-K PLAIN GEMM + tg_splitsum pair (an extra launch); else the RESID epilogue
    _, ks = _plan("plain", m, d, (hid,), k.f32, k.nsm, True)
    route = SQUEEZE_ROUTE or ("cublas" if ks > 1 else "resid")
    if route == "cublas":
        out = torch.addmm(x, h, ws.t())                 # y = h Ws^T + x, the residual as beta * C
    elif route == "split":
        bn, ks = _plan("plain", m, d, (hid,), k.f32, k.nsm, True)
        mp = -(-m // 128) * 128
        part = torch.empty((ks * mp, d), dtype=torch.float32, device=x.device)
        out = torch.empty_like(x)
        k.gemm("plain", m, d, h, ws, part, k0=hid, bn=bn, ks=max(ks, 2), mstride=mp)
        k.rows("tg_splitsum", max(1, min(-(-m * d // 1024), k.nsm * 4)), (part, x, out, m, d, max(ks, 2), mp * d))
    else:
        out = torch.empty_like(x)
        k.gemm("resid", m, d, h, ws, out, res=x, k0=hid, ldres=d, bn=_plan("resid", m, d, (hid,), k.f32, k.nsm, False)[0])
    return out, xn, mean, rstd, ab


def _mm_f32(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """A weight-gradient GEMM with an fp32 result (cuBLAS; bf16 operands accumulate in fp32 and are not rounded through bf16)."""
    if a.dtype is torch.float32:
        return torch.mm(a, b)
    try:
        return torch.mm(a, b, out_dtype=torch.float32)
    except TypeError:
        return torch.mm(a, b).float()


def _ln_bwd_blocks(m: int, rows_per_warp: int, nsm: int) -> int:
    """LayerNorm-backward blocks: one row group per warp (8 warps) up to four blocks per SM; a few row tiles then spread over ~M / 8
    blocks instead of 64-row blocks each walking eight rows in sequence (with the slot-parallel block reduction, [768, 384] fp32 alone:
    15.6 -> 3.9 us; [90 000, 256] unchanged)."""
    return max(1, min(-(-m // (8 * rows_per_warp)), nsm * 4))


def _bwd_launch_fake(dy, x, xn, mean, rstd, gamma, wa, wb, wst, wabt, ab):
    """Output structure of ``_bwd_launch``: (dx, dgamma, dbeta, [dWa; dWb], dWs), the weight gradients fp32."""
    h, d = wa.shape
    f32 = dict(dtype=torch.float32, device=x.device)
    return (torch.empty_like(x), torch.empty((d,), **f32), torch.empty((d,), **f32), torch.empty((2 * h, d), **f32),
            torch.empty((d, h), **f32))


@opaque(fake=_bwd_launch_fake, name="transition_generic_bwd_sm100a")
def _bwd_launch(dy: torch.Tensor, x: torch.Tensor, xn: torch.Tensor, mean: torch.Tensor, rstd: torch.Tensor, gamma: torch.Tensor,
                wa: torch.Tensor, wb: torch.Tensor, wst: torch.Tensor, wabt: torch.Tensor, ab: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """(dx, dgamma, dbeta, [dWa; dWb] [2H, D], dWs [D, H]); ``dx`` carries the residual branch, the weight gradients are fp32. ``ab``: the
    forward's [a | b] where it kept them (GATE_AB), else a placeholder (GATE recomputes a, b from xn)."""
    if _is_fake(dy, x):
        return _bwd_launch_fake(dy, x, xn, mean, rstd, gamma, wa, wb, wst, wabt, ab)
    k = _K(x)
    m, d = x.shape
    hid = wa.shape[0]
    dyg = dy
    if k.f32:   # dy as the GATE GEMM's A operand on the TF32 grid (the MMA would truncate it); the residual branch keeps dy itself
        dyg = torch.empty_like(dy)
        k.rows("tg_round_tf32", max(1, min(-(-dy.numel() // 1024), k.nsm * 8)), (dy, dyg, dy.numel() // 4))
    h = torch.empty((m, hid), dtype=x.dtype, device=x.device)
    dab = torch.empty((m, 2 * hid), dtype=x.dtype, device=x.device)
    if ab.shape[0] == m and ab.shape[1] == 2 * hid:
        k.gemm("gate_ab", m, hid, dyg, wst, h, o1=dab, res=ab, ldres=2 * hid, k0=d, hoff=hid,
               bn=_plan("gate_ab", m, hid, (d,), k.f32, k.nsm, False)[0])
    else:
        k.gemm("gate", m, hid, dyg, wst, h, a1=xn, b1=wa, b2=wb, o1=dab, k0=d, k1=d, hoff=hid,
               bn=_plan("gate", m, hid, (d, d), k.f32, k.nsm, False)[0])
    dws = _mm_f32(dy.t(), h)            # [D, H]
    dwab = _mm_f32(dab.t(), xn)         # [2H, D]: dWa then dWb
    del h
    bn, ks = _plan("plain", m, d, (2 * hid,), k.f32, k.nsm, True)
    if DXN_ROUTE == "cublas":
        dxn, ks, mp = _mm_f32(dab, wabt.t()), 1, 0
    else:
        if DXN_ROUTE == "nosplit":
            ks = 1
        mp = -(-m // 128) * 128 if ks > 1 else 0   # split K: fp32 partial slabs of mp rows, added in order by the LayerNorm backward
        dxn = torch.empty((ks * mp if ks > 1 else m, d), dtype=torch.float32, device=x.device)
        k.gemm("plain", m, d, dab, wabt, dxn, k0=2 * hid, bn=bn, ks=ks, mstride=mp)
    del dab
    sfx, rpw = _ln_variant(d, k.f32)
    nb = _ln_bwd_blocks(m, rpw, k.nsm)
    part = torch.empty((nb, 2 * d), dtype=torch.float32, device=x.device)
    dx = torch.empty_like(x)
    k.rows(f"tg_ln_bwd{'s' if ks > 1 else ''}_{sfx}", nb, (dxn, x, dy, mean, rstd, gamma, dx, part, m, d, ks, mp * d))
    dgb = torch.empty((2 * d,), dtype=torch.float32, device=x.device)
    k.rows("tg_colsum", -(-2 * d // 32), (part, dgb, nb, 2 * d))
    return dx, dgb[:d], dgb[d:].clone(), dwab, dws


class _GenericTransitionSM100A(torch.autograd.Function):
    """``y = transition(x) + x``; the forward saves x, xn and the statistics. The parameters keep their dtype (bf16, fp32 masters): the
    kernels get copies in the activation dtype made here, outside autograd, and the parameters get fp32 gradients in their own dtype."""

    @staticmethod
    def forward(ctx, x, gamma, beta, wa, wb, ws, eps):
        shape = x.shape
        ctx.param_dtypes = (gamma.dtype, beta.dtype, wa.dtype, wb.dtype, ws.dtype)
        flat = x.reshape(-1, shape[-1]).contiguous()
        gf, bf = gamma.float().contiguous(), beta.float().contiguous()
        wa_, wb_, ws_, wst, wabt = _prep(wa, wb, ws, x.dtype is torch.float32, True)
        out, xn, mean, rstd, ab = _fwd_launch(flat, gf, bf, wa_, wb_, ws_, float(eps), True)
        ctx.save_for_backward(flat, xn, mean, rstd, gf, wa_, wb_, wst, wabt, ab)
        ctx.shape = shape
        return out.reshape(shape)

    @staticmethod
    def backward(ctx, dy):
        flat, xn, mean, rstd, gf, wa_, wb_, wst, wabt, ab = ctx.saved_tensors
        gdt, bdt, adt, bwdt, sdt = ctx.param_dtypes
        dx, dgam, dbeta, dwab, dws = _bwd_launch(dy.reshape(-1, dy.shape[-1]).to(flat.dtype).contiguous(), flat, xn, mean, rstd, gf,
                                                 wa_, wb_, wst, wabt, ab)
        hid = wa_.shape[0]
        return (dx.reshape(ctx.shape), dgam.to(gdt), dbeta.to(bdt), dwab[:hid].to(adt), dwab[hid:].to(bwdt), dws.to(sdt), None)


def transition_generic_sm100a(x, gamma, beta, wa, wb, ws, eps):
    """Module-facing entry, same signature as ``fused_sm100a.transition_fused_sm100a``. Call ``available()`` first: this raises rather
    than falling back. Inference (grad mode off, or nothing requiring grad) runs the forward without the saves, with the weight copies
    made once per parameter version (``_capture.param_forms``)."""
    if not (torch.is_grad_enabled() and any(t.requires_grad for t in (x, gamma, beta, wa, wb, ws))):
        from miniworld_engine.kernels import _capture

        shape = x.shape
        g, b, ww = _capture.param_forms(
            f"transition_generic_sm100a_{x.dtype}",
            (gamma, beta, wa, wb, ws),
            lambda: (gamma.float().contiguous(), beta.float().contiguous(), _prep(wa, wb, ws, x.dtype is torch.float32, False)[:3]),
        )
        out, *_ = _fwd_launch(x.reshape(-1, shape[-1]).contiguous(), g, b, *ww, float(eps), False)
        return out.reshape(shape)
    return _GenericTransitionSM100A.apply(x, gamma, beta, wa, wb, ws, eps)
