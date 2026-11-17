"""EXPERIMENTAL, opt-in (off by default): FP8 GEMMs in the PWA TRAINING step (forward + backward) on H100 (cuBLASLt).

Nothing here runs unless the caller asks for it (``MINIWORLD_PWA_FP8_TRAIN`` / ``MINIWORLD_FP8_TRAIN=1`` / :data:`PLACES`); with it off the
training route is the 2.5.14 one, bit for bit. Not part of the inference ``safe`` preset (``miniworld_engine.fp8``).

Scope: ``pwa_train`` (d_msa 64, 8 heads x 32, any d_pair it serves), sm_90a, batch 1, N a multiple of 128 up to 1024, S even. The route runs
the native (``-DMW_PWA_CUDA_AUX``) kernels at every such shape (the bf16 default runs them at the six qualified shapes only).

The three key contractions of the step, each ``2 H N^2 S C`` FLOPs, run as cuBLASLt FP8 batched GEMMs (fp32 accumulation, bf16 out)::

    fwd  o[h]  [N_i, S*C] = w[h]  [N_i, N_j] . vT[h] [S*C, N_j]^T      w e4m3 (fixed 2^8), v e4m3
    dv   dv[h] [N_j, S*C] = wT[h] [N_j, N_i] . doT[h][S*C, N_i]^T      w e4m3, do e5m2 (or e4m3)
    dw   dw[h] [N_i, N_j] = do[h] [N_i, S*C] . v[h]  [N_j, S*C]^T      do e5m2 (or e4m3), v e4m3

Hopper's FP8 MMA takes K-major operands only, so each operand exists in the layout its GEMM contracts over: v as vT (fwd) and natural (dw),
do as natural (dw) and doT (dv), w as w (fwd) and wT (dv). ``IMPL``:

* ``fused`` (default): the producers write them -- ``ln_vt8`` built with ``-DVT_NAT=1`` (``csrc/pwa_vt8.cu``) writes y, vT e4m3 and v e4m3
  (natural) instead of ``ln_vg``'s bf16 v; ``pwa_glue3_f8`` (``csrc/pwa_glue3_f8.cu``, ``pwa_glue3`` with an FP8 epilogue) writes do and
  doT in FP8 instead of the bf16 do; ``w_quant_t`` writes wT e4m3 in one pass (w e4m3: ``pwa_fp8``'s ``w_quant``).
* ``proto``: the default producers (bf16) and PyTorch quantize / transpose passes (same rounding and scales; for accuracy work).

Scales (per tensor, power of two, dequant factors as device scalars given to cuBLASLt as the A / B scale pointers): w fixed 2^-8 (w <= 1);
v and do delayed: ``scale = 2^ceil(log2(MARGIN * max(amax history) / FMAX))`` over the last :data:`HISTORY` calls of that site (module), the
producer records this call's amax, the next call's scale takes it (no host sync, CUDA-graph safe). The first call of a site seeds the history
from the tensor itself (eager). A value past FMAX saturates (satfinite) and counts in :func:`overflow_count`; the history makes the next
call's scale cover it.
"""

from __future__ import annotations

import ctypes
import os
import weakref
from pathlib import Path
from typing import cast

import torch

from miniworld_engine.kernels.trimul_inproj.cuda import _h100_fp8 as _F


def _places(v: str | None) -> frozenset[str]:
    v = (v or "").strip().lower()
    if v in ("", "0", "off", "none"):
        return frozenset()
    if v in ("1", "all", "on"):
        return frozenset({"fwd", "dv", "dw"})
    out = frozenset(p.strip() for p in v.split(",") if p.strip())
    bad = out - {"fwd", "dv", "dw"}
    if bad:
        raise ValueError(f"MINIWORLD_PWA_FP8_TRAIN: unknown placement(s) {sorted(bad)} (use fwd, dv, dw, or 1 for all)")
    return out


#: FP8 placements in the training step ({"fwd", "dv", "dw"}); empty = off (the default route)
PLACES: frozenset[str] = _places(os.environ.get("MINIWORLD_PWA_FP8_TRAIN", "all" if os.environ.get("MINIWORLD_FP8_TRAIN") == "1" else ""))
#: incoming-gradient format: "e5m2" (default) or "e4m3"
DO_FMT = os.environ.get("MINIWORLD_PWA_FP8_TRAIN_DO", "e5m2")
#: "fused" (producers write the FP8 operands) or "proto" (PyTorch quantize passes)
IMPL = os.environ.get("MINIWORLD_PWA_FP8_TRAIN_IMPL", "fused")
HISTORY = int(os.environ.get("MINIWORLD_PWA_FP8_TRAIN_HISTORY", "16"))
MARGIN = float(os.environ.get("MINIWORLD_PWA_FP8_TRAIN_MARGIN", "2"))
FAST_ACCUM = os.environ.get("MINIWORLD_PWA_FP8_TRAIN_FASTACC") == "1"

E4M3_MAX, E5M2_MAX = 448.0, 57344.0
W_SCALE = 2.0**-8
_DT = {"e4m3": 28, "e5m2": 29}  # cudaDataType
_TORCH = {"e4m3": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2}
_FMAX = {"e4m3": E4M3_MAX, "e5m2": E5M2_MAX}


#: measurement only: route the call through this path with no FP8 placement (the native bf16 kernels at a shape the default serves with
#: the plain builds), to separate the native-kernel part of a gain from the FP8 part
NATIVE_ONLY = False


def active() -> bool:
    return bool(PLACES)


def serves(msa: torch.Tensor) -> bool:
    """True when the FP8 training route takes this grad-enabled pwa_train call (8 x 32 widths checked by the caller)."""
    if not (PLACES or NATIVE_ONLY) or not msa.is_cuda or msa.ndim != 4 or msa.shape[0] != 1 or msa.shape[3] != 64:
        return False
    s, n = msa.shape[1], msa.shape[2]
    return torch.cuda.get_device_capability(msa.device) == (9, 0) and n % 128 == 0 and 0 < n <= 1024 and s % 2 == 0 and s > 0


# ---------------------------------------------------------------------------------------------------------------- cuBLASLt FP8
class _Plan:
    """out[b] (row-major [M, N], bf16) = (sa A[b]) (sb B[b])^T with A [M, K], B [N, K] row-major FP8 (both K-major)."""

    def __init__(self, dev: int, bt: int, m: int, n: int, k: int, ta: str, tb: str, fast: bool):
        lib, handles, _ = _F._lib()
        if dev not in handles:
            hd = ctypes.c_void_p()
            if lib.cublasLtCreate(ctypes.byref(hd)):
                raise RuntimeError("cublasLtCreate")
            handles[dev] = hd
        self.handle = handles[dev]
        desc = ctypes.c_void_p()
        lib.cublasLtMatmulDescCreate.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        lib.cublasLtMatmulDescCreate(ctypes.byref(desc), 68, 0)  # CUBLAS_COMPUTE_32F, fp32 scale type
        _F._set(lib.cublasLtMatmulDescSetAttribute, desc, 3, 1, ctypes.c_int32)  # op(lt A = B) = T
        _F._set(lib.cublasLtMatmulDescSetAttribute, desc, 4, 0, ctypes.c_int32)  # op(lt B = A) = N
        if fast:
            _F._set(lib.cublasLtMatmulDescSetAttribute, desc, 25, 1, ctypes.c_int8)
        self.desc = desc

        def layout(dt, rows, cols, ld, stride):
            lay = ctypes.c_void_p()
            lib.cublasLtMatrixLayoutCreate(ctypes.byref(lay), dt, ctypes.c_uint64(rows), ctypes.c_uint64(cols), ctypes.c_int64(ld))
            _F._set(lib.cublasLtMatrixLayoutSetAttribute, lay, 5, bt, ctypes.c_int32)
            _F._set(lib.cublasLtMatrixLayoutSetAttribute, lay, 6, stride, ctypes.c_int64)
            return lay

        self.la = layout(_DT[tb], k, n, k, n * k)  # B: column-major K x N == row-major [N, K]
        self.lb = layout(_DT[ta], k, m, k, m * k)  # A: column-major K x M == row-major [M, K]
        self.lc = layout(14, n, m, n, m * n)  # out bf16: column-major N x M == row-major [M, N]
        from miniworld_engine.kernels.integrations import pwa_fp8 as _inf

        if dev not in _inf._WORK:
            _inf._WORK[dev] = torch.empty(32 << 20, dtype=torch.uint8, device=f"cuda:{dev}")
        self.ws = _inf._WORK[dev]
        pref = ctypes.c_void_p()
        lib.cublasLtMatmulPreferenceCreate(ctypes.byref(pref))
        _F._set(lib.cublasLtMatmulPreferenceSetAttribute, pref, 1, self.ws.numel(), ctypes.c_uint64)
        res = (_F._Heur * 8)()
        cnt = ctypes.c_int(0)
        r = lib.cublasLtMatmulAlgoGetHeuristic(self.handle, desc, self.la, self.lb, self.lc, self.lc, pref, 8, res, ctypes.byref(cnt))
        if r or cnt.value == 0:
            raise RuntimeError(f"no cuBLASLt FP8 algorithm for batch {bt} M {m} N {n} K {k} ({ta} x {tb})")
        self.algo = (ctypes.c_uint64 * 8)(*res[0].algo)
        self.scales = (0, 0)


def _fmt(t: torch.Tensor) -> str:
    return "e5m2" if t.dtype == torch.float8_e5m2 else "e4m3"


def mm8(a: torch.Tensor, b: torch.Tensor, sa: torch.Tensor, sb: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """out[i] = (sa a[i]) (sb b[i])^T in bf16; a [Bt, M, K], b [Bt, N, K] FP8 contiguous; sa / sb fp32 [1] device dequant factors."""
    lib, _, plans = _F._lib()
    bt, m, k = a.shape
    n = b.shape[1]
    if out is None:
        out = torch.empty((bt, m, n), dtype=torch.bfloat16, device=a.device)
    dev = a.device.index
    key = ("pwa8t", dev, bt, m, n, k, _fmt(a), _fmt(b), FAST_ACCUM)
    if key not in plans:
        plans[key] = _Plan(dev, bt, m, n, k, _fmt(a), _fmt(b), FAST_ACCUM)
    p = plans[key]
    ptrs = (sb.data_ptr(), sa.data_ptr())
    if p.scales != ptrs:
        _F._set(lib.cublasLtMatmulDescSetAttribute, p.desc, 17, ptrs[0], ctypes.c_void_p)  # lt A (= b) scale
        _F._set(lib.cublasLtMatmulDescSetAttribute, p.desc, 18, ptrs[1], ctypes.c_void_p)  # lt B (= a) scale
        p.scales = ptrs
    one, zero = ctypes.c_float(1.0), ctypes.c_float(0.0)
    r = lib.cublasLtMatmul(
        p.handle,
        p.desc,
        ctypes.byref(one),
        ctypes.c_void_p(b.data_ptr()),
        p.la,
        ctypes.c_void_p(a.data_ptr()),
        p.lb,
        ctypes.byref(zero),
        ctypes.c_void_p(out.data_ptr()),
        p.lc,
        ctypes.c_void_p(out.data_ptr()),
        p.lc,
        p.algo,
        ctypes.c_void_p(p.ws.data_ptr()),
        ctypes.c_size_t(p.ws.numel()),
        ctypes.c_void_p(torch.cuda.current_stream(a.device).cuda_stream),
    )
    if r:
        raise RuntimeError(f"cublasLtMatmul FP8 (PWA training) failed: {r}")
    return out


# ------------------------------------------------------------------------------------------------------------- delayed scales
_SITES: dict = {}  # (kind, key tensor address) -> _Site
_WS: dict = {}  # device -> W_SCALE fp32 [1]
_OVF: dict = {}  # device -> fp32 [1]: number of calls whose amax passed FMAX at its scale


class _Site:
    """Delayed power-of-two scale of one FP8 tensor stream (v or do of one module)."""

    def __init__(self, dev: torch.device, fmt: str):
        self.fmt = fmt
        self.hist = torch.zeros(HISTORY, device=dev)
        self.amax = torch.zeros(1, device=dev)  # max |x / scale| of the last call (scaled units), written by the producer
        self.scale = torch.ones(1, device=dev)  # dequant factor of the last call (persistent: CUDA-graph replays update it in place)
        self.ovf = _OVF.setdefault(dev, torch.zeros(1, device=dev))
        self.calls = 0

    def next_scale(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Push the last call's amax into the history; this call's (scale, 1 / scale) as fresh fp32 [1] tensors."""
        fmax = _FMAX[self.fmt]
        if fused():  # one launch
            out = _glue8().scale_update(self.hist, self.amax, self.scale, self.ovf, fmax, MARGIN, self.calls > 0)
            self.calls += 1
            return out[0:1], out[1:2]
        if self.calls:
            self.ovf.add_((self.amax > fmax).float())
            self.hist.copy_(torch.cat([(self.amax * self.scale), self.hist[:-1]]))
            self.amax.zero_()
        peak = self.hist.max().clamp_min(2.0**-60)
        scale = torch.exp2(torch.ceil(torch.log2(peak * (MARGIN / fmax)))).reshape(1)
        self.scale.copy_(scale)
        self.calls += 1
        return scale, 1.0 / scale  # fresh tensors: the backward of this call reads its own scale

    def seed(self, amax_true: torch.Tensor) -> None:
        """First call: the history is this tensor's own amax (eager)."""
        self.hist.fill_(0.0)
        self.hist[0] = amax_true.reshape(())


def _site(kind: str, key_tensor: torch.Tensor, fmt: str) -> _Site:
    key = (kind, key_tensor.data_ptr())
    s = _SITES.get(key)
    if s is None:
        s = _SITES[key] = _Site(key_tensor.device, fmt)
        weakref.finalize(key_tensor, _SITES.pop, key, None)
    return s


def _wscale(dev) -> torch.Tensor:
    t = _WS.get(dev)
    if t is None:
        t = _WS[dev] = torch.full((1,), W_SCALE, device=dev)
    return t


def overflow_count() -> float:
    """Calls (all sites) whose FP8 tensor saturated at its delayed scale since the last reset (one host sync)."""
    pending = sum(float((s.amax > _FMAX[s.fmt]).float().item()) for s in _SITES.values())  # the last call of each site (not yet pushed)
    return sum(float(t.item()) for t in _OVF.values()) + pending


def reset() -> None:
    """Forget every site's history (the next call of each site re-seeds) and the overflow count."""
    _SITES.clear()
    for t in _OVF.values():
        t.zero_()


# ---------------------------------------------------------------------------------------------------------------- extensions
_EXT: dict = {}


def _vt8_nat():
    if "vt8n" not in _EXT:
        from miniworld_engine.kernels._nvcc import load_extension as load

        _EXT["vt8n"] = load(
            name="miniworld_pwa_vt8_nat",
            sources=[str(Path(__file__).with_name("csrc") / "pwa_vt8.cu")],
            extra_cuda_cflags=[
                "-O3",
                "-gencode=arch=compute_90a,code=sm_90a",
                "--use_fast_math",
                "-Xcompiler=-fvisibility=hidden",
                "-DVT_NAT=1",
            ],
            extra_cflags=["-O3", "-fvisibility=hidden"],
        )
    return _EXT["vt8n"]


def _glue8():
    if "glue8" not in _EXT:
        from miniworld_engine.kernels._nvcc import load_extension as load

        _EXT["glue8"] = load(
            name="miniworld_pwa_glue3_f8",
            sources=[str(Path(__file__).with_name("csrc") / "pwa_glue3_f8.cu")],
            extra_cuda_cflags=[
                "-O3",
                "-gencode=arch=compute_90a,code=sm_90a",
                "--use_fast_math",
                "-Xcompiler=-fvisibility=hidden",
                "-DMW_PWA_CUDA_AUX=1",
            ],
            extra_cflags=["-O3", "-fvisibility=hidden"],
        )
    return _EXT["glue8"]


def _q(x: torch.Tensor, inv: torch.Tensor, fmt: str) -> torch.Tensor:
    """PyTorch quantize (proto): satfinite cast of x * inv."""
    fmax = _FMAX[fmt]
    return (x.float() * inv).clamp_(-fmax, fmax).to(_TORCH[fmt])


def _t8(x: torch.Tensor) -> torch.Tensor:
    """[B, R, C] FP8 -> [B, C, R] contiguous (byte transpose)."""
    return x.view(torch.uint8).transpose(1, 2).contiguous().view(x.dtype)


# ------------------------------------------------------------------------------------------------------------------ forward
ALL = frozenset({"fwd", "dv", "dw"})


def fused() -> bool:
    """The fused producers serve the full placement set only; a partial set (ablation) runs the PyTorch quantize passes."""
    return IMPL == "fused" and PLACES == ALL


def forward_values(
    k, m, lnm_w, lnm_b, wv, wv16, eps_m, w16
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """(o, v_bf, v_keep, y, v scale): o = the FP8 contraction (head-major bf16) when "fwd" is on, else None and the caller contracts
    the bf16 ``v_bf``; ``v_keep`` = v for the backward (e4m3 natural when "dw" is on, else bf16); the v scale fp32 [1] (None: no FP8 v)."""
    from miniworld_engine.kernels.integrations.pwa_train import _h100_float

    mw, mb = _h100_float(lnm_w.detach(), lnm_b.detach())
    if not PLACES & {"fwd", "dw"}:
        v, y = k["lnvg"].ln_vg(m, mw, mb, wv16, eps_m, 2, 3, 1)
        return None, v, v, y, None
    site = _site("v", wv, "e4m3")
    fused_ = fused()
    if site.calls == 0:  # seed the history from this call's |v| (eager)
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("PWA FP8 training: the first call of a module must be eager (it seeds the v scale)")
        if fused_:
            probe = torch.zeros(1, device=m.device)
            _vt8_nat().ln_vt8n(m, mw, mb, wv16, eps_m, torch.ones(1, device=m.device), probe, 2, 3)
            site.seed(probe)
        else:
            v, _ = k["lnvg"].ln_vg(m, mw, mb, wv16, eps_m, 2, 3, 1)
            site.seed(v.abs().amax().float())
    vs, vinv = site.next_scale()
    v_bf = None
    if fused_:
        vt8, v8, y = _vt8_nat().ln_vt8n(m, mw, mb, wv16, eps_m, vinv, site.amax, 2, 3)
    else:
        v_bf, y = k["lnvg"].ln_vg(m, mw, mb, wv16, eps_m, 2, 3, 1)
        site.amax.copy_((v_bf.abs().amax().float() * vinv).reshape(1))
        v8 = _q(v_bf, vinv, "e4m3")
        vt8 = _t8(v8) if "fwd" in PLACES else None
    o = None
    if "fwd" in PLACES:
        from miniworld_engine.kernels.integrations import pwa_fp8 as _inf

        assert vt8 is not None
        o = mm8(_inf._ext().w_quant(w16), vt8, _wscale(m.device), vs)
        del vt8
    return o, v_bf, cast(torch.Tensor, v8 if "dw" in PLACES else v_bf), y, vs


# ------------------------------------------------------------------------------------------------------------------ backward
def backward_glue(k, o, y, dres0, wg16, wot, dgv, dmask, dscale, wo, ohm):
    """(do bf16 or None, do8 natural or None, doT8 or None, do scale or None, dWo) from the saved o (``ohm``: head-major)."""
    if not PLACES & {"dv", "dw"}:
        d_o, _, dWo = k["glue3"].pwa_glue3(o, y, dres0, wg16, wot, dgv, 2, 1, dmask, dscale, ohm)
        return d_o, None, None, None, dWo
    site = _site("do", wo, DO_FMT)
    fused_ = fused() and ohm
    if site.calls == 0:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("PWA FP8 training: the first backward of a module must be eager (it seeds the do scale)")
        if fused_:
            probe = torch.zeros(1, device=o.device)
            _glue8().pwa_glue3_f8(o, y, dres0, wg16, wot, dgv, 2, 1, dmask, dscale, torch.ones(1, device=o.device), probe, DO_FMT == "e5m2")
            site.seed(probe)
        else:
            d_o, _, _ = k["glue3"].pwa_glue3(o, y, dres0, wg16, wot, dgv, 2, 1, dmask, dscale, ohm)
            site.seed(d_o.abs().amax().float())
    ds, dinv = site.next_scale()
    if fused_:
        do8, doT8, _, dWo = _glue8().pwa_glue3_f8(o, y, dres0, wg16, wot, dgv, 2, 1, dmask, dscale, dinv, site.amax, DO_FMT == "e5m2")
        return None, do8, doT8, ds, dWo
    d_o, _, dWo = k["glue3"].pwa_glue3(o, y, dres0, wg16, wot, dgv, 2, 1, dmask, dscale, ohm)
    site.amax.copy_((d_o.abs().amax().float() * dinv).reshape(1))
    do8 = _q(d_o, dinv, DO_FMT)
    doT8 = _t8(do8) if "dv" in PLACES else None
    return d_o, do8, doT8, ds, dWo


def backward_dv(w16, d_o, doT8, ds):
    """dv head-major [H, N, S*C] bf16."""
    if "dv" not in PLACES:
        return torch.bmm(w16.transpose(1, 2), d_o)
    from miniworld_engine.kernels.integrations import pwa_fp8 as _inf

    wT8 = _glue8().w_quant_t(w16) if fused() else _t8(_inf._ext().w_quant(w16))
    return mm8(wT8, doT8, _wscale(w16.device), ds)


def backward_dw(d_o, do8, ds, v, vs):
    """dw [H, N, N] bf16."""
    if "dw" not in PLACES:
        return torch.bmm(d_o, v.transpose(1, 2))
    return mm8(do8, v, ds, vs)
