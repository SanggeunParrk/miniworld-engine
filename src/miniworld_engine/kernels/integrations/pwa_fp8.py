"""EXPERIMENTAL, opt-in (off by default): the PWA key contraction ``o_h = w_h v_h`` with e4m3 operands on H100 (cuBLASLt).

Nothing here runs unless the caller asks for it; the bf16 route is untouched.

Layout. Hopper's e4m3 tensor cores take K-major operands only. The contraction is over the key index j, so
  w  [H, N_i, N_j]        e4m3, row-major (K = j contiguous: the softmax output as it is)
  vT [H, S*C, N_j]        e4m3, j contiguous (v transposed: the producer writes it so)
  o  [H, N_i, S*C]        bf16, head-major (the layout ``pwa_gateout`` reads)
In cuBLASLt's column-major terms ``o^T (S*C x N_i) = vT^T . w^T`` with op(A) = T on vT and op(B) = N on w.

Scales. ``w`` are probabilities (<= 1): a fixed power-of-two scale ``W_SCALE`` (2^8: w * 256 <= 256 < 448, no calibration, no
overflow). ``v`` takes the engine's static power-of-two scale (calibrated on the first eager call, headroom as TriMul / TriAttn).
Both scales are device scalars handed to cuBLASLt as the A / B scale pointers: the bf16 output is the true o (no rescale pass), and
``rescale_all`` updates them in place (captured graphs follow).
"""

import ctypes
import os

import torch

from miniworld_engine.kernels.trimul_inproj.cuda import _h100_fp8 as _F

E4M3_MAX = _F.E4M3_MAX
W_SCALE = 2.0**-8  # dequant factor of w (w stored as w * 256)
FAST_ACCUM = os.environ.get("MINIWORLD_PWA_FP8_FASTACC") == "1"
REQUESTED = 8
ALGO_INDEX: dict = {}


class _Plan:
    def __init__(self, dev: int, h: int, n: int, ms: int, fast: bool, out_dtype: int):
        lib, handles, _ = _F._lib()
        if dev not in handles:
            hd = ctypes.c_void_p()
            if lib.cublasLtCreate(ctypes.byref(hd)):
                raise RuntimeError("cublasLtCreate")
            handles[dev] = hd
        self.handle = handles[dev]
        desc = ctypes.c_void_p()
        lib.cublasLtMatmulDescCreate.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        lib.cublasLtMatmulDescCreate(ctypes.byref(desc), 68, 0)  # CUBLAS_COMPUTE_32F, scale type fp32
        _F._set(lib.cublasLtMatmulDescSetAttribute, desc, 3, 1, ctypes.c_int32)  # op(A) = T (vT: K-major)
        _F._set(lib.cublasLtMatmulDescSetAttribute, desc, 4, 0, ctypes.c_int32)  # op(B) = N (w: K-major)
        if fast:
            _F._set(lib.cublasLtMatmulDescSetAttribute, desc, 25, 1, ctypes.c_int8)
        self.desc = desc

        def layout(dt, rows, cols, ld, stride):
            lay = ctypes.c_void_p()
            lib.cublasLtMatrixLayoutCreate(ctypes.byref(lay), dt, ctypes.c_uint64(rows), ctypes.c_uint64(cols), ctypes.c_int64(ld))
            _F._set(lib.cublasLtMatrixLayoutSetAttribute, lay, 5, h, ctypes.c_int32)
            _F._set(lib.cublasLtMatrixLayoutSetAttribute, lay, 6, stride, ctypes.c_int64)
            return lay

        self.la = layout(28, n, ms, n, ms * n)  # vT: column-major (K x M) == row-major [S*C, N]
        self.lb = layout(28, n, n, n, n * n)  # w: column-major (K x N_i) == row-major [N_i, N_j]
        self.lc = layout(14, ms, n, ms, ms * n)  # C (bf16; beta = 0, never read): column-major (M x N_i) == row-major [N_i, S*C]
        self.ld = layout(out_dtype, ms, n, ms, ms * n)  # o: bf16, or e4m3 (o / scale_D) with out_dtype 28
        pref = ctypes.c_void_p()
        lib.cublasLtMatmulPreferenceCreate(ctypes.byref(pref))
        _F._set(lib.cublasLtMatmulPreferenceSetAttribute, pref, 1, 32 << 20, ctypes.c_uint64)
        self.ws = torch.empty(32 << 20, dtype=torch.uint8, device=f"cuda:{dev}")
        res = (_F._Heur * REQUESTED)()
        cnt = ctypes.c_int(0)
        r = lib.cublasLtMatmulAlgoGetHeuristic(self.handle, desc, self.la, self.lb, self.lc, self.ld, pref, REQUESTED, res, ctypes.byref(cnt))
        if r or cnt.value == 0:
            raise RuntimeError(f"no cuBLASLt e4m3 algorithm for H={h} N={n} S*C={ms}")
        self.algos = [(ctypes.c_uint64 * 8)(*res[i].algo) for i in range(cnt.value)]
        self.algo = self.algos[min(ALGO_INDEX.get((n, ms), 0), cnt.value - 1)]
        self.scales = None  # (vscale ptr, wscale ptr) last set on the descriptor


def contract(
    wq: torch.Tensor, vt: torch.Tensor, out: torch.Tensor, vscale: torch.Tensor, wscale: torch.Tensor, dinv: torch.Tensor | None = None
) -> torch.Tensor:
    """out[h, i, :] = sum_j (wscale wq[h, i, j]) (vscale vt[h, :, j]); e4m3 operands, fp32 accumulation (promoted unless FAST_ACCUM).
    ``out`` bf16, or e4m3 holding ``dinv * o`` (``dinv`` given: the D scale). ``vscale`` / ``wscale`` / ``dinv``: fp32 device scalars,
    read by the kernel at run time."""
    lib, _, plans = _F._lib()
    dev = wq.device.index
    h, n, ms = wq.shape[0], wq.shape[-1], vt.shape[1]
    o8 = out.dtype == torch.float8_e4m3fn
    assert o8 == (dinv is not None)
    key = ("pwa", dev, h, n, ms, FAST_ACCUM, o8)
    if key not in plans:
        plans[key] = _Plan(dev, h, n, ms, FAST_ACCUM, 28 if o8 else 14)
    p = plans[key]
    ptrs = (vscale.data_ptr(), wscale.data_ptr(), dinv.data_ptr() if dinv is not None else 0)
    if p.scales != ptrs:
        _F._set(lib.cublasLtMatmulDescSetAttribute, p.desc, 17, ptrs[0], ctypes.c_void_p)  # A (vT) scale
        _F._set(lib.cublasLtMatmulDescSetAttribute, p.desc, 18, ptrs[1], ctypes.c_void_p)  # B (w) scale
        if o8:
            _F._set(lib.cublasLtMatmulDescSetAttribute, p.desc, 20, ptrs[2], ctypes.c_void_p)  # D scale
        p.scales = ptrs
    one, zero = ctypes.c_float(1.0), ctypes.c_float(0.0)
    r = lib.cublasLtMatmul(
        p.handle,
        p.desc,
        ctypes.byref(one),
        ctypes.c_void_p(vt.data_ptr()),
        p.la,
        ctypes.c_void_p(wq.data_ptr()),
        p.lb,
        ctypes.byref(zero),
        ctypes.c_void_p(out.data_ptr()),
        p.lc,
        ctypes.c_void_p(out.data_ptr()),
        p.ld,
        p.algo,
        ctypes.c_void_p(p.ws.data_ptr()),
        ctypes.c_size_t(p.ws.numel()),
        ctypes.c_void_p(torch.cuda.current_stream(wq.device).cuda_stream),
    )
    if r:
        raise RuntimeError(f"cublasLtMatmul e4m3 (PWA) failed: {r}")
    return out


# ------------------------------------------------------------------------------------------------------------------ the route
#: opt-in switch (environment read at import, or set the attribute): the MiniWorld-width H100 inference forward (d_msa 64, 8 x 32, any
#: d_pair the pwa_train path serves) runs pair_fwd3 -> w_quant -> ln_vt8 -> e4m3 contraction -> pwa_gateout.  Off: the bf16 route, unchanged.
ENABLED = os.environ.get("MINIWORLD_PWA_FP8") == "1"
#: "sync" (default): a saturating call recalibrates (x RESCALE) and redoes itself (one host sync per call); "lazy": no host sync (CUDA
#: graphs) -- the caller runs ``take_overflow()`` / ``rescale_all()`` / redo once per forward, outside the graph; "off": no check.
GUARD = os.environ.get("MINIWORLD_PWA_FP8_GUARD", "sync")
HEADROOM = float(os.environ.get("MINIWORLD_PWA_FP8_HEADROOM", "16"))
#: shortest N served (below it the bf16 route runs); 0 = no gate
MIN_N = int(os.environ.get("MINIWORLD_PWA_FP8_MIN_N", "0"))
#: smallest MSA depth served
MIN_S = int(os.environ.get("MINIWORLD_PWA_FP8_MIN_S", "0"))
RESCALE = 4.0
#: EXPERIMENTAL: the contraction writes o as e4m3 too (cuBLASLt D scale = 1 / v scale) and the gate / out pass reads it (`pwa_gateout.cu` -DPWA_O8=1)
O8 = os.environ.get("MINIWORLD_PWA_FP8_O8") == "1"
LNVT8_CFG = (2, 4)  # ln_vt8 (blocks per SM, TMA stages)

_EXT: dict = {}
_VS: dict = {}  # key (the value weight's address) -> static scale fp32 [1] (dequant factor of v)
_VINV: dict = {}  # key -> 1 / scale fp32 [1]
_AMAX: dict = {}  # device -> fp32 [1]: max |v / scale| since the last take_overflow()
_WS: dict = {}  # device -> W_SCALE fp32 [1]


def _ext():
    if "vt8" not in _EXT:
        from pathlib import Path

        from miniworld_engine.kernels._nvcc import load_extension as load

        _EXT["vt8"] = load(
            name="miniworld_pwa_vt8",
            sources=[str(Path(__file__).with_name("csrc") / "pwa_vt8.cu")],
            extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "--use_fast_math", "-Xcompiler=-fvisibility=hidden"],
            extra_cflags=["-O3", "-fvisibility=hidden"],
        )
    return _EXT["vt8"]


def _ext_o8():
    if "o8" not in _EXT:
        from pathlib import Path

        from miniworld_engine.kernels._nvcc import load_extension as load

        _EXT["o8"] = load(
            name="miniworld_pwa_gateout8",
            sources=[str(Path(__file__).with_name("csrc") / "pwa_gateout.cu")],
            extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "--use_fast_math", "-Xcompiler=-fvisibility=hidden", "-DPWA_O8=1"],
            extra_cflags=["-O3", "-fvisibility=hidden"],
        )
    return _EXT["o8"]


def serves(s: int, n: int, d_pair: int) -> bool:
    return ENABLED and n % 64 == 0 and n >= MIN_N and s >= MIN_S


def _forget(key) -> None:
    _VS.pop(key, None)
    _VINV.pop(key, None)


def take_overflow() -> bool:
    """True when a call saturated an e4m3 v value since the last call (clears; one host sync per device)."""
    hit = False
    for a in _AMAX.values():
        hit |= float(a.item()) > E4M3_MAX
        a.zero_()
    return hit


def overflow_peak() -> float:
    """max |v / scale| since the last take_overflow() (does not clear)."""
    return max((float(a.item()) for a in _AMAX.values()), default=0.0)


def rescale_all(factor: float = RESCALE) -> None:
    """Raise every calibrated v scale by ``factor`` in place (captured graphs follow)."""
    for key, t in _VS.items():
        t.mul_(factor)
        _VINV[key].copy_(1.0 / t)


def forward(m, z, pm, key_tensor, lnz_w, lnm_w, lnm_b, wb16, wv16, wg16, wo16, eps_m, eps_z, pair_fn):
    """msa update of the MiniWorld-width inference call with the e4m3 contraction; ``pair_fn()`` makes w [H, N, N] bf16 (the default
    route's pair kernel). Returns out [S, N, 64]."""
    import weakref

    k = _ext()
    dev = m.device
    amax = _AMAX.get(dev)
    if amax is None:
        amax = _AMAX[dev] = torch.zeros(1, device=dev)
    ws = _WS.get(dev)
    if ws is None:
        ws = _WS[dev] = torch.full((1,), W_SCALE, device=dev)
    key = key_tensor.data_ptr()
    w16 = pair_fn()
    wq = k.w_quant(w16)
    if key not in _VS:  # calibration (eager): scale 1, read |v| max, then the static power of two with HEADROOM
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("PWA FP8: the first call of a module must be eager (calibration)")
        from miniworld_engine.kernels.triangle_attention.cuda.h100_sm90 import vscale_from_amax

        probe = torch.zeros(1, device=dev)
        k.ln_vt8(m, lnm_w, lnm_b, wv16, eps_m, torch.ones(1, device=dev), probe, *LNVT8_CFG)
        _VS[key] = vscale_from_amax(probe, HEADROOM).float().contiguous()
        _VINV[key] = (1.0 / _VS[key]).contiguous()
        weakref.finalize(key_tensor, _forget, key)
    from miniworld_engine.kernels.integrations import pwa_train as P

    for _attempt in range(4):
        vt, y = k.ln_vt8(m, lnm_w, lnm_b, wv16, eps_m, _VINV[key], amax, *LNVT8_CFG)
        if O8:  # o as e4m3 too: |o| <= max |v| (a convex combination), so v's scale serves o (no calibration of its own)
            o = torch.empty((8, m.shape[1], m.shape[0] * 32), dtype=torch.float8_e4m3fn, device=dev)
            contract(wq, vt, o, _VS[key], ws, _VINV[key])
            del vt
            out = _ext_o8().pwa_gateout8(o, _VS[key], y, wg16, wo16, m, 1, 4)
        else:
            o = torch.empty((8, m.shape[1], m.shape[0] * 32), dtype=torch.bfloat16, device=dev)
            contract(wq, vt, o, _VS[key], ws)
            del vt
            out = P._k(True)["gateout"].pwa_gateout(o, y, wg16, wo16, m, 1, 4, None, 1.0)
        if GUARD != "sync" or not take_overflow():
            return out
        rescale_all()
    return out
