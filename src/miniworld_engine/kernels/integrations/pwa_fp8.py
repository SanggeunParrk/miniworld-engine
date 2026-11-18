"""EXPERIMENTAL, opt-in (off by default): the PWA key contraction ``o_h = w_h v_h`` with e4m3 operands on H100 (cuBLASLt).

Nothing here runs unless the caller asks for it (``ENABLED`` / ``MINIWORLD_PWA_FP8=1``); the bf16 route is untouched.

Route (MiniWorld MSA widths: d_msa 64, 8 heads x 32, any d_pair ``pwa_train`` serves; inference, one sample)::

    pair_fwd3 (the default pair kernel, w bf16) -> w_quant (e4m3 of 256 w) -> ln_vt8 (y bf16 = ln_vg's bytes, v as e4m3 K-major)
      -> cuBLASLt e4m3 batched GEMM (o bf16, head-major) -> pwa_gateout (the default gate / out pass, unchanged)

Wide route (``pwa_wide`` widths, 8 heads x 8 / 16: AF3 / OpenFold3 / Protenix-v1, Protenix-v2, OpenDDE, ESMFold2; ``forward_wide``)::

    pair_fwd3 (pwa_wide's pair kernel) -> w_quant -> ln_vt8 built at the width (v as e4m3 K-major, no y; N a multiple of 16)
      -> the same cuBLASLt e4m3 contraction (M = S*C per head) -> gate_out_w (pwa_wide's default inference gate / out pass, unchanged)

BIG build (``pwa_wide`` 256 x 8 x 32, d_pair 512: IntelliFold2; same ``forward_wide``)::

    pair_fwd3 -> w_quant -> ln_vt8 at d_msa 256 (v as e4m3 K-major, y kept when N % 64 == 0; one block per SM, two TMA stages)
      -> the same contraction -> gate_w (reads y, or normalises m again) -> out_w (the default BIG inference gate / out passes, unchanged)

Layout. Hopper's e4m3 tensor cores take K-major operands only. The contraction is over the key index j, so
  w  [H, N_i, N_j]        e4m3, row-major (K = j contiguous: the softmax output as it is)
  vT [H, S*C, N_j]        e4m3, j contiguous (v transposed: ``ln_vt8`` writes it so)
  o  [H, N_i, S*C]        bf16, head-major (the layout ``pwa_gateout`` reads)
In cuBLASLt's column-major terms ``o^T (S*C x N_i) = vT^T . w^T`` with op(A) = T on vT and op(B) = N on w.

Scales. ``w`` are probabilities (<= 1): a fixed power-of-two scale (w * 256 <= 256 < 448: no calibration, no overflow). ``v`` takes the
engine's static power-of-two scale (``vscale_from_amax`` of the TriAttn FP8 route, ``HEADROOM`` x the absmax of the first, eager call).
Both are device scalars handed to cuBLASLt as the A / B scale pointers: the bf16 output is the true o (no rescale pass), and
``rescale_all`` updates them in place (captured graphs follow). Guard: ``ln_vt8`` records max |v / scale| (``overflow_peak``); the cast
saturates above 448 -- ``take_overflow()`` / ``rescale_all()`` / redo, as the TriAttn / TriMul FP8 routes.
"""

import ctypes
import os
import weakref
from pathlib import Path

import torch

from miniworld_engine.fp8 import env as _fp8_env
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_fp8 as _F

E4M3_MAX = _F.E4M3_MAX
W_SCALE = 2.0**-8  # dequant factor of w (w stored as w * 256)

#: opt-in switch (environment read at import, or set the attribute)
ENABLED = _fp8_env("MINIWORLD_PWA_FP8") == "1"
#: "sync" (default): a saturating call raises the scale (x RESCALE) and redoes itself (one host sync per call); "lazy": no host sync
#: (required with CUDA graphs) -- the caller runs ``take_overflow()`` / ``rescale_all()`` / redo once per forward, outside the graph
GUARD = os.environ.get("MINIWORLD_PWA_FP8_GUARD", "sync")
HEADROOM = float(os.environ.get("MINIWORLD_PWA_FP8_HEADROOM", "16"))
#: shortest N served (below it the bf16 route runs; measured on 2.5.12: N256 +8 %, N384 -5 % at S1024); 0 = no gate
MIN_N = int(os.environ.get("MINIWORLD_PWA_FP8_MIN_N", "384"))
#: smallest MSA depth served below N 512 (N384: S256 +8 %, S512 0 %, S1024 -5 %; from N512 every depth measured gains); 0 = no gate
MIN_S = int(os.environ.get("MINIWORLD_PWA_FP8_MIN_S", "1024"))
#: cuBLASLt fast accumulation (no fp32 promotion of the e4m3 accumulator): module 1-3 % faster at N >= 768
FAST_ACCUM = os.environ.get("MINIWORLD_PWA_FP8_FASTACC") == "1"
RESCALE = 4.0
LNVT8_CFG = (2, 4)  # ln_vt8 (blocks per SM, TMA stages): measured best of {2, 3} x {2, 3, 4}

REQUESTED = 8
ALGO_INDEX: dict = {}  # (n, S*C) -> index of the heuristic candidate (default 0)
_EXT: dict = {}
_VS: dict = {}  # key (the value weight's address) -> static scale fp32 [1] (dequant factor of v)
_VINV: dict = {}  # key -> 1 / scale fp32 [1]
_AMAX: dict = {}  # device -> fp32 [1]: max |v / scale| since the last take_overflow()
_WS: dict = {}  # device -> W_SCALE fp32 [1]
_WORK: dict = {}  # device -> cuBLASLt workspace


def _ext(d_msa: int = 64, c: int = 32):
    """``pwa_vt8.cu`` built at (d_msa, channels per head): the MiniWorld widths 64 / 32 (the default build), or a ``pwa_wide`` width."""
    key = "vt8" if (d_msa, c) == (64, 32) else f"vt8_{d_msa}_{c}"
    if key not in _EXT:
        from miniworld_engine.kernels._nvcc import load_extension as load

        width = [] if key == "vt8" else [f"-DVT_D={d_msa}", f"-DVT_C={c}"]
        _EXT[key] = load(
            name="miniworld_pwa_" + key,
            sources=[str(Path(__file__).with_name("csrc") / "pwa_vt8.cu")],
            extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "--use_fast_math", "-Xcompiler=-fvisibility=hidden", *width],
            extra_cflags=["-O3", "-fvisibility=hidden"],
        )
    return _EXT[key]


# ---------------------------------------------------------------------------------------------------------------- cuBLASLt e4m3
class _Plan:
    def __init__(self, dev: int, h: int, n: int, ms: int, fast: bool):
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
            _F._set(lib.cublasLtMatmulDescSetAttribute, desc, 25, 1, ctypes.c_int8)  # FAST_ACCUM
        self.desc = desc

        def layout(dt, rows, cols, ld, stride):
            lay = ctypes.c_void_p()
            lib.cublasLtMatrixLayoutCreate(ctypes.byref(lay), dt, ctypes.c_uint64(rows), ctypes.c_uint64(cols), ctypes.c_int64(ld))
            _F._set(lib.cublasLtMatrixLayoutSetAttribute, lay, 5, h, ctypes.c_int32)  # batch count
            _F._set(lib.cublasLtMatrixLayoutSetAttribute, lay, 6, stride, ctypes.c_int64)  # batch stride
            return lay

        self.la = layout(28, n, ms, n, ms * n)  # vT e4m3: column-major (K x M) == row-major [S*C, N]
        self.lb = layout(28, n, n, n, n * n)  # w e4m3: column-major (K x N_i) == row-major [N_i, N_j]
        self.lc = layout(14, ms, n, ms, ms * n)  # o bf16: column-major (M x N_i) == row-major [N_i, S*C]
        if dev not in _WORK:
            _WORK[dev] = torch.empty(32 << 20, dtype=torch.uint8, device=f"cuda:{dev}")
        self.ws = _WORK[dev]
        pref = ctypes.c_void_p()
        lib.cublasLtMatmulPreferenceCreate(ctypes.byref(pref))
        _F._set(lib.cublasLtMatmulPreferenceSetAttribute, pref, 1, self.ws.numel(), ctypes.c_uint64)
        res = (_F._Heur * REQUESTED)()
        cnt = ctypes.c_int(0)
        r = lib.cublasLtMatmulAlgoGetHeuristic(
            self.handle, desc, self.la, self.lb, self.lc, self.lc, pref, REQUESTED, res, ctypes.byref(cnt)
        )
        if r or cnt.value == 0:
            raise RuntimeError(f"no cuBLASLt e4m3 algorithm for H={h} N={n} S*C={ms}")
        self.algos = [(ctypes.c_uint64 * 8)(*res[i].algo) for i in range(cnt.value)]
        self.algo = self.algos[min(ALGO_INDEX.get((n, ms), 0), cnt.value - 1)]
        self.scales = (0, 0)  # (vscale ptr, wscale ptr) last set on the descriptor


def contract(wq: torch.Tensor, vt: torch.Tensor, out: torch.Tensor, vscale: torch.Tensor, wscale: torch.Tensor) -> torch.Tensor:
    """out[h, i, :] = sum_j (wscale wq[h, i, j]) (vscale vt[h, :, j]); e4m3 operands, bf16 out, fp32 accumulation (promoted unless
    FAST_ACCUM). ``vscale`` / ``wscale``: fp32 device scalars (dequant factors), read by the kernel at run time."""
    lib, _, plans = _F._lib()
    dev = wq.device.index
    h, n, ms = wq.shape[0], wq.shape[-1], vt.shape[1]
    key = ("pwa", dev, h, n, ms, FAST_ACCUM)
    if key not in plans:
        plans[key] = _Plan(dev, h, n, ms, FAST_ACCUM)
    p = plans[key]
    ptrs = (vscale.data_ptr(), wscale.data_ptr())
    if p.scales != ptrs:
        _F._set(lib.cublasLtMatmulDescSetAttribute, p.desc, 17, ptrs[0], ctypes.c_void_p)  # A (vT) scale pointer
        _F._set(lib.cublasLtMatmulDescSetAttribute, p.desc, 18, ptrs[1], ctypes.c_void_p)  # B (w) scale pointer
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
        p.lc,
        p.algo,
        ctypes.c_void_p(p.ws.data_ptr()),
        ctypes.c_size_t(p.ws.numel()),
        ctypes.c_void_p(torch.cuda.current_stream(wq.device).cuda_stream),
    )
    if r:
        raise RuntimeError(f"cublasLtMatmul e4m3 (PWA) failed: {r}")
    return out


# ------------------------------------------------------------------------------------------------------------------ the route
def serves(s: int, n: int) -> bool:
    """True when the opt-in route takes this call (MiniWorld MSA widths; the caller checks the width and the pair kernel's range)."""
    return ENABLED and n % 128 == 0 and s % 2 == 0 and n >= MIN_N and (n >= 512 or s >= MIN_S)


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
    """max |v / scale| since the last take_overflow() (does not clear; one host sync)."""
    return max((float(a.item()) for a in _AMAX.values()), default=0.0)


def rescale_all(factor: float = RESCALE) -> None:
    """Multiply every calibrated v scale by ``factor`` in place (a power of two keeps the scales exact; captured graphs follow)."""
    for key, t in _VS.items():
        t.mul_(factor)
        _VINV[key].copy_(1.0 / t)


def _scales(k, m, key_tensor, lnm_w, lnm_b, wv16, eps_m, cfg, store_y):
    """(amax, w scale, v-scale key) of this call; the first call of a module calibrates its static v scale (eager)."""
    dev = m.device
    amax = _AMAX.get(dev)
    if amax is None:
        amax = _AMAX[dev] = torch.zeros(1, device=dev)
    ws = _WS.get(dev)
    if ws is None:
        ws = _WS[dev] = torch.full((1,), W_SCALE, device=dev)
    key = key_tensor.data_ptr()
    if key not in _VS:  # calibration (eager): scale 1, read max |v|, then the static power of two with HEADROOM
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("PWA FP8: the first call of a module must be eager (it calibrates the v scale)")
        from miniworld_engine.kernels.triangle_attention.cuda.h100_sm90 import vscale_from_amax

        probe = torch.zeros(1, device=dev)
        k.ln_vt8(m, lnm_w, lnm_b, wv16, eps_m, torch.ones(1, device=dev), probe, *cfg, store_y)
        _VS[key] = vscale_from_amax(probe, HEADROOM).float().contiguous()
        _VINV[key] = (1.0 / _VS[key]).contiguous()
        weakref.finalize(key_tensor, _forget, key)
    return amax, ws, key


def forward(m, key_tensor, lnm_w, lnm_b, wv16, wg16, wo16, eps_m, w16):
    """The msa output (residual included) [S, N, 64] of one inference call from the pair kernel's w [H, N, N] bf16; ``key_tensor``
    (the module's value weight) keys the static v scale."""
    k = _ext()
    amax, ws, key = _scales(k, m, key_tensor, lnm_w, lnm_b, wv16, eps_m, LNVT8_CFG, True)
    wq = k.w_quant(w16)
    from miniworld_engine.kernels.integrations import pwa_train as P

    gate = P._k(True)["gateout"]
    out = m
    for _attempt in range(4):
        vt, y = k.ln_vt8(m, lnm_w, lnm_b, wv16, eps_m, _VINV[key], amax, *LNVT8_CFG)
        o = torch.empty((8, m.shape[1], m.shape[0] * 32), dtype=torch.bfloat16, device=m.device)
        contract(wq, vt, o, _VS[key], ws)
        del vt
        out = gate.pwa_gateout(o, y, wg16, wo16, m, 1, 4, None, 1.0)
        if GUARD != "sync" or not take_overflow():
            break
        rescale_all()
    return out


# ------------------------------------------------------------------------------------------------- the wide widths (pwa_wide)
#: the pwa_wide widths served, (d_msa, channels per head) with 8 heads: AF3 / OpenFold3 / Protenix-v1 (64, 8), Protenix-v2 / OpenDDE (128, 8),
#: ESMFold2 (128, 16), and the BIG build (256, 32) of IntelliFold2 (d_pair 512; its gate / out pass is ``gate_w`` + ``out_w``).
WIDE = ((64, 8), (128, 8), (128, 16), (256, 32))
#: BIG build: ``ln_vt8`` also stores y (the bytes of ``ln_vg_w``'s y) and ``gate_w`` reads it, like the default route's ``BIG_INFER_Y``; needs
#: N % 64 == 0, other lengths (N a multiple of 16) take the LayerNorm-again ``gate_w`` mode.  Module FP8 / default at S1024: with y 0.88 (N384) /
#: 0.87 (N768), LayerNorm-again 0.98 / 0.97 (N400, a length without y: 1.03).  "0": always the LayerNorm-again mode
BIG_Y = os.environ.get("MINIWORLD_PWA_FP8_BIG_Y", "1") != "0"
#: the BIG build is served from N % BIG_N_MULT == 0 (64: the y-keeping route; 16 admits the other multiples of 16, which lose at N400)
BIG_N_MULT = int(os.environ.get("MINIWORLD_PWA_FP8_BIG_N_MULT", "64"))
#: smallest S * N served on the BIG build (every measured shape from S64 N256 = 2^14 gains: 0.95-0.97, larger 0.83-0.91); 0 = no gate
BIG_MIN_SN = int(os.environ.get("MINIWORLD_PWA_FP8_BIG_MIN_SN", str(1 << 14)))
#: shortest N served on the wide widths (AF3 64 x 8, module FP8 / default: N128 1.04 and N192 1.01 at S1024; N256 0.97); 0 = no gate
WIDE_MIN_N = int(os.environ.get("MINIWORLD_PWA_FP8_WIDE_MIN_N", "256"))
#: smallest S * N served on the wide widths (S512 N256 1.03, S256 N384 1.01, S128 N768 1.02; S1024 N256, S512 N384, S256 N768 0.96-0.97);
#: 0 = no gate
WIDE_MIN_SN = int(os.environ.get("MINIWORLD_PWA_FP8_WIDE_MIN_SN", str(3 << 16)))


def wide_cfg(d_msa: int, c: int) -> tuple[int, int]:
    """ln_vt8 (blocks per SM, TMA stages) at a wide width: measured best of {1, 2, 3, 4, 6, 8} x {2, 3, 4} (S1024 N768; 64 x 8 also S4096 N1536)."""
    if d_msa == 256:  # BIG: Wv (128 KiB) + two 32 KiB tiles + the 16 KiB v staging = 209 KiB of shared memory: one block per SM, two stages
        return (1, 2)
    return {64: (3, 3), 128: (3, 2) if c == 8 else (2, 3)}[d_msa]


def serves_wide(d_msa: int, c: int, s: int, n: int) -> bool:
    """True when the opt-in route takes this pwa_wide inference call (N padded to a multiple of 16 by the caller)."""
    if not (ENABLED and (d_msa, c) in WIDE):
        return False
    if d_msa == 256:  # BIG: its own gates (no N floor: S1024 N128 0.89)
        return n % BIG_N_MULT == 0 and n % 16 == 0 and s * n >= BIG_MIN_SN
    return n % 16 == 0 and n >= WIDE_MIN_N and s * n >= WIDE_MIN_SN


def forward_wide(m, key_tensor, lnm_w, lnm_b, wv16, wg16, wo16, eps_m, w16, kv):
    """pwa_wide's inference output (residual included) [S, N, d_msa] from its pair kernel's w [H, N, N] bf16: ln_vt8 (no y) -> e4m3
    contraction -> ``kv.gate_out_w`` in its LayerNorm mode (the default inference gate pass, unchanged). ``kv``: the pwa_wide build."""
    d, c = m.shape[2], wv16.shape[0] // 8
    k, cfg = _ext(d, c), wide_cfg(d, c)
    big = d == 256  # the BIG build: gate_w + out_w instead of gate_out_w
    keep_y = big and BIG_Y and m.shape[1] % 64 == 0
    amax, ws, key = _scales(k, m, key_tensor, lnm_w, lnm_b, wv16, eps_m, cfg, keep_y)
    wq = k.w_quant(w16)
    out = m
    for _attempt in range(4):
        vt, y = k.ln_vt8(m, lnm_w, lnm_b, wv16, eps_m, _VINV[key], amax, *cfg, keep_y)
        o = torch.empty((8, m.shape[1], m.shape[0] * c), dtype=torch.bfloat16, device=m.device)
        contract(wq, vt, o, _VS[key], ws)
        del vt
        if big:
            a, _ = kv.gate_w(o, y, wg16, False) if keep_y else kv.gate_w(o, m, wg16, False, lnm_w, lnm_b, eps_m)
            out = kv.out_w(a, m, wo16, None, 1.0)
        else:
            out = kv.gate_out_w(o, m, m, wg16, wo16, None, 1.0, lnm_w, lnm_b, eps_m)
        if GUARD != "sync" or not take_overflow():
            break
        rescale_all()
    return out
