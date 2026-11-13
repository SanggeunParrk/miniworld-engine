"""Batched bf16 GEMM through cuBLAS with a zero workspace.

``torch.bmm`` hands cuBLAS the per-handle workspace; the heuristic then picks stream-K style kernels that need a semaphore reset
(a ``Memset`` graph node, ~0.7 us plus a launch gap, before every contraction). With no workspace cuBLAS picks the same
tiles as static-schedule kernels at the same speed. ATen re-installs its workspace each time it fetches the handle, so the
setting stays local to these calls.
"""

import ctypes

import torch

_CUDA_R_16BF = 14
_COMPUTE_32F = 68
_GEMM_DEFAULT_TENSOR_OP = 99
_lib = None


def _cublas():
    global _lib
    if _lib is None:
        lib = ctypes.CDLL("libcublas.so.12", mode=ctypes.RTLD_GLOBAL)
        lib.cublasSetWorkspace_v2.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        lib.cublasGemmStridedBatchedEx.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_longlong,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_longlong,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_longlong,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        _lib = lib
    return _lib


def _layout(t):
    """(transposed?, leading dimension) of a [B, R, C] bf16 operand whose rows or columns are unit stride."""
    if t.stride(-1) == 1 and t.stride(-2) >= t.shape[-1]:
        return False, t.stride(-2)
    if t.stride(-2) == 1 and t.stride(-1) >= t.shape[-2]:
        return True, t.stride(-1)
    raise ValueError("bmm operand needs a unit-stride row or column")


def supports(a, b, out):
    try:
        _layout(a), _layout(b)
    except ValueError:
        return False
    return (
        a.dtype == b.dtype == out.dtype == torch.bfloat16
        and a.ndim == b.ndim == out.ndim == 3
        and out.is_contiguous()
        and a.shape[0] == b.shape[0] == out.shape[0]
        and a.stride(0) > 0
        and b.stride(0) > 0
    )


def bmm_out(a, b, out):
    """out[B, M, N] = a[B, M, K] @ b[B, K, N] (bf16, fp32 accumulation), no cuBLAS workspace."""
    lib = _cublas()
    batch, m, k = a.shape
    n = b.shape[2]
    a_t, lda = _layout(a)
    b_t, ldb = _layout(b)
    handle = torch.cuda.current_blas_handle()
    one = ctypes.c_float(1.0)
    zero = ctypes.c_float(0.0)
    lib.cublasSetWorkspace_v2(handle, None, 0)
    # row-major C = A B  <=>  column-major C^T (n x m) = B^T (n x k) A^T (k x m)
    status = lib.cublasGemmStridedBatchedEx(
        handle,
        1 if b_t else 0,
        1 if a_t else 0,
        n,
        m,
        k,
        ctypes.byref(one),
        b.data_ptr(),
        _CUDA_R_16BF,
        ldb,
        b.stride(0),
        a.data_ptr(),
        _CUDA_R_16BF,
        lda,
        a.stride(0),
        ctypes.byref(zero),
        out.data_ptr(),
        _CUDA_R_16BF,
        n,
        out.stride(0),
        batch,
        _COMPUTE_32F,
        _GEMM_DEFAULT_TENSOR_OP,
    )
    if status:
        raise RuntimeError("cublasGemmStridedBatchedEx failed: %d" % status)
    return out


_enabled = None


def bmm(a, b, out):
    """``torch.bmm(a, b, out=out)``, without the cuBLAS workspace when the operands are plain bf16."""
    global _enabled
    if _enabled is None:
        try:
            _cublas()
            _enabled = True
        except OSError:
            _enabled = False
    if _enabled and a.is_cuda and supports(a, b, out):
        return bmm_out(a, b, out)
    return torch.bmm(a, b, out=out)


# ---------------------------------------------------------------------------------------------------------------------------- cuBLASLt
#: The TriMul contraction through a measured cuBLASLt configuration. cuBLAS's default heuristic (torch.bmm, and ``bmm`` above) is
#: 5-18 % slower than another configuration of the same kernel family (algorithm 66, one k-pass: no split-K, the same fp32
#: accumulation per output, outputs bitwise equal) at several TriMul lengths on H100; which one wins, and whether the heuristic offers
#: it, changes from length to length, so only lengths measured twice in separate processes are listed (2026-10-31, record
#: docs/records/trimul_h100_d256_20261031.md). Key: (L, a transposed (incoming), planes per launch) -> (tile id, custom option,
#: cluster shape id) as cublasLtMatmulAlgoConfigGetAttribute reports them for the heuristic candidate. Anything else: ``bmm``.
LT_CONFIGS: dict = {
    # D256 one direction, one sample (256 planes per launch); ratios vs the default pick, two processes each
    (256, False, 256): (20, 1, 3),
    (256, True, 256): (20, 1, 3),  # 0.94 / 0.94
    (512, False, 256): (20, 1, 3),
    (512, True, 256): (20, 1, 3),  # 0.91 / 0.92
    (640, False, 256): (29, 3, 5),
    (640, True, 256): (29, 3, 5),  # 0.84 / 0.86
    (768, False, 256): (20, 1, 3),
    (768, True, 256): (20, 1, 3),  # 0.88 / 0.82
    (896, True, 256): (29, 3, 5),  # 0.88
    (1024, False, 256): (20, 1, 3),
    (1024, True, 256): (20, 1, 3),  # 0.87 / 0.87
    (1280, False, 256): (29, 1, 3),  # 0.89
    (1536, False, 256): (201, 2, 5),  # 0.94
    (2560, False, 256): (29, 1, 3),  # 0.94
}

_LT_ALGO = 66
_lt_state = None


class _HeurResult(ctypes.Structure):
    _fields_ = [
        ("algo", ctypes.c_uint64 * 8),
        ("ws", ctypes.c_size_t),
        ("state", ctypes.c_int),
        ("waves", ctypes.c_float),
        ("reserved", ctypes.c_int * 4),
    ]


def _lt():
    """(library, {device: handle}, {key: plan}) or None when libcublasLt cannot be loaded."""
    global _lt_state
    if _lt_state is None:
        try:
            lib = ctypes.CDLL("libcublasLt.so.12", mode=ctypes.RTLD_GLOBAL)
            _lt_state = (lib, {}, {})
        except OSError:
            _lt_state = False
    return _lt_state or None


def _lt_set(fn, obj, attr, value, ctype):
    v = ctype(value)
    if fn(obj, attr, ctypes.byref(v), ctypes.sizeof(v)):
        raise RuntimeError("cuBLASLt attribute %d" % attr)


def _lt_plan(lib, handle, n, batch, ta, tb, want):
    """Descriptor, layout and the heuristic candidate with configuration ``want`` (no workspace), or None."""
    desc, lay, pref = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()
    if lib.cublasLtMatmulDescCreate(ctypes.byref(desc), _COMPUTE_32F, 0):
        return None
    _lt_set(lib.cublasLtMatmulDescSetAttribute, desc, 3, int(ta), ctypes.c_int32)
    _lt_set(lib.cublasLtMatmulDescSetAttribute, desc, 4, int(tb), ctypes.c_int32)
    if lib.cublasLtMatrixLayoutCreate(ctypes.byref(lay), _CUDA_R_16BF, ctypes.c_uint64(n), ctypes.c_uint64(n), ctypes.c_int64(n)):
        return None
    _lt_set(lib.cublasLtMatrixLayoutSetAttribute, lay, 5, batch, ctypes.c_int32)
    _lt_set(lib.cublasLtMatrixLayoutSetAttribute, lay, 6, n * n, ctypes.c_int64)
    if lib.cublasLtMatmulPreferenceCreate(ctypes.byref(pref)):
        return None
    _lt_set(lib.cublasLtMatmulPreferenceSetAttribute, pref, 1, 0, ctypes.c_uint64)
    results = (_HeurResult * 32)()
    count = ctypes.c_int(0)
    if lib.cublasLtMatmulAlgoGetHeuristic(handle, desc, lay, lay, lay, lay, pref, 32, results, ctypes.byref(count)):
        return None
    for i in range(count.value):
        r = results[i]
        if r.state or r.ws:
            continue
        got = []
        for attr, ctype in ((0, ctypes.c_int32), (1, ctypes.c_uint32), (5, ctypes.c_uint32), (8, ctypes.c_uint16), (2, ctypes.c_uint32)):
            v, written = ctype(0), ctypes.c_size_t(0)
            lib.cublasLtMatmulAlgoConfigGetAttribute(r.algo, attr, ctypes.byref(v), ctypes.sizeof(v), ctypes.byref(written))
            got.append(v.value)
        if got == [_LT_ALGO, *want, 1]:
            algo = (ctypes.c_uint64 * 8)(*r.algo)
            return desc, lay, algo
    return None


def _square_planes(t):
    """(transposed?) for a [B, n, n] bf16 view of contiguous planes (plain or with its last two axes swapped), else None."""
    n = t.shape[-1]
    if t.ndim != 3 or t.shape[1] != n or t.dtype != torch.bfloat16 or t.stride(0) != n * n:
        return None
    if t.stride(1) == n and t.stride(2) == 1:
        return False
    if t.stride(1) == 1 and t.stride(2) == n:
        return True
    return None


def contract(a, b, out, *, bounded=False, mask=None, samples=1):
    """``bmm(a, b, out)`` for the TriMul contraction (a, b: [P, n, n] views of contiguous bf16 planes, one of them may be transposed;
    out contiguous [P, n, n]), through the measured cuBLASLt configuration for (n, a transposed, P) when ``LT_CONFIGS`` has one and
    the heuristic offers it; the same products and fp32 accumulation either way.

    ``bounded`` (inference): on a cell of ``_bounded.CELLS`` the bounded contraction instead, which stops at the live extents of
    ``mask`` (the token mask [samples * n] or [samples, n]; plane l is sample l % ``samples``; None: the full product)."""
    if bounded:
        from . import _bounded

        ta = _square_planes(a)
        cfg = _bounded.cell(a.shape[0], a.shape[-1], int(bool(ta)), a.device) if ta is not None else None
        if cfg is not None:
            _bounded.contract(a, b, out, _bounded.live_bounds(mask, samples), samples, cfg)
            return out
    ta, tb = _square_planes(a), _square_planes(b)
    key = (a.shape[-1], ta, a.shape[0])
    want = LT_CONFIGS.get(key)
    state = _lt() if want is not None and ta is not None and tb is not None and out.is_contiguous() and a.is_cuda else None
    if state is None:
        return bmm(a, b, out)
    lib, handles, plans = state
    dev = a.device.index if a.device.index is not None else torch.cuda.current_device()
    handle = handles.get(dev)
    if handle is None:
        handle = ctypes.c_void_p()
        if lib.cublasLtCreate(ctypes.byref(handle)):
            return bmm(a, b, out)
        handles[dev] = handle
    pkey = (dev, *key, tb)
    if pkey not in plans:
        # row-major out = a @ b <=> column-major out^T = b^T a^T: cuBLAS's first operand is b's memory (op T when b is the transposed
        # view), the second a's memory (op T when a is)
        plans[pkey] = _lt_plan(lib, handle, a.shape[-1], a.shape[0], tb, ta, want)
    plan = plans[pkey]
    if plan is None:
        return bmm(a, b, out)
    desc, lay, algo = plan
    one, zero = ctypes.c_float(1.0), ctypes.c_float(0.0)
    status = lib.cublasLtMatmul(
        handle,
        desc,
        ctypes.byref(one),
        ctypes.c_void_p(b.data_ptr()),
        lay,
        ctypes.c_void_p(a.data_ptr()),
        lay,
        ctypes.byref(zero),
        ctypes.c_void_p(out.data_ptr()),
        lay,
        ctypes.c_void_p(out.data_ptr()),
        lay,
        algo,
        None,
        ctypes.c_size_t(0),
        ctypes.c_void_p(torch.cuda.current_stream(a.device).cuda_stream),
    )
    if status:
        raise RuntimeError("cublasLtMatmul failed: %d" % status)
    return out
