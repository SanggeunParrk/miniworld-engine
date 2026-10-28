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
