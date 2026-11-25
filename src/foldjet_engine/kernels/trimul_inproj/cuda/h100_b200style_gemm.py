"""Layout-specific cuBLASLt selection for the B200-style backward candidate."""

import ctypes

import torch

from . import h100_wide_training as W

WORKSPACE = W.WORKSPACE


def _selection(d, h, n, role):
    return None


def mm(a, b, workspace, d, h, n, role, fp32=False):
    choice = _selection(d, h, n, role)
    if choice is None:
        return torch.mm(a, b, out_dtype=torch.float32) if fp32 else torch.mm(a, b)
    out = torch.empty(
        (a.shape[0], b.shape[1]),
        device=a.device,
        dtype=torch.float32 if fp32 else a.dtype,
    )
    aa, bb, oo = (v.unsqueeze(0) for v in (a, b, out))
    problem = W._lt_problem(*(W._layout_key(v) for v in (aa, bb, oo)))
    if choice not in problem.indices:
        raise ValueError(f"unsupported cuBLASLt heuristic {choice} for {role}")
    algorithm = problem.heuristics[choice].algo
    one, zero = ctypes.c_float(1), ctypes.c_float(0)
    ad, bd, dd = problem.layouts
    W._lt_check(
        W._lt_library().cublasLtMatmul(
            W._lt_handle(),
            problem.desc,
            ctypes.byref(one),
            a.data_ptr(),
            ad,
            b.data_ptr(),
            bd,
            ctypes.byref(zero),
            out.data_ptr(),
            dd,
            out.data_ptr(),
            dd,
            ctypes.byref(algorithm),
            workspace.data_ptr(),
            workspace.numel(),
            torch.cuda.current_stream().cuda_stream,
        ),
        "B200-style " + role,
    )
    return out
