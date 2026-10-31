"""d-aware dispatch for the post-AdaLN ConditionedTransition tail (inference + training).

Inference routes by d_hidden:
  - atom  (d_hidden <= 128): the fully-fused single-kernel b2b (``inference.py``) — the
    whole (M, ND) SwiGLU + out_acc[BM, D] live in registers, h never hits HBM. Fastest
    when K fits one BLOCK_K and the working set fits smem.
  - token (d_hidden >= 256): the composed two-kernel path (``composed.py``) — K-tiled
    expand+SwiGLU writes h:(M, ND) to HBM, then K-tiled squeeze+gate. Always compiles. fp32 takes
    cuBLAS (TF32) for the two GEMMs and the Triton SwiGLU / gate row passes instead
    (``composed.cond_transition_inference_cublas``): the Triton TF32 GEMMs lost to cuBLAS by 1.5x there.

Training (forward that saves for backward + backward) lives in ``training.py`` as a
torch.autograd.Function; the inference paths save nothing.
"""

from __future__ import annotations

import torch

ATOM_D_MAX = 128  # d_hidden <= this -> fused b2b; else composed
#: B200 fp32 (TF32) at the atom width: None (the fused b2b kernel, as on every card) or "cublas" (the token path's cuBLAS GEMMs + row passes).
#: Measured (B200, module inference in a CUDA graph, A = 5, d128 c128, us): L1024 / 8192 51.1 / 215.0 -> 42.8 / 128.8 (the b2b kernel's TF32
#: dots run far below this card's tensor cores).
B200_F32_ATOM_ROUTE = "cublas"


def cond_transition_inference_dispatch(x, cond, wa, wb, ws, wsc, bsc, length=None, residual=None):
    """Forward-only (inference) ConditionedTransition tail, routed by d_hidden.

    ``length`` is L -- the ATOM count A of the activation BEFORE the caller flattened it to
    (M, K); it is the autotune shape key's input and is passed straight through to whichever
    path d_hidden selects. None lets that path fall back to its row count M, which is A only
    when the batch axis is 1.

    ``residual`` (optional, (M, D) in ``x``'s dtype): the stream the update is added to, ``residual + y``, rounded as the unfused
    ``residual + y.to(dtype)`` -- folded into the squeeze+gate epilogue on the token path, one add on the atom path.
    """
    d_hidden = x.shape[1]
    if d_hidden <= ATOM_D_MAX and B200_F32_ATOM_ROUTE == "cublas" and x.dtype is torch.float32 and x.is_cuda and (
            torch.cuda.get_device_capability(x.device) == (10, 0)):
        from .composed import cond_transition_inference_cublas

        return cond_transition_inference_cublas(x, cond, wa, wb, ws, wsc, bsc, length, residual)
    if d_hidden <= ATOM_D_MAX:
        from .inference import cond_transition_inference

        y = cond_transition_inference(x, cond, wa, wb, ws, wsc, bsc, length)
        return y if residual is None else residual + y
    if x.dtype is torch.float32:
        from .composed import cond_transition_inference_cublas

        return cond_transition_inference_cublas(x, cond, wa, wb, ws, wsc, bsc, length, residual)
    from .composed import cond_transition_inference_composed

    return cond_transition_inference_composed(x, cond, wa, wb, ws, wsc, bsc, length, residual=residual)
