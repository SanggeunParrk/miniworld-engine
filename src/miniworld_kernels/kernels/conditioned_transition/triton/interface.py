"""d-aware dispatch for the post-AdaLN ConditionedTransition tail (inference + training).

Inference routes by d_hidden:
  - atom  (d_hidden <= 128): the fully-fused single-kernel b2b (``inference.py``) — the
    whole (M, ND) SwiGLU + out_acc[BM, D] live in registers, h never hits HBM. Fastest
    when K fits one BLOCK_K and the working set fits smem.
  - token (d_hidden >= 256): the composed two-kernel path (``composed.py``) — K-tiled
    expand+SwiGLU writes h:(M, ND) to HBM, then K-tiled squeeze+gate. Always compiles.

Training (forward that saves for backward + backward) lives in ``training.py`` as a
torch.autograd.Function; the inference paths save nothing.
"""

from __future__ import annotations

ATOM_D_MAX = 128  # d_hidden <= this -> fused b2b; else composed


def cond_transition_inference_dispatch(x, cond, wa, wb, ws, wsc, bsc):
    """Forward-only (inference) ConditionedTransition tail, routed by d_hidden."""
    d_hidden = x.shape[1]
    if d_hidden <= ATOM_D_MAX:
        from .inference import cond_transition_inference

        return cond_transition_inference(x, cond, wa, wb, ws, wsc, bsc)
    from .composed import cond_transition_inference_composed

    return cond_transition_inference_composed(x, cond, wa, wb, ws, wsc, bsc)
