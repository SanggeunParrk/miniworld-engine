"""H100 TriangleAttention projection backward (BF16 C128 H4).

Forward keeps five independent Linear operations. Only their shared input
gradient is fused: four TMA/WGMMA products and the narrow bias product accumulate
in FP32, then store BF16 once. Parameter gradients remain cuBLAS GEMMs.
"""
from __future__ import annotations

import functools
import importlib.util
import json
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

from miniworld_engine.kernels._compile import device_constant, opaque

_ROOT = Path(__file__).resolve().parent
_EXT = None


@functools.lru_cache(maxsize=1)
def _artifact_ready() -> bool:
    manifest = _ROOT / "manifest.json"
    if not manifest.is_file():
        return False
    data = json.loads(manifest.read_text())
    if data["python_abi"] != sys.implementation.cache_tag or data["torch"] != str(torch.__version__):
        return False
    return (_ROOT / data["binary"]).is_file()


@device_constant
def _available(device: torch.device) -> bool:
    return (
        torch.cuda.is_available() and _artifact_ready()
        and torch.cuda.get_device_capability(device) == (9, 0)
        and "H100" in torch.cuda.get_device_name(device)
    )


def can_use(pair: torch.Tensor, weights: tuple[torch.Tensor, ...]) -> bool:
    return (
        pair.is_cuda and pair.dtype == torch.bfloat16 and pair.is_contiguous()
        and (not torch.is_autocast_enabled("cuda") or torch.get_autocast_dtype("cuda") == torch.bfloat16)
        and pair.ndim == 4 and pair.shape[0] == 1
        and pair.shape[1] in (384, 768, 1024)
        and pair.shape[2] == pair.shape[1] and pair.shape[3] == 128
        and len(weights) == 5
        and all(w.dtype == pair.dtype and w.device == pair.device and w.is_contiguous() for w in weights)
        and all(w.shape == (128, 128) for w in weights[:4])
        and weights[4].shape == (4, 128)
        and _available(pair.device)
    )


def _extension():
    global _EXT
    if _EXT is None:
        data = json.loads((_ROOT / "manifest.json").read_text())
        spec = importlib.util.spec_from_file_location(data["module_name"], _ROOT / data["binary"])
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _EXT = module
    return _EXT


def _fake(dy, weights):
    return torch.empty((dy[0].numel() // 128, 128), device=dy[0].device, dtype=dy[0].dtype)


@opaque(fake=_fake, name="triangle_projection_dgrad_cuda")
def _dgrad(dy: list[torch.Tensor], weights: list[torch.Tensor]) -> torch.Tensor:
    return _extension().dgrad(dy, weights, 64)


class _Projections(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, wq, wk, wv, wg, wb):
        ctx.save_for_backward(x, wq, wk, wv, wg, wb)
        return tuple(F.linear(x, w) for w in (wq, wk, wv, wg, wb))

    @staticmethod
    def backward(ctx, *grad):
        x, *weights = ctx.saved_tensors
        dy = [g.reshape(-1, w.shape[0]).contiguous() for g, w in zip(grad, weights)]
        dx = None
        if ctx.needs_input_grad[0]:
            if torch.is_grad_enabled():
                # Keep the linear part differentiable for create_graph callers.
                dx = dy[0] @ weights[0]
                for g, w in zip(dy[1:], weights[1:]):
                    dx = dx + g @ w
            else:
                dx = _dgrad(dy, weights)
            dx = dx.reshape_as(x)
        xx = x.reshape(-1, 128)
        dw = [g.T @ xx if needed else None for g, needed in zip(dy, ctx.needs_input_grad[1:])]
        return dx, *dw


def projections(x, wq, wk, wv, wg, wb):
    """Return query, key, value, gate and bias without changing parameter storage."""
    return _Projections.apply(x, wq, wk, wv, wg, wb)
