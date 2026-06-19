# vendored from team-gm psk/benchmark@e085d6d : src/team_gm/modules/kernels/cuda/transition/__init__.py
"""CUDA implementation of Transition."""

import importlib.util
import os
import subprocess
import sys
from functools import lru_cache
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path

import torch

_dir = Path(__file__).parent
_ext_name = "transition_cuda_ext_v2"


def _find_extension_binary() -> Path | None:
    search_roots = [_dir, *sorted(_dir.glob("build*/lib.*"))]
    for root in search_roots:
        for suffix in EXTENSION_SUFFIXES:
            matches = sorted(root.glob(f"{_ext_name}*{suffix}"))
            if matches:
                return matches[0]
    return None


@lru_cache(maxsize=1)
def _load_extension():
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0;8.6;8.9;9.0")

    if _find_extension_binary() is None:
        subprocess.run(
            [sys.executable, "setup.py", "build_ext"],
            cwd=_dir,
            env=os.environ.copy(),
            check=True,
        )

    extension_path = _find_extension_binary()
    if extension_path is None:
        msg = f"{_ext_name} build completed but no extension binary was found"
        raise ImportError(msg)

    spec = importlib.util.spec_from_file_location(_ext_name, extension_path)
    if spec is None or spec.loader is None:
        msg = f"Failed to load extension spec from {extension_path}"
        raise ImportError(msg)

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CUDATransitionFunction(torch.autograd.Function):
    @staticmethod
    @torch.compiler.disable()
    def forward(ctx, x, expand_a_weight, expand_b_weight, squeeze_weight, n):
        orig_shape = x.shape
        x_2d = x.reshape(-1, orig_shape[-1]).contiguous()
        expand_a_weight = expand_a_weight.contiguous()
        expand_b_weight = expand_b_weight.contiguous()
        squeeze_weight = squeeze_weight.contiguous()

        ext = _load_extension()
        out = ext.forward(
            x_2d,
            expand_a_weight,
            expand_b_weight,
            squeeze_weight,
            n,
        )

        ctx.save_for_backward(
            x_2d,
            expand_a_weight,
            expand_b_weight,
            squeeze_weight,
        )
        ctx.orig_shape = orig_shape
        ctx.n = n

        return out.reshape(orig_shape)

    @staticmethod
    @torch.compiler.disable()
    def backward(ctx, grad_output):
        x, expand_a_weight, expand_b_weight, squeeze_weight = ctx.saved_tensors

        grad_output_2d = grad_output.reshape(-1, grad_output.shape[-1]).contiguous()
        if grad_output_2d.dtype != x.dtype:
            grad_output_2d = grad_output_2d.to(x.dtype)

        ext = _load_extension()
        dx, grad_a_weight, grad_b_weight, grad_squeeze_weight = ext.backward(
            grad_output_2d,
            x,
            expand_a_weight,
            expand_b_weight,
            squeeze_weight,
            ctx.n,
        )

        return (
            dx.reshape(ctx.orig_shape),
            grad_a_weight,
            grad_b_weight,
            grad_squeeze_weight,
            None,
        )


cuda_transition = CUDATransitionFunction.apply
