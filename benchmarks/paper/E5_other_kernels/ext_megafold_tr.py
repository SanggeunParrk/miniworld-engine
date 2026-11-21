"""E5 external backend ``megafold_tr``: MegaFold's fused Transition (MIT, Triton, forward + backward): ``LayernormLinear`` (LayerNorm fused into the expand GEMM) +
``FusedSwiGLU`` + the squeeze GEMM (torch). Upstream: github.com/Supercomputing-System-AI-Lab/MegaFold, ``megafold/model/FusedLayernormLinear`` and
``FusedTransition`` (loaded as source files; they import liger-kernel's Triton helpers: ``third_party/megafold/fetch.sh`` installs liger-kernel 0.8.5 next to the tree).
Serves ``Transition`` (the module's residual is added here): bf16; MegaFold's ``calculate_config_layernorm_linear`` decides the shapes it tunes (the upstream module
silently falls back to torch otherwise: here that is ``unsupported``).
"""

from __future__ import annotations

import math
import os
import sys

import ext_attn_common as common
import torch

NAME = "megafold_tr"


def _mods():
    sys.path.insert(0, os.path.join(common.ext_root(), "liger-kernel")) if os.path.join(
        common.ext_root(), "liger-kernel"
    ) not in sys.path else None
    root = os.path.join(common.upstream("megafold"), "megafold", "model")
    # fused_transition does `from megafold.model.FusedLayernormLinear.fused_layernorm_linear import LayernormLinear`: register the two files under that name
    import types

    for p in ("megafold", "megafold.model", "megafold.model.FusedLayernormLinear"):
        sys.modules.setdefault(p, types.ModuleType(p)).__path__ = []
    sys.modules["megafold.model.FusedLayernormLinear"].helper = common.load_file(
        "megafold.model.FusedLayernormLinear.helper", os.path.join(root, "FusedLayernormLinear", "helper.py")
    )
    ln = common.load_file(
        "megafold.model.FusedLayernormLinear.fused_layernorm_linear",
        os.path.join(root, "FusedLayernormLinear", "fused_layernorm_linear.py"),
    )
    sys.modules["megafold.model.FusedLayernormLinear"].fused_layernorm_linear = ln
    ft = common.load_file("megafold_fused_transition", os.path.join(root, "FusedTransition", "fused_transition.py"))
    return ln, ft


def _supports(self, x, *args, **kwargs):
    try:
        ln, _ = _mods()
    except Exception as exc:
        return f"megafold_tr: {type(exc).__name__}: {str(exc)[:200]} (see third_party/megafold)"
    dt = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else x.dtype
    if dt is not torch.bfloat16:
        return f"dtype {dt}: the fused kernels cast to bf16"
    m = math.prod(x.shape[:-1])
    if ln.calculate_config_layernorm_linear(m, 2 * self.d_hidden * self.n, self.d_hidden, 2) is None:
        return f"M {m}, K {self.d_hidden}, N {2 * self.d_hidden * self.n}: no tuned LayernormLinear config (MegaFold falls back to torch there)"
    return None


def _forward(self, x):
    import torch.nn.functional as F
    from miniworld_engine.modules.primitives import master_autocast

    if self.dead and not (self.training or torch.is_grad_enabled()):
        return x
    with master_autocast(x, self):
        ln, ft = _mods()
        cfg = ln.calculate_config_layernorm_linear(math.prod(x.shape[:-1]), 2 * self.d_hidden * self.n, self.d_hidden, 2)
        w = (
            torch.cat([self.expand_b.weight, self.expand_a.weight], 0).to(x.dtype).t().contiguous()
        )  # [K, 2 * inner]: value columns then gate columns
        g = self.ln_in.weight.to(x.dtype)
        b = None if self.ln_in.bias is None else self.ln_in.bias.to(x.dtype)
        h = ln.LayernormLinearFunction.apply(x, w, None, g, b, b is not None, False, cfg)
        y = ft.FusedSwiGLUFunction.apply(h)
        return x + F.linear(y, self.squeeze.weight.to(x.dtype))


def install() -> None:
    from miniworld_engine.modules import transition

    cls = transition.Transition
    if NAME not in cls.backends:
        cls.backends = (*cls.backends, NAME)
    setattr(cls, f"_supports_{NAME}", _supports)
    setattr(cls, f"_forward_{NAME}", _forward)
