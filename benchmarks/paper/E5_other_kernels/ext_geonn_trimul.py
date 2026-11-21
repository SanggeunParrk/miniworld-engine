"""E5 adapter: AMD ``geometric_nn_ops_torch.triangle.triangle_multiplicative_update`` (ROCm/geometric-nn-ops 1.0.0, Apache-2.0).

Same function and signature as ``cuequivariance_torch.triangle_multiplicative_update`` (LN_in, [a; b] gated projections, optional pair mask,
contraction, LN_out, output projection x output gate; no residual), so this is the module's ``_forward_cueq`` with the library swapped. Triton
forward kernels (fp32 / bf16 / fp16); the library's backward is its PyTorch reference (no kernel), so a call that needs gradients is recorded as
unsupported rather than measured as PyTorch. Validated by AMD on MI300 only; here the Triton source runs on NVIDIA unmodified.
"""

from __future__ import annotations

import torch
from ext_trimul_registry import add_paths

add_paths("geometric-nn-ops-trimul")
PROBE = "geometric_nn_ops_torch.triangle"


def supports(self, pair, mask, dropout_p):
    if pair.dtype not in (torch.bfloat16, torch.float32, torch.float16):
        return f"dtype {pair.dtype}: the library's Triton kernels take fp32 / bf16 / fp16"
    if torch.is_grad_enabled() and (pair.requires_grad or any(p.requires_grad for p in self.parameters())):
        return "training: the library has no backward kernel (its autograd path is its PyTorch reference)"
    return None


def forward(self, pair, mask, dropout_p):
    from geometric_nn_ops_torch.triangle import triangle_multiplicative_update

    from miniworld_engine.modules.functional import dropout_scale

    if self.dead and not (self.training or torch.is_grad_enabled()):
        return pair
    dropout_p = self.p_drop if dropout_p is None else dropout_p
    scale = dropout_scale(pair, dropout_p, 1) if dropout_p and dropout_p > 0.0 and self.training else None
    w = lambda lin: lin.weight.to(pair.dtype)
    pm = None if mask is None else (mask[:, :, None] & mask[:, None, :]).to(pair.dtype)
    update = triangle_multiplicative_update(
        pair,
        direction="outgoing" if self.outgoing else "incoming",
        mask=pm,
        norm_in_weight=self.ln_pair.weight,
        norm_in_bias=self.ln_pair.bias,
        p_in_weight=torch.cat([w(self.to_left), w(self.to_right)]),  # rows [a; b]
        g_in_weight=torch.cat([w(self.to_left_gate), w(self.to_right_gate)]),
        norm_out_weight=self.ln_out.weight,
        norm_out_bias=self.ln_out.bias,
        p_out_weight=w(self.to_out),
        g_out_weight=w(self.to_gate),
        eps=self.ln_pair.eps,
    )
    if scale is not None:
        update = update * scale
    return update + pair
