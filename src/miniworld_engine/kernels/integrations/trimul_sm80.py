"""Automatic dispatch to the A100 (sm_80) hand-CUDA TriMul, inference and training.

Two kernel sets, one gate (``serves``):

* ``kernels.trimul_inproj.cuda.sm80`` -- the fused D128 path (bf16, one direction or bidirectional): literal tile shapes, the on-chip backward;
* ``kernels.trimul_inproj.cuda.sm80_wide`` -- every other width D = 64 / 256 / 384 (and the dtypes the fused path does not take): the LayerNorm-folded
  GEMM kernels ``k1w`` / ``k3w`` with the cuBLAS contraction, and the backward as cuBLAS GEMMs between row kernels.
  D512 uses this set for bf16 inference; ``sm80_wide512`` adds native backward rows at hidden 512 or 1024.
"""

from __future__ import annotations

import torch


def _wide_hs(module) -> int:
    """Hidden channels of the plane pair: the leading size of the front matrices (D one direction, 2 D bidirectional)."""
    return module.to_left.weight.shape[0]


def _wide_kernels(pair):
    from miniworld_engine.kernels.trimul_inproj.cuda import sm80_wide

    if pair.shape[-1] == 512 and torch.is_grad_enabled():
        from miniworld_engine.kernels.trimul_inproj.cuda import sm80_wide512

        return sm80_wide512
    return sm80_wide


def serves(module, pair: torch.Tensor, mask: torch.Tensor | None) -> bool:
    """``refusal`` is None."""
    return refusal(module, pair, mask) is None


def refusal(module, pair: torch.Tensor, mask: torch.Tensor | None) -> str | None:
    """Why the A100 kernels cannot take this call (the ``cuda`` backend of the TriMul classes states it on an A100), or ``None``."""
    if pair.dtype != torch.bfloat16:
        return f"dtype {pair.dtype}: the A100 TriMul kernels are bf16 only (no fp32 TriMul, by policy)"
    if module.ln_pair.eps != module.ln_out.eps:
        return "LN_in and LN_out must share one epsilon"
    # the input LayerNorm's gradients are written by the kernel, in the parameters' dtype (bf16 or fp32, both alike)
    if module.ln_pair.weight.dtype != module.ln_pair.bias.dtype or module.ln_pair.weight.dtype not in (torch.bfloat16, torch.float32):
        return "the input LayerNorm's weight and bias must share one dtype, bf16 or fp32"
    # the six weight matrices share one dtype (bf16, or fp32 masters over bf16 activations); the kernels cast them inside the op and write the
    # weight gradients in that dtype
    wd = module.to_left.weight.dtype
    if wd not in (torch.bfloat16, torch.float32) or any(
        w.weight.dtype != wd
        for w in (
            module.to_left_gate,
            module.to_right,
            module.to_right_gate,
            module.to_gate,
            module.to_out,
        )
    ):
        return "the six weight matrices must share one dtype, bf16 or fp32 masters"
    if pair.dim() != 4 or pair.shape[0] < 1 or not pair.is_contiguous():
        return "pair must be a contiguous [B, L, L, d_pair] stack"
    from miniworld_engine.kernels.trimul_inproj.cuda import sm80

    # the kernels take one square plane (a batch runs plane by plane in ``update``): the gate is asked for the first one
    plane, plane_mask = pair[:1], None if mask is None else mask[:1]
    if sm80.available(plane, module.d_hidden, plane_mask):
        return None
    if _wide_kernels(plane).available(plane, module.d_hidden, plane_mask, hs=_wide_hs(module)):
        return None
    return (
        f"d_pair {pair.shape[-1]}, d_hidden {module.d_hidden}, L {pair.shape[1]}, grad {torch.is_grad_enabled()}: outside the A100 TriMul "
        "kernels (sm80: D128 fused; sm80_wide / sm80_wide512: D 64 .. 512) or their extensions failed to build"
    )


def update(
    module,
    pair: torch.Tensor,
    mask: torch.Tensor | None,
    dropscale: torch.Tensor | None,
    *,
    bidirectional: bool,
) -> torch.Tensor:
    """``pair + drop_row(trimul(pair))``.  The parameters go to the op as leaves; it casts the weights to the kernels' dtype outside autograd and returns
    each gradient in its parameter's own dtype (fp32 masters get the kernels' fp32 accumulators unrounded)."""
    if pair.shape[0] > 1:  # the kernels take one square plane: a batch runs plane by plane (autograd sums the weight gradients)
        return torch.cat(
            [
                update(
                    module,
                    pair[i : i + 1],
                    None if mask is None else mask[i : i + 1],
                    None if dropscale is None else (dropscale if dropscale.shape[0] == 1 else dropscale[i : i + 1]),
                    bidirectional=bidirectional,
                )
                for i in range(pair.shape[0])
            ],
            dim=0,
        )
    from miniworld_engine.kernels.trimul_inproj.cuda import sm80

    n = pair.shape[1]
    direction = sm80.BIDIR if bidirectional else sm80.OUTGOING if module.outgoing else sm80.INCOMING
    ds = pair.new_empty((0,)) if dropscale is None else dropscale.reshape(n, pair.shape[-1]).to(pair.dtype).contiguous()
    kernels = sm80 if sm80.supports(pair, module.d_hidden, mask) else _wide_kernels(pair)
    return kernels.trimul(
        pair,
        module.to_left.weight,
        module.to_left_gate.weight,
        module.to_right.weight,
        module.to_right_gate.weight,
        module.to_gate.weight,
        module.to_out.weight,
        module.ln_pair.weight,
        module.ln_pair.bias,
        module.ln_out.weight,
        module.ln_out.bias,
        sm80.token_mask(mask, n, pair.device),
        ds,
        direction,
        module.ln_pair.eps,
        module.ln_out.eps,
    )
