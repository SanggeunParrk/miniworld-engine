"""SWA inference gate/output fusion with a row-based cache key.

Reuse the tested pair-attention GEMM body, while measuring SWA workloads in their
own cache. The gate linear remains separate. Training keeps the existing path.
"""

from __future__ import annotations
import torch
import triton
import triton.language as tl
from miniworld_engine.autotune.configs import configs_for
from miniworld_engine.autotune.shape_key import both_key
from miniworld_engine.kernels._compile import opaque
from miniworld_engine.kernels._tiles import tile_grid
from miniworld_engine.kernels.bias_only_attention.triton import gate_out

_forward_body = gate_out._gate_out_fwd.fn


@triton.autotune(configs=configs_for("swa_gate_out_fwd_triton"), key=["shape_key"])
@triton.jit
def swa_gate_out_fwd_kernel(
    gate_ptr,
    outr_ptr,
    wo_ptr,
    o_ptr,
    M,
    N,
    DH: tl.constexpr,
    stride_gm,
    stride_gd,
    stride_om,
    stride_od,
    stride_wn,
    stride_wd,
    stride_cm,
    stride_cn,
    BLOCK_M1: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    shape_key,
    GROUP_M: tl.constexpr,
):
    _forward_body(
        gate_ptr,
        outr_ptr,
        wo_ptr,
        o_ptr,
        M,
        N,
        DH,
        stride_gm,
        stride_gd,
        stride_om,
        stride_od,
        stride_wn,
        stride_wd,
        stride_cm,
        stride_cn,
        BLOCK_M1,
        BLOCK_N,
        BLOCK_K,
        shape_key,
        GROUP_M,
    )


def _forward_fake(gate, out, weight, key):
    """Allocate the projected shape for tracing without executing the GEMM."""
    return gate.new_empty((gate.shape[0], weight.shape[0]))


@opaque(fake=_forward_fake, name="swa_gate_out_fwd")
def _forward(
    gate: torch.Tensor, out: torch.Tensor, weight: torch.Tensor, key: int
) -> torch.Tensor:
    """Run sigmoid-gated output projection using the shared GEMM body."""
    m, dh = gate.shape
    n = weight.shape[0]
    result = _forward_fake(gate, out, weight, key)
    if m:
        swa_gate_out_fwd_kernel[
            lambda cfg: tile_grid(m, n, cfg["BLOCK_M1"], cfg["BLOCK_N"])
        ](
            gate,
            out,
            weight,
            result,
            m,
            n,
            dh,
            *gate.stride(),
            *out.stride(),
            *weight.stride(),
            *result.stride(),
            shape_key=key,
        )
    return result


def swa_gate_out_inference(gate, out, weight):
    """BF16 ``(sigmoid(gate) * out) @ weight.T``; call under no_grad/inference_mode."""
    if torch.is_grad_enabled() and any(t.requires_grad for t in (gate, out, weight)):
        raise RuntimeError("swa_gate_out_inference requires gradients to be disabled")
    if gate.shape != out.shape or gate.ndim < 2:
        raise ValueError("gate and out must have matching [..., D] shapes")
    if weight.ndim != 2 or weight.shape[1] != gate.shape[-1]:
        raise ValueError("weight must have shape [output_width, D]")
    if not gate.is_cuda or gate.device != out.device or gate.device != weight.device:
        raise ValueError("gate, out and weight must share a CUDA device")
    if any(t.dtype != torch.bfloat16 for t in (gate, out, weight)):
        raise ValueError("SWA gate-output fusion requires BF16 tensors")
    shape = gate.shape
    g = gate.reshape(-1, shape[-1]).contiguous()
    a = out.reshape_as(g).contiguous()
    key = both_key(g.shape[0], N=weight.shape[0], DH=weight.shape[1])
    return _forward(g, a, weight.contiguous(), key).reshape(
        *shape[:-1], weight.shape[0]
    )
