"""PyTorch reference of the fused SWA atom DiT block: what "correct" means for kernel family ``swa_dit``.

The equations of ``interface`` (team-gm ``swa_fused_triton`` docstring, commit 14f2c73), written as plain tensor ops in
the dtype of the inputs -- the checkers hand it fp32 copies of the kernel's bf16 operands, so none of the kernel's bf16
rounding points (x, the projections, the normalised q/k, the gated attention, y, h) happen here. That difference is the
error the declared band prices.

    mod rows            row (a*B + b)*S + s of the flattened [N = A*B, S] sequence reads modulation row b*S + s
    x   = rmsnorm(q) * (1 + scale_a) + shift_a                         rmsnorm over C with ``eps``, no affine
    q_h, k_h, v_h = heads(x Wqkv^T);  q_h, k_h = rope(rmsnorm_D(.))    per-head RMSNorm with fp32's eps, then RoPE
    o_i = softmax_j(q_i k_j / sqrt(D)) v_j over |i - j| <= half_window and j < seqused[n]; o_i = 0 for i >= seqused[n]
    q1  = q + gate_a * ((sigmoid(x Wg^T) * o) Wo^T)
    out = q1 + gate_f * ((silu(y Wu[:H]^T) * (y Wu[H:]^T)) Wd^T),  y = rmsnorm(q1) * (1 + scale_f) + shift_f

RoPE rotates the two halves of each head (x1 | x2) by the per-atom angles: (x1 c - x2 s | x2 c + x1 s), which is
``modules.swa_atom_attention.apply_rotary_emb_3d`` with all D/2 frequencies active. Padding rows (s >= seqused) still
run the FFN; only their attention output is zero -- as in the kernels and in the module's flash path.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

FP32_EPS = float(torch.finfo(torch.float32).eps)


def swa_dit_hoist_modulation_reference(c_base: torch.Tensor, wmod: torch.Tensor) -> torch.Tensor:
    """[B*S, 6C] = silu(c_base) @ Wmod^T, in the inputs' dtype."""
    return F.silu(c_base).reshape(-1, c_base.shape[-1]) @ wmod.t()


def _rms(x: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def _rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x [N, S, H, D]; cos / sin [N, S, D/2], shared by the heads."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    c, s = cos.unsqueeze(2), sin.unsqueeze(2)
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)


def swa_dit_block_reference(q: torch.Tensor, mod: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                            seqused: torch.Tensor, wqkv: torch.Tensor, wg: torch.Tensor, wo: torch.Tensor, wu: torch.Tensor,
                            wd: torch.Tensor, B: int, half_window: int = 64, n_head: int = 4,
                            eps: float = FP32_EPS) -> torch.Tensor:
    """The block, differentiable in every floating input. Same arguments as ``interface.swa_dit_block``.

    q [N = A*B, S, C]; mod [B*S, 6C]; cos / sin [B*S, C/n_head/2] or [B, S, ...]; seqused [N]. Dense [S, S] attention
    per (row, head), so this is for checker-sized S.
    """
    N, S, C = q.shape
    A = N // B
    D = C // n_head

    def per_row(t: torch.Tensor) -> torch.Tensor:
        """[B*S, k] (b, atom) rows -> [N, S, k]: row a*B + b of the flattened sequence reads batch element b."""
        return t.reshape(B, S, t.shape[-1]).repeat(A, 1, 1)

    shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = per_row(mod).chunk(6, dim=-1)
    cs, sn = per_row(cos), per_row(sin)
    x = _rms(q, eps) * (1 + scale_a) + shift_a
    p_q, p_k, p_v = (x @ wqkv.t()).chunk(3, dim=-1)
    g = x @ wg.t()
    qh = _rope(_rms(p_q.reshape(N, S, n_head, D), FP32_EPS), cs, sn)
    kh = _rope(_rms(p_k.reshape(N, S, n_head, D), FP32_EPS), cs, sn)
    vh = p_v.reshape(N, S, n_head, D)

    pos = torch.arange(S, device=q.device)
    valid = pos.view(1, S) < seqused.to(torch.long).view(N, 1)                        # [N, S]
    band = (pos.view(S, 1) - pos.view(1, S)).abs() <= half_window                     # [S, S]
    allowed = band.view(1, 1, S, S) & valid.view(N, 1, 1, S)
    # A padding query row may have no allowed key at all; give it its own position so the softmax stays finite. Its
    # output is replaced by zero below, and a valid row already allows its own position.
    allowed = allowed | torch.eye(S, dtype=torch.bool, device=q.device).view(1, 1, S, S)
    scores = torch.einsum("nihd,njhd->nhij", qh, kh) * D ** -0.5
    p = torch.softmax(scores.masked_fill(~allowed, float("-inf")), dim=-1)
    o = torch.einsum("nhij,njhd->nihd", p, vh)
    o = torch.where(valid.view(N, S, 1, 1), o, torch.zeros_like(o)).reshape(N, S, C)

    q1 = q + gate_a * ((torch.sigmoid(g) * o) @ wo.t())
    y = _rms(q1, eps) * (1 + scale_f) + shift_f
    hidden = wd.shape[1]
    h = F.silu(y @ wu[:hidden].t()) * (y @ wu[hidden:].t())
    return q1 + gate_f * (h @ wd.t())
