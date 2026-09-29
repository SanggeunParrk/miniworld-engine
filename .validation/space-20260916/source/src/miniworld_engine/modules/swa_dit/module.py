"""ESMFold2 atom DiT: adaLN-Zero, sliding-window 3D RoPE, and SwiGLU.

Matches MiniWorld's ``block_style="esmfold2"`` with magnitude-preserving options
disabled: each residual branch has its own zero-initialized conditioning gate.
There is no atom-pair representation. Attention parameters are built externally
with ``modules.swa_atom_attention.build_attention_params``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float

from miniworld_engine.modules.exceptions import ImplementationType
from miniworld_engine.modules.primitives import Linear
from miniworld_engine.modules.swa_atom_attention import SWA3DRoPEAttention


class SwiGLUFFN(nn.Module):
    """ESMFold2 SwiGLU with the hidden width rounded up to a multiple of 256."""

    def __init__(
        self, d_model: int, expansion_ratio: int = 2, *,
        implementation: ImplementationType = ImplementationType.PYTORCH,
    ) -> None:
        super().__init__()
        self.implementation = ImplementationType(implementation)
        hidden = ((expansion_ratio * (d_model // 3) * 2) + 255) // 256 * 256
        self.w_up = Linear(d_model, 2 * hidden, bias=False, init="normal")
        self.w_down = Linear(hidden, d_model, bias=False, init="normal")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.implementation != ImplementationType.PYTORCH:
            from miniworld_engine import ops

            wa, wb = self.w_up.weight.chunk(2, dim=0)
            return ops.swiglu_ffn(x, wa, wb, self.w_down.weight)
        x1, x2 = self.w_up(x).chunk(2, dim=-1)
        return self.w_down(F.silu(x1) * x2)


class SWADiTBlock(nn.Module):
    """MiniWorld ESMFold2 SWAAtomBlock, with engine attention dispatch.

    ``n`` is MiniWorld's ``expansion_ratio``. Inputs have atom-length layout
    ``[N, S, d]``, where ``N = A * B``. Modulation predicts shift, scale, and
    residual gate for each of attention and FFN. Both gates start at zero, so
    the block is initially the identity.

    Parameter names follow the ESMFold2 block. Checkpoints from the former
    AF3-AdaLN/ConditionedTransition composition are structurally incompatible.
    """

    def __init__(
        self,
        d_atom: int = 128,
        d_cond: int = 128,
        n_head: int = 4,
        n: int = 2,
        half_window: int = 64,
        *,
        implementation: ImplementationType = ImplementationType.PYTORCH,
    ) -> None:
        super().__init__()
        self.implementation = ImplementationType(implementation)
        self.attn_norm = nn.RMSNorm(d_atom, elementwise_affine=False)
        self.ffn_norm = nn.RMSNorm(d_atom, elementwise_affine=False)
        self.adaln_modulation = nn.Sequential(
            nn.SiLU(),
            Linear(d_cond, 6 * d_atom, bias=False, init="zero"),
        )
        self.attn = SWA3DRoPEAttention(
            d_atom, n_head, half_window=half_window, implementation=implementation,
        )
        self.ffn = SwiGLUFFN(d_atom, expansion_ratio=n, implementation=self.implementation)

    def forward(
        self,
        x: Float[torch.Tensor, "N S d_atom"],
        cond: Float[torch.Tensor, "N S d_cond"],
        attention_params: tuple,
    ) -> Float[torch.Tensor, "N S d_atom"]:
        if self.implementation != ImplementationType.PYTORCH:
            from miniworld_engine import ops

            activated = self.adaln_modulation[0](cond)
            projection = self.adaln_modulation[1]
            assert isinstance(projection, nn.Linear)
            sh_a, sc_a, g_a, sh_f, sc_f, g_f = projection.weight.chunk(6, dim=0)
            # nn.RMSNorm(eps=None) uses the opmath epsilon (fp32 for BF16).
            eps = torch.finfo(torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype).eps
            attn_in, gate_a = ops.rms_norm_modulation(
                x, activated, sc_a, sh_a, g_a, eps=eps,
            )
            x = ops.gated_residual(x, gate_a, self.attn(attn_in, attention_params))
            ffn_in, gate_f = ops.rms_norm_modulation(
                x, activated, sc_f, sh_f, g_f, eps=eps,
            )
            return ops.gated_residual(x, gate_f, self.ffn(ffn_in))

        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = (
            self.adaln_modulation(cond).chunk(6, dim=-1)
        )
        attn_in = self.attn_norm(x) * (1 + scale_a) + shift_a
        x = x + gate_a * self.attn(attn_in, attention_params)
        ffn_in = self.ffn_norm(x) * (1 + scale_f) + shift_f
        return x + gate_f * self.ffn(ffn_in)
