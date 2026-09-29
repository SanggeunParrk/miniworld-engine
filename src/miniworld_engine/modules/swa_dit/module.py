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

    Any implementation other than PYTORCH first offers the call to the fused block
    (``kernels.swa_dit``: Triton everywhere, hand-CUDA wgmma stages on sm_90), which
    serves bf16 or fp32 d_atom 128 / 4 heads / half_window 64 / SwiGLU hidden 256 and
    computes the whole block in 3 forward and 5-6 backward kernels (plus the
    weight-gradient GEMMs). Anything it
    refuses (:meth:`fused_refusal` says why) runs the per-op path below, and
    ``settings.swa_dit_fused=False`` turns it off. :meth:`forward_hoisted` takes the
    augment-invariant conditioning [B, S, d_cond] instead and computes the adaLN
    modulation once per batch element.
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

    def fused_refusal(
        self, x: torch.Tensor, cond: torch.Tensor, attention_params: tuple,
    ) -> str | None:
        """None if the fused block (``kernels.swa_dit``) can run this call, else why not.

        ``cond`` is either the per-row conditioning [N, S, d_cond] or the augment-invariant one [B, S, d_cond]
        (:meth:`forward_hoisted`). Reads tensor metadata only, so it traces without a graph break.
        """
        from miniworld_engine import settings

        if self.implementation == ImplementationType.PYTORCH:
            return "implementation is pytorch"
        if not settings.current().swa_dit_fused:
            return "settings.swa_dit_fused is off"
        if len(attention_params) != 6:
            return f"attention_params must be the 6-tuple of build_attention_params, got {len(attention_params)}"
        projection = self.adaln_modulation[1]
        if not isinstance(projection, nn.Linear):
            return "the adaLN projection is not a Linear"
        linears = (projection, self.attn.Wqkv, self.attn.gate_proj, self.attn.out_proj,
                   self.ffn.w_up, self.ffn.w_down)
        if any(type(m) is not Linear or m.bias is not None for m in linears):
            return "the fused block reads bias-free engine Linear weights"
        if any(m._forward_hooks or m._forward_pre_hooks
               for m in (self.attn, self.ffn, *linears)):
            return "a forward hook is registered on the attention / FFN / a projection"
        from miniworld_engine.kernels.swa_dit.interface import refusal

        cos, sin, seqused = attention_params[:3]
        return refusal(
            x, cos, sin, seqused, self.attn.Wqkv.weight, self.attn.gate_proj.weight,
            self.attn.out_proj.weight, self.ffn.w_up.weight, self.ffn.w_down.weight,
            n_head=self.attn.n_heads, half_window=self.attn.half_window,
            cond=cond, wmod=projection.weight,
        )

    def _fused(
        self, x: torch.Tensor, cond: torch.Tensor, attention_params: tuple,
    ) -> torch.Tensor:
        """The fused block with the modulation computed once per row of ``cond`` ([B, S, d_cond], N % B == 0)."""
        from miniworld_engine.kernels.swa_dit.interface import (
            swa_dit_block,
            swa_dit_hoist_modulation,
        )

        projection = self.adaln_modulation[1]
        assert isinstance(projection, nn.Linear)
        b = cond.shape[0]
        cos, sin, seqused = attention_params[:3]
        # build_attention_params repeats the per-batch RoPE over the augments, so rows [:B] are the per-batch angles.
        mod = swa_dit_hoist_modulation(cond, projection.weight)
        return swa_dit_block(
            x, mod, cos[:b], sin[:b], seqused, self.attn.Wqkv.weight,
            self.attn.gate_proj.weight, self.attn.out_proj.weight, self.ffn.w_up.weight,
            self.ffn.w_down.weight, b, half_window=self.attn.half_window,
        )

    def forward_hoisted(
        self,
        x: Float[torch.Tensor, "N S d_atom"],
        c_base: Float[torch.Tensor, "B S d_cond"],
        attention_params: tuple,
    ) -> Float[torch.Tensor, "N S d_atom"]:
        """The block over ``N = A * B`` rows whose conditioning repeats over the augments: ``cond[a*B + b] == c_base[b]``.

        The adaLN modulation depends only on (b, atom), so the fused block computes it once per batch element instead
        of once per augment (the team-gm ``SWAAtomTransformer`` contract). Where the fused block cannot serve the call,
        this is ``forward(x, c_base.repeat(A, 1, 1), attention_params)`` -- the same function.
        """
        if self.fused_refusal(x, c_base, attention_params) is None:
            return self._fused(x, c_base, attention_params)
        return self(x, c_base.repeat(x.shape[0] // c_base.shape[0], 1, 1), attention_params)

    def forward(
        self,
        x: Float[torch.Tensor, "N S d_atom"],
        cond: Float[torch.Tensor, "N S d_cond"],
        attention_params: tuple,
    ) -> Float[torch.Tensor, "N S d_atom"]:
        if self.implementation != ImplementationType.PYTORCH:
            if self.fused_refusal(x, cond, attention_params) is None:
                # One modulation row per sequence row (B = N): no augment structure is known here.
                return self._fused(x, cond, attention_params)
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
