"""Bidirectional triangular multiplicative update — outgoing + incoming in one block.

A single module shares one input LayerNorm and projects the pair to ``2 * d_hidden``
channels; the hidden channels split in half — first ``d_hidden`` compute the
**outgoing** product (``bikd,bjkd->bijd``), the second ``d_hidden`` the **incoming**
product (``bkid,bkjd->bijd``). The two are concatenated to ``2 * d_hidden`` and
projected down to ``d_pair``.

PYTORCH is the reference. The fused path (CUTE) reuses the trimul_inproj pipeline
with bidirectional dims: one wider gated GEMM front (left/right each ``2*d_hidden``),
two einsums (outgoing on the first half, incoming on the second), then the split
back (cute LayerNormLinear over ``2*d_hidden`` + triton GateElem). See
``kernels/trimul_inproj/cute/bidirectional.py``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from cuequivariance_torch import triangle_multiplicative_update
from jaxtyping import Bool, Float

from miniworld_kernels._typecheck import typecheck
from miniworld_kernels.modules.dispatch import (
    KernelBackend,
    resolve_triangle_multiplication as _resolve_trimul_backend,
)
from miniworld_kernels.modules.triangle_multiplication.dispatch import (
    resolve_out_layout as _resolve_trimul_out_layout,
)
from miniworld_kernels.modules.exceptions import (
    ImplementationType,
    InvalidImplementationError,
)
from miniworld_kernels.modules.ops import sigmoid_gate
from miniworld_kernels.modules.primitives import LayerNorm, Linear


class BidirectionalTriangleMultiplication(nn.Module):
    """Triangular multiplicative update computing outgoing+incoming in one block."""

    def __init__(
        self,
        d_pair: int = 128,
        d_hidden: int | None = None,
        *,
        implementation: ImplementationType = ImplementationType.PYTORCH,
    ) -> None:
        super().__init__()
        # Keep the PUBLIC option on self.implementation (contract: modules never overwrite it
        # with the resolved backend). 'miniworld' (auto) -> concrete backend for the running
        # GPU arch is resolved ONCE into self._backend; forward routes on that.
        self.implementation = ImplementationType(implementation)
        self._backend = _resolve_trimul_backend(implementation)  # concrete KernelBackend
        self.d_pair = d_pair
        self.d_hidden = d_hidden if d_hidden is not None else d_pair
        d2 = 2 * self.d_hidden

        self.ln_pair = LayerNorm(d_pair, implementation=implementation)
        # Doubled-width left/right projections: [outgoing | incoming] channels.
        self.to_left = Linear(d_pair, d2, bias=False, init="default")
        self.to_left_gate = Linear(d_pair, d2, bias=False, init="zero")
        self.to_right = Linear(d_pair, d2, bias=False, init="default")
        self.to_right_gate = Linear(d_pair, d2, bias=False, init="zero")

        self.ln_out = LayerNorm(d2, implementation=implementation)
        self.to_gate = Linear(d_pair, d_pair, bias=False, init="zero")
        self.to_out = Linear(d2, d_pair, bias=False, init="zero")

    @typecheck
    def forward(
        self,
        pair: Float[torch.Tensor, "B L L d_pair"],
        mask: Bool[torch.Tensor, "B L"] | None = None,
    ) -> Float[torch.Tensor, "B L L d_pair"]:
        """Forward pass. Routes on the resolved backend (self._backend); self.implementation
        stays the public option."""
        if self._backend == KernelBackend.CUEQUIVARIANCE:
            return self._forward_cuequivariance(pair, mask)
        if self._backend == KernelBackend.CUTE:
            # Inference: forward-only fused sm100 kernels. Training (grad): the
            # v6-faithful fused bidirectional training kernel (BidirV6TriMulSm100,
            # trimul_bidir_b200 v4), wired here so dispatch stays inside the module.
            if torch.is_grad_enabled():
                return self._forward_cute_train(pair, mask)
            return self._forward_cute(pair, mask)
        if self._backend == KernelBackend.TRITON:
            # Composed-from-unidirectional TRITON path (fwd + autograd bwd): reuses
            # the per-direction triton_tm1 front + triton GateElem back. One code
            # path serves inference and training (grad flows through the composed
            # autograd pieces). See kernels/trimul_inproj/triton/bidirectional.py.
            return self._forward_triton(pair, mask)
        if self._backend != KernelBackend.PYTORCH:
            raise InvalidImplementationError(self.implementation)

        pair = self.ln_pair(pair)
        left = sigmoid_gate(self.to_left_gate(pair), self.to_left(pair))
        right = sigmoid_gate(self.to_right_gate(pair), self.to_right(pair))

        if mask is not None:
            mask_2d = mask.unsqueeze(-1) & mask.unsqueeze(-2)
            left = left * mask_2d[..., None]
            right = right * mask_2d[..., None]

        # Split hidden channels: first half -> outgoing, second half -> incoming.
        h = self.d_hidden
        left_out, left_in = left[..., :h], left[..., h:]
        right_out, right_in = right[..., :h], right[..., h:]

        out_outgoing = torch.einsum("bikd,bjkd->bijd", left_out, right_out)
        out_incoming = torch.einsum("bkid,bkjd->bijd", left_in, right_in)
        out = torch.cat([out_outgoing, out_incoming], dim=-1)

        out = self.ln_out(out)
        return sigmoid_gate(self.to_gate(pair), self.to_out(out))

    def _forward_cuequivariance(
        self,
        pair: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """cuequivariance bidirectional baseline. cuequiv has no fused-bidir kernel,
        so this is the standard AF3 way: two single-direction
        ``triangle_multiplicative_update`` calls (outgoing on the first ``d_hidden``
        channels, incoming on the second), summed — i.e. the two residual updates a
        pairformer block would apply. Weights are split from this module's doubled
        (``2*d_hidden``) projections. Not bit-identical to the fused formulation
        (the fused path shares one LayerNorm over the 2h concat); it is the
        representative cuequiv cost/quality for a bidirectional update."""
        h = self.d_hidden
        mask_2d = None
        if mask is not None:
            mask_2d = mask.unsqueeze(-1) & mask.unsqueeze(-2)

        def _one(direction: str, sl: slice) -> torch.Tensor:
            return triangle_multiplicative_update(
                pair,
                direction=direction,
                mask=mask_2d,
                norm_in_weight=self.ln_pair.weight,
                norm_in_bias=self.ln_pair.bias,
                p_in_weight=torch.cat(
                    [self.to_left.weight[sl], self.to_right.weight[sl]], dim=0
                ),
                g_in_weight=torch.cat(
                    [self.to_left_gate.weight[sl], self.to_right_gate.weight[sl]], dim=0
                ),
                norm_out_weight=self.ln_out.weight[sl],
                norm_out_bias=self.ln_out.bias[sl],
                p_out_weight=self.to_out.weight[:, sl],
                g_out_weight=self.to_gate.weight,
            )

        return _one("outgoing", slice(0, h)) + _one("incoming", slice(h, 2 * h))

    @torch.compiler.disable
    def _forward_triton(
        self,
        pair: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """TRITON bidirectional path (fwd + autograd bwd) — composed from the
        unidirectional triton pieces (per-direction ``triton_tm1`` front + triton
        ``GateElem`` back) plus torch einsum/cat/LayerNorm. Same code path for
        inference and training; every stage is autograd-capable so the backward is
        obtained by composition. Requires ``d_hidden == d_pair`` (as the
        single-direction triton trimul does). bf16 / fp32, B>=1."""
        from miniworld_kernels.kernels.trimul_inproj.triton.bidirectional import (
            bidirectional_trimul_triton,
        )

        return bidirectional_trimul_triton(
            pair,
            self.to_left.weight, self.to_left_gate.weight,
            self.to_right.weight, self.to_right_gate.weight,
            self.to_gate.weight, self.to_out.weight,
            self.ln_pair.weight, self.ln_pair.bias,
            self.ln_out.weight, self.ln_out.bias,
            self.ln_pair.eps, self.ln_out.eps, self.d_hidden,
            mask=mask,
        )

    def _forward_cute_train(
        self,
        pair: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """MINIWORLD (ours) TRAINING path: the v6-faithful fused-bidirectional trimul
        training kernel (fwd+bwd, autograd-capable) — sm_100 ``bidir_forward_sm100`` on
        Blackwell, else sm90 ``bidir_forward``. Same kernel stack as the single-direction
        v6 path (m-major front, te LN_out+@Wp, fused gate; 0 transposes), applied to both
        directions with a shared 2h back-half.

        Calls the weights-as-args kernel with THIS module's OWN parameters by reference
        (the ``.t().contiguous()`` transposes stay in the autograd graph), so gradients
        flow straight back to ``self.to_left.weight`` / ``ln_pair.weight`` / … and the
        optimizer trains them. The former ``BidirV6TriMul*`` wrapper cloned the weights
        into fresh, first-forward-created Parameters — leaving this module's registered
        params grad-less (dead) and the live copies invisible to an optimizer built over
        ``model.parameters()`` before the first forward. bf16, B=1."""
        major = (
            torch.cuda.get_device_capability(pair.device)[0]
            if torch.cuda.is_available()
            else 0
        )
        if major >= 10:
            from miniworld_kernels.kernels.trimul_inproj.cute.bidir_training_sm100 import (  # noqa: E501
                bidir_forward_sm100 as _fwd,
                prepack_lr_operand_sm100 as _prepack,
            )
        else:
            from miniworld_kernels.kernels.trimul_inproj.cute.bidir_training import (  # noqa: E501
                bidir_forward as _fwd,
                prepack_lr_operand as _prepack,
            )
        # By-reference (differentiable) transposes of the module's own projection weights.
        WL = self.to_left.weight.t().contiguous()
        WLg = self.to_left_gate.weight.t().contiguous()
        WR = self.to_right.weight.t().contiguous()
        WRg = self.to_right_gate.weight.t().contiguous()
        Wg = self.to_gate.weight.t().contiguous()
        b_lr = _prepack(WL, WLg, WR, WRg)
        row_scale = None
        if mask is not None:
            m = mask.unsqueeze(-1) & mask.unsqueeze(-2)  # [B, L, L]
            row_scale = m.reshape(-1).to(pair.dtype)     # [M]
        return _fwd(
            pair, WL, WLg, WR, WRg, Wg, self.to_out.weight,
            self.ln_pair.weight, self.ln_pair.bias,
            self.ln_out.weight, self.ln_out.bias,
            self.ln_pair.eps, b_lr, self.d_hidden, row_scale,
        )

    @torch.compiler.disable
    def _forward_cute(
        self,
        pair: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """CUTE bidirectional path: compose the single-direction tm1 ``bdll_sm100``
        gate-GEMM+einsum for BOTH directions (outgoing on the first ``d_hidden``
        channels, incoming on the second), then the SHARED ln_out(2h) + to_out + gate
        back. Avoids the broken quack gated-M-major front of trimul_inproj/cute.

        incoming = outgoing with the k<->contraction index flipped, handled directly
        by the incoming einsum (no input transpose needed since we control the einsum).
        Same math as the pytorch reference; bf16 in / fp32 acc / bf16 out.
        """
        import os as _os
        major = (
            torch.cuda.get_device_capability(pair.device)[0]
            if torch.cuda.is_available() else 0
        )
        _free_default = "1" if major >= 10 else "0"
        if _os.environ.get(
            "MINIWORLD_TRIMUL_CUEQUIV_FREE", _free_default
        ) != "0":
            # free path now folds the pair-mask into LN_in (row_scale), so it serves
            # masked/padded inputs too — no longer gated on `mask is None`.
            return self._forward_cute_free(pair, mask)

        from miniworld_kernels.modules.triangle_multiplication.module import _load_cute_fns

        tm1_cute_forward, _tm2, fused_ln_mask, layer_norm_transpose = _load_cute_fns()
        b, l1, l2, d = pair.shape
        h = self.d_hidden
        M = b * l1 * l2

        if mask is not None:
            mask_2d = mask.unsqueeze(-1) & mask.unsqueeze(-2)
            x = fused_ln_mask(pair, self.ln_pair.weight, self.ln_pair.bias, mask_2d)
        else:
            o = layer_norm_transpose(
                pair.reshape(M, d), self.ln_pair.weight, self.ln_pair.bias,
                eps=self.ln_pair.eps, layout="nd->nd")
            x = (o[0] if isinstance(o, tuple) else o).view(b, l1, l2, d)

        def _front(sl: slice):
            return tm1_cute_forward(
                x,
                self.to_left.weight[sl].T.contiguous(),
                self.to_left_gate.weight[sl].T.contiguous(),
                self.to_right.weight[sl].T.contiguous(),
                self.to_right_gate.weight[sl].T.contiguous(),
                out_layout=_resolve_trimul_out_layout(pair.device),
            )

        left_out, right_out = _front(slice(0, h))          # outgoing half, [B,h,L,L]
        left_in, right_in = _front(slice(h, 2 * h))        # incoming half, [B,h,L,L]
        out_o = torch.einsum("bdik,bdjk->bdij", left_out, right_out)   # outgoing
        out_i = torch.einsum("bdki,bdkj->bdij", left_in, right_in)     # incoming
        tri = torch.cat([out_o, out_i], dim=1)             # [B, 2h, L, L]

        tri_dbn = tri.permute(1, 0, 2, 3).reshape(2 * h, b, l1 * l2)
        oo = layer_norm_transpose(
            tri_dbn, self.ln_out.weight, self.ln_out.bias,
            eps=self.ln_out.eps, layout="dbn->bnd")
        out_normed = (oo[0] if isinstance(oo, tuple) else oo).view(b, l1, l2, 2 * h)

        # shared back: sigmoid(x @ to_gate.T) * (out_normed @ to_out.T)  (gate K=d, out K=2h)
        gate = torch.sigmoid(x.reshape(M, d) @ self.to_gate.weight.T)
        proj = out_normed.reshape(M, 2 * h) @ self.to_out.weight.T
        return (gate * proj).view(b, l1, l2, d)

    @torch.compiler.disable
    def _forward_cute_free(
        self, pair: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """CUEQUIV-FREE sm100 (B200) bidirectional path — the current sm100 kernels.

        Mirrors the single-direction ``TriangleMultiplication._forward_cute_free``
        (triton LN_in -> tm1 ``bdll_sm100`` front -> cuBLAS einsum -> sm100
        LayerNormLinear + triton gate_elem), applied to BOTH directions with a
        SHARED back-half over the 2h concatenation. NO cuequiv / quack LN. B=1.

        ``mask`` [B, L] (residue mask) is folded into LN_in as a per-row scale
        (row_scale = mask_i & mask_j over the M=L*L rows) — free masking on the fast path.
        """
        from miniworld_kernels.kernels.trimul_inproj.cute.bidirectional_sm100 import (
            bidirectional_trimul_sm100,
        )
        from miniworld_kernels.modules.triangle_multiplication.module import (
            _load_cute_fns,
        )

        tm1_cute_forward, _tm2, _flm, _lnt = _load_cute_fns()
        out_layout = _resolve_trimul_out_layout(pair.device)
        row_scale = None
        if mask is not None:
            m = mask.unsqueeze(-1) & mask.unsqueeze(-2)        # [B, L, L]
            row_scale = m.reshape(-1).to(pair.dtype)           # [M]
        return bidirectional_trimul_sm100(
            pair,
            self.to_left.weight, self.to_left_gate.weight,
            self.to_right.weight, self.to_right_gate.weight,
            self.to_gate.weight, self.to_out.weight,
            self.ln_pair.weight, self.ln_pair.bias,
            self.ln_out.weight, self.ln_out.bias,
            self.ln_pair.eps, self.ln_out.eps, self.d_hidden,
            tm1_cute_forward, out_layout,
            row_scale=row_scale,
        )
