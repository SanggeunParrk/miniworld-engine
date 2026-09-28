"""Bidirectional triangular multiplicative update — outgoing + incoming in one block.

A single module shares one input LayerNorm and projects the pair to ``2 * d_hidden``
channels; the hidden channels split in half — first ``d_hidden`` compute the
**outgoing** product (``bikd,bjkd->bijd``), the second ``d_hidden`` the **incoming**
product (``bkid,bkjd->bijd``). The two are concatenated to ``2 * d_hidden`` and
projected down to ``d_pair``.

PYTORCH is the reference. On H100 the hand-CUDA kernels (``integrations.trimul_h100``)
serve the qualified training/inference shapes; everything else runs the Triton pipeline
(``kernels/trimul_inproj/triton/bidirectional.py``): one wider gated GEMM front
(left/right each ``2*d_hidden``), two contractions, and a shared ``2*d_hidden`` back.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from jaxtyping import Bool, Float

from miniworld_engine._typecheck import typecheck
from miniworld_engine.integrations import anthropic_trimul as _anthropic
from miniworld_engine.integrations import trimul_h100 as _h100
from miniworld_engine.modules.dispatch import (
    KernelBackend,
)
from miniworld_engine.modules.dispatch import (
    resolve_triangle_multiplication as _resolve_trimul_backend,
)
from miniworld_engine.modules.exceptions import (
    ImplementationType,
    InvalidImplementationError,
)
from miniworld_engine.modules.functional import sigmoid_gate
from miniworld_engine.modules.primitives import LayerNorm, Linear


class BidirectionalTriangleMultiplication(nn.Module):
    """Triangular multiplicative update computing outgoing+incoming in one block."""

    def __init__(
        self,
        d_pair: int = 128,
        d_hidden: int | None = None,
        *,
        implementation: ImplementationType = ImplementationType.PYTORCH,
        p_drop: float = 0.25,
    ) -> None:
        super().__init__()
        # Keep the PUBLIC option on self.implementation (contract: modules never overwrite it
        # with the resolved backend). 'miniworld' (auto) -> concrete backend for the running
        # GPU arch is resolved ONCE into self._backend; forward routes on that.
        self.implementation = ImplementationType(implementation)
        self._backend = _resolve_trimul_backend(implementation)  # concrete KernelBackend
        # ======================================================================================
        # THIS MODULE ALWAYS APPLIES THE RESIDUAL: y = pair + drop_row(bidir_trimul(pair)).
        # The residual connection is UNCONDITIONAL (AF3 default; residual is the domain standard) —
        # there is deliberately NO flag to turn it off. The row-broadcast DROPOUT is OPTIONAL:
        # ``p_drop`` (drop_row, broadcast_dim=1) applies only in ``self.training`` and DEFAULTS ON
        # at AF3's 0.25; p_drop=0 / eval
        # => residual only. The block just calls ``module(pair, mask)``.
        # WHY IT'S FUSED IN (SPEED): the residual add + dropout scale are done inside the trimul
        # gate/back kernel's output epilogue (no separate elementwise op / [B,L,L,D] HBM round-trip),
        # which is the whole point of the fusion; an unconditional residual is what lets the kernel
        # own that fused epilogue. See the single-dir TriangleMultiplication for the full rationale.
        # >>> There is no way to turn it off, not even by editing a local: the residual is part
        # >>> of what this module IS. For the raw op in isolation -- benchmarking, or a caller
        # >>> that owns its own residual -- use ``ops.bidirectional_triangle_multiplicative_update``, the
        # >>> weights-as-args facade that mirrors cuequivariance's signature.
        # ======================================================================================
        self.p_drop = p_drop
        self.d_pair = d_pair
        self.d_hidden = d_hidden if d_hidden is not None else d_pair
        d2 = 2 * self.d_hidden

        # The LayerNorm primitives are plumbing for the non-native paths; "anthropic" names a TriMul payload,
        # not a LayerNorm one, so they take the engine's own auto choice in that case.
        ln_impl = (ImplementationType.MINIWORLD if self.implementation == ImplementationType.ANTHROPIC
                   else self.implementation)
        self.ln_pair = LayerNorm(d_pair, implementation=ln_impl)
        # Doubled-width left/right projections: [outgoing | incoming] channels.
        self.to_left = Linear(d_pair, d2, bias=False, init="default")
        self.to_left_gate = Linear(d_pair, d2, bias=False, init="zero")
        self.to_right = Linear(d_pair, d2, bias=False, init="default")
        self.to_right_gate = Linear(d_pair, d2, bias=False, init="zero")

        self.ln_out = LayerNorm(d2, implementation=ln_impl)
        self.to_gate = Linear(d_pair, d_pair, bias=False, init="zero")
        self.to_out = Linear(d2, d_pair, bias=False, init="zero")

        if self.implementation == ImplementationType.MINIWORLD and d_pair == self.d_hidden == 128:
            # Keep the public [out, in] shapes and checkpoint keys, but store
            # the four front matrices in the [in, out] order consumed by the
            # native backward and Triton GEMMs. Their transpose is now a view.
            # Construct real leaf Parameters before the optimizer is created;
            # no detached copies or first-forward parameter replacement.
            for projection in (self.to_left, self.to_left_gate, self.to_right, self.to_right_gate):
                weight = projection.weight
                projection.weight = nn.Parameter(
                    weight.detach().t().contiguous().t(), requires_grad=weight.requires_grad
                )

    @typecheck
    def _make_drop_row_scale(self, pair: torch.Tensor, p: float) -> torch.Tensor:
        """drop_row (``Dropout(broadcast_dim=1)``) scale [B,1,L,D] = (rand>p)/(1-p)."""
        b, _l1, l2, d = pair.shape
        keep = torch.rand(b, 1, l2, d, device=pair.device, dtype=pair.dtype) > p
        return keep.to(pair.dtype) / (1.0 - p)

    def forward(
        self,
        pair: Float[torch.Tensor, "B L L d_pair"],
        mask: Bool[torch.Tensor, "B L"] | None = None,
        dropout_p: float | None = None,
    ) -> Float[torch.Tensor, "B L L d_pair"]:
        """Forward pass. ALWAYS returns the residual output ``pair + drop_row(bidir_trimul(pair))``,
        fused in the gate/back (see the constructor comment). Routes on the resolved
        backend (self._backend); self.implementation stays the public option. The residual is
        UNCONDITIONAL (no flag — domain standard, fused FOR SPEED). The row-broadcast DROPOUT is
        OPTIONAL: ``dropout_p`` overrides the instance ``p_drop`` per call (None -> ``self.p_drop``)
        and is active only in ``self.training``.
        >>> The raw op without the residual is ``ops.bidirectional_triangle_multiplicative_update``,
        not a flag on this module."""
        dropout_p = self.p_drop if dropout_p is None else dropout_p
        _pair_in = pair
        _ds = (self._make_drop_row_scale(pair, dropout_p)
               if dropout_p and dropout_p > 0.0 and self.training else None)
        def _r(out):
            if _ds is not None:
                out = out * _ds
            return out + _pair_in

        if (torch.is_grad_enabled() or _ds is not None) and _h100.serves(self, pair):
            return _h100.update(self, pair, mask, _ds)

        if _h100.serves_inference(self, pair, bidirectional=True, dropscale=_ds):
            return _h100.update_inference(self, pair, mask, bidirectional=True)

        # The Anthropic TriMul payload, when TRIMUL_NATIVE_BUILD_DIR names one that can run this forward
        # (sm_90, bf16, one square plane, no grad, no live dropout scale, a unit for this width).  An explicit
        # `implementation="anthropic"` refuses with the reason; `miniworld` uses it where it fits and falls
        # back to the backends below where it does not.  See integrations.anthropic_trimul.
        if _anthropic.wanted(self.implementation):
            _native = {"grad": torch.is_grad_enabled(), "dropout": _ds is not None}
            if self.implementation == ImplementationType.ANTHROPIC:
                _anthropic.require(pair, pair.shape[-1], 2 * self.d_hidden, **_native)   # explicit: the reason, never a reroute
            if _anthropic.serves(pair, pair.shape[-1], 2 * self.d_hidden, **_native):
                return _anthropic.update_bidirectional(self, pair, mask)

        if self._backend == KernelBackend.CUEQUIVARIANCE:
            return _r(self._forward_cuequivariance(pair, mask))
        if self._backend == KernelBackend.TRITON:
            # Composed-from-unidirectional TRITON path (fwd + autograd bwd): reuses
            # the per-direction triton_tm1 front + triton GateElem back. One code
            # path serves inference and training (grad flows through the composed
            # autograd pieces). See kernels/trimul_inproj/triton/bidirectional.py.
            # residual + row-broadcast dropout are FUSED into the triton gate store
            # (gate_elem epilogue) — no external _r() add.
            return self._forward_triton(pair, mask, _ds)
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
        return _r(sigmoid_gate(self.to_gate(pair), self.to_out(out)))

    def _forward_cuequivariance(
        self,
        pair: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Compose vendor primitives with the same shared 2h output normalization.

        cuEquivariance's public update supports one direction. Two full updates
        normalize each half separately and implement a different function.
        Here the vendor input norm/gated projection and output norm surround both
        contractions. The output projection and gate use torch because the vendor
        dual-input GEMM requires equal input widths (ours are d_pair and 2h).
        """
        from cuequivariance_ops_torch.fused_layer_norm_torch import layer_norm_transpose
        from cuequivariance_ops_torch.gated_gemm_torch import (
            fused_sigmoid_gated_dual_gemm,
        )

        normalized = layer_norm_transpose(
            pair, self.ln_pair.weight, self.ln_pair.bias,
            eps=self.ln_pair.eps, layout="bijd->bijd")
        pair_mask = None if mask is None else mask.unsqueeze(-1) & mask.unsqueeze(-2)
        projected = fused_sigmoid_gated_dual_gemm(
            normalized,
            torch.cat((self.to_left_gate.weight, self.to_right_gate.weight)),
            torch.cat((self.to_left.weight, self.to_right.weight)),
            mask=pair_mask, transpose_out=True)
        left, right = projected.chunk(2, dim=0)
        h = self.d_hidden
        outgoing = torch.einsum("dbik,dbjk->dbij", left[:h], right[:h])
        incoming = torch.einsum("dbki,dbkj->dbij", left[h:], right[h:])
        contraction = torch.cat((outgoing, incoming), dim=0)
        output = layer_norm_transpose(
            contraction, self.ln_out.weight, self.ln_out.bias,
            eps=self.ln_out.eps, layout="dbij->bijd")
        return sigmoid_gate(self.to_gate(normalized), self.to_out(output))

    # No wrapper: every launch reachable from here is an ``opaque`` op at its own definition,
    # so Dynamo traces straight through this. It could never have BEEN an op itself -- see
    # ``kernels._compile`` -- but it does not need to be.
    def _forward_triton(
        self,
        pair: torch.Tensor,
        mask: torch.Tensor | None,
        dropscale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """TRITON bidirectional path (fwd + autograd bwd) — composed from the
        unidirectional triton pieces (per-direction ``triton_tm1`` front + triton
        ``GateElem`` back) plus torch einsum/cat/LayerNorm. Same code path for
        inference and training; every stage is autograd-capable so the backward is
        obtained by composition. Requires ``d_hidden == d_pair`` (as the
        single-direction triton trimul does). bf16 / fp32.

        The Triton kernels take one square plane (B=1); a batched input is run plane by plane
        here so the public module keeps its ``[B, L, L, d]`` contract on every arch."""
        if pair.shape[0] > 1:
            return torch.cat([
                self._forward_triton(
                    pair[i:i + 1],
                    None if mask is None else mask[i:i + 1],
                    None if dropscale is None else
                    (dropscale if dropscale.shape[0] == 1 else dropscale[i:i + 1]),
                ) for i in range(pair.shape[0])
            ], dim=0)
        from miniworld_engine.kernels.trimul_inproj.triton.bidirectional import (
            bidirectional_trimul_triton,
        )

        return bidirectional_trimul_triton(
            pair,
            self.to_left.weight.to(pair.dtype), self.to_left_gate.weight.to(pair.dtype),
            self.to_right.weight.to(pair.dtype), self.to_right_gate.weight.to(pair.dtype),
            self.to_gate.weight.to(pair.dtype), self.to_out.weight.to(pair.dtype),
            self.ln_pair.weight, self.ln_pair.bias,
            self.ln_out.weight, self.ln_out.bias,
            self.ln_pair.eps, self.ln_out.eps, self.d_hidden,
            mask=mask,
            dropscale=dropscale,
        )
