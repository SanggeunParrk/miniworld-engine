"""Bidirectional bias-only triangle attention — starting + ending in one block.

Mirrors ``BidirectionalTriangleMultiplication``: one shared input LayerNorm, the
value/bias projections doubled so the channels split in half — first half computes
the **starting** attention, second half the **ending** attention — then the two are
concatenated to ``2 * d_hidden`` and projected down to ``d_pair``.

This is the bias-only case (``use_self_attention=False``): the attention weights
come only from a learned bias projection (no Q/K). The two directions are different
contractions of the *same* shape (no input transpose):

    starting:  out_s[i,j,d] = sum_k softmax_k(bias_s[j,k]) * value_s[i,k,d]   (contract axis 2)
    ending:    out_e[i,j,d] = sum_k softmax_k(bias_e[k,i]) * value_e[k,j,d]   (contract axis 1)

The ending form is the transpose-derived dual of starting (run starting on the
spatially-transposed pair, transpose the result back), expressed directly as an
einsum so no ``.transpose()`` of the pair is needed.

Gate convention follows ``TriangleAttention`` (gate on the hidden-width output,
applied *before* ``to_out``), not the trimul convention. PYTORCH is the reference;
the TRITON path reuses the bias-only fusion machinery (layernorm_kernel,
fused_gate_out / split, the inference LN+proj concat, and the per-GPU dispatch).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Bool, Float

from miniworld_kernels import kernels
from miniworld_kernels._typecheck import typecheck
from miniworld_kernels.kernels.bias_only_attention import dispatch as _bo_dispatch
from miniworld_kernels.modules.exceptions import (
    ImplementationType,
    InvalidImplementationError,
)
from miniworld_kernels.modules.ops import sigmoid_gate
from miniworld_kernels.modules.primitives import LayerNorm, Linear


class BidirectionalTriangleAttention(nn.Module):
    """Bias-only triangular gated attention computing starting+ending in one block."""

    def __init__(
        self,
        d_pair: int = 128,
        n_head: int = 4,
        *,
        d_hidden: int | None = None,
        implementation: ImplementationType = ImplementationType.PYTORCH,
    ) -> None:
        super().__init__()
        self.implementation = implementation
        self.n_head = n_head
        self.d_pair = d_pair
        self.d_hidden = d_hidden if d_hidden is not None else d_pair

        if self.d_hidden % n_head != 0:
            msg = f"d_hidden ({self.d_hidden}) must be divisible by n_head ({n_head})"
            raise ValueError(msg)

        d2 = 2 * self.d_hidden

        self.ln_pair = LayerNorm(d_pair, implementation=implementation)
        # Doubled-width value/bias projections: [starting | ending] channels.
        self.to_value = Linear(d_pair, d2, bias=False, init="glorot")
        self.to_bias = Linear(d_pair, 2 * n_head, bias=False, init="default")
        self.to_gate = Linear(d_pair, d2, bias=False, init="gating")
        self.to_out = Linear(d2, d_pair, bias=False, init="zero")

    # ── shared machinery (bias-only fusion, reused from TriangleAttention) ──────────
    def _layernorm(self, pair: torch.Tensor) -> torch.Tensor:
        if self.implementation == ImplementationType.TRITON and _bo_dispatch.use_kernels(
            pair.shape[1]
        ):
            return kernels.layernorm_kernel(
                pair, self.ln_pair.weight, self.ln_pair.bias, self.ln_pair.eps
            )
        return self.ln_pair(pair)

    def _gate_out(self, gate: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        """sigmoid(gate) * out @ to_out, backend chosen per-GPU. The fused-vs-split
        choice keys on the OUTPUT width (d_pair), not the 2*d_hidden gate width — so
        at d_pair<=128 the fused kernel is used even though d_hidden=2h is large."""
        dh = gate.shape[-1]
        m = gate.shape[:-1].numel()
        if _bo_dispatch.gate_use_fused(dh, self.to_out.weight.shape[0], m,
                                       gate.device, gate.dtype):
            return kernels.fused_gate_out(gate, out, self.to_out.weight)
        return self.to_out(kernels.sigmoid_gate_fused(gate, out))

    def _inproj_weight(self) -> torch.Tensor:
        """Concatenated [value|bias|gate] projection weight for the fused inference
        path, cached and rebuilt only when a projection weight changes."""
        w = (self.to_value.weight, self.to_bias.weight, self.to_gate.weight)
        ver = tuple(t._version for t in w)
        if getattr(self, "_wcat_ver", None) != ver:
            self._wcat = torch.cat(w, dim=0)
            self._wcat_ver = ver
        return self._wcat

    def _attend(
        self, value: torch.Tensor, bias: torch.Tensor, mask: torch.Tensor | None
    ) -> torch.Tensor:
        """The two direction einsums on the split channels -> [B, L, L, 2*d_hidden].

        value: [B, L, L, 2*d_hidden]; bias: [B, L, L, 2*n_head] (strided views ok).
        """
        H = self.n_head
        value = rearrange(value, "B L L2 (G D) -> B G L L2 D", G=2 * H)
        bias = rearrange(bias, "B L L2 G -> B G L L2", G=2 * H)
        value_s, value_e = value[:, :H], value[:, H:]
        bias_s, bias_e = bias[:, :H], bias[:, H:]

        if mask is not None:
            bias_s = bias_s.masked_fill(~mask[:, None, None, :], float("-inf"))
            bias_e = bias_e.masked_fill(~mask[:, None, :, None], float("-inf"))

        out_s = torch.einsum("bhjk,bhikd->bhijd", F.softmax(bias_s, dim=-1), value_s)
        out_e = torch.einsum("bhki,bhkjd->bhijd", F.softmax(bias_e, dim=-2), value_e)
        out = torch.cat([out_s, out_e], dim=1)  # [B, 2H, L, L, D]
        return rearrange(out, "B G L L2 D -> B L L2 (G D)")  # [B, L, L, 2*d_hidden]

    def _inference(self, pair: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        """Inference-only max-fusion: fold LN into the [value|bias|gate] concat
        projection (pln never materializes), then the two einsums and gate+to_out."""
        from miniworld_kernels.kernels.layernorm_linear import layernorm_linear_triton

        B, L, _, d = pair.shape
        dv = self.to_value.weight.shape[0]
        db = self.to_bias.weight.shape[0]
        dg = self.to_gate.weight.shape[0]
        proj = layernorm_linear_triton(
            pair.reshape(-1, d), self.ln_pair.weight, self.ln_pair.bias,
            self._inproj_weight(), None, self.ln_pair.eps,
        )
        value, bias, gate = proj.split([dv, db, dg], dim=-1)
        out = self._attend(value.view(B, L, L, dv), bias.view(B, L, L, db), mask)
        return self._gate_out(gate.view(B, L, L, dg), out)

    @typecheck
    def forward(
        self,
        pair: Float[torch.Tensor, "B L L d_pair"],
        mask: Bool[torch.Tensor, "B L"] | None = None,
    ) -> Float[torch.Tensor, "B L L d_pair"]:
        """Forward pass."""
        if self.implementation == ImplementationType.PYTORCH:
            pln = self.ln_pair(pair)
            out = self._attend(self.to_value(pln), self.to_bias(pln), mask)
            return self.to_out(sigmoid_gate(self.to_gate(pln), out))

        if self.implementation == ImplementationType.TRITON:
            # Inference max-fusion (LN folded into the projections), gated like the
            # single-dir path: needs L past the kernel crossover and the concat
            # projection narrow enough to stay tensor-core-friendly.
            if (
                not torch.is_grad_enabled()
                and _bo_dispatch.use_kernels(pair.shape[1])
                and _bo_dispatch.use_infer_concat(self.to_value.weight.shape[0])
            ):
                return self._inference(pair, mask)

            pln = self._layernorm(pair)
            out = self._attend(self.to_value(pln), self.to_bias(pln), mask)
            return self._gate_out(self.to_gate(pln), out)

        raise InvalidImplementationError(self.implementation)
