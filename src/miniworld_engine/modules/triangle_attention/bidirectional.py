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

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Bool, Float

from miniworld_engine import kernels
from miniworld_engine._typecheck import typecheck
from miniworld_engine.kernels.bias_only_attention import dispatch as _bo_dispatch
from miniworld_engine.modules import dispatch as _dispatch
from miniworld_engine.modules.dispatch import (
    KernelBackend,
    resolve_triangle_attention,
)
from miniworld_engine.modules.exceptions import (
    ImplementationType,
    InvalidImplementationError,
)
from miniworld_engine.modules.functional import sigmoid_gate
from miniworld_engine.modules.primitives import LayerNorm, Linear


class BidirectionalTriangleAttention(nn.Module):
    """Bias-only triangular gated attention computing starting+ending in one block."""

    def __init__(
        self,
        d_pair: int = 128,
        n_head: int = 4,
        *,
        d_hidden: int | None = None,
        use_self_attention: bool = False,
        implementation: ImplementationType = ImplementationType.PYTORCH,
    ) -> None:
        super().__init__()
        # 'miniworld' (auto) -> the TRITON family (fused bidir attention kernels).
        self.implementation = ImplementationType(implementation)
        self._backend = resolve_triangle_attention(self.implementation)
        self.n_head = n_head
        self.d_pair = d_pair
        self.d_hidden = d_hidden if d_hidden is not None else d_pair
        self.use_self_attention = use_self_attention

        if self.d_hidden % n_head != 0:
            msg = f"d_hidden ({self.d_hidden}) must be divisible by n_head ({n_head})"
            raise ValueError(msg)

        d2 = 2 * self.d_hidden

        self.ln_pair = LayerNorm(d_pair, implementation=self.implementation)
        # Doubled-width value/bias projections: [starting | ending] channels.
        self.to_value = Linear(d_pair, d2, bias=False, init="glorot")
        self.to_bias = Linear(d_pair, 2 * n_head, bias=False, init="default")
        self.to_gate = Linear(d_pair, d2, bias=False, init="gating")
        self.to_out = Linear(d2, d_pair, bias=False, init="zero")
        if use_self_attention:
            # Doubled-width Q/K projections: [starting | ending] channels, like value.
            self.to_query = Linear(d_pair, d2, bias=False, init="glorot")
            self.to_key = Linear(d_pair, d2, bias=False, init="glorot")

    # ── shared machinery (bias-only fusion, reused from TriangleAttention) ──────────
    def _layernorm(self, pair: torch.Tensor, backend: KernelBackend) -> torch.Tensor:
        if backend == KernelBackend.TRITON and _bo_dispatch.use_kernels(
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
        m = math.prod(gate.shape[:-1])
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

    def _attn_one(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        bias: torch.Tensor,
        backend: KernelBackend,
    ) -> torch.Tensor:
        """One-direction (starting-frame) triangle self-attention: q,k,v [B,H,L,L,D],
        bias [B,H,L,L] -> out [B,H,L,L,D]. PYTORCH = the reference einsum; TRITON reuses the
        single-direction fused kernel (consumes strided q/k/v; bias made contiguous inside)."""
        if backend == KernelBackend.TRITON:
            return kernels.triton_triangle_attention_pair_bias(q, k, v, bias)
        scale = q.shape[-1] ** -0.5
        attn = torch.einsum("bhijd,bhikd->bhijk", q * scale, k)
        attn = attn + bias[:, :, None, :, :]
        attn = F.softmax(attn, dim=-1)
        return torch.einsum("bhijk,bhikd->bhijd", attn, v)

    def _attend_sa(
        self, pln: torch.Tensor, mask: torch.Tensor | None, backend: KernelBackend
    ) -> torch.Tensor:
        """Self-attention (Q/K/V) bidirectional attend on the split channels -> [B,L,L,2*d_hidden].

        Both directions use ONE LayerNorm'd pair and the doubled q/k/v/bias projections. The
        ENDING direction is the transpose-dual of STARTING: run the SAME single-direction
        attention on the (i<->j)-transposed ending-half tensors and transpose the result back
        (identical to what the single-dir module does via `rearrange(pair, "B I J D -> B J I D")`,
        but expressed on the already-projected tensors so the input pair is never transposed).

        TRITON routes through the fused ``bidir_triangle_attention_pair_bias`` Function: both
        directions run the v7 kernels writing straight into a shared concat buffer (no
        transpose/cat/split materialization). PYTORCH below is the reference einsum."""
        H = self.n_head
        if backend == KernelBackend.TRITON:
            from miniworld_engine.kernels.triangle_attention.triton.bidirectional import (
                bidir_triangle_attention_pair_bias,
            )

            q = self.to_query(pln)
            k = self.to_key(pln)
            v = self.to_value(pln)
            bias = self.to_bias(pln)  # [B, L, L, 2*n_head]
            if mask is not None:
                # starting half masks over the key axis (j); ending half over i (its dual).
                bs = bias[..., :H].masked_fill(~mask[:, None, :, None], float("-inf"))
                be = bias[..., H:].masked_fill(~mask[:, :, None, None], float("-inf"))
                bias = torch.cat([bs, be], dim=-1)
            return bidir_triangle_attention_pair_bias(q, k, v, bias, H)  # [B, L, L, 2*d_hidden]

        q = rearrange(self.to_query(pln), "B L L2 (G D) -> B G L L2 D", G=2 * H)
        k = rearrange(self.to_key(pln), "B L L2 (G D) -> B G L L2 D", G=2 * H)
        v = rearrange(self.to_value(pln), "B L L2 (G D) -> B G L L2 D", G=2 * H)
        bias = rearrange(self.to_bias(pln), "B L L2 G -> B G L L2", G=2 * H)

        # starting
        qs, ks, vs, bs = q[:, :H], k[:, :H], v[:, :H], bias[:, :H]
        if mask is not None:
            bs = bs.masked_fill(~mask[:, None, None, :], float("-inf"))
        out_s = self._attn_one(qs, ks, vs, bs, backend)

        # ending: (i<->j)-transpose the projected tensors -> starting-frame, attend, transpose back
        qe = q[:, H:].transpose(2, 3)
        ke = k[:, H:].transpose(2, 3)
        ve = v[:, H:].transpose(2, 3)
        be = bias[:, H:].transpose(2, 3)
        if mask is not None:
            be = be.masked_fill(~mask[:, None, None, :], float("-inf"))
        out_e = self._attn_one(qe, ke, ve, be, backend).transpose(2, 3)

        out = torch.cat([out_s, out_e], dim=1)  # [B, 2H, L, L, D]
        return rearrange(out, "B G L L2 D -> B L L2 (G D)")

    def _inference(self, pair: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        """Inference-only max-fusion: fold LN into the [value|bias|gate] concat
        projection (pln never materializes), then the two einsums and gate+to_out."""
        from miniworld_engine.kernels.layernorm_linear import layernorm_linear_triton

        B, L, _, _d = pair.shape
        dv = self.to_value.weight.shape[0]
        db = self.to_bias.weight.shape[0]
        dg = self.to_gate.weight.shape[0]
        # `pair`, NOT pair.reshape(-1, d). The wrapper flattens internally and reshapes the
        # result back, so pre-flattening changes nothing except that `length_of` then sees
        # M = L*L instead of L -- and both_key clamps L*L to the top bucket at any L >= 91, so
        # every sequence length shared one cached config. Measured here at L=384: M = 147,456
        # recorded as shape_key=8192.
        proj = layernorm_linear_triton(
            pair, self.ln_pair.weight, self.ln_pair.bias,
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
        """Forward pass. Routes on the resolved internal backend, degrading to the
        pytorch reference (with a warning) on a dtype the fused kernels can't run."""
        backend = _dispatch.guard_dtype(
            self._backend, pair.dtype, op="BidirectionalTriangleAttention"
        )
        # cuequivariance ships a SINGLE-direction triangle-attention kernel only; there is no
        # bidirectional one, so a CUEQUIVARIANCE request here runs the pytorch reference -- the
        # honest baseline -- exactly as it did before `triangle_attention` joined `_CUEQ_OPS`.
        # Without this arm the request falls past both branches and raises: the bidirectional class
        # resolves under the same op name as the single-direction one, so adding that name to the
        # cueq set re-routed it to a backend its forward never handled.
        if backend in (KernelBackend.PYTORCH, KernelBackend.CUEQUIVARIANCE):
            pln = self.ln_pair(pair)
            if self.use_self_attention:
                out = self._attend_sa(pln, mask, backend)
            else:
                out = self._attend(self.to_value(pln), self.to_bias(pln), mask)
            return self.to_out(sigmoid_gate(self.to_gate(pln), out))

        if backend == KernelBackend.TRITON:
            # Inference max-fusion (LN folded into the projections), gated like the
            # single-dir path: needs L past the kernel crossover and the concat
            # projection narrow enough to stay tensor-core-friendly. Bias-only only
            # (the fold packs value|bias|gate; self-attention adds Q/K).
            if (
                not self.use_self_attention
                and not torch.is_grad_enabled()
                and _bo_dispatch.use_kernels(pair.shape[1])
                and _bo_dispatch.use_infer_concat(self.to_value.weight.shape[0])
            ):
                return self._inference(pair, mask)

            pln = self._layernorm(pair, backend)
            if self.use_self_attention:
                out = self._attend_sa(pln, mask, backend)
            else:
                out = self._attend(self.to_value(pln), self.to_bias(pln), mask)
            return self._gate_out(self.to_gate(pln), out)

        raise InvalidImplementationError(self.implementation)
