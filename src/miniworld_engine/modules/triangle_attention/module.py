# vendored from team-gm psk/benchmark : src/team_gm/modules/layers/triangle_updates.py
"""Triangle (gated self-)attention — model-level op connecting the fused
triangle-attention kernel (and a cuequivariance baseline)."""

from contextlib import contextmanager

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
from miniworld_engine.modules.ops import sigmoid_gate
from miniworld_engine.modules.primitives import LayerNorm, Linear


@contextmanager
def _nvtx_range(name: str, enabled: bool):
    if enabled:
        torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        if enabled:
            torch.cuda.nvtx.range_pop()


class TriangleAttention(nn.Module):
    """Unified implementation of triangular gated self-attention.

    Parameters
    ----------
    d_pair : int
        Dimension of pair representation.
    n_head : int
        Number of attention heads.
    d_hidden : int | None
        Total hidden dimension for QKV projections.  Defaults to ``d_pair`` when *None*.
        Must be divisible by ``n_head``.
    starting : bool
        Whether the attention is around the "starting" node.
    use_self_attention : bool
        Whether to use self-attention.
    use_qk_norm : bool
        Whether to apply RMSNorm to query and key projections.
    implementation : ImplementationType
        Implementation to use.

    """

    def __init__(
        self,
        d_pair: int = 128,
        n_head: int = 4,
        *,
        d_hidden: int | None = None,
        starting: bool = True,
        use_self_attention: bool = True,
        use_qk_norm: bool = False,
        implementation: ImplementationType = ImplementationType.PYTORCH,
        p_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.n_head = n_head
        self.starting = starting
        self.use_self_attention = use_self_attention
        self.use_qk_norm = use_qk_norm
        # ======================================================================================
        # THIS MODULE ALWAYS APPLIES THE RESIDUAL: y = pair + drop(triangle_attention(pair)).
        # The residual connection is UNCONDITIONAL (AF3 pairformer default
        # ``pair = pair + drop_row/col(tri_attention(pair))``); residual connections are the
        # domain standard, so there is deliberately NO flag to turn it off. The DROPOUT is
        # OPTIONAL: ``p_drop`` applies only in ``self.training`` on the AF3 broadcast axis for this
        # module's role — starting => drop_row (broadcast over i, dim=1); ending => drop_col
        # (broadcast over j, dim=2). p_drop=0 (default) / eval => residual only.
        #
        # NOTE — NOT KERNEL-FUSED YET (unlike Transition / TriangleMultiplication): the residual +
        # dropout are an EXPLICIT ``pair + drop(out)`` after the attention op, not folded into the
        # kernel epilogue. This unifies the module CONTRACT now (the block just calls
        # ``module(pair, mask)``); fusing them into the attention output kernel for the speed win is
        # a separate, later task.
        # >>> To run WITHOUT the residual, EDIT THE CODE: flip the ``_ADD_RESIDUAL`` local in forward().
        # ======================================================================================
        self.p_drop = p_drop
        self._drop_dim = 1 if starting else 2  # drop_row (i) for starting, drop_col (j) for ending
        # 'miniworld' (ours, auto) resolves to the TRITON family: the repo's only
        # tensor-core triangle-attention kernels are the triton ones (which
        # themselves per-GPU dispatch fused vs split via _bo_dispatch). Resolution
        # lives in modules.dispatch; forward routes on self._backend.
        self.implementation = ImplementationType(implementation)
        self._backend = resolve_triangle_attention(self.implementation)
        position = "starting" if starting else "ending"
        self.nvtx_enabled = False
        self.nvtx_name = f"triangle_attention/{position}"

        if d_hidden is None:
            d_hidden = d_pair

        if d_hidden % n_head != 0:
            msg = f"d_hidden ({d_hidden}) must be divisible by n_head ({n_head})"
            raise ValueError(msg)

        self.ln_pair = LayerNorm(d_pair)
        if use_self_attention:
            self.to_query = Linear(d_pair, d_hidden, bias=False, init="glorot")
            self.to_key = Linear(d_pair, d_hidden, bias=False, init="glorot")

            if use_qk_norm:
                d_head = d_hidden // n_head
                self.norm_query = nn.RMSNorm(d_head)
                self.norm_key = nn.RMSNorm(d_head)

        self.to_value = Linear(d_pair, d_hidden, bias=False, init="glorot")
        self.to_bias = Linear(d_pair, n_head, bias=False, init="default")
        self.to_gate = Linear(d_pair, d_hidden, bias=False, init="gating")
        self.to_out = Linear(d_hidden, d_pair, bias=False, init="zero")

    def _kernel_triangle_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        bias: torch.Tensor,
        backend: KernelBackend,
    ) -> torch.Tensor:
        if backend == KernelBackend.PYTORCH:
            query.mul_(query.shape[-1] ** -0.5)
            attention = torch.einsum("bhijd,bhikd->bhijk", query, key)
            attention = attention + bias[:, :, None, :, :]
            attention = F.softmax(attention, dim=-1)
            return torch.einsum("bhijk,bhikd->bhijd", attention, value)

        if backend == KernelBackend.TRITON:
            return kernels.triton_triangle_attention_pair_bias(
                query,
                key,
                value,
                bias,
            )

        if backend == KernelBackend.CUEQUIVARIANCE:
            # cuequiv backend (opt-in): lazy import so the default path never needs cuequiv.
            from cuequivariance_torch import triangle_attention
            # Feed cuequiv CONTIGUOUS inputs in its native (B, L1, H, L2, D) layout: its kernel is
            # ~35% slower on the strided rearrange view (measured H100), so a strided input
            # unfairly penalizes the baseline. .contiguous() pays one copy but hits cuequiv's fast
            # path (net faster). bf16 vs fp32 bias is a wash, so keep the fp32 bias it expects.
            q = rearrange(query, "B H L1 L2 D -> B L1 H L2 D").contiguous()
            k = rearrange(key, "B H L1 L2 D -> B L1 H L2 D").contiguous()
            v = rearrange(value, "B H L1 L2 D -> B L1 H L2 D").contiguous()
            out = triangle_attention(q, k, v, bias.unsqueeze(1).float())
            return rearrange(out, "B L1 H L2 D -> B H L1 L2 D")  # ty: ignore[invalid-return-type]

        raise InvalidImplementationError(self.implementation)

    def _layernorm(self, pair: torch.Tensor, backend: KernelBackend) -> torch.Tensor:
        """LayerNorm via this repo's standalone kernel for the TRITON impl.

        Uses ``kernels.layernorm_kernel`` (the repo's own developed LayerNorm, not
        the legacy vendored ``triton_layernorm``); it is fully autograd-aware. It
        wins at large L but its dispatch overhead regresses at small L, so fall
        back to torch's native LayerNorm below the per-GPU threshold (see dispatch).
        """
        if backend == KernelBackend.TRITON and _bo_dispatch.use_kernels(
            pair.shape[1]
        ):
            return kernels.layernorm_kernel(
                pair, self.ln_pair.weight, self.ln_pair.bias, self.ln_pair.eps
            )
        return self.ln_pair(pair)

    def _gate_out(self, gate: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        """sigmoid(gate) * out @ to_out. Backend chosen per-GPU (dispatch): the fused
        GEMM (gate folded into to_out) at small d_hidden vs the one-pass sigmoid*mul +
        cuBLAS to_out at large d_hidden, where the wide fused tile degrades."""
        dh = gate.shape[-1]
        M = gate.shape[:-1].numel()
        if _bo_dispatch.gate_use_fused(dh, self.to_out.weight.shape[0], M,
                                       gate.device, gate.dtype):
            return kernels.fused_gate_out(gate, out, self.to_out.weight)
        return self.to_out(kernels.sigmoid_gate_fused(gate, out))

    def _kernel_bias_only_attention(
        self,
        value: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        # softmax(bias) is independent of i; this lowers (via opt_einsum) to a
        # single big GEMM per (b,h) -- already optimal, no custom kernel beats it.
        # `value` is a strided view (no .contiguous()): einsum folds the permute
        # into the GEMM prep, which is cheaper than a separate copy.
        attention = F.softmax(bias, dim=-1)
        return torch.einsum("bhjk,bhikd->bhijd", attention, value)

    def _inproj_weight(self) -> torch.Tensor:
        """Concatenated [value|bias|gate] projection weight for the fused inference
        path, cached across calls and rebuilt only when a projection weight changes
        (keyed on the parameters' version counters). Avoids a per-forward torch.cat."""
        w = (self.to_value.weight, self.to_bias.weight, self.to_gate.weight)
        ver = tuple(t._version for t in w)
        if getattr(self, "_wcat_ver", None) != ver:
            self._wcat = torch.cat(w, dim=0)
            self._wcat_ver = ver
        return self._wcat

    def _bias_only_inference(
        self,
        pair: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Inference-only fused bias-only path (no autograd).

        Fuses LayerNorm into the value/bias/gate projection (one ``layernorm_linear``
        over a concatenated weight, so the normalized pair never materializes) and
        the gate+to_out into ``fused_gate_out``. This wins for inference but its
        fused backward loses to (layernorm_kernel + cuBLAS) for training, so it is
        gated on ``not torch.is_grad_enabled()``.
        """
        from miniworld_engine.kernels.layernorm_linear import layernorm_linear_triton

        H = self.n_head
        B, L, _, d = pair.shape
        dv = self.to_value.weight.shape[0]
        db = self.to_bias.weight.shape[0]
        w_cat = self._inproj_weight()
        # `pair`, NOT pair.reshape(-1, d) -- see the note in bidirectional.py: pre-flattening
        # leaves `length_of` reading M = L*L, which both_key clamps to the top bucket.
        proj = layernorm_linear_triton(
            pair, self.ln_pair.weight, self.ln_pair.bias, w_cat, None,
            self.ln_pair.eps,
        )
        value, bias, gate = proj.split([dv, db, dv], dim=-1)
        value = rearrange(value.view(B, L, L, dv), "B L L2 (H D) -> B H L L2 D", H=H)
        bias = rearrange(bias.view(B, L, L, db), "B L L2 H -> B H L L2")
        if mask is not None:
            bias = bias.masked_fill(~mask[:, None, None, :], torch.finfo(bias.dtype).min)
        out = self._kernel_bias_only_attention(value, bias)
        out = rearrange(out, "B H L L2 D -> B L L2 (H D)")
        return self._gate_out(gate.view(B, L, L, dv), out)

    @typecheck
    def _make_drop_scale(self, pair: torch.Tensor, p: float) -> torch.Tensor:
        """drop_row/drop_col scale = (rand>p)/(1-p), broadcast over i (starting, dim=1) or j
        (ending, dim=2) — matches modules.primitives.Dropout(broadcast_dim=self._drop_dim)."""
        shape = list(pair.shape)
        shape[self._drop_dim] = 1
        keep = torch.rand(shape, device=pair.device, dtype=pair.dtype) > p
        return keep.to(pair.dtype) / (1.0 - p)

    @typecheck
    def forward(
        self,
        pair: Float[torch.Tensor, "B L L d_pair"],
        mask: Bool[torch.Tensor, "B L"] | None = None,
    ) -> Float[torch.Tensor, "B L L d_pair"]:
        """Forward pass. ALWAYS returns the residual output ``pair + drop(tri_attention(pair))``
        (residual UNCONDITIONAL; dropout optional via ``p_drop``, active only in ``self.training``).
        The residual/dropout are applied EXPLICITLY here (NOT kernel-fused yet — see the constructor
        comment); fusing them into the attention epilogue for the speed win is a later task.
        >>> To disable the residual (benchmarking the raw op), EDIT the ``_ADD_RESIDUAL`` line."""
        _ADD_RESIDUAL = True  # UNCONDITIONAL residual (explicit add; not fused yet). Edit to False to disable.
        out = self._attention(pair, mask)
        if self.p_drop > 0.0 and self.training:
            out = out * self._make_drop_scale(pair, self.p_drop)
        return pair + out if _ADD_RESIDUAL else out

    def _attention(
        self,
        pair: Float[torch.Tensor, "B L L d_pair"],
        mask: Bool[torch.Tensor, "B L"] | None = None,
    ) -> Float[torch.Tensor, "B L L d_pair"]:
        """Raw triangle-attention op (NO residual/dropout) — the residual is added by forward()."""
        with _nvtx_range(self.nvtx_name, self.nvtx_enabled):
            # Degrade to the pytorch reference (with a warning) on a dtype the fused
            # triton kernels can't run; bf16 (production) keeps the fast path.
            backend = _dispatch.guard_dtype(
                self._backend, pair.dtype, op="TriangleAttention"
            )
            if not self.starting:
                pair = rearrange(pair, "B I J D -> B J I D").contiguous()
            assert pair.is_contiguous()

            # Inference-only max-fusion path (folds LN into the projections). Gated by
            # dispatch: needs L past the kernel-launch crossover and d_hidden small
            # enough that the LN+proj concat GEMM stays tensor-core-friendly (it
            # regresses once the concat is too wide).
            if (
                not self.use_self_attention
                and not torch.is_grad_enabled()
                and backend == KernelBackend.TRITON
                and _bo_dispatch.use_kernels(pair.shape[1])
                and _bo_dispatch.use_infer_concat(self.to_value.weight.shape[0])
            ):
                out = self._bias_only_inference(pair, mask)
                if not self.starting:
                    out = rearrange(out, "B J I D -> B I J D").contiguous()
                return out

            pair = self._layernorm(pair, backend)
            value = self.to_value(pair)
            bias = self.to_bias(pair)

            # No .contiguous(): the bias-only einsum and the self-attention kernels
            # consume these strided views directly (the triton kernel re-packs
            # internally), so the explicit permute copy is pure overhead.
            value = rearrange(value, "B L L2 (H D) -> B H L L2 D", H=self.n_head)
            bias = rearrange(bias, "B L L2 H -> B H L L2")
            if mask is not None:
                bias = bias.masked_fill(~mask[:, None, None, :], torch.finfo(bias.dtype).min)

            if self.use_self_attention:
                query = self.to_query(pair)
                key = self.to_key(pair)

                # No .contiguous(): the triton attention kernel consumes these strided
                # (B,H,L,L2,D) views directly via explicit strides (head_dim D is stride-1,
                # so loads stay coalesced) — the transpose copy was pure overhead.
                query = rearrange(query, "B L L2 (H D) -> B H L L2 D", H=self.n_head)
                key = rearrange(key, "B L L2 (H D) -> B H L L2 D", H=self.n_head)

                if self.use_qk_norm:
                    query = self.norm_query(query)
                    key = self.norm_key(key)

                out = self._kernel_triangle_attention(query, key, value, bias, backend)
            else:
                out = self._kernel_bias_only_attention(value, bias)

            # sigmoid_gate is elementwise and materializes a contiguous result, so
            # an explicit .contiguous() on this transpose view is redundant.
            out = rearrange(out, "B H L L2 D -> B L L2 (H D)")
            if backend == KernelBackend.TRITON and _bo_dispatch.use_kernels(
                pair.shape[1]
            ):
                # Fuse sigmoid(to_gate(pair)) * out + the to_out projection (gated
                # tensor never hits HBM). Backend chosen per-GPU in _gate_out: fused
                # GEMM at small DH, split (sigmoid*mul + cuBLAS to_out) at large DH.
                out = self._gate_out(self.to_gate(pair), out)
            else:
                out = sigmoid_gate(self.to_gate(pair), out)
                out = self.to_out(out)

            if not self.starting:
                out = rearrange(out, "B J I D -> B I J D").contiguous()
            return out


class TrianglePairAttention(nn.Module):
    """Triangular gated self-attention with a pair-attention contraction."""

    def __init__(
        self,
        d_pair: int = 128,
        n_head: int = 4,
        *,
        d_hidden: int | None = None,
        starting: bool = True,
        use_self_attention: bool = True,
        use_qk_norm: bool = False,
        implementation: ImplementationType = ImplementationType.PYTORCH,
    ) -> None:
        super().__init__()
        self.n_head = n_head
        self.starting = starting
        self.use_self_attention = use_self_attention
        self.use_qk_norm = use_qk_norm
        # This variant's forward is the pure-torch pair-attention contraction
        # regardless of backend; resolution is kept for API consistency.
        self.implementation = ImplementationType(implementation)
        self._backend = resolve_triangle_attention(self.implementation)
        position = "starting" if starting else "ending"
        self.nvtx_enabled = False
        self.nvtx_name = f"triangle_attention/{position}"

        if d_hidden is None:
            d_hidden = d_pair

        if d_hidden % n_head != 0:
            msg = f"d_hidden ({d_hidden}) must be divisible by n_head ({n_head})"
            raise ValueError(msg)

        self.ln_pair = LayerNorm(d_pair)

        self.to_value = Linear(d_pair, d_hidden, bias=False, init="glorot")
        self.to_bias = Linear(d_pair, n_head, bias=False, init="default")
        self.to_gate = Linear(d_pair, d_hidden, bias=False, init="gating")
        self.to_out = Linear(d_hidden, d_pair, bias=False, init="zero")

    @typecheck
    def forward(
        self,
        pair: Float[torch.Tensor, "B L L d_pair"],
        mask: Bool[torch.Tensor, "B L"] | None = None,
    ) -> Float[torch.Tensor, "B L L d_pair"]:
        """Forward pass."""
        if not self.starting:
            pair = pair.transpose(1, 2).contiguous()
        pair = self.ln_pair(pair)
        value = self.to_value(pair)
        bias = self.to_bias(pair)

        value = value.view(
            value.shape[0], value.shape[1], value.shape[2], self.n_head, -1
        )
        if mask is not None:
            bias = bias.masked_fill(~mask[:, None, :, None], torch.finfo(bias.dtype).min)

        attention = F.softmax(bias, dim=-2)
        out = torch.einsum("bjkh,bikhd->bijhd", attention, value).contiguous()
        out = out.view(out.shape[0], out.shape[1], out.shape[2], -1)

        out = sigmoid_gate(self.to_gate(pair), out)
        out = self.to_out(out)

        if not self.starting:
            out = out.transpose(1, 2).contiguous()
        return out
