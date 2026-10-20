# vendored from team-gm psk/benchmark : src/team_gm/modules/layers/triangle_updates.py
"""Triangle (gated self-)attention — model-level op connecting the fused
triangle-attention kernel (and a cuequivariance baseline). On B200 the sm_100a kernels
(``integrations.triattn_b200``: d_pair 128 / 4 heads fused, inference and training; other widths 64-512 with 16- or
32-channel heads, inference and training; any L >= 16, B samples per call in deterministic chunks) serve the whole module."""

import math
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Bool, Float

from miniworld_engine import kernels, settings
from miniworld_engine._typecheck import typecheck
from miniworld_engine.integrations import h100_master as _master
from miniworld_engine.integrations import triattn_b200 as _b200
from miniworld_engine.kernels import _capture
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
    transposed_bias : bool
        Which orientation the ending node (``starting=False``) takes its pair bias from. ``True`` (default): the bias is
        projected from the transposed pair, the orientation the queries and keys are in (the supplementary-information
        and OpenFold convention). ``False``: the bias is projected from the pair as it is and read in the transposed
        frame, i.e. the transpose of the ``True`` bias (AlphaFold 3's released code). Starting-node attention has one
        orientation and ignores it. Training is served by the composed legs; the H100 hand-CUDA kernels serve the
        ``False`` orientation for inference only, the B200 kernels not at all.
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
        transposed_bias: bool = True,
        implementation: ImplementationType = ImplementationType.PYTORCH,
        p_drop: float = 0.25,
        anthropic_row: str = "k2b",
        gate_bias: bool = False,
        out_bias: bool = False,
        untransposed_output: bool = False,
        dual_output: bool = False,
    ) -> None:
        super().__init__()
        self.n_head = n_head
        self.starting = starting
        self.transposed_bias = transposed_bias
        #: Ending node on AlphaFold 3's bias orientation: the composed legs transpose the bias, the fused whole-module
        #: legs below are told to keep out of the way (the H100 sm90 kernels do it themselves, inference only).
        self._bias_untransposed = not starting and not transposed_bias
        self.use_self_attention = use_self_attention
        self.use_qk_norm = use_qk_norm
        # ======================================================================================
        # THIS MODULE ALWAYS APPLIES THE RESIDUAL: y = pair + drop(triangle_attention(pair)).
        # The residual connection is UNCONDITIONAL (AF3 pairformer default
        # ``pair = pair + drop_row/col(tri_attention(pair))``); residual connections are the
        # domain standard, so there is deliberately NO flag to turn it off. The DROPOUT is
        # OPTIONAL: ``p_drop`` applies only in ``self.training`` on the AF3 broadcast axis for this
        # module's role — starting => drop_row (broadcast over i, dim=1); ending => drop_col
        # (broadcast over j, dim=2). It DEFAULTS ON at AF3's 0.25 -- training a pairformer
        # block without it is the unusual case, so it is the value you have to ask to turn off,
        # not the one you have to ask for. At eval (`self.training` False) it is identity, which
        # is why dropout stays a variable where the residual does not.
        #
        # NOTE — NOT KERNEL-FUSED YET (unlike Transition / TriangleMultiplication): the residual +
        # dropout are an EXPLICIT ``pair + drop(out)`` after the attention op, not folded into the
        # kernel epilogue. This unifies the module CONTRACT now (the block just calls
        # ``module(pair, mask)``); fusing them into the attention output kernel for the speed win is
        # a separate, later task.
        # >>> There is no way to turn it off, not even by editing a local. The raw op without the
        # >>> residual is ``_attention()``, which is what ``ops.triangle_attention`` exposes.
        # ======================================================================================
        self.p_drop = p_drop
        self._drop_dim = (
            1 if starting else 2
        )  # drop_row (i) for starting, drop_col (j) for ending
        # 'miniworld' (ours, auto) resolves to the PORTABLE family: the repo's only
        # tensor-core triangle-attention kernels are the triton ones (which
        # themselves per-GPU dispatch fused vs split via _bo_dispatch). Resolution
        # lives in modules.dispatch; forward routes on self._backend.
        self.implementation = ImplementationType(implementation)
        self._backend = resolve_triangle_attention(self.implementation)
        self._train_gate = _dispatch.train_gate("triangle_attention", self.implementation, self._backend)
        self.anthropic_row = anthropic_row
        if self._backend == KernelBackend.ANTHROPIC and not use_self_attention:
            raise ValueError(
                "Anthropic TriangleAttention adapter requires use_self_attention=True"
            )
        if self._backend == KernelBackend.ANTHROPIC and self._bias_untransposed:
            raise ValueError(
                "Anthropic TriangleAttention adapter implements the transposed bias only"
            )
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
        # RoseTTAFold3 gives the gate and the out-projection a bias (``gate_bias`` / ``out_bias``): the H100 sm90 kernels serve
        # them for inference, anything else runs ``_biased_reference``.
        self.to_gate = Linear(d_pair, d_hidden, bias=gate_bias, init="gating")
        self.to_out = Linear(d_hidden, d_pair, bias=out_bias, init="zero")
        # Chai-1: the ending node returns its update as the transposed pair has it (``untransposed_output``); the confidence head adds
        # a second out-projection whose update lands transposed (``dual_output``: starting kept + other^T, ending kept^T + other).
        self.untransposed_output = untransposed_output and not dual_output
        self.dual_output = dual_output
        if dual_output:
            self.to_out_t = Linear(d_hidden, d_pair, bias=out_bias, init="zero")
        # Independent switches for qualified H100 projection and grouped-bias backward.
        # Each switch also selects its corresponding benchmark baseline.
        self._fuse_projection_backward = True
        self._fuse_bias_backward = True
        self._fuse_dq_backward = True
        self._fuse_front_backward = True
        self._fuse_gate_backward = True
        # B200 (sm_100a) whole-module CUDA path (integrations.triattn_b200); False keeps the Triton path there.
        self._b200_cuda = True
        # H100 (sm_90a) whole-module CUDA path (integrations.triattn_h100_sm90: raw-PTX wgmma / TMA, every registered width,
        # inference and training); False falls through to the b200style path below (the benchmark baseline).
        self._h100_cuda = True

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

        if backend == KernelBackend.PORTABLE:
            if (
                getattr(self, "_fuse_bias_backward", True)
                and torch.is_grad_enabled()
                and not self.use_qk_norm
                and not settings.current().force_portable_kernels  # hand-CUDA grouped-bias backward: not on the portable rung
            ):
                from miniworld_engine.kernels.triangle_attention.cuda.bias_backward import (
                    attention,
                    can_use,
                )

                if can_use(query, key, value, bias):
                    return attention(
                        query,
                        key,
                        value,
                        bias,
                        native_dq=getattr(self, "_fuse_dq_backward", True),
                    )
            return kernels.triton_triangle_attention_pair_bias(
                query,
                key,
                value,
                bias,
            )

        if backend == KernelBackend.ANTHROPIC:
            from miniworld_engine.integrations.anthropic import triangle_attention

            q, k, v = (
                t.permute(0, 2, 1, 3, 4).contiguous() for t in (query, key, value)
            )
            out = triangle_attention(
                q, k, v, bias.unsqueeze(1).float(), row=self.anthropic_row
            )
            return out.permute(0, 2, 1, 3, 4)

        if backend == KernelBackend.CUEQUIVARIANCE:
            # cuequiv backend (opt-in): lazy import so the default path never needs cuequiv.
            from cuequivariance_torch import (  # optional cuequivariance backend
                triangle_attention,
            )

            # Feed cuequiv CONTIGUOUS inputs in its native (B, L1, H, L2, D) layout: its kernel is
            # ~35% slower on the strided rearrange view (measured H100), so a strided input
            # unfairly penalizes the baseline. .contiguous() pays one copy but hits cuequiv's fast
            # path (net faster). bf16 vs fp32 bias is a wash, so keep the fp32 bias it expects.
            q = rearrange(query, "B H L1 L2 D -> B L1 H L2 D").contiguous()
            k = rearrange(key, "B H L1 L2 D -> B L1 H L2 D").contiguous()
            v = rearrange(value, "B H L1 L2 D -> B L1 H L2 D").contiguous()
            out = triangle_attention(q, k, v, bias.unsqueeze(1).float())
            assert isinstance(
                out, torch.Tensor
            )  # cuequivariance returns the attention output alone (no LSE here)
            return out.permute(0, 2, 1, 3, 4)  # B L1 H L2 D -> B H L1 L2 D

        raise InvalidImplementationError(self.implementation)

    def _layernorm(self, pair: torch.Tensor, backend: KernelBackend) -> torch.Tensor:
        """LayerNorm via this repo's standalone kernel for the PORTABLE impl.

        Uses ``kernels.layernorm_kernel`` (the repo's own developed LayerNorm, not
        the legacy vendored ``triton_layernorm``); it is fully autograd-aware. It
        wins at large L but its dispatch overhead regresses at small L, so fall
        back to torch's native LayerNorm below the per-GPU threshold (see dispatch).
        """
        if backend == KernelBackend.PORTABLE and _bo_dispatch.use_kernels(pair.shape[1]):
            return kernels.layernorm_kernel(
                pair, self.ln_pair.weight, self.ln_pair.bias, self.ln_pair.eps
            )
        return self.ln_pair(pair)

    def _gate_out(self, gate: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        """sigmoid(gate) * out @ to_out. Backend chosen per-GPU (dispatch): the fused
        GEMM (gate folded into to_out) at small d_hidden vs the one-pass sigmoid*mul +
        cuBLAS to_out at large d_hidden, where the wide fused tile degrades."""
        dh = gate.shape[-1]
        M = math.prod(gate.shape[:-1])
        if _bo_dispatch.gate_use_fused(
            dh, self.to_out.weight.shape[0], M, gate.device, gate.dtype
        ):
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
        ver = _capture.scoped(
            tuple(t._version for t in w)
        )  # never an eager copy inside a CUDA-graph capture
        if ver is None or getattr(self, "_wcat_ver", None) != ver:
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
        B, L, _, _d = pair.shape
        dv = self.to_value.weight.shape[0]
        db = self.to_bias.weight.shape[0]
        w_cat = self._inproj_weight()
        # `pair`, NOT pair.reshape(-1, d) -- see the note in bidirectional.py: pre-flattening
        # leaves `length_of` reading M = L*L, which both_key clamps to the top bucket.
        proj = layernorm_linear_triton(
            pair,
            self.ln_pair.weight,
            self.ln_pair.bias,
            w_cat,
            None,
            self.ln_pair.eps,
        )
        value, bias, gate = proj.split([dv, db, dv], dim=-1)
        value = rearrange(value.view(B, L, L, dv), "B L L2 (H D) -> B H L L2 D", H=H)
        bias = rearrange(bias.view(B, L, L, db), "B L L2 H -> B H L L2")
        if self._bias_untransposed:
            bias = bias.transpose(-1, -2).contiguous()
        if mask is not None:
            bias = bias.masked_fill(
                ~mask[:, None, None, :], torch.finfo(bias.dtype).min
            )
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
        (residual UNCONDITIONAL; dropout via ``p_drop``, default 0.25, active only in
        ``self.training``).
        The residual/dropout are applied EXPLICITLY here (NOT kernel-fused yet — see the constructor
        comment); fusing them into the attention epilogue for the speed win is a later task.
        >>> The raw op without the residual is ``_attention()`` / ``ops.triangle_attention``,
        not a flag on this module."""
        variant_output = self.untransposed_output or self.dual_output
        if variant_output or self.to_gate.bias is not None or self.to_out.bias is not None:
            # RoseTTAFold3's biases, Chai-1's output variants: the H100 kernels for inference, else the torch statements
            from miniworld_engine.integrations import triattn_h100_sm90
            from miniworld_engine.kernels.triangle_attention.cuda import h100_dh

            with _nvtx_range(self.nvtx_name, self.nvtx_enabled):
                if not variant_output and triattn_h100_sm90.serves(self, pair, mask):
                    return triattn_h100_sm90.forward(self, pair, mask)
                if h100_dh.serves(self, pair, mask):
                    return h100_dh.forward(self, pair, mask)
            out = self._triton_variant(pair, mask)
            if out is None:
                out = self._biased_reference(pair, mask)
            if self.p_drop > 0.0 and self.training:
                out = out * self._make_drop_scale(pair, self.p_drop)
            return pair + out
        if self._backend == KernelBackend.ANTHROPIC:
            if self.anthropic_row.startswith("block:"):
                from miniworld_engine.integrations.anthropic import (
                    module_triangle_attention,
                )

                return module_triangle_attention(self, pair, mask)
            from miniworld_engine.integrations.anthropic import _inference

            _inference()
            if self.training and self.p_drop:
                raise RuntimeError(
                    "Anthropic TriangleAttention requires eval() when dropout is enabled"
                )
        # the B200 kernels project the ending node's bias from the transposed pair only (transposed_bias=True)
        if not self._bias_untransposed and _b200.kind_of(self, pair, mask) is not None:
            with _nvtx_range(self.nvtx_name, self.nvtx_enabled):
                return _b200.forward(self, pair, mask)
        from miniworld_engine.integrations import triattn_h100, triattn_h100_sm90

        if triattn_h100_sm90.serves(self, pair, mask):
            with _nvtx_range(self.nvtx_name, self.nvtx_enabled):
                return triattn_h100_sm90.forward(self, pair, mask)
        from miniworld_engine.kernels.triangle_attention.cuda import h100_dh

        if h100_dh.head_dim(self) == 64 and h100_dh.serves(self, pair, mask):  # 64-channel heads (Chai-1): the sm90 kernels take 16 / 32
            with _nvtx_range(self.nvtx_name, self.nvtx_enabled):
                return h100_dh.forward(self, pair, mask)
        requested_h100 = not self._bias_untransposed and (
            triattn_h100.native_requested(self, pair)
            or getattr(self, "_h100_b200style_cuda", False)
        )
        if requested_h100:
            if triattn_h100.serves(self, pair, mask):
                with _nvtx_range(self.nvtx_name, self.nvtx_enabled):
                    return triattn_h100.forward(self, pair, mask)
            # A call outside the native contract (heads of 64+ channels, an unqualified bucket, ...) falls back to the
            # composed path below; only an explicitly forced native route (the candidate flag) keeps the error.
            if getattr(self, "_h100_b200style_cuda", False):
                raise ValueError(
                    "requested H100 CUDA TriangleAttention does not support this shape or dtype"
                )
        if (
            not self._bias_untransposed
            and getattr(self, "_fuse_front_backward", True)
            and getattr(self, "_fuse_projection_backward", True)
            and torch.is_grad_enabled()
            and self._backend == KernelBackend.PORTABLE
            and self.use_self_attention
            and not self.use_qk_norm
            and self.n_head == 4
            and _bo_dispatch.use_kernels(pair.shape[1])
        ):
            from miniworld_engine.kernels.triangle_attention.cuda import ln_backward

            weights = (
                self.to_query.weight,
                self.to_key.weight,
                self.to_value.weight,
                self.to_gate.weight,
                self.to_bias.weight,
            )
            if ln_backward.can_use(
                pair, weights, self.ln_pair.weight, self.ln_pair.bias
            ):
                with _nvtx_range(self.nvtx_name, self.nvtx_enabled):
                    return ln_backward.forward(self, pair, mask)
        out = self._attention(pair, mask)
        if self.p_drop > 0.0 and self.training:
            out = out * self._make_drop_scale(pair, self.p_drop)
        return pair + out

    def _triton_variant(self, pair: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor | None:
        """The variant update (no residual) on the portable Triton core (``kernels.triangle_attention.triton.variants``), or None when
        this call is not served there (the caller runs ``_biased_reference``). Served on every card the standard Triton leg is: a card with
        a record for the op (``dispatch.call_backend``), and any card under ``force_portable_kernels``. bf16 only; forward and backward."""
        if (
            _dispatch.call_backend(self, self._backend, pair) != KernelBackend.PORTABLE
            or self.implementation != ImplementationType.MINIWORLD
            or not self.use_self_attention
            or _dispatch.restricted_card(pair.device)
        ):
            return None
        from miniworld_engine.kernels.triangle_attention.triton.variants import (
            refusal,
            triton_triangle_attention_variant,
        )

        weights = (
            self.to_query.weight, self.to_key.weight, self.to_value.weight, self.to_bias.weight, self.to_gate.weight,
            self.to_out.weight, self.to_gate.bias, self.to_out.bias,
            *((self.to_out_t.weight, self.to_out_t.bias) if self.dual_output else ()),
        )
        why = refusal(pair, weights, self.n_head, self.to_query.weight.shape[0], use_qk_norm=self.use_qk_norm)
        if why is not None:
            if pair.is_cuda:
                _dispatch.warn_refused("triangle_attention", pair.device, why)
            return None
        return triton_triangle_attention_variant(
            pair, mask, self.ln_pair.weight, self.ln_pair.bias, self.to_query.weight, self.to_key.weight, self.to_value.weight,
            self.to_bias.weight, self.to_gate.weight, self.to_gate.bias, self.to_out.weight, self.to_out.bias,
            self.to_out_t.weight if self.dual_output else None, self.to_out_t.bias if self.dual_output else None,
            n_head=self.n_head, starting=self.starting, transposed_bias=self.transposed_bias,
            untransposed_output=self.untransposed_output, dual_output=self.dual_output, eps=float(self.ln_pair.eps),
        )

    def _biased_reference(
        self,
        pair: Float[torch.Tensor, "B L L d_pair"],
        mask: Bool[torch.Tensor, "B L"] | None = None,
    ) -> Float[torch.Tensor, "B L L d_pair"]:
        """The triangle-attention update (no residual) with the gate / out-projection biases, in torch (products in the parameters'
        dtype, softmax in fp32). The ending node attends over the transposed pair; its bias comes from the transposed pair
        (``transposed_bias``) or from the pair as it is."""
        B, L, _, _ = pair.shape
        H = self.n_head
        x = self.ln_pair(pair)
        xt = x if self.starting else x.transpose(1, 2)
        bias = self.to_bias(x if (self.starting or not self.transposed_bias) else xt).permute(0, 3, 1, 2)  # [B, H, L, L]
        q, k, v = (proj(xt).view(B, L, L, H, -1) for proj in (self.to_query, self.to_key, self.to_value))
        d = q.shape[-1]
        acc = torch.promote_types(pair.dtype, torch.float32)
        logits = torch.einsum("bijhd,bikhd->bihjk", q.to(acc), k.to(acc)) * d**-0.5 + bias.to(acc)[:, None]
        if mask is not None:
            logits = logits.masked_fill(~mask.bool()[:, None, None, None, :], float("-inf"))
        o = torch.einsum("bihjk,bikhd->bijhd", torch.nan_to_num(logits.softmax(-1)), v.to(acc)).reshape(B, L, L, -1).to(pair.dtype)
        u = torch.sigmoid(self.to_gate(xt)) * o
        out = self.to_out(u)
        if self.dual_output:
            other = self.to_out_t(u)
            return out + other.transpose(1, 2) if self.starting else out.transpose(1, 2) + other
        return out if self.starting or self.untransposed_output else out.transpose(1, 2)

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
                _dispatch.call_backend(self, self._backend, pair), pair.dtype, op="TriangleAttention"
            )
            master_family = backend  # the fp32-master projections below (cuBLAS casts, no Triton) stay on for the engine family
            if (
                backend == KernelBackend.PORTABLE
                and self.implementation == ImplementationType.MINIWORLD
                and _dispatch.restricted_card(pair.device)
            ):
                backend = KernelBackend.PYTORCH  # H100 / B200: no Triton fallback; the native CUDA routes were tried in forward()
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
                and backend == KernelBackend.PORTABLE
                and _bo_dispatch.use_kernels(pair.shape[1])
                and _bo_dispatch.use_infer_concat(self.to_value.weight.shape[0])
            ):
                out = self._bias_only_inference(pair, mask)
                if not self.starting:
                    out = rearrange(out, "B J I D -> B I J D").contiguous()
                return out

            pair = self._layernorm(pair, backend)
            master = (
                master_family == KernelBackend.PORTABLE
                and pair.dtype == torch.bfloat16
                and self.to_value.weight.dtype == torch.float32
                and _master.is_h100(pair.device)
            )
            if master:
                wv, wb, wg, wo = _master.pack(
                    [
                        self.to_value.weight,
                        self.to_bias.weight,
                        self.to_gate.weight,
                        self.to_out.weight,
                    ]
                )
                if self.use_self_attention:
                    wq, wk = _master.pack([self.to_query.weight, self.to_key.weight])
            fused_projection = False
            if (
                getattr(self, "_fuse_projection_backward", True)
                and torch.is_grad_enabled()
                and backend == KernelBackend.PORTABLE
                and self.use_self_attention
                and not self.use_qk_norm
                and self.n_head == 4
            ):
                from miniworld_engine.kernels.triangle_attention.cuda import (
                    can_use,
                    projections,
                )

                weights = (
                    self.to_query.weight,
                    self.to_key.weight,
                    self.to_value.weight,
                    self.to_gate.weight,
                    self.to_bias.weight,
                )
                fused_projection = can_use(pair, weights)
            if fused_projection:
                query, key, value, gate, bias = projections(pair, *weights)
            else:
                value = (
                    _master.linear(pair, self.to_value, wv)
                    if master
                    else self.to_value(pair)
                )
                bias = (
                    _master.linear(pair, self.to_bias, wb)
                    if master
                    else self.to_bias(pair)
                )

            # No .contiguous(): the bias-only einsum and the self-attention kernels
            # consume these strided views directly (the triton kernel re-packs
            # internally), so the explicit permute copy is pure overhead.
            value = rearrange(value, "B L L2 (H D) -> B H L L2 D", H=self.n_head)
            bias = rearrange(bias, "B L L2 H -> B H L L2")
            if self._bias_untransposed:
                # AF3: the bias of the pair as it is, read in the transposed frame (see ``transposed_bias``).
                bias = bias.transpose(-1, -2).contiguous()
            if mask is not None:
                bias = bias.masked_fill(
                    ~mask[:, None, None, :], torch.finfo(bias.dtype).min
                )

            if self.use_self_attention:
                if not fused_projection:
                    query = (
                        _master.linear(pair, self.to_query, wq)
                        if master
                        else self.to_query(pair)
                    )
                    key = (
                        _master.linear(pair, self.to_key, wk)
                        if master
                        else self.to_key(pair)
                    )

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
            if backend == KernelBackend.PORTABLE and _bo_dispatch.use_kernels(
                pair.shape[1]
            ):
                # Fuse sigmoid(to_gate(pair)) * out + the to_out projection (gated
                # tensor never hits HBM). Backend chosen per-GPU in _gate_out: fused
                # GEMM at small DH, split (sigmoid*mul + cuBLAS to_out) at large DH.
                gate = (
                    gate
                    if fused_projection
                    else (
                        _master.linear(pair, self.to_gate, wg)
                        if master
                        else self.to_gate(pair)
                    )
                )
                if master and not _bo_dispatch.gate_use_fused(
                    gate.shape[-1],
                    self.to_out.weight.shape[0],
                    math.prod(gate.shape[:-1]),
                    gate.device,
                    gate.dtype,
                ):
                    out = _master.linear(
                        kernels.sigmoid_gate_fused(gate, out), self.to_out, wo
                    )
                else:
                    out = self._gate_out(gate, out)
            else:
                gate = (
                    _master.linear(pair, self.to_gate, wg)
                    if master
                    else self.to_gate(pair)
                )
                out = sigmoid_gate(gate, out)
                out = (
                    _master.linear(out, self.to_out, wo) if master else self.to_out(out)
                )

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
        """Forward pass. ALWAYS returns the residual output ``pair + pair_attention(pair)``."""
        pair_res = pair  # residual == the ORIGINAL input (before the transpose / ln_pair rebind `pair`)
        if not self.starting:
            pair = pair.transpose(1, 2).contiguous()
        pair = self.ln_pair(pair)
        value = self.to_value(pair)
        bias = self.to_bias(pair)

        value = value.view(
            value.shape[0], value.shape[1], value.shape[2], self.n_head, -1
        )
        if mask is not None:
            bias = bias.masked_fill(
                ~mask[:, None, :, None], torch.finfo(bias.dtype).min
            )

        attention = F.softmax(bias, dim=-2)
        out = torch.einsum("bjkh,bikhd->bijhd", attention, value).contiguous()
        out = out.view(out.shape[0], out.shape[1], out.shape[2], -1)

        out = sigmoid_gate(self.to_gate(pair), out)
        out = self.to_out(out)

        if not self.starting:
            out = out.transpose(1, 2).contiguous()
        return pair_res + out
