
# vendored from team-gm psk/benchmark : src/team_gm/modules/layers/triangle_updates.py
"""Triangle multiplicative update — the model-level op over the hand-CUDA H100 kernels, the
Triton fused pipeline (portable fallback), the PyTorch reference and the cuequivariance /
Anthropic comparison baselines."""

from contextlib import contextmanager

import torch
import torch.nn as nn
from jaxtyping import Bool, Float

from miniworld_engine._typecheck import typecheck
from miniworld_engine.integrations import anthropic_trimul as _anthropic
from miniworld_engine.integrations import trimul_b200 as _b200
from miniworld_engine.integrations import trimul_h100 as _h100
from miniworld_engine.integrations import trimul_sm80 as _sm80
from miniworld_engine.modules import dispatch as _dispatch
from miniworld_engine.modules.dispatch import (
    KernelBackend,
    resolve_triangle_multiplication,
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


class TriangleMultiplication(nn.Module):
    """Unified implementation of triangular multiplicative update.

    Parameters
    ----------
    d_pair : int
        Dimension of pair representation.
    d_hidden : int | None
        Hidden dimension for left/right projections. Defaults to ``d_pair`` when *None*.
        Not supported with the TRITON implementation.
    outgoing : bool
        Whether to use outgoing edges.
    implementation : ImplementationType
        Implementation to use.

    """

    def __init__(
        self,
        d_pair: int = 128,
        *,
        d_hidden: int | None = None,
        outgoing: bool = True,
        implementation: ImplementationType = ImplementationType.PYTORCH,
        ln_implementation: ImplementationType = ImplementationType.PYTORCH,
        p_drop: float = 0.25,
    ) -> None:
        super().__init__()
        self.outgoing = outgoing
        # ======================================================================================
        # THIS MODULE ALWAYS APPLIES THE RESIDUAL: y = pair + drop_row(trimul(pair)).
        # The residual connection is UNCONDITIONAL (AF3 pairformer default
        # ``pair = pair + drop_row(trimul(pair))``); residual connections are the standard in this
        # domain, so there is deliberately NO flag to turn it off. The row-broadcast DROPOUT is
        # OPTIONAL: ``p_drop`` (drop_row, broadcast_dim=1) is applied only in ``self.training``;
        # it DEFAULTS ON at AF3's 0.25, and at eval it is identity. The block just calls
        # ``module(pair, mask)``.
        #
        # WHY IT'S FUSED IN (SPEED): both the residual add AND the dropout scale are done INSIDE
        # the trimul back/gate kernel's output epilogue — not as separate ``out*ds + pair``
        # elementwise ops. That removes extra kernel launches and their [B,L,L,D] HBM round-trips
        # (the gate output + residual input are already resident at store time), which is the whole
        # point of the fusion. Making the residual unconditional is what lets the kernel own that
        # fused epilogue; a runtime residual toggle would force the slow separate-add path.
        # >>> There is no way to turn it off, not even by editing a local: the residual is part
        # >>> of what this module IS. For the raw op in isolation -- benchmarking, or a caller
        # >>> that owns its own residual -- use ``ops.triangle_multiplicative_update``, the
        # >>> weights-as-args facade that mirrors cuequivariance's signature.
        # ======================================================================================
        self.p_drop = p_drop
        # 'miniworld' (auto) -> concrete backend for the running GPU arch. The
        # public option is kept on self.implementation; forward routes on _backend.
        self.implementation = ImplementationType(implementation)
        self._backend = resolve_triangle_multiplication(self.implementation)
        self.ln_implementation = ln_implementation
        direction = "outgoing" if outgoing else "incoming"
        self.nvtx_enabled = False
        self.nvtx_name = f"triangle_multiplication/{direction}"

        if d_hidden is None:
            d_hidden = d_pair

        if d_hidden != d_pair and implementation == ImplementationType.TRITON:
            msg = (
                f"d_hidden != d_pair ({d_hidden} != {d_pair}) is not "
                f"supported with TRITON implementation"
            )
            raise ValueError(msg)

        # The front projects d_pair -> d_hidden, and `to_out` projects d_hidden -> d_pair
        # (AF3 Alg. 12). These four were `Linear(d_pair, d_pair)`, which ignored `d_hidden`
        # entirely and left the module internally inconsistent the moment the two differed: the
        # front emitted d_pair channels and `to_out` expected d_hidden inputs. Every path broke
        # on it -- the pytorch reference with `mat1 and mat2 shapes cannot be multiplied`, and
        # the fused kernels with an out-of-bounds READ (the back half indexes `to_out.weight.T`
        # as (d_pair, d_pair) when it is (d_hidden, d_pair)), which corrupted silently far more
        # often than it trapped. `d_hidden` was a parameter this module accepted and did not
        # implement. With d_hidden == d_pair -- every shape the model runs -- nothing changes.
        self.d_hidden = d_hidden
        self.ln_pair = LayerNorm(d_pair, implementation=ln_implementation)
        self.to_left = Linear(d_pair, d_hidden, bias=False, init="default")
        self.to_left_gate = Linear(d_pair, d_hidden, bias=False, init="zero")
        self.to_right = Linear(d_pair, d_hidden, bias=False, init="default")
        self.to_right_gate = Linear(d_pair, d_hidden, bias=False, init="zero")

        # LN_out normalises the CONTRACTION output, whose width is d_hidden, not d_pair.
        self.ln_out = LayerNorm(d_hidden, implementation=ln_implementation)
        self.to_gate = Linear(d_pair, d_pair, bias=False, init="zero")
        self.to_out = Linear(d_hidden, d_pair, bias=False, init="zero")

    @typecheck
    def forward(
        self,
        pair: Float[torch.Tensor, "B L L d_pair"],
        mask: Bool[torch.Tensor, "B L"] | None = None,
        dropout_p: float | None = None,
    ) -> Float[torch.Tensor, "B L L d_pair"]:
        """Forward pass. ALWAYS returns the residual output ``pair + drop_row(trimul(pair))``.
        Routes on the resolved internal backend, degrading to the pytorch reference (with a
        warning) on a dtype the fused kernels can't run.

        The residual is UNCONDITIONAL and fused into the kernel epilogue FOR SPEED (see the
        constructor comment) — there is intentionally no flag to disable it (residual is the
        domain standard). The row-broadcast DROPOUT is OPTIONAL: ``dropout_p`` overrides the
        instance ``p_drop`` per call (None -> ``self.p_drop``) and is active only in
        ``self.training`` (standard nn.Module semantics); inference is identity.
        >>> The raw op without the residual is ``ops.triangle_multiplicative_update``,
        not a flag on this module."""
        dropout_p = self.p_drop if dropout_p is None else dropout_p
        with _nvtx_range(self.nvtx_name, self.nvtx_enabled):
            from miniworld_engine.integrations import a100_families
            backend = self._backend if a100_families.serves(self, pair) else _dispatch.guard_dtype(
                self._backend, pair.dtype, op="TriangleMultiplication"
            )
            _pair_in = pair  # original (pre-LN) input == the residual; torch path rebinds `pair`
            # row-broadcast dropout scale (== drop_row mask/(1-p), [B,1,L,D]); training only.
            _ds = (self._make_drop_row_scale(pair, dropout_p)
                   if dropout_p and dropout_p > 0.0 and self.training else None)
            def _r(out):  # explicit residual+dropout for paths that don't fold it in-kernel
                if _ds is not None:
                    out = out * _ds
                return out + _pair_in

            # Hand-CUDA B200 inference (integrations.trimul_b200 states its contract).
            if _b200.serves_inference(self, pair, bidirectional=False, dropscale=_ds):
                return _b200.update_inference(self, pair, mask, _ds, bidirectional=False)
            if _b200.serves_train(self, pair, bidirectional=False):
                return _b200.update_train(self, pair, mask, _ds, bidirectional=False)
            # Hand-CUDA H100 kernels first (integrations.trimul_h100 states their contract).
            if _h100.serves_inference(self, pair, bidirectional=False, dropscale=_ds):
                return _h100.update_inference(self, pair, mask, bidirectional=False)
            if _h100.serves_single(self, pair):
                return _h100.update(self, pair, mask, _ds, bidirectional=False)
            # Hand-CUDA A100 kernels (integrations.trimul_sm80): inference and training, D128.
            if _sm80.serves(self, pair, mask):
                return _sm80.update(self, pair, mask, _ds, bidirectional=False)
            if a100_families.serves(self, pair):
                return a100_families.trimul_module(self, pair, mask, _ds)
            # The Anthropic TriMul payload, when TRIMUL_NATIVE_BUILD_DIR names one that can run this forward
            # (sm_90, bf16, one square plane, no grad, no live dropout scale, a unit for this width).  An explicit
            # `implementation="anthropic"` refuses with the reason; `miniworld` uses it where it fits and falls
            # back to the backends below where it does not.  See integrations.anthropic_trimul.
            if _anthropic.wanted(self.implementation):
                _native = {"grad": torch.is_grad_enabled(), "dropout": _ds is not None}
                if self.implementation == ImplementationType.ANTHROPIC:
                    _anthropic.require(pair, pair.shape[-1], self.d_hidden, **_native)   # explicit: the reason, never a reroute
                if _anthropic.serves(pair, pair.shape[-1], self.d_hidden, **_native):
                    return _anthropic.update_unidirectional(self, pair, mask)

            if backend == KernelBackend.CUEQUIVARIANCE:
                return _r(self._forward_cuequivariance(pair, mask))

            if backend == KernelBackend.TRITON:
                # Fused BDLL pipeline: LN_in -> gated BDLL front (transposed store, no permute) -> ONE
                # bmm contraction -> fused output stages. BF16 training uses
                # LN_out + F567; inference also folds LN_out into the back kernel.
                # Requires d_hidden == d_pair. See
                # kernels/trimul_inproj/triton/unidirectional.py.
                # residual + row-broadcast dropout are FUSED into the triton gate store
                # (gate_elem epilogue) — no external _r() add.
                return self._forward_triton(pair, mask, _ds)

            if backend != KernelBackend.PYTORCH:
                raise InvalidImplementationError(self.implementation)

            pair = self.ln_pair(pair)
            left = sigmoid_gate(self.to_left_gate(pair), self.to_left(pair))
            right = sigmoid_gate(self.to_right_gate(pair), self.to_right(pair))

            if mask is not None:
                mask_2d = mask.unsqueeze(-1) & mask.unsqueeze(-2)
                left = left * mask_2d[..., None]
                right = right * mask_2d[..., None]

            if self.outgoing:
                out = torch.einsum("bikd,bjkd->bijd", left, right)
            else:
                out = torch.einsum("bkid,bkjd->bijd", left, right)

            out = self.ln_out(out)
            return _r(sigmoid_gate(self.to_gate(pair), self.to_out(out)))

    # No wrapper: every launch reachable from here is an ``opaque`` op at its own definition,
    # so Dynamo traces straight through this. It could never have BEEN an op itself -- see
    # ``kernels._compile`` -- but it does not need to be.
    def _forward_triton(
        self,
        pair: torch.Tensor,
        mask: torch.Tensor | None,
        dropscale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """TRITON single-direction path (fwd + autograd bwd) — the fused BDLL pipeline
        (LN_in -> gated BDLL front -> ONE bmm contraction -> fused output stages).
        BF16 training shares F567 and the two backward fusions with bidirectional
        TriMul. The front emits left/right in channel-major BDLL directly (transposed
        store, no permute) so the contraction lowers to a tensor-core cuBLAS bmm on
        contiguous operands. Requires ``d_hidden == d_pair`` (the front produces per-side
        width d_hidden). bf16 / fp32. See kernels/trimul_inproj/triton/unidirectional.py.

        The Triton back half takes one square plane (B=1, ``trimul_back_triton``); a batched
        input is run plane by plane here so the module keeps its ``[B, L, L, d]`` contract."""
        if pair.shape[0] > 1:
            return torch.cat([
                self._forward_triton(
                    pair[i:i + 1],
                    None if mask is None else mask[i:i + 1],
                    None if dropscale is None else
                    (dropscale if dropscale.shape[0] == 1 else dropscale[i:i + 1]),
                ) for i in range(pair.shape[0])
            ], dim=0)
        from miniworld_engine.kernels.trimul_inproj.triton.unidirectional import (
            trimul_triton,
        )

        return trimul_triton(
            pair,
            self.to_left.weight, self.to_left_gate.weight,
            self.to_right.weight, self.to_right_gate.weight,
            self.to_gate.weight, self.to_out.weight,
            self.ln_pair.weight, self.ln_pair.bias,
            self.ln_out.weight, self.ln_out.bias,
            self.ln_pair.eps, self.ln_out.eps,
            self.to_left.weight.shape[0],   # d_hidden
            self.outgoing,
            mask=mask,
            dropscale=dropscale,
        )

    def _forward_cuequivariance(
        self,
        pair: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        # cuequiv backend (opt-in): lazy import so the default miniworld path never needs cuequiv.
        from cuequivariance_torch import (  # ty: ignore[unresolved-import]  # optional cuequivariance backend
            triangle_multiplicative_update,
        )

        mask_2d = None
        if mask is not None:
            mask_2d = mask.unsqueeze(-1) & mask.unsqueeze(-2)

        return triangle_multiplicative_update(
            pair,
            direction="outgoing" if self.outgoing else "incoming",
            mask=mask_2d,
            norm_in_weight=self.ln_pair.weight,
            norm_in_bias=self.ln_pair.bias,
            p_in_weight=torch.cat(
                [self.to_left.weight, self.to_right.weight],
                dim=0,
            ),
            g_in_weight=torch.cat(
                [self.to_left_gate.weight, self.to_right_gate.weight],
                dim=0,
            ),
            norm_out_weight=self.ln_out.weight,
            norm_out_bias=self.ln_out.bias,
            p_out_weight=self.to_out.weight,
            g_out_weight=self.to_gate.weight,
        )

    def _make_drop_row_scale(self, pair: torch.Tensor, p: float) -> torch.Tensor:
        """drop_row (``Dropout(broadcast_dim=1)``) scale [B,1,L,D] = (rand>p)/(1-p), broadcast
        over the i-index — matches modules.primitives.Dropout for the pair track."""
        b, _l1, l2, d = pair.shape
        keep = torch.rand(b, 1, l2, d, device=pair.device, dtype=pair.dtype) > p
        return keep.to(pair.dtype) / (1.0 - p)
