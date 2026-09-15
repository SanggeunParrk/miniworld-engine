
# vendored from team-gm psk/benchmark : src/team_gm/modules/layers/triangle_updates.py
"""Triangle multiplicative update — the model-level op that connects the tm1 / tm2
fused kernels (and a cuequivariance baseline)."""

from contextlib import contextmanager

import torch
import torch.nn as nn
from jaxtyping import Bool, Float

from miniworld_engine import kernels
from miniworld_engine._typecheck import typecheck
from miniworld_engine.modules import dispatch as _dispatch
from miniworld_engine.modules.dispatch import (
    KernelBackend,
    resolve_triangle_multiplication,
)
from miniworld_engine.modules.dispatch import (
    trimul_out_layout as _resolve_trimul_out_layout,
)
from miniworld_engine.modules.exceptions import (
    ImplementationType,
    InvalidImplementationError,
)
from miniworld_engine.modules.functional import sigmoid_gate
from miniworld_engine.modules.primitives import LayerNorm, Linear

_CUTE_FNS = None


def _load_cute_fns():
    """Lazily import the CuTeDSL kernels used by the cute composition.

    Kept lazy so importing this module doesn't require the cute toolchain
    (cutlass-dsl + quack), which only exists in the dedicated cute env. The cute
    kernels live under ``kernels/{tm1,tm2,fused_ln_mask}/cute`` and cross-import
    by bare name, so we put those dirs on ``sys.path`` first.
    """
    global _CUTE_FNS
    if _CUTE_FNS is not None:
        return _CUTE_FNS
    import sys
    from pathlib import Path

    src_root = Path(__file__).resolve()
    while src_root.name != "src" and src_root.parent != src_root:
        src_root = src_root.parent
    for d in (
        src_root / "miniworld_engine" / "kernels" / "tm1" / "cute",
        src_root / "miniworld_engine" / "kernels" / "tm2" / "cute",
        src_root / "miniworld_engine" / "kernels" / "fused_ln_mask" / "cute",
    ):
        if str(d) not in sys.path:
            sys.path.insert(0, str(d))
    # These two resolve only through the sys.path entries added just above, so the `ty: ignore`
    # has to sit on the `from` line -- an import sorter that wraps the statement in parentheses
    # moves the comment onto the name and the suppression stops applying.
    from fused_ln_mask import fused_ln_mask  # ty: ignore[unresolved-import]
    from launch import tm1_cute_forward  # ty: ignore[unresolved-import]

    from miniworld_engine.kernels.layernorm.triton.transpose import layer_norm_transpose

    # NOTE: tm2 (cuequiv-backed gated GEMM) is NOT imported here — the default cute path is
    # cuequiv-free. The legacy cuequiv tm2 path lazy-imports it itself (see _forward_cute).
    _CUTE_FNS = (tm1_cute_forward, fused_ln_mask, layer_norm_transpose)
    return _CUTE_FNS


@contextmanager
def _nvtx_range(name: str, enabled: bool):
    if enabled:
        torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        if enabled:
            torch.cuda.nvtx.range_pop()


def _require_square_widths(kind: str, d_pair: int, d_hidden: int) -> None:
    """Refuse ``d_hidden != d_pair`` at the module boundary, for the kernel paths.

    Every fused trimul back half in this repo folds ``LN(tri)`` and the output gate on ``x_n``
    into ONE pass, which makes tri's channel axis and the pair width the SAME axis. When they
    differ there is no correct thing for those kernels to compute.

    ``trimul_triton`` says so and raises (unidirectional.py). The cute paths did not, and one of
    them -- ``_forward_cute_free`` -- reached ``trimul_back_triton`` with a ``(d_hidden, d_pair)``
    weight where the kernel indexes ``(K, N) = (d_pair, d_pair)``, i.e. an out-of-bounds READ of
    the missing rows. It did not reliably crash: the read lands inside whatever the caching
    allocator holds next, so it usually returned another tensor's bytes and corrupted the result
    silently, and only trapped when it cleared the mapping. Both the A100 and the A6000 session
    chased it as an "illegal memory access" that appeared at whatever kernel ran next.

    So the check belongs HERE, once, in front of every kernel path -- not in each kernel, where
    it can only be a last line of defence (``trimul_back_triton`` now validates too).
    The pytorch and cuequivariance backends are shape-general and never reach this.
    """
    if d_hidden != d_pair:
        msg = (f"{kind}: d_hidden ({d_hidden}) != d_pair ({d_pair}) is not supported by the "
               f"fused trimul kernels. Their back half normalises the contraction over tri's "
               f"channel axis and gates on the pair over the same axis, so the two widths are "
               f"one axis. Use implementation='pytorch' for asymmetric widths.")
        raise ValueError(msg)


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

        if implementation == ImplementationType.CUTE:
            _load_cute_fns()

    def _kernel_tm1(
        self, pair: torch.Tensor, backend: KernelBackend
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if backend == KernelBackend.PYTORCH:
            left = sigmoid_gate(self.to_left_gate(pair), self.to_left(pair))
            right = sigmoid_gate(self.to_right_gate(pair), self.to_right(pair))
            return left, right

        if backend == KernelBackend.TRITON:
            return kernels.triton_tm1(
                pair,
                self.to_left.weight.T,
                self.to_left_gate.weight.T,
                self.to_right.weight.T,
                self.to_right_gate.weight.T,
            )

        raise InvalidImplementationError(self.implementation)

    def _kernel_tm2(
        self, pair: torch.Tensor, out: torch.Tensor, backend: KernelBackend
    ) -> torch.Tensor:
        if backend == KernelBackend.PYTORCH:
            return sigmoid_gate(self.to_gate(pair), self.to_out(out))

        if backend == KernelBackend.TRITON:
            return kernels.triton_tm2(
                pair,
                out,
                self.to_gate.weight.T,
                self.to_out.weight.T,
            )

        raise InvalidImplementationError(self.implementation)

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
            backend = _dispatch.guard_dtype(
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

            if backend == KernelBackend.CUEQUIVARIANCE:
                return _r(self._forward_cuequivariance(pair, mask))

            if backend == KernelBackend.CUTE:
                # The cute inference path (_forward_cute) is forward-only (no saved
                # stats / no autograd graph). Under grad (training) OR whenever a dropout
                # scale is live, dispatch to the autograd-capable sm100/sm90 v6 merged
                # training kernel (it is the path that consumes _ds) — keeping all backend
                # selection inside the module.
                if torch.is_grad_enabled() or _ds is not None:
                    # training: fuse residual + row-broadcast dropout into the v6 back (sm90).
                    return self._forward_cute_train(pair, mask, _ds)
                return self._forward_cute(pair, mask)

            if backend == KernelBackend.TRITON:
                # Fused BDLL pipeline (mirrors cute's single-direction dispatch):
                # LN_in -> gated BDLL front (transposed store, no permute) -> ONE
                # bmm contraction -> te-style LN_out+@Wp -> triton output gate. One
                # code path serves inference (forward-only) and training (merged
                # autograd Function). Requires d_hidden == d_pair. See
                # kernels/trimul_inproj/triton/unidirectional.py.
                # residual + row-broadcast dropout are now FUSED into the triton gate store
                # (same gate_elem epilogue the cute path uses) — no external _r() add.
                return self._forward_triton(pair, mask, _ds)

            pair = self.ln_pair(pair)
            left, right = self._kernel_tm1(pair, backend)

            if mask is not None:
                mask_2d = mask.unsqueeze(-1) & mask.unsqueeze(-2)
                left = left * mask_2d[..., None]
                right = right * mask_2d[..., None]

            if self.outgoing:
                out = torch.einsum("bikd,bjkd->bijd", left, right)
            else:
                out = torch.einsum("bkid,bkjd->bijd", left, right)

            out = self.ln_out(out)
            return _r(self._kernel_tm2(pair, out, backend))

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
        mirroring cute's dispatch (LN_in -> gated BDLL front -> ONE bmm contraction ->
        te-style LN_out+@Wp -> triton output gate). Same code path for inference and
        training; the front emits left/right in channel-major BDLL directly (transposed
        store, no permute) so the contraction lowers to a tensor-core cuBLAS bmm on
        contiguous operands. Requires ``d_hidden == d_pair`` (the front produces per-side
        width d_hidden). bf16 / fp32, B>=1. See
        kernels/trimul_inproj/triton/unidirectional.py."""
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
        from cuequivariance_torch import triangle_multiplicative_update

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

    def _forward_cute_train(
        self,
        pair: torch.Tensor,
        mask: torch.Tensor | None,
        dropscale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """MINIWORLD (ours) TRAINING path: the v6 merged trimul training kernel
        (fwd+bwd, autograd-capable) — sm_100 ``V6TriMulMergedSm100`` on Blackwell,
        else sm90 ``V6TriMulMerged``. Built lazily from this module's own weights and
        cached (the cute inference kernels have no backward). bf16.

        The residual and ``dropscale`` fuse into the sm90 v6 back's store epilogue; on sm100
        (no fused path yet) they are applied explicitly."""
        _require_square_widths(
            type(self).__name__, pair.shape[-1], self.d_hidden)
        if _dispatch.is_sm90(pair.device):
            from miniworld_engine.kernels.trimul_inproj.cute.launch import (
                prepack_lr_operand,
            )
            from miniworld_engine.kernels.trimul_inproj.cute.v6_training_merged import (
                v6_forward_merged,
            )

            wl, wlg = self.to_left.weight.t(), self.to_left_gate.weight.t()
            wr, wrg = self.to_right.weight.t(), self.to_right_gate.weight.t()
            row_scale = None if mask is None else (mask.unsqueeze(-1) & mask.unsqueeze(-2)).reshape(-1)
            return v6_forward_merged(
                pair, wl, wlg, wr, wrg, self.to_gate.weight.t(), self.to_out.weight,
                self.ln_pair.weight, self.ln_pair.bias, self.ln_out.weight, self.ln_out.bias,
                self.ln_pair.eps, prepack_lr_operand(wl, wlg, wr, wrg),
                "out" if self.outgoing else "in", row_scale, dropscale=dropscale,
                eps_out=self.ln_out.eps)
        impl = getattr(self, "_train_impl", None)
        if impl is None:
            direction = "out" if self.outgoing else "in"
            self._train_fused = not _dispatch.is_sm100(pair.device)  # sm90 v6 fuses residual+dropout
            if _dispatch.is_sm100(pair.device):
                # v6_merged sm100 — the faster single-direction training kernel
                # (cuBLAS-centric merged backward; beats train_b200 v14 by 1.3x+ at
                # L<=1024). Baseline for further optimization.
                from miniworld_engine.kernels.trimul_inproj.cute.v6_training_merged_sm100 import (
                    V6TriMulMergedSm100 as _Impl,
                )
            else:
                from miniworld_engine.kernels.trimul_inproj.cute.v6_training_merged import (
                    V6TriMulMerged as _Impl,
                )
            # Built from this module's params (copied into the kernel's packed
            # layout); a benchmark/forward-eval wrapper, not a param-sharing one.
            impl = _Impl(self, direction=direction).to(pair.device)
            self._train_impl = impl
        if self._train_fused:
            return impl(pair, mask, dropscale=dropscale)
        # sm100 v6 (tcgen05): no fused residual+dropout path -> apply explicitly.
        out = impl(pair, mask)
        if dropscale is not None:
            out = out * dropscale
        return out + pair

    # No wrapper: every launch reachable from here is an ``opaque`` op at its own definition,
    # so Dynamo traces straight through this. It could never have BEEN an op itself -- see
    # ``kernels._compile`` -- but it does not need to be.
    def _forward_cute(
        self,
        pair: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """CuTeDSL path: connects the tm1 / tm2 / fused-LN cute kernels.

            fused_ln(+mask) -> tm1_cute (bdll) -> bmm contraction
                -> fused LN(dbn->bnd) -> tm2_cute

        Requires the cute env (cutlass-dsl + quack). Outgoing direction only.
        """
        _require_square_widths(
            type(self).__name__, pair.shape[-1], self.d_hidden)
        # cuequiv-FREE, unconditionally: our from-scratch tm2 (tm2_dual_from_scratch) is the tm2
        # kernel — the legacy cuequiv A/B path (and the MINIWORLD_TRIMUL_CUEQUIV_FREE gate) were
        # DELETED 2026-08-04. cuequivariance is a comparison-only baseline (pyproject [baselines]).
        return self._forward_cute_free(pair, mask)

    # No wrapper: every launch reachable from here is an ``opaque`` op at its own definition,
    # so Dynamo traces straight through this. It could never have BEEN an op itself -- see
    # ``kernels._compile`` -- but it does not need to be.
    def _forward_cute_free(
        self,
        pair: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """CUEQUIV-FREE cute path (B200 / sm_100), mirroring the H100 trimul_inproj
        design:

            LN_in(triton) -> tm1 front (bdll_sm100, ours tcgen05) -> einsum(cuBLAS)
              -> sm100 LayerNormLinear (triton M-major LN + tm1 tcgen05 proj GEMM)
              -> triton GateElem

        NO cuequiv kernels on this path (verified by nsys: no layer_norm_transpose /
        fused_sigmoid_gated_dual_gemm). Selected by default on sm_100; set
        MINIWORLD_TRIMUL_CUEQUIV_FREE=0 to fall back to the cuequiv-reusing
        _forward_cute for comparison. B=1, bf16.
        """
        _require_square_widths(
            type(self).__name__, pair.shape[-1], self.d_hidden)
        from miniworld_engine.kernels.layernorm.triton.main import triton_layernorm
        tm1_cute_forward, _fused_ln_mask, _lnt = _load_cute_fns()

        x_normed = triton_layernorm(
            pair, self.ln_pair.weight, self.ln_pair.bias, self.ln_pair.eps)

        left_bdll, right_bdll = tm1_cute_forward(
            x_normed,
            self.to_left.weight.T,
            self.to_left_gate.weight.T,
            self.to_right.weight.T,
            self.to_right_gate.weight.T,
            out_layout=_resolve_trimul_out_layout(pair.device),
        )
        if mask is not None:
            scale = (mask.unsqueeze(-1) & mask.unsqueeze(-2))[:, None]
            left_bdll, right_bdll = left_bdll * scale, right_bdll * scale
        if self.outgoing:
            tri = torch.einsum("bdik,bdjk->bdij", left_bdll, right_bdll)  # (B,D,L,L)
        else:
            tri = torch.einsum("bdki,bdkj->bdij", left_bdll, right_bdll)
        # Back-half (LN_out + proj + output-gate), all ours. sm100 keeps its tcgen05-tuned
        # split; sm90 uses the fused triton back (also folds LN_out -> no dbn->bnd LN needed).
        if _dispatch.is_sm100(pair.device):
            from miniworld_engine.kernels.trimul_inproj.cute.back_split_sm100 import (
                trimul_back_split_sm100,
            )
            return trimul_back_split_sm100(
                tri, x_normed, self.to_out.weight, self.to_gate.weight.T,
                self.ln_out.weight, self.ln_out.bias, self.ln_out.eps,
                residual=pair,  # FUSED into the gate store, as on every other back half
            )
        from miniworld_engine.kernels.trimul_inproj.triton.back import (
            trimul_back_triton,
        )
        return trimul_back_triton(
            tri, x_normed, self.to_out.weight.T, self.to_gate.weight.T,
            self.ln_out.weight, self.ln_out.bias, self.ln_out.eps,
            residual=pair,  # FUSED residual add in the store epilogue
        )
