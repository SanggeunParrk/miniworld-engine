# vendored from team-gm psk/benchmark : src/team_gm/modules/layers/triangle_updates.py
"""Triangle multiplicative update — the model-level op that connects the tm1 / tm2
fused kernels (and a cuequivariance baseline)."""

from contextlib import contextmanager

import torch
import torch.nn as nn
from cuequivariance_torch import triangle_multiplicative_update
from jaxtyping import Bool, Float

from miniworld_kernels import kernels
from miniworld_kernels._typecheck import typecheck
from miniworld_kernels.modules import dispatch as _dispatch
from miniworld_kernels.modules.dispatch import (
    KernelBackend,
    resolve_triangle_multiplication,
    trimul_out_layout as _resolve_trimul_out_layout,
)
from miniworld_kernels.modules.exceptions import (
    ImplementationType,
    InvalidImplementationError,
)
from miniworld_kernels.modules.ops import sigmoid_gate
from miniworld_kernels.modules.primitives import LayerNorm, Linear


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
        src_root / "miniworld_kernels" / "kernels" / "tm1" / "cute",
        src_root / "miniworld_kernels" / "kernels" / "tm2" / "cute",
        src_root / "miniworld_kernels" / "kernels" / "fused_ln_mask" / "cute",
    ):
        if str(d) not in sys.path:
            sys.path.insert(0, str(d))
    from cuequivariance_ops_torch.fused_layer_norm_torch import layer_norm_transpose
    from fused_ln_mask import fused_ln_mask  # pyright: ignore[reportMissingImports]
    from launch import tm1_cute_forward  # pyright: ignore[reportMissingImports]
    from tm2_cute import tm2_cute_forward  # pyright: ignore[reportMissingImports]

    _CUTE_FNS = (tm1_cute_forward, tm2_cute_forward, fused_ln_mask, layer_norm_transpose)
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
    ) -> None:
        super().__init__()
        self.outgoing = outgoing
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

        self.ln_pair = LayerNorm(d_pair, implementation=ln_implementation)
        self.to_left = Linear(d_pair, d_pair, bias=False, init="default")
        self.to_left_gate = Linear(d_pair, d_pair, bias=False, init="zero")
        self.to_right = Linear(d_pair, d_pair, bias=False, init="default")
        self.to_right_gate = Linear(d_pair, d_pair, bias=False, init="zero")

        self.ln_out = LayerNorm(d_pair, implementation=ln_implementation)
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
    ) -> Float[torch.Tensor, "B L L d_pair"]:
        """Forward pass. Routes on the resolved internal backend, degrading to the
        pytorch reference (with a warning) on a dtype the fused kernels can't run."""
        with _nvtx_range(self.nvtx_name, self.nvtx_enabled):
            backend = _dispatch.guard_dtype(
                self._backend, pair.dtype, op="TriangleMultiplication"
            )
            if backend == KernelBackend.CUEQUIVARIANCE:
                return self._forward_cuequivariance(pair, mask)

            if backend == KernelBackend.CUTE:
                # The cute inference path (_forward_cute) is forward-only (no saved
                # stats / no autograd graph). Under grad (training), dispatch to the
                # autograd-capable sm100/sm90 v6 merged training kernel instead —
                # keeping all backend selection inside the module.
                if torch.is_grad_enabled():
                    return self._forward_cute_train(pair, mask)
                return self._forward_cute(pair, mask)

            if backend == KernelBackend.TRITON:
                # Fused BDLL pipeline (mirrors cute's single-direction dispatch):
                # LN_in -> gated BDLL front (transposed store, no permute) -> ONE
                # bmm contraction -> te-style LN_out+@Wp -> triton output gate. One
                # code path serves inference (forward-only) and training (merged
                # autograd Function). Requires d_hidden == d_pair. See
                # kernels/trimul_inproj/triton/unidirectional.py.
                return self._forward_triton(pair, mask)

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
            return self._kernel_tm2(pair, out, backend)

    @torch.compiler.disable
    def _forward_triton(
        self,
        pair: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """TRITON single-direction path (fwd + autograd bwd) — the fused BDLL pipeline
        mirroring cute's dispatch (LN_in -> gated BDLL front -> ONE bmm contraction ->
        te-style LN_out+@Wp -> triton output gate). Same code path for inference and
        training; the front emits left/right in channel-major BDLL directly (transposed
        store, no permute) so the contraction lowers to a tensor-core cuBLAS bmm on
        contiguous operands. Requires ``d_hidden == d_pair`` (the front produces per-side
        width d_hidden). bf16 / fp32, B>=1. See
        kernels/trimul_inproj/triton/unidirectional.py."""
        from miniworld_kernels.kernels.trimul_inproj.triton.unidirectional import (
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
        )

    def _forward_cuequivariance(
        self,
        pair: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
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

    def _forward_cute_train(
        self,
        pair: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """MINIWORLD (ours) TRAINING path: the v6 merged trimul training kernel
        (fwd+bwd, autograd-capable) — sm_100 ``V6TriMulMergedSm100`` on Blackwell,
        else sm90 ``V6TriMulMerged``. Built lazily from this module's own weights and
        cached (the cute inference kernels have no backward). bf16."""
        impl = getattr(self, "_train_impl", None)
        if impl is None:
            direction = "out" if self.outgoing else "in"
            if _dispatch.is_sm100(pair.device):
                # v6_merged sm100 — the faster single-direction training kernel
                # (cuBLAS-centric merged backward; beats train_b200 v14 by 1.3x+ at
                # L<=1024). Baseline for further optimization.
                from miniworld_kernels.kernels.trimul_inproj.cute.v6_training_merged_sm100 import (  # noqa: E501
                    V6TriMulMergedSm100 as _Impl,
                )
            else:
                from miniworld_kernels.kernels.trimul_inproj.cute.v6_training_merged import (  # noqa: E501
                    V6TriMulMerged as _Impl,
                )
            # Built from this module's params (copied into the kernel's packed
            # layout); a benchmark/forward-eval wrapper, not a param-sharing one.
            impl = _Impl(self, direction=direction).to(pair.device)
            self._train_impl = impl
        return impl(pair, mask)

    @torch.compiler.disable
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
        import os as _os
        _free_default = "1" if _dispatch.is_sm100(pair.device) else "0"
        if _os.environ.get("MINIWORLD_TRIMUL_CUEQUIV_FREE", _free_default) != "0":
            return self._forward_cute_free(pair, mask)
        tm1_cute_forward, tm2_cute_forward, fused_ln_mask, layer_norm_transpose = (
            _load_cute_fns()
        )
        x = pair
        b, l1, l2, d = x.shape
        ln_in_w, ln_in_b = self.ln_pair.weight, self.ln_pair.bias
        ln_out_w, ln_out_b = self.ln_out.weight, self.ln_out.bias

        if mask is not None:
            mask_2d = mask.unsqueeze(-1) & mask.unsqueeze(-2)
            x_normed = fused_ln_mask(x, ln_in_w, ln_in_b, mask_2d)
        else:
            out = layer_norm_transpose(
                x.reshape(b * l1 * l2, d), ln_in_w, ln_in_b, eps=self.ln_pair.eps, layout="nd->nd"
            )
            x_normed = (out[0] if isinstance(out, tuple) else out).view(b, l1, l2, d)

        # Zero-copy [B,D,L,L] store (no transpose). bdll_direct now writes the
        # side output straight into [B,D,L,L] via two non-gated M-major GEMMs +
        # a pointwise sigmoid-gate (quack 0.3.11's gated epilogue can't do an
        # M-major postact; its non-gated store can — see launch.py). This drops
        # the ~4.5ms permute the plain "bdll" path incurs at L=1024.
        left_bdll, right_bdll = tm1_cute_forward(
            x_normed,
            self.to_left.weight.T,
            self.to_left_gate.weight.T,
            self.to_right.weight.T,
            self.to_right_gate.weight.T,
            out_layout=_resolve_trimul_out_layout(pair.device),
        )
        if self.outgoing:
            tri_out_bdij = torch.einsum("bdik,bdjk->bdij", left_bdll, right_bdll)
        else:
            tri_out_bdij = torch.einsum("bdki,bdkj->bdij", left_bdll, right_bdll)
        tri_dbn = tri_out_bdij.permute(1, 0, 2, 3).reshape(d, b, l1 * l2)
        out = layer_norm_transpose(
            tri_dbn, ln_out_w, ln_out_b, eps=self.ln_out.eps, layout="dbn->bnd"
        )
        out_normed = (out[0] if isinstance(out, tuple) else out).view(b, l1, l2, d)
        # tm2 cute wants weights in (N, K) = nn.Linear form (already so).
        return tm2_cute_forward(x_normed, out_normed, self.to_gate.weight, self.to_out.weight)

    @torch.compiler.disable
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
        from miniworld_kernels.kernels.layernorm.triton.main import triton_layernorm
        from miniworld_kernels.kernels.trimul_inproj.cute.back_split_sm100 import (
            trimul_back_split_sm100,
        )
        tm1_cute_forward, _, fused_ln_mask, _ = _load_cute_fns()
        b, l1, l2, d = pair.shape

        if mask is not None:
            mask_2d = mask.unsqueeze(-1) & mask.unsqueeze(-2)
            x_normed = fused_ln_mask(pair, self.ln_pair.weight, self.ln_pair.bias, mask_2d)
        else:
            x_normed = triton_layernorm(
                pair.reshape(b * l1 * l2, d), self.ln_pair.weight, self.ln_pair.bias,
                self.ln_pair.eps,
            ).view(b, l1, l2, d)

        left_bdll, right_bdll = tm1_cute_forward(
            x_normed,
            self.to_left.weight.T,
            self.to_left_gate.weight.T,
            self.to_right.weight.T,
            self.to_right_gate.weight.T,
            out_layout=_resolve_trimul_out_layout(pair.device),
        )
        if self.outgoing:
            tri = torch.einsum("bdik,bdjk->bdij", left_bdll, right_bdll)  # (B,D,L,L)
        else:
            tri = torch.einsum("bdki,bdkj->bdij", left_bdll, right_bdll)
        return trimul_back_split_sm100(
            tri, x_normed, self.to_out.weight, self.to_gate.weight.T,
            self.ln_out.weight, self.ln_out.bias, self.ln_out.eps,
        )
