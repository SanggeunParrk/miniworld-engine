"""OuterProduct / OuterProductMean — MSA/single -> pair building blocks (ported from team-gm)."""

import torch
import torch.nn as nn
from einops import rearrange
from jaxtyping import Bool, Float, Int

from miniworld_engine.baselines import anthropic_msa as _anthropic
from miniworld_engine.kernels.integrations import opm_train as _opm_train
from miniworld_engine.modules import dispatch as _dispatch
from miniworld_engine.modules.dispatch import KernelBackend, resolve
from miniworld_engine.modules.exceptions import ImplementationType
from miniworld_engine.modules.primitives import (
    ImplementationSwitch,
    LayerNorm,
    Linear,
    MasterWeights,
)
from miniworld_engine.utils.typecheck import typecheck


class OuterProduct(nn.Module):
    """Outer product single represetation to pair representation.

    Parameters
    ----------
    d_single : int
        Dimension of single representation.
    d_pair : int
        Dimension of pair representation.
    d_hidden : int
        Dimension of hidden layer.

    """

    def __init__(
        self,
        d_single: int,
        d_pair: int,
        d_hidden: int = 32,
    ) -> None:
        super().__init__()

        self.ln_single = nn.LayerNorm(d_single)
        self.to_left = Linear(d_single, d_hidden, bias=False, init="default")
        self.to_right = Linear(d_single, d_hidden, bias=False, init="default")
        self.to_out = Linear(d_hidden * d_hidden, d_pair, bias=True, init="zero")

    @typecheck
    def forward(
        self,
        single: Float[torch.Tensor, "B L d_single"],
    ) -> Float[torch.Tensor, "B L L d_pair"]:
        """Forward pass."""
        single = self.ln_single(single)
        left = self.to_left(single)
        right = self.to_right(single)

        out = torch.einsum("bid,bje->bijde", left, right)
        out = rearrange(out, "B L1 L2 d1 d2 -> B L1 L2 (d1 d2)")
        return self.to_out(out)


class OuterProductMean(MasterWeights, ImplementationSwitch):
    """Outer product mean of MSA representation to pair representation.

    Normalization uses ``clamp(min=1)`` on the mask count instead of adding a
    small epsilon, following the Boltz approach.  AlphaFold3 and Protenix add
    ``eps`` (1e-3) to the denominator.

    ``normalize_before_proj=False`` matches ESMFold2: divide after the output
    projection so that its learned bias is divided by the valid-pair count too.
    The default preserves the AF3 ordering and checkpoint behavior.

    ``count_offset`` (default ``None``) switches the divisor to AlphaFold 3's ``count + count_offset`` (``1e-3`` in the released
    code); with ``normalize_before_proj=False`` that is AF3's ``(W out + b) / (count + 1e-3)``.

    ``count_offset`` changes only the per-pair divisor: the H100 CUDA paths (``kernels.integrations.opm_train``, ``kernels.outer_product_mean.composed``)
    take it as a pair of kernel scalars (``divisor = max(count, dmin) + doff``), inference and training; a shape they do not serve runs the PyTorch statements.

    Parameters
    ----------
    d_msa : int
        Dimension of MSA representation.
    d_pair : int
        Dimension of pair representation.
    d_hidden : int
        Dimension of hidden layer.

    """

    #: A numerically dead instance of a checkpoint (set by a model's checkpoint loader; the engine no longer ships that loader): inference returns the residual input.
    dead: bool = False

    def __init__(
        self,
        d_msa: int,
        d_pair: int,
        d_hidden: int = 32,
        *,
        mask_interchain: bool = False,
        normalize_before_proj: bool = True,
        projection_bias: bool = False,
        groups: int = 1,
        product_norm_eps: float = 0.1,
        count_offset: float | None = None,
        implementation: ImplementationType = ImplementationType.PYTORCH,
        master_weights: bool = False,
    ) -> None:
        super().__init__()
        # AlphaFold 3 / Protenix divisor: ``count + count_offset`` (1e-3 in the released code) instead of ``clamp(count, 1)``. None keeps
        # the clamp. The H100 CUDA paths take it (see the class docstring);
        # a call they do not serve REFUSES (a KernelRefusedWarning on a GPU) and runs the PyTorch statements.
        if count_offset is not None and groups > 1:
            raise ValueError("count_offset needs groups == 1: the grouped (Chai-1) form does not divide by the pair count")
        self.count_offset = count_offset

        self.mask_interchain = mask_interchain
        self.implementation = ImplementationType(implementation)
        # "anthropic" names an MSA payload, not a LayerNorm one: the primitives below are plumbing for
        # the fallback path and take the engine's own auto choice in that case (integrations.anthropic_msa).
        implementation = self._child_implementation(None, self.implementation)
        self.normalize_before_proj = normalize_before_proj
        # LN over the MSA feature dim — route to the fused miniworld_engine LN (bf16) under
        # the engine implementation; a raw nn.LayerNorm runs fp32-native under autocast and,
        # at MSA depth 2048, is the single biggest kernel in the MSA module.
        self.ln_msa = LayerNorm(d_msa, implementation=implementation)
        # RoseTTAFold3: the left / right projections carry a bias (``projection_bias``; masked rows lose it too). Chai-1: ``groups`` = 8
        # outer products of d_hidden / groups wide halves, their G (d_hidden / G)^2 products LayerNorm'd (``product_norm``, eps
        # ``product_norm_eps``) before the out-projection, no division by the pair counts. Both run kernels.outer_product_mean.composed
        # on H100 (inference), the portable Triton path (kernels.outer_product_mean.triton) on the other GPUs, and
        # ``_variant_reference`` where neither serves.
        if d_hidden % groups:
            raise ValueError(f"d_hidden ({d_hidden}) must be divisible by groups ({groups})")
        self.groups, self.group_width = groups, d_hidden // groups
        self.to_left = Linear(d_msa, d_hidden, bias=projection_bias)
        self.to_right = Linear(d_msa, d_hidden, bias=projection_bias)
        products = groups * self.group_width**2
        self.to_out = Linear(products, d_pair, bias=True, init="zero")
        if groups > 1:
            self.product_norm = nn.LayerNorm(products, eps=product_norm_eps)
        if master_weights:
            self.set_master_weights()

    def _child_implementation(self, child, implementation):
        # "anthropic" names an MSA payload, not a LayerNorm one: the norms take the engine's own auto choice
        return ImplementationType.MINIWORLD if implementation == ImplementationType.ANTHROPIC else implementation

    @classmethod
    def from_released(cls, state_dict, *, prefix: str = "", dtype: torch.dtype | None = None, norm_eps: float = 1e-5,
                      bias_after_norm: bool = False, clamped_norm: bool = False, **options) -> "OuterProductMean":
        """The module from a released checkpoint's outer product mean (``layer_norm_input`` / ``left,right_projection`` / ``output_w`` /
        ``output_b``; see ``modules.released``). The released divisors: AlphaFold 3's ``(W out + b) / (count + 1e-3)`` (default),
        ``bias_after_norm`` (``W out / (count + 1e-3) + b``), ``clamped_norm`` (``clamp(count, 1)`` instead of ``count + 1e-3``); a grouped
        (Chai-1) checkpoint does not divide. ``groups`` and ``projection_bias`` follow from the weights."""
        from miniworld_engine.modules import released

        converted = released.outer_product_mean_from_released(state_dict, prefix=prefix)
        w = released._sub(state_dict, prefix)["output_w"]
        groups = w.shape[0] if w.ndim == 4 else 1
        options.setdefault("projection_bias", "to_left.bias" in converted)
        options.setdefault("groups", groups)
        options.setdefault("normalize_before_proj", bias_after_norm)
        if groups == 1:
            options.setdefault("count_offset", None if clamped_norm else 1e-3)
        d_hidden, d_msa = converted["to_left.weight"].shape
        d_pair = converted["to_out.weight"].shape[0]
        return released.load(cls(d_msa, d_pair, d_hidden, **options), converted, dtype=dtype, norm_eps=norm_eps)

    def _variant_reference(self, msa, mask, token_asym_id, residual):
        """The projection-bias / grouped OPM in torch (products in the parameters' dtype)."""
        B, S, N, _ = msa.shape
        G, K = self.groups, self.group_width
        m = self.ln_msa(msa)
        left, right = self.to_left(m), self.to_right(m)
        if mask is None:
            mask = torch.ones(msa.shape[:3], dtype=torch.bool, device=msa.device)
        left, right = left * mask[..., None], right * mask[..., None]
        if G > 1:
            prod = torch.einsum("bsigk,bsjgl->bijgkl", left.view(B, S, N, G, K), right.view(B, S, N, G, K)).reshape(B, N, N, -1)
            pair = self.to_out(self.product_norm(prod))
        else:
            out = rearrange(torch.einsum("bmid,bmje->bijde", left, right), "B L1 L2 D1 D2 -> B L1 L2 (D1 D2)")
            norm = torch.einsum("bmi,bmj->bij", mask.float(), mask.float())
            norm = (norm + self.count_offset if self.count_offset is not None else norm.clamp(min=1))[..., None]
            if self.normalize_before_proj:
                pair = self.to_out((out / norm).to(left.dtype))
            else:
                pair = (self.to_out(out) / norm).to(left.dtype)
        if self.mask_interchain and token_asym_id is not None:
            pair = pair * (token_asym_id[:, :, None] == token_asym_id[:, None, :])[..., None].to(pair.dtype)
        return residual + pair if residual is not None else pair

    @typecheck
    def forward(
        self,
        msa: Float[torch.Tensor, "B M L d_msa"],
        mask: Bool[torch.Tensor, "B M L"] | None = None,
        token_asym_id: Int[torch.Tensor, "B L"] | None = None,
        residual: Float[torch.Tensor, "B L L d_pair"] | None = None,
    ) -> Float[torch.Tensor, "B L L d_pair"]:
        """Forward pass. When ``residual`` (the running pair) is given, ALWAYS returns
        ``residual + OPM(msa)``. This is a CROSS-TENSOR residual — the residual (pair) is a
        DIFFERENT tensor than the input (msa), so unlike the self-residual layers it is passed in
        explicitly by the block (``pair = opm(msa, ..., residual=pair)``) rather than being the
        module's own input; the add is unconditional when ``residual`` is provided (no dropout on
        the OPM branch). ``residual=None`` returns the raw OPM output (standalone / benchmarking)."""
        if self.dead and residual is not None and not (self.training or torch.is_grad_enabled()):
            return residual
        # ``count_offset``: only the pair divisor differs. The H100 opm_train / composed paths take it; the Anthropic
        # payload and the portable Triton kernel do not (never ignored: they step aside), and a call no kernel serves runs the PyTorch statements.
        custom = self.count_offset is not None
        if self.groups > 1 or self.to_left.bias is not None:  # RoseTTAFold3 / Chai-1 (see the constructor)
            from miniworld_engine.kernels.outer_product_mean import composed

            # The H100 CUDA paths below step aside for the A/B switch (force_portable_kernels), which runs the Triton family below.
            native = self.implementation != ImplementationType.PYTORCH and not _dispatch.portable_forced(self.implementation)
            # RoseTTAFold3 training (a grad-enabled call): the wide fused composition, the projection biases in the prologue and their
            # gradients off dA / dB (``opm_train``, either normalisation order)
            if (
                native
                and self.groups == 1
                and self.to_left.bias is not None
                and torch.is_grad_enabled()
                and _opm_train.wanted(self.implementation)
                and msa.is_cuda
                and torch.cuda.get_device_capability(msa.device) == (9, 0)
                and _opm_train.serves(msa, *(self.to_left.weight.shape[1], self.to_left.weight.shape[0], self.to_out.weight.shape[0]),
                                      interchain=bool(self.mask_interchain and token_asym_id is not None),
                                      normalize_before_proj=self.normalize_before_proj, custom_divisor=custom)
            ):
                _m = mask if mask is not None else torch.ones(msa.shape[:3], dtype=torch.bool, device=msa.device)
                return _opm_train.outer_product_mean(self, msa, _m, residual=residual)
            if native and self.to_left.bias is not None and composed.biased_wide_serves(self, msa, mask):
                return composed.biased_wide(self, msa, mask, token_asym_id, residual)
            if (
                native
                and self.groups == 1
                and _opm_train.wanted(self.implementation)
                and _opm_train.serves(msa, *(self.to_left.weight.shape[1], self.to_left.weight.shape[0], self.to_out.weight.shape[0]),
                                      interchain=False, normalize_before_proj=self.normalize_before_proj, custom_divisor=custom)
                and not torch.is_grad_enabled()
            ):
                return composed.biased_on_fused(self, msa, mask, token_asym_id, residual)
            if native and self.groups > 1:   # Chai-1: the fused H100 passes (kernels.outer_product_mean.chai), inference and training
                from miniworld_engine.kernels.outer_product_mean import chai as _chai

                if _chai.serves(self, msa, mask, training=torch.is_grad_enabled(),
                                interchain=bool(self.mask_interchain and token_asym_id is not None)):
                    return _chai.forward(self, msa, mask, token_asym_id, residual)
            if native and composed.serves(self, msa, mask):
                return composed.inference(self, msa, mask, token_asym_id, residual)
            # The portable Triton path (projection biases folded into the projection kernel, groups as a leading batch axis of the
            # outer product), routed like the standard leg below: a card with a record (dispatch.resolve), force_portable_kernels. A
            # call the kernels refuse warns once (KernelRefusedWarning) and runs the PyTorch statements.
            if (
                not custom   # the Triton kernel divides by the clamped count only
                and self.implementation == ImplementationType.MINIWORLD
                and resolve("outer_product_mean", self.implementation, msa.device,
                            training=_dispatch.needs_backward(self, msa)) == KernelBackend.PORTABLE
                and _dispatch.portable_leg_allowed(msa.device)
            ):
                from miniworld_engine.kernels.outer_product_mean.interface import (
                    refusal,
                    triton_outer_product_mean,
                )

                why = refusal(
                    msa,
                    self.to_left.weight.shape[0],
                    self.to_out.weight.shape[0],
                    interchain=bool(self.mask_interchain and token_asym_id is not None),
                    residual=residual,
                    groups=self.groups,
                )
                if why is not None and msa.is_cuda:
                    _dispatch.warn_refused("outer_product_mean", msa.device, why)
                if why is None:
                    options = {}
                    if self.groups > 1:
                        options = {
                            "groups": self.groups,
                            "product_norm_weight": self.product_norm.weight,
                            "product_norm_bias": self.product_norm.bias,
                            "product_norm_eps": float(self.product_norm.eps),
                        }
                    return triton_outer_product_mean(
                        msa,
                        mask,
                        self.ln_msa.weight,
                        self.ln_msa.bias,
                        self.to_left.weight,
                        self.to_right.weight,
                        self.to_out.weight,
                        self.to_out.bias,
                        residual,
                        eps=float(self.ln_msa.eps),
                        normalize_before_proj=self.normalize_before_proj,
                        b_left=self.to_left.bias,
                        b_right=self.to_right.bias,
                        **options,
                    )
            return self._variant_reference(msa, mask, token_asym_id, residual)
        # The packaged H100 OPM forward/backward (integrations.opm_train) serves a
        # supported call automatically, including the optional pair residual.
        if _opm_train.wanted(self.implementation):
            _dims = (
                self.to_left.weight.shape[1],
                self.to_left.weight.shape[0],
                self.to_out.weight.shape[0],
            )
            if _opm_train.serves(
                msa,
                *_dims,
                interchain=bool(self.mask_interchain and token_asym_id is not None),
                normalize_before_proj=self.normalize_before_proj,
                custom_divisor=custom,
            ):
                _m = (
                    mask
                    if mask is not None
                    else torch.ones(msa.shape[:3], dtype=torch.bool, device=msa.device)
                )
                return _opm_train.outer_product_mean(self, msa, _m, residual=residual)
        # `implementation="anthropic"` refuses with the reason; `miniworld` uses the fused path where it
        # fits and falls through to the statements below where it does not. See integrations.anthropic_msa.
        if not custom and _anthropic.wanted(self.implementation):
            _fused = {
                "grad": torch.is_grad_enabled(),
                "interchain": bool(self.mask_interchain and token_asym_id is not None),
                "normalize_before_proj": self.normalize_before_proj,
            }
            _dims = (self.to_left.weight.shape[0], self.to_out.weight.shape[0])
            if self.implementation == ImplementationType.ANTHROPIC:
                _anthropic.require_opm(
                    msa, *_dims, **_fused
                )  # explicit: the reason, never a reroute
            if _anthropic.serves_opm(msa, *_dims, **_fused):
                _m = (
                    mask
                    if mask is not None
                    else torch.ones(msa.shape[:3], dtype=torch.bool, device=msa.device)
                )
                _pair = _anthropic.outer_product_mean(self, msa, _m)
                return residual + _pair if residual is not None else _pair
        if custom:
            # no CUDA path serves this call (shape / dtype / device outside opm_train's contract) and the portable Triton kernel and the
            # Anthropic payload divide by the clamped count only: refuse (never ignore the option) and run the PyTorch statements
            if self.implementation == ImplementationType.MINIWORLD and msa.is_cuda:
                _dispatch.warn_refused("outer_product_mean", msa.device,
                                       "count_offset: this call is outside the CUDA kernels' contract (bf16, batch 1, d_msa/d_pair of opm_train.SHAPES)")
            return self._variant_reference(msa, mask, token_asym_id, residual)
        # The portable Triton path (kernels.outer_product_mean) for every call the native paths above did not take:
        # other GPUs with a record (dispatch.resolve), other widths, force_portable_kernels. A card without a record resolves to
        # PYTORCH there (one warning); H100 / B200 take the PyTorch statements below.
        if (
            self.implementation == ImplementationType.MINIWORLD
            and resolve("outer_product_mean", self.implementation, msa.device,
                        training=_dispatch.needs_backward(self, msa)) == KernelBackend.PORTABLE
            and _dispatch.portable_leg_allowed(msa.device)   # H100 / B200: no Triton fallback, the PyTorch statements below run
        ):
            from miniworld_engine.kernels.outer_product_mean.interface import (
                refusal,
                triton_outer_product_mean,
            )

            why = refusal(
                msa,
                self.to_left.weight.shape[0],
                self.to_out.weight.shape[0],
                interchain=bool(self.mask_interchain and token_asym_id is not None),
                residual=residual,
            )
            if why is not None and msa.is_cuda:
                _dispatch.warn_refused("outer_product_mean", msa.device, why)
            if why is None:
                return triton_outer_product_mean(
                    msa,
                    mask,
                    self.ln_msa.weight,
                    self.ln_msa.bias,
                    self.to_left.weight,
                    self.to_right.weight,
                    self.to_out.weight,
                    self.to_out.bias,
                    residual,
                    eps=float(self.ln_msa.eps),
                    normalize_before_proj=self.normalize_before_proj,
                )
        with _dispatch.fallback_autocast(msa, self, self.implementation == ImplementationType.MINIWORLD and _dispatch.restricted_card(msa.device)):
            msa = self.ln_msa(msa)
            left = self.to_left(msa)
            right = self.to_right(msa)

            if mask is None:
                mask = torch.ones(msa.shape[:3], dtype=torch.bool, device=msa.device)

            left = left * mask[..., None]
            right = right * mask[..., None]
            out = torch.einsum("bmid,bmje->bijde", left, right)
            out = rearrange(out, "B L1 L2 D1 D2 -> B L1 L2 (D1 D2)")

            norm = torch.einsum("bmi,bmj->bij", mask.float(), mask.float())
            # norm stays fp32 for count precision (>256 counts aren't bf16-exact), but cast the
            # normalized result back to the projection dtype so to_out runs in native bf16
            # (no autocast in the distogram trunk; a bf16/fp32 mismatch would otherwise crash).
            norm = norm.clamp(min=1)[..., None]
            if self.normalize_before_proj:
                pair = self.to_out((out / norm).to(left.dtype))
            else:
                # ESMFold2 normalizes AFTER projection, including its learned bias.
                pair = (self.to_out(out) / norm).to(left.dtype)
            if self.mask_interchain and token_asym_id is not None:
                same_chain = token_asym_id[:, :, None] == token_asym_id[:, None, :]
                pair = pair * same_chain[..., None].to(pair.dtype)
            return residual + pair if residual is not None else pair
