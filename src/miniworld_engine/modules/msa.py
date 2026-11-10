"""MSA -> pair and MSA <- pair: ``OuterProduct`` (single -> pair), ``OuterProductMean`` (MSA -> pair) and ``MSAPairWeightedAveraging``
(AlphaFold 3 Algorithm 10: pair-weighted averaging over the MSA, with the residual and the row dropout).

Each is a :class:`~miniworld_engine.modules.primitives.BackendModule`; ``_forward_pytorch`` is the reference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from miniworld_engine.modules.functional import sigmoid_gate
from miniworld_engine.modules.norm import LayerNorm
from miniworld_engine.modules.primitives import BackendModule, Dropout, Linear, MasterWeights, key_mask, master_autocast, random_input, wide


class OuterProduct(BackendModule):
    """Outer product single representation to pair representation.

    Parameters
    ----------
    d_single : int
        Dimension of single representation.
    d_pair : int
        Dimension of pair representation.
    d_hidden : int
        Dimension of hidden layer.

    """

    backends = ("pytorch",)

    def __init__(
        self,
        d_single: int,
        d_pair: int,
        d_hidden: int = 32,
        *,
        backend: str = "pytorch",
    ) -> None:
        super().__init__()

        self.ln_single = nn.LayerNorm(d_single)
        self.to_left = Linear(d_single, d_hidden, bias=False, init="default")
        self.to_right = Linear(d_single, d_hidden, bias=False, init="default")
        self.to_out = Linear(d_hidden * d_hidden, d_pair, bias=True, init="zero")
        self.set_backend(backend)

    def _forward_pytorch(self, single: torch.Tensor) -> torch.Tensor:
        """``single`` ``[B, L, d_single]`` -> ``[B, L, L, d_pair]``."""
        single = self.ln_single(single)
        left = self.to_left(single)
        right = self.to_right(single)

        out = torch.einsum("bid,bje->bijde", left, right)
        out = rearrange(out, "B L1 L2 d1 d2 -> B L1 L2 (d1 d2)")
        return self.to_out(out)


class OuterProductMean(BackendModule, MasterWeights):
    """Outer product mean of MSA representation to pair representation::

        m = LN(msa); a = mask * W_l m (+ b_l); b = mask * W_r m (+ b_r)
        o[i, j] = sum_s a[s, i] (x) b[s, j]                 # [d_hidden * d_hidden]
        pair = W_out (o / n) + b_out                        # normalize_before_proj (AF3 SI Alg. 9 ordering, default; AF3 code divides after the projection: normalize_before_proj=False, count_offset=1e-3)
        pair = (W_out o + b_out) / n                        # normalize_before_proj=False (ESMFold2: the bias is divided too)

    ``n[i, j] = sum_s mask[s, i] mask[s, j]`` (fp32), as ``clamp(n, 1)`` (the Boltz approach) or, with ``count_offset``, AlphaFold 3's
    ``n + count_offset`` (``1e-3`` in the released code).

    ``projection_bias`` (RoseTTAFold3): the left / right projections carry a bias (masked rows lose it too). ``groups`` (Chai-1): ``groups``
    outer products of ``d_hidden / groups`` wide halves, their ``groups * (d_hidden / groups)**2`` products LayerNorm'd (``product_norm``, eps
    ``product_norm_eps``) before the out-projection, no division by the pair counts. ``mask_interchain``: pairs of different chains
    (``token_asym_id``) are zeroed.

    Parameters
    ----------
    d_msa : int
        Dimension of MSA representation.
    d_pair : int
        Dimension of pair representation.
    d_hidden : int
        Dimension of hidden layer.

    """
    version = 3   # human-managed: bump when a change alters this class's accuracy or speed results

    #: A numerically dead instance of a checkpoint (set by a model's checkpoint loader): inference returns the residual input.
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
        master_weights: bool = False,
        backend: str = "pytorch",
    ) -> None:
        super().__init__()
        if count_offset is not None and groups > 1:
            raise ValueError("count_offset needs groups == 1: the grouped (Chai-1) form does not divide by the pair count")
        self.count_offset = count_offset
        self.mask_interchain = mask_interchain
        self.normalize_before_proj = normalize_before_proj
        self.ln_msa = LayerNorm(d_msa)
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
        self.set_backend(backend)

    def _forward_pytorch(
        self,
        msa: torch.Tensor,
        mask: torch.Tensor | None = None,
        token_asym_id: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``msa`` ``[B, M, L, d_msa]``, ``mask`` ``[B, M, L]`` (bool), ``token_asym_id`` ``[B, L]`` -> ``[B, L, L, d_pair]``.

        When ``residual`` (the running pair) is given, ALWAYS returns ``residual + OPM(msa)``: a CROSS-TENSOR residual, passed in explicitly
        by the block (``pair = opm(msa, ..., residual=pair)``); ``residual=None`` returns the raw OPM output."""
        if self.dead and residual is not None and not (self.training or torch.is_grad_enabled()):
            return residual
        with master_autocast(msa, self):
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
                # the counts in at least fp32 (>256 counts are not bf16-exact); the normalized product goes back to the projection dtype
                norm = torch.einsum("bmi,bmj->bij", mask.to(wide(out)), mask.to(wide(out)))
                norm = (norm + self.count_offset if self.count_offset is not None else norm.clamp(min=1))[..., None]
                pair = self.to_out((out / norm).to(left.dtype)) if self.normalize_before_proj else (self.to_out(out) / norm).to(left.dtype)
            if self.mask_interchain and token_asym_id is not None:
                pair = pair * (token_asym_id[:, :, None] == token_asym_id[:, None, :])[..., None].to(pair.dtype)
            return residual + pair if residual is not None else pair

    # ------------------------------------------------------------------------------------------------------------- engine kernels
    #: ``"cuda"``: the H100 (sm_90a) kernels, bf16 activations (bf16 or fp32-master parameters), one call routed by the module's form:
    #:   standard (groups 1, no projection bias)  ``kernels.integrations.opm_train`` (WGMMA / TMA prologue, cuBLAS contraction, CUDA epilogue;
    #:       forward + backward) at d_hidden 32, (d_msa, d_pair) in ``opm_train.SHAPES`` (``opm_train.TRAIN_ONLY``: training only), batch 1;
    #:       inference of other widths:
    #:       ``kernels.outer_product_mean.composed.inference`` (cuBLAS + the ``cuda.gather`` pass; d_hidden % 8, d_hidden^2 in 512 / 1024)
    #:   projection biases (RoseTTAFold3)  training: ``opm_train`` (biases in the prologue); inference: ``composed.biased_wide``
    #:       (S % 256, N % 64, d_msa 64 / 128, d_pair 128 / 256 / 384), else ``composed.inference``
    #:   groups (Chai-1, 8 x 8)  ``kernels.outer_product_mean.chai`` (fused CUDA passes, forward + backward; d_msa 64, N % 64)
    #: ``"triton"``: ``kernels.outer_product_mean.triton.main.triton_outer_product_mean`` (forward + backward, bf16 activations; d_msa /
    #: d_hidden powers of two >= 16, d_pair % 32; divides by the clamped count only: no ``count_offset``; no ``mask_interchain``).
    backends = ("pytorch", "cuda", "triton")

    def _interchain(self, token_asym_id) -> bool:
        return bool(self.mask_interchain and token_asym_id is not None)

    def _common_refusal(self, msa, mask, residual) -> str | None:
        if not msa.is_cuda:
            return "the kernels run on CUDA tensors only"
        if msa.dtype != torch.bfloat16:
            return f"dtype {msa.dtype}: the kernels take bfloat16 activations (no fp32 OuterProductMean kernel)"
        if msa.dim() != 4:
            return f"msa must be [B, M, L, d_msa], got {tuple(msa.shape)}"
        if mask is not None and (mask.dtype != torch.bool or tuple(mask.shape) != tuple(msa.shape[:3])):
            return "mask must be a bool [B, M, L] mask"
        if residual is not None and residual.dtype != msa.dtype:
            return "the pair residual must be in the msa's dtype"
        return None

    def _cuda_route(self, msa, mask, token_asym_id, residual):
        """``(kernel name, None)`` for the H100 path that serves the call, or ``(None, why not)``."""
        why = self._common_refusal(msa, mask, residual)
        if why is not None:
            return None, why
        if torch.cuda.get_device_capability(msa.device) == (8, 0):   # A100: kernels.integrations.opm_sm80 (forward + backward)
            from miniworld_engine.kernels.integrations import opm_sm80

            why = opm_sm80.refusal(self, msa, mask, token_asym_id, residual)
            return ("sm80", None) if why is None else (None, f"opm_sm80: {why}")
        if torch.cuda.get_device_capability(msa.device) != (9, 0):
            return None, "the cuda backend is the H100 (sm_90a) kernels"
        from miniworld_engine.kernels.integrations import opm_train
        from miniworld_engine.kernels.outer_product_mean import chai, composed

        grad = torch.is_grad_enabled()
        interchain = self._interchain(token_asym_id)
        dims = (self.to_left.weight.shape[1], self.to_left.weight.shape[0], self.to_out.weight.shape[0])
        custom = self.count_offset is not None
        # composed.inference runs the module's own LayerNorm and torch.mm on the parameters: outside autocast they must be in msa's dtype
        same_dtype = torch.is_autocast_enabled("cuda") or self.to_left.weight.dtype == msa.dtype
        if self.groups > 1:
            if chai.serves(self, msa, mask, training=grad, interchain=interchain):
                return "chai", None
            return None, (f"grouped OPM (groups {self.groups} x {self.group_width}, d_msa {msa.shape[-1]}, N {msa.shape[2]}, batch {msa.shape[0]}"
                          f", grad {grad}): the Chai-1 kernels take 8 x 8 groups, d_msa 64, N % 64, batch 1 (training without interchain)")
        why = opm_train.refusal(msa, *dims, interchain=interchain if grad or self.to_left.bias is None else False,
                                normalize_before_proj=self.normalize_before_proj, custom_divisor=custom)
        if self.to_left.bias is None:
            if why is None:
                return "opm_train", None
            if not grad and same_dtype and composed.serves(self, msa, mask):
                return "composed", None
            return None, f"opm_train: {why}" + ("" if not grad else " (training: no other OPM backward on H100)")
        if grad:
            return ("opm_train", None) if why is None else (None, f"projection-bias OPM training: opm_train: {why}")
        if composed.biased_wide_serves(self, msa, mask):
            return "biased_wide", None
        if same_dtype and composed.serves(self, msa, mask):
            return "composed", None
        return None, "projection-bias OPM inference: neither the wide composition nor the composed path serves it"

    def _supports_cuda(self, msa, mask=None, token_asym_id=None, residual=None) -> str | None:
        return self._cuda_route(msa, mask, token_asym_id, residual)[1]

    def _forward_cuda(self, msa, mask=None, token_asym_id=None, residual=None) -> torch.Tensor:
        if self.dead and residual is not None and not (self.training or torch.is_grad_enabled()):
            return residual
        from miniworld_engine.kernels.integrations import opm_train
        from miniworld_engine.kernels.outer_product_mean import chai, composed

        route, _ = self._cuda_route(msa, mask, token_asym_id, residual)
        if route == "sm80":
            from miniworld_engine.kernels.integrations import opm_sm80

            if torch.is_grad_enabled():
                return opm_sm80.update_train(self, msa, mask, residual)
            return opm_sm80.update_inference(self, msa, mask, residual)
        if route == "chai":
            return chai.forward(self, msa, mask, token_asym_id, residual)
        if route == "composed":
            return composed.inference(self, msa, mask, token_asym_id, residual)
        if route == "biased_wide":
            return composed.biased_wide(self, msa, mask, token_asym_id, residual)
        # opm_train (never with the interchain mask: _cuda_route refuses it there); the residual is fused in its epilogue
        full = mask if mask is not None else torch.ones(msa.shape[:3], dtype=torch.bool, device=msa.device)
        return opm_train.outer_product_mean(self, msa, full, residual=residual)

    def _supports_triton(self, msa, mask=None, token_asym_id=None, residual=None) -> str | None:
        why = self._common_refusal(msa, mask, residual)
        if why is not None:
            return why
        if self.count_offset is not None:
            return "count_offset: the Triton kernel divides by the clamped pair count only"
        from miniworld_engine.kernels.outer_product_mean.triton.main import refusal

        return refusal(msa, self.to_left.weight.shape[0], self.to_out.weight.shape[0], interchain=self._interchain(token_asym_id),
                       residual=residual, groups=self.groups)

    def _forward_triton(self, msa, mask=None, token_asym_id=None, residual=None) -> torch.Tensor:
        if self.dead and residual is not None and not (self.training or torch.is_grad_enabled()):
            return residual
        from miniworld_engine.kernels.outer_product_mean.triton.main import triton_outer_product_mean

        options = {}
        if self.groups > 1:
            options = {"groups": self.groups, "product_norm_weight": self.product_norm.weight, "product_norm_bias": self.product_norm.bias,
                       "product_norm_eps": float(self.product_norm.eps)}
        return triton_outer_product_mean(
            msa, mask, self.ln_msa.weight, self.ln_msa.bias, self.to_left.weight, self.to_right.weight, self.to_out.weight, self.to_out.bias,
            residual, eps=float(self.ln_msa.eps), normalize_before_proj=self.normalize_before_proj, b_left=self.to_left.bias,
            b_right=self.to_right.bias, **options,
        )

    def make_inputs(self, *, length, samples, depth, dtype, device, generator, stream=None) -> dict:
        depth, d_msa = depth or 1, self.ln_msa.normalized_shape[0]
        return {"msa": random_input((samples, depth, length, d_msa), dtype=dtype, device=device, generator=generator),
                "mask": key_mask(samples * depth, length, device=device).view(samples, depth, length)}


class MSAPairWeightedAveraging(BackendModule, MasterWeights):
    """MSA pair-weighted averaging::

        v = W_v LN(msa)                                   # [B, M, L, H, D]
        w = softmax_j(W_b LN_p(pair) masked)              # [B, H, L, L], masked keys at the dtype's min
        o = sigmoid(W_g LN(msa)) * sum_j w[h, i, j] v[m, j, h]
        return msa + drop_msa(W_o o)                      # row-broadcast dropout (training, p_drop)

    ``ln_pair`` has no offset: it feeds only the softmax logits (``to_bias`` has no bias), and an offset beta adds the same ``Wb_h . beta``
    to every key's logit of head h, which softmax over keys cancels exactly.

    Parameters
    ----------
    d_msa : int
        Dimension of MSA representation.
    d_pair : int
        Dimension of pair representation.
    n_head : int
        Number of heads.
    d_hidden : int, optional
        Per-head hidden dimension. Defaults to ``d_msa // n_head``.
    p_drop : float
        Row dropout of the update (training only).

    """
    version = 2   # human-managed: bump when a change alters this class's accuracy or speed results

    #: "cuda": the H100 kernels (kernels.integrations.pwa_train / pwa_wide); "triton": kernels.pair_weighted_averaging.triton (see the
    #: _supports_ docstrings for the ranges).
    backends = ("pytorch", "cuda", "triton")

    #: A numerically dead instance of a checkpoint (set by a model's checkpoint loader): inference returns the residual input.
    dead: bool = False

    def __init__(
        self,
        d_msa: int,
        d_pair: int,
        n_head: int = 8,
        d_hidden: int | None = None,
        *,
        p_drop: float = 0.0,
        master_weights: bool = False,
        backend: str = "pytorch",
    ) -> None:
        super().__init__()
        self.n_head = n_head
        self.drop_msa = Dropout(broadcast_dim=1, p_drop=p_drop)

        if d_hidden is None:
            d_hidden = d_msa // n_head

        self.ln_msa = LayerNorm(d_msa)
        self.to_value = Linear(d_msa, d_hidden * n_head, bias=False, init="glorot")
        self.ln_pair = LayerNorm(d_pair, bias=False)
        self.to_bias = Linear(d_pair, n_head, bias=False, init="default")
        self.to_gate = Linear(d_msa, d_hidden * n_head, bias=False, init="gating")
        self.to_out = Linear(d_hidden * n_head, d_msa, bias=False, init="zero")
        if master_weights:
            self.set_master_weights()
        self.set_backend(backend)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # checkpoints written before the offset was removed carry ln_pair.bias; it had no effect on the output, so dropping it is exact
        state_dict.pop(prefix + "ln_pair.bias", None)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def _forward_pytorch(
        self,
        msa: torch.Tensor,
        pair: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``msa`` ``[B, M, L, d_msa]``, ``pair`` ``[B, L, L, d_pair]``, ``mask`` ``[B, L]`` (bool). ALWAYS returns the residual output
        ``msa + drop_msa(pwa(msa, pair))``."""
        if self.dead and not (self.training or torch.is_grad_enabled()):
            return msa
        with master_autocast(msa, self):
            msa_res = msa  # residual == the ORIGINAL input (before ln_msa rebinds `msa`)
            msa = self.ln_msa(msa)
            value = self.to_value(msa)
            value = rearrange(value, "B M L (H D) -> B M L H D", H=self.n_head)

            pair = self.ln_pair(pair)
            bias = self.to_bias(pair)
            bias = rearrange(bias, "B L1 L2 H -> B H L1 L2")

            if mask is not None:
                bias = bias.masked_fill(~mask[:, None, None, :], torch.finfo(bias.dtype).min)

            attention = F.softmax(bias, dim=-1)
            out = torch.einsum("bhij,bmjhd->bmihd", attention, value)

            gate = self.to_gate(msa)
            out = rearrange(out, "B M L H D -> B M L (H D)")
            out = sigmoid_gate(gate, out)
            out = self.drop_msa(self.to_out(out))
            return msa_res + out

    # ------------------------------------------------------------------------------------------------------------- engine kernels
    def _widths(self) -> tuple[int, int, int, int]:
        """(d_msa, d_pair, n_head, d_hidden per head)."""
        return (self.to_value.weight.shape[1], self.to_bias.weight.shape[1], self.n_head, self.to_value.weight.shape[0] // self.n_head)

    def _trains(self) -> bool:
        """The call takes the kernels' autograd function: grad recording, or the live row dropout of training mode."""
        return torch.is_grad_enabled() or (self.training and self.drop_msa.p_drop > 0)

    def _common_refusal(self, msa, pair, mask) -> str | None:
        if not msa.is_cuda:
            return "the kernels run on CUDA tensors only"
        if msa.dtype != torch.bfloat16 or pair.dtype != torch.bfloat16:
            return f"dtype {msa.dtype} / {pair.dtype}: the kernels take bfloat16 activations (no fp32 MSAPairWeightedAveraging kernel)"
        if msa.dim() != 4 or pair.dim() != 4:
            return "msa must be [B, M, L, d_msa] and pair [B, L, L, d_pair]"
        if mask is not None and (mask.dtype != torch.bool or tuple(mask.shape) != (msa.shape[0], msa.shape[2])):
            return "mask must be a bool [B, L] key mask"
        return None

    def _supports_cuda(self, msa, pair, mask=None) -> str | None:
        """``kernels.integrations.pwa_train`` (d_msa 64, d_pair 128, 8 heads x 32: N % 128, S even, N <= 2048 inference / 1024 training;
        d_pair 256 / 384 / 512 with pwa_wide's pair kernels: N <= 1024) and ``pwa_wide`` (d_msa 64 / 128, d_pair 128 / 256 / 384, 8 heads x
        8 / 16, or d_msa 256, d_pair 512, 8 heads x 32 (the BIG build); N <= 1024): bf16 activations, batch 1, sm_90a, forward + backward
        (the residual and the row dropout in the kernels)."""
        why = self._common_refusal(msa, pair, mask)
        if why is not None:
            return why
        if torch.cuda.get_device_capability(msa.device) == (8, 0):   # A100: kernels.integrations.pwa_sm80 (forward + backward)
            from miniworld_engine.kernels.integrations import pwa_sm80

            why = pwa_sm80.refusal(self, msa, pair, mask)
            return None if why is None else f"pwa_sm80: {why}"
        if torch.cuda.get_device_capability(msa.device) != (9, 0):
            return "the cuda backend is the H100 (sm_90a) kernels"
        from miniworld_engine.kernels.integrations import pwa_train

        return pwa_train.refusal(msa, pair, *self._widths())

    def _forward_cuda(self, msa, pair, mask=None) -> torch.Tensor:
        if self.dead and not (self.training or torch.is_grad_enabled()):
            return msa
        if torch.cuda.get_device_capability(msa.device) == (8, 0):
            from miniworld_engine.kernels.integrations import pwa_sm80

            if self._trains():
                return pwa_sm80.update_train(self, msa, pair, mask)
            return pwa_sm80.update_inference(self, msa, pair, mask)
        from miniworld_engine.kernels.integrations import pwa_train

        if self._trains():
            return pwa_train.pair_weighted_averaging(self, msa, pair, mask)
        return pwa_train.pair_weighted_averaging_inference(self, msa, pair, mask)

    def _supports_triton(self, msa, pair, mask=None) -> str | None:
        """``kernels.pair_weighted_averaging.triton.main`` (forward + backward, the residual and the row dropout fused): bf16 activations,
        d_msa a power of two >= 16, d_hidden 1..128, n_head * d_hidden % 16, d_pair >= 16."""
        why = self._common_refusal(msa, pair, mask)
        if why is not None:
            return why
        from miniworld_engine.kernels.pair_weighted_averaging.triton.main import refusal

        return refusal(msa, pair, self.n_head, self._widths()[3])

    def _forward_triton(self, msa, pair, mask=None) -> torch.Tensor:
        if self.dead and not (self.training or torch.is_grad_enabled()):
            return msa
        from miniworld_engine.kernels.pair_weighted_averaging.triton.main import triton_pair_weighted_averaging

        p_drop = float(self.drop_msa.p_drop) if self.training else 0.0
        keep = None
        if p_drop > 0:   # the keep-mask Dropout(broadcast_dim=1) draws, broadcast over the MSA rows
            keep = torch.rand((msa.shape[0], msa.shape[2], msa.shape[3]), device=msa.device, dtype=msa.dtype) > p_drop
        return triton_pair_weighted_averaging(
            msa, pair, mask, self.ln_msa.weight, self.ln_msa.bias, self.to_value.weight, self.to_gate.weight, self.ln_pair.weight,
            self.to_bias.weight, self.to_out.weight, eps_msa=float(self.ln_msa.eps), eps_pair=float(self.ln_pair.eps), keep=keep, p_drop=p_drop,
        )

    def make_inputs(self, *, length, samples, depth, dtype, device, generator, stream=None) -> dict:
        d_msa, d_pair = self.ln_msa.normalized_shape[0], self.ln_pair.normalized_shape[0]
        return {"msa": random_input((samples, depth or 1, length, d_msa), dtype=dtype, device=device, generator=generator),
                "pair": random_input((samples, length, length, d_pair), dtype=dtype, device=device, generator=generator),
                "mask": key_mask(samples, length, device=device)}
