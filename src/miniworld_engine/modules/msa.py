"""MSA -> pair and MSA <- pair: ``OuterProduct`` (single -> pair), ``OuterProductMean`` (MSA -> pair) and ``MSAPairWeightedAveraging``
(AlphaFold 3 Algorithm 10: pair-weighted averaging over the MSA, with the residual and the row dropout).

Each is a :class:`~miniworld_engine.modules.primitives.BackendModule`; ``_forward_pytorch`` is the reference.
"""

from typing import ClassVar, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from miniworld_engine.modules.functional import sigmoid_gate
from miniworld_engine.modules.norm import LayerNorm
from miniworld_engine.modules.primitives import BackendModule, Dropout, Linear, MasterWeights, key_mask, master_autocast, random_input, wide

#: Release directories of the model kits' own MSA kernels (the ``"anthropic"`` backends; ``modules/_anthropic.load(..., path=...)``).
_AF3T_KERNELS = "af3_torch/opt/forward/af3t/kernels/third_party"  # af3t_opm.py: torch + triton only
_EF2_DRIVER = "esmfold2/opt/forward/fast_inference/driver"  # ef2_msa_v2.py: torch + triton only


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

    version = 4  # human-managed: bump when a change alters this class's accuracy or speed results

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
    #: d_hidden powers of two >= 16, d_pair % 32; the pair divisor clamp(n, 1) or, with ``count_offset``, n + count_offset; no ``mask_interchain``).
    #: ``"anthropic"`` (inference only; the residual and the interchain mask are the module's): the model kit's own OPM kernel by the
    #: module's form (the module is model-agnostic: the kit of the models its rows serve):
    #:   normalize_before_proj, groups 1, d_pair 128 (Boltz-2, MiniWorld; RoseTTAFold3's biased projections in the same form)  the boltz2
    #:       kit's ``fpf_msa.boltz2.outer_product_mean`` (``opt_core.ops.msa_opm`` form ``mask_norm``: LN + a | b (+ bias) x mask prologue,
    #:       the outer GEMM over the depth, / clamp(n, 1) before the projection, proj_o + bias in its epilogue)
    #:   divide after the projection, d_msa 64, d_hidden 32, d_pair 128, no projection bias (AlphaFold 3, OpenFold3, Protenix-v1)  the
    #:       af3_torch kit's ``af3t_opm.ln_proj2`` + cuBLAS outer GEMM + ``af3t_opm.opm_out`` (the kit's ``af3t_msa.opm_kernels`` driver
    #:       restated: that module imports xfold); the kernel divides by ``bf16(norm + eps)`` of a caller-given norm, so the module's
    #:       ``clamp(n, 1)`` is handed in with eps 0 (``count_offset``: the unclamped n with eps = count_offset, AF3's own form)
    #:   divide after the projection, d_msa 128, d_hidden 32, d_pair % 256, no projection bias (ESMFold2, Protenix-v2)  the esmfold2 kit's
    #:       ``ef2_msa_v2.opm_fused`` (LN + W x row mask producer, cuBLAS, projection + bias / clamp(n, 1) + pair residual epilogue)
    backends = ("pytorch", "anthropic", "cuda", "triton")
    backend_versions: ClassVar[dict[str, int]] = {"cuda:a100_sm80": 3, "triton": 2}  # 2026-11-10: sm_80 paths; 2026-11-11: count_offset in the triton / sm_80 kernels (FoldForge R2)

    # ------------------------------------------------------------------------------------------------------------------ anthropic
    def _anthropic_route(self, msa, mask, residual) -> tuple[str | None, str | None]:
        """``(route, None)``: ``"boltz2"`` | ``"af3"`` | ``"ef2"``, else ``(None, reason)``."""
        from miniworld_engine.modules import _anthropic

        if torch.is_grad_enabled():
            return None, "inference only (no kit carries an OuterProductMean backward): call under torch.no_grad() / inference_mode"
        if not msa.is_cuda:
            return None, "the release's kernels run on CUDA tensors only"
        if msa.dtype != torch.bfloat16:
            return None, f"dtype {msa.dtype}: the kits' OPM kernels take bfloat16 activations (bf16 autocast)"
        if msa.dim() != 4:
            return None, f"msa must be [B, M, L, d_msa], got {tuple(msa.shape)}"
        if mask is not None and (mask.dtype != torch.bool or tuple(mask.shape) != tuple(msa.shape[:3])):
            return None, "mask must be a bool [B, M, L] mask"
        if residual is not None and residual.dtype != msa.dtype:
            return None, "the pair residual must be in the msa's dtype"
        if self.groups > 1:
            return None, (
                "grouped OPM (Chai-1: 8 groups, product LayerNorm, no count division): the chai1 kit has no OPM kernel (its eager "
                "trunk restates the TorchScript graph in torch) and no other kit computes the grouped form"
            )
        d_msa, d_hidden, d_pair = self.to_left.weight.shape[1], self.to_left.weight.shape[0], self.to_out.weight.shape[0]
        biased = self.to_left.bias is not None
        if self.normalize_before_proj:
            if self.count_offset is not None:
                return None, "count_offset with normalize_before_proj: the boltz2 kit's mask_norm form divides by clamp(n, 1) only"
            if (d_hidden, d_pair) != (32, 128):
                return None, f"d_hidden {d_hidden}, d_pair {d_pair}: the boltz2 kit pins its mask_norm cell to (c_hidden, c_z) = (32, 128)"
            if d_msa & (d_msa - 1) or d_msa < 16:
                return None, f"d_msa {d_msa}: the fused prologue takes a power-of-two width"
            return ("boltz2", None) if (why := _anthropic.missing("fpf_msa.boltz2", path="boltz2/opt/forward")) is None else (None, why)
        if biased:
            return None, (
                "projection biases with the division after the projection: the af3_torch / esmfold2 kernels have no projection bias, "
                "and RoseTTAFold3's kit divides by the MSA depth (a scalar) instead of the pair counts"
            )
        if (d_msa, d_hidden, d_pair) == (64, 32, 128):
            return ("af3", None) if (why := _anthropic.missing("af3t_opm", path=_AF3T_KERNELS)) is None else (None, why)
        if d_msa == 128 and d_hidden == 32 and d_pair % 256 == 0:
            if self.count_offset is not None:
                return None, "count_offset: the esmfold2 kernel divides by clamp(n, 1)"
            return ("ef2", None) if (why := _anthropic.missing("ef2_msa_v2", path=_EF2_DRIVER)) is None else (None, why)
        if d_pair % 256 == 0 and (d_hidden * d_hidden) % 64 == 0 and self.count_offset is None:
            # the other widths of this form (IntelliFold2 256 / 32 / 512: no kit of its own): the esmfold2 kit's T12 lever
            # (ef2_msa.opm_core_t12: the module's LN and a | b GEMM, the cuBLAS outer GEMM, the Triton projection + bias / clamp(n, 1) epilogue)
            return ("ef2_t12", None) if (why := _anthropic.missing("ef2_msa", path=_EF2_DRIVER)) is None else (None, why)
        return None, (
            f"(d_msa, d_hidden, d_pair) = ({d_msa}, {d_hidden}, {d_pair}) dividing after the projection: the kits' kernels take "
            "(64, 32, 128) (af3_torch) or d_pair a multiple of 256 (esmfold2); OpenDDE's 384 has no kit"
        )

    def _supports_anthropic(self, msa, mask=None, token_asym_id=None, residual=None) -> str | None:
        return self._anthropic_route(msa, mask, residual)[1]

    def _anthropic_key(self) -> tuple:
        return tuple((p.data_ptr(), p._version, p.dtype) for p in self.parameters())

    def _forward_anthropic(self, msa, mask=None, token_asym_id=None, residual=None) -> torch.Tensor:
        if self.dead and residual is not None:  # inference only: a dead instance returns the residual input (as the reference does)
            return residual
        route, _ = self._anthropic_route(msa, mask, residual)
        if mask is None:
            mask = torch.ones(msa.shape[:3], dtype=torch.bool, device=msa.device)
        interchain = self._interchain(token_asym_id)
        if route == "boltz2":
            pair = self._anthropic_boltz2(msa, mask)
        elif route == "af3":
            pair = self._anthropic_af3(msa, mask)
        elif route == "ef2_t12":
            pair = self._anthropic_ef2_t12(msa, mask)
        else:
            pair = self._anthropic_ef2(msa, mask, None if interchain else residual)
            if not interchain and residual is not None:
                return pair
        if interchain:
            pair = pair * (cast(torch.Tensor, token_asym_id)[:, :, None] == cast(torch.Tensor, token_asym_id)[:, None, :])[..., None].to(
                pair.dtype
            )
        return residual + pair if residual is not None else pair

    def _anthropic_boltz2(self, msa, mask):
        """The boltz2 kit's entry point on a namespace of Boltz-2's attribute schema (norm / proj_a / proj_b / proj_o / c_hidden); the kit
        caches its weight pack on that object, so a new one is made when the parameters change."""
        from types import SimpleNamespace

        from miniworld_engine.modules import _anthropic

        kit = _anthropic.load("fpf_msa.boltz2", path="boltz2/opt/forward")
        key = self._anthropic_key()
        if getattr(self, "_anthropic_ns_key", None) != key:
            self._anthropic_ns = SimpleNamespace(
                norm=self.ln_msa, proj_a=self.to_left, proj_b=self.to_right, proj_o=self.to_out, c_hidden=self.group_width
            )
            self._anthropic_ns_key = key
        with master_autocast(msa, self):
            return kit.outer_product_mean(self._anthropic_ns, msa.contiguous(), mask.to(msa.dtype).contiguous(), None)

    def _anthropic_af3(self, msa, mask):
        """The af3_torch kit's OPM kernels, driven as its ``af3t_msa.opm_kernels`` does (per sample: ln_proj2, then per chunk of 256 left
        tokens one cuBLAS bf16 GEMM and opm_out)."""
        from miniworld_engine.modules import _anthropic

        ko = _anthropic.load("af3t_opm", path=_AF3T_KERNELS)
        key = self._anthropic_key()
        if getattr(self, "_anthropic_w_key", None) != key:
            with torch.no_grad():
                self._anthropic_w = {
                    "WT": torch.cat([self.to_left.weight, self.to_right.weight], 0).to(torch.bfloat16).t().contiguous(),  # [c_m, 2c]
                    "W16": self.to_out.weight.to(torch.bfloat16).t().contiguous(),  # [c*c, c_z], rows (c, e)
                    "bias": self.to_out.bias.float().contiguous(),
                    "lnw": self.ln_msa.weight.float().contiguous(),
                    "lnb": self.ln_msa.bias.float().contiguous(),
                }
            self._anthropic_w_key = key
        W = self._anthropic_w
        B, S, N, _ = msa.shape
        CO, F = self.group_width, self.to_out.weight.shape[0]
        rows = min(256, N)
        eps = 0.0 if self.count_offset is None else float(self.count_offset)
        outs = []
        with torch.autocast("cuda", enabled=False):
            for b in range(B):
                m_b = mask[b].to(torch.bfloat16).contiguous()
                LT, R = ko.ln_proj2(msa[b].contiguous(), W["lnw"], W["lnb"], W["WT"], m_b, float(self.ln_msa.eps), BA=64, num_warps=4)
                norm = torch.mm(m_b.t(), m_b)  # exact integer counts in bf16 (the kit's _mask_norm)
                norm16 = (norm.clamp(min=1) if self.count_offset is None else norm).contiguous()
                A, Bm = LT.view(N * CO, S), R.view(S, N * CO)
                T = torch.empty((rows * CO, N * CO), device=msa.device, dtype=torch.bfloat16)
                dst = torch.empty((N, N, F), device=msa.device, dtype=torch.float32)
                for b0 in range(0, N, rows):
                    nb = min(rows, N - b0)
                    torch.mm(A[b0 * CO : (b0 + nb) * CO], Bm, out=T[: nb * CO])
                    ko.opm_out(T[: nb * CO], W["W16"], W["bias"], norm16, eps, b0, nb, N, dst, False, BD=64, num_warps=4)
                outs.append(dst.to(msa.dtype))
        return torch.stack(outs, 0)

    def _anthropic_ef2_t12(self, msa, mask):
        """The esmfold2 kit's ``ef2_msa.opm_core_t12`` per sample on its token-major layout ([1, L, M, c_m]); ESMFold2's attribute schema
        (norm / W = [a | b] / Wout / d_hidden) on a namespace (the kit caches its bf16 Wout on it)."""
        from types import SimpleNamespace

        from miniworld_engine.modules import _anthropic

        kit = _anthropic.load("ef2_msa", path=_EF2_DRIVER)
        key = self._anthropic_key()
        if getattr(self, "_anthropic_t12_key", None) != key:
            with torch.no_grad():
                w = torch.cat([self.to_left.weight, self.to_right.weight], 0)  # ESMFold2's W: [a | b] rows
            self._anthropic_t12 = SimpleNamespace(
                W=lambda x, _w=w: F.linear(x, _w.to(x.dtype)), Wout=self.to_out, norm=self.ln_msa, d_hidden=self.group_width
            )
            self._anthropic_t12_key = key
        outs = []
        with master_autocast(msa, self):
            for b in range(msa.shape[0]):
                m_t = msa[b].transpose(0, 1).contiguous()[None]  # [1, L, M, c_m]
                mk = mask[b].transpose(0, 1).float().contiguous()[None]  # [1, L, M]
                outs.append(kit.opm_core_t12(self._anthropic_t12, m_t, mk))
        return torch.cat(outs, 0)

    def _anthropic_ef2(self, msa, mask, residual):
        """The esmfold2 kit's ``opm_fused`` on its token-major layout ([1, L, M, c_m]) per sample; the pair residual in its epilogue."""
        from types import SimpleNamespace

        from miniworld_engine.modules import _anthropic

        kit = _anthropic.load("ef2_msa_v2", path=_EF2_DRIVER)
        key = self._anthropic_key()
        if getattr(self, "_anthropic_ns_key", None) != key:
            with torch.no_grad():
                w = torch.cat([self.to_left.weight, self.to_right.weight], 0)  # ESMFold2's W: [a | b] rows
            self._anthropic_ns = SimpleNamespace(
                W=SimpleNamespace(weight=w, bias=None),
                Wout=self.to_out,
                norm=self.ln_msa,
                d_hidden=self.group_width,
                _chunk_size=None,
                divide_outer_before_proj=False,
            )
            self._anthropic_ns_key = key
        outs = []
        with master_autocast(msa, self):
            for b in range(msa.shape[0]):
                m_t = msa[b].transpose(0, 1).contiguous()[None]  # [1, L, M, c_m]
                mk = mask[b].transpose(0, 1).float().contiguous()[None]  # [1, L, M]
                pair = None if residual is None else residual[b : b + 1].contiguous()
                outs.append(kit.opm_fused(self._anthropic_ns, m_t, mk, pair))
        return torch.cat(outs, 0)

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
        if torch.cuda.get_device_capability(msa.device) == (8, 0):  # A100: kernels.integrations.opm_sm80 (forward + backward)
            if self.groups > 1:  # Chai-1: kernels.outer_product_mean.chai on its sm_80 passes (cuda/chai_sm80)
                from miniworld_engine.kernels.outer_product_mean import chai

                if chai.serves(self, msa, mask, training=torch.is_grad_enabled(), interchain=self._interchain(token_asym_id)):
                    return "chai", None
                return None, (
                    f"grouped OPM (groups {self.groups} x {self.group_width}, d_msa {msa.shape[-1]}, d_pair {self.to_out.weight.shape[0]}, N "
                    f"{msa.shape[2]}, batch {msa.shape[0]}): the A100 Chai-1 passes take 8 x 8 groups, d_msa 64, d_pair a multiple of 256, N % 64, batch 1"
                )
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
            return None, (
                f"grouped OPM (groups {self.groups} x {self.group_width}, d_msa {msa.shape[-1]}, N {msa.shape[2]}, batch {msa.shape[0]}"
                f", grad {grad}): the Chai-1 kernels take 8 x 8 groups, d_msa 64, N % 64, batch 1 (training without interchain)"
            )
        why = opm_train.refusal(
            msa,
            *dims,
            interchain=interchain if grad or self.to_left.bias is None else False,
            normalize_before_proj=self.normalize_before_proj,
            custom_divisor=custom,
        )
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
        from miniworld_engine.kernels.outer_product_mean.triton.main import refusal

        return refusal(
            msa,
            self.to_left.weight.shape[0],
            self.to_out.weight.shape[0],
            interchain=self._interchain(token_asym_id),
            residual=residual,
            groups=self.groups,
        )

    def _forward_triton(self, msa, mask=None, token_asym_id=None, residual=None) -> torch.Tensor:
        if self.dead and residual is not None and not (self.training or torch.is_grad_enabled()):
            return residual
        from miniworld_engine.kernels.outer_product_mean.triton.main import triton_outer_product_mean

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
            count_offset=self.count_offset,
            **options,
        )

    def make_inputs(self, *, length, samples, depth, dtype, device, generator, stream=None) -> dict:
        depth, d_msa = depth or 1, self.ln_msa.normalized_shape[0]
        return {
            "msa": random_input((samples, depth, length, d_msa), dtype=dtype, device=device, generator=generator),
            "mask": key_mask(samples * depth, length, device=device).view(samples, depth, length),
        }


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

    version = 2  # human-managed: bump when a change alters this class's accuracy or speed results
    backend_versions: ClassVar[dict[str, int]] = {
        "cuda:a100_sm80": 2
    }  # 2026-11-10: the BIG width (d_msa 256 / d_pair 512) on sm_80 (pwa_wide_sm80)

    #: "cuda": the H100 kernels (kernels.integrations.pwa_train / pwa_wide); "triton": kernels.pair_weighted_averaging.triton (see the
    #: _supports_ docstrings for the ranges). "anthropic" (inference only): the model kits' own masked PWA kernels (see _supports_anthropic).
    backends = ("pytorch", "anthropic", "cuda", "triton")

    # ------------------------------------------------------------------------------------------------------------------ anthropic
    def _anthropic_route(self, msa, pair, mask) -> tuple[str | None, str | None]:
        """``("boltz2" | "ef2", None)`` or ``(None, reason)``. The module is model-agnostic: the masked kernels of the kits whose models its
        rows serve --
          d_msa 64, 8 heads x 32 (Boltz-2, RoseTTAFold3, MiniWorld)  the boltz2 kit's ``fpf_msa.boltz2.msa_pair_weighted_avg``
              (``opt_core.ops.msa_pwa`` form ``masked``: torch LN_z / proj_z / masked fp32 softmax, the fused LN_m -> v | sigmoid(g)
              prologue, the contraction + gate + proj_o kernel);
          d_msa 128, d_pair 256, 8 heads x 2^k (ESMFold2; Protenix-v2 at 8 x 8: its own kit's MSA cells take no mask)  the esmfold2 kit's
              ``ef2_msa_v2.pwa_fused`` (LN + Wv | Wg producer, LN_z + bias + key mask + fp32 softmax kernel, cuBLAS bmm, gate + Wout +
              residual epilogue).
        Not served: the af3_torch kit fuses only the LN prologues (the rest is stock torch), Protenix's / AF3's 8 x 8 core form takes no mask,
        Chai-1's kit has no PWA kernel."""
        from miniworld_engine.modules import _anthropic

        if torch.is_grad_enabled():
            return None, "inference only (no kit carries a pair-weighted-averaging backward): call under torch.no_grad() / inference_mode"
        if not msa.is_cuda:
            return None, "the release's kernels run on CUDA tensors only"
        if msa.dtype != torch.bfloat16 or pair.dtype != torch.bfloat16:
            return None, f"dtype {msa.dtype} / {pair.dtype}: the kits' PWA kernels take bfloat16 activations (bf16 autocast)"
        if msa.dim() != 4 or pair.dim() != 4:
            return None, "msa must be [B, M, L, d_msa] and pair [B, L, L, d_pair]"
        if mask is not None and (mask.dtype != torch.bool or tuple(mask.shape) != (msa.shape[0], msa.shape[2])):
            return None, "mask must be a bool [B, L] key mask"
        if self.training and self.drop_msa.p_drop > 0:
            return None, "the row dropout is live (training mode): inference only"
        d_msa, d_pair, H, D = self._widths()
        if self.anthropic_word == "exact" and (d_msa, d_pair, H, D) == (64, 128, 8, 32):
            # the exact tier: opt_core.ops.msa_pwa2 (the boltz2 kit's pwa2x cell: torch LN / proj_z / masked softmax, Triton v | g, the
            # K = N contraction in a cuBLAS summation structure, proj_o)
            return ("pwa2", None) if (why := _anthropic.missing("opt_core.ops.msa_pwa2")) is None else (None, why)
        if (d_msa, H, D) == (64, 8, 32):
            return ("boltz2", None) if (why := _anthropic.missing("fpf_msa.boltz2", path="boltz2/opt/forward")) is None else (None, why)
        if d_msa == 128 and d_pair == 256 and not (H * D) & (H * D - 1) and H * D <= 256 and not D & (D - 1):
            return ("ef2", None) if (why := _anthropic.missing("ef2_msa_v2", path=_EF2_DRIVER)) is None else (None, why)
        if d_msa == 64 and d_pair == 128 and H <= 16 and 2 * H * D >= 16 and not (2 * H * D) & (2 * H * D - 1):
            # AlphaFold 3 / OpenFold3 / Protenix-v1 (8 x 8): the af3_torch kit's MSAAttention path (levers pwa_lnl + pwa_msa: lnl_fused.ln_linear
            # for LN(pair) -> logits planes and LN(msa) -> v | gate; softmax, the weighted sum, the gate and the out-projection its torch statements)
            return ("af3", None) if (why := _anthropic.missing("lnl_fused", path=_AF3T_KERNELS)) is None else (None, why)
        if D and not D & (D - 1) and D <= 64:
            # the other widths (OpenDDE 128 / 384, IntelliFold2 256 / 512: no kit of their own): the esmfold2 kit's T13 lever
            # (ef2_msa.pwa_core_t13: the module's LN / bias projection / masked fp32 softmax / v / gate, the Triton weighted-sum x gate kernel,
            # the module's out-projection)
            return ("ef2_t13", None) if (why := _anthropic.missing("ef2_msa", path=_EF2_DRIVER)) is None else (None, why)
        return None, (
            f"d_msa {d_msa}, d_pair {d_pair}, {H} heads x {D}: the kits' masked PWA kernels take d_msa 64 with 8 x 32 (boltz2), "
            "d_msa 64 / d_pair 128 (af3_torch's fused prologues), d_msa 128 / d_pair 256 (esmfold2 v2) or power-of-two heads <= 64 "
            "(esmfold2 T13)"
        )

    def _supports_anthropic(self, msa, pair, mask=None) -> str | None:
        return self._anthropic_route(msa, pair, mask)[1]

    def _anthropic_af3(self, msa, pair, mask):
        """af3_torch's ``_msaattn_forward`` with both fused prologues on (per sample: the kit's kernels are unbatched), the module's key mask
        added as the kit adds its MSA-derived one (``+ 1e9 * (mask - 1)``)."""
        from miniworld_engine.modules import _anthropic

        kit = _anthropic.load("lnl_fused", path=_AF3T_KERNELS)
        H, D = self.n_head, self._widths()[3]
        key = tuple((p.data_ptr(), p._version, p.dtype) for p in self.parameters())
        if getattr(self, "_anthropic_af3_key", None) != key:
            with torch.no_grad():
                C = self.to_bias.weight.shape[1]
                wb = torch.zeros(max(16, 1 << (H - 1).bit_length()), C, device=msa.device, dtype=torch.bfloat16)  # zero-padded logit planes
                wb[:H] = self.to_bias.weight.to(torch.bfloat16)
                self._anthropic_af3_w = {
                    "Wb": wb,
                    "lnw_z": self.ln_pair.weight.float().contiguous(),
                    "lnb_z": torch.zeros_like(self.ln_pair.weight, dtype=torch.float32),
                    "Wvg": torch.cat([self.to_value.weight, self.to_gate.weight], 0).to(torch.bfloat16).contiguous(),
                    "lnw_m": self.ln_msa.weight.float().contiguous(),
                    "lnb_m": self.ln_msa.bias.float().contiguous(),
                }
            self._anthropic_af3_key = key
        W = self._anthropic_af3_w
        HD = H * D
        outs = []
        with master_autocast(msa, self):
            for b in range(msa.shape[0]):
                _, planes = kit.ln_linear(pair[b].contiguous(), W["lnw_z"], W["lnb_z"], W["Wb"], eps=self.ln_pair.eps, planes=True)
                logits = planes[:H].float()
                if mask is not None:
                    logits = logits + 1e9 * (mask[b].float() - 1.0)
                weights = torch.softmax(logits, dim=-1)  # [H, i, j] over the keys j
                _, vg = kit.ln_linear(msa[b].contiguous(), W["lnw_m"], W["lnb_m"], W["Wvg"], eps=self.ln_msa.eps)  # [M, L, 2 H D] bf16
                v, gate = vg[..., :HD].unflatten(-1, (H, D)), vg[..., HD:]
                v_avg = torch.einsum("hqk,bkhc->bqhc", weights, v).reshape(*v.shape[:2], HD)
                outs.append(msa[b] + self.to_out(v_avg * torch.sigmoid(gate)))
        return torch.stack(outs, 0)

    #: The release tier the ``"anthropic"`` backend asks for: ``"fast"`` (default, the tier every anthropic backend takes) or ``"exact"``
    #: (the 64 / 128 / 8 x 32 cell on ``opt_core.ops.msa_pwa2``, the exact-tier replica of the stock statements).
    anthropic_word: str = "fast"

    def _anthropic_ef2_t13(self, msa, pair, mask):
        """The esmfold2 kit's ``ef2_msa.pwa_core_t13`` per sample on its token-major layout; ESMFold2's attribute schema on a namespace
        (compute_bias = LN_z + the bias projection); it returns the update (``Wout(o)``), the residual is the module's."""
        from types import SimpleNamespace

        from miniworld_engine.modules import _anthropic

        kit = _anthropic.load("ef2_msa", path=_EF2_DRIVER)
        if getattr(self, "_anthropic_t13", None) is None:
            self._anthropic_t13 = SimpleNamespace(
                n_heads=self.n_head,
                head_width=self._widths()[3],
                Wv=self.to_value,
                Wgate=self.to_gate,
                Wout=self.to_out,
                norm_single=self.ln_msa,
                compute_bias=lambda z: self.to_bias(self.ln_pair(z)),
            )
        B, _, L, _ = msa.shape
        if mask is None:
            mask = torch.ones((B, L), dtype=torch.bool, device=msa.device)
        outs = []
        with master_autocast(msa, self):
            for b in range(B):
                m_t = msa[b].transpose(0, 1).contiguous()[None]  # [1, L, M, c_m]
                pm = mask[b][None, None, :].expand(1, L, L)  # row i's keys j
                upd = kit.pwa_core_t13(self._anthropic_t13, m_t, pair[b : b + 1].contiguous(), pm)  # [1, L, M, c_m]
                outs.append(msa[b] + upd[0].transpose(0, 1))
        return torch.stack(outs, 0)

    def _anthropic_pwa2(self, msa, pair, pair_mask):
        """``opt_core.ops.msa_pwa2.pwa_forward`` per sample (Boltz-2's attribute schema; batch 1 per call), the unchunked path. The kit's
        adapter (boltz2 ``msa2.py`` pwa2x) picks the K = N summation structure by a run-time bit-compare against Boltz's stock statements
        and pins torch 2.12 / CUDA 13.0 as its vouch envelope; the kernels themselves are Triton JIT with no version check, and the
        structure here is the cell's own first candidate (``distinct_candidates(N)[0]``, cuBLAS's likeliest for the class)."""
        from miniworld_engine.modules import _anthropic

        P = _anthropic.load("opt_core.ops.msa_pwa2")
        ns = self._anthropic_ns("boltz2")
        kmode, R = P.distinct_candidates(int(msa.shape[2]))[0]
        outs = []
        with master_autocast(msa, self):
            for b in range(msa.shape[0]):
                upd = P.pwa_forward(
                    ns,
                    msa[b : b + 1].contiguous(),
                    pair[b : b + 1].contiguous(),
                    pair_mask[b : b + 1].to(torch.float32),
                    False,
                    kmode=kmode,
                    R=R,
                )
                outs.append(msa[b : b + 1] + upd)
        return torch.cat(outs, 0)

    def _anthropic_ns(self, kind: str):
        from types import SimpleNamespace

        key = (kind, *((p.data_ptr(), p._version, p.dtype) for p in self.parameters()))
        if getattr(self, "_anthropic_ns_key", None) != key:  # the kits cache their weight packs on this object: a new one per parameter set
            H, D = self.n_head, self._widths()[3]
            if kind == "boltz2":
                ns = SimpleNamespace(
                    norm_m=self.ln_msa,
                    proj_m=self.to_value,
                    proj_g=self.to_gate,
                    norm_z=self.ln_pair,
                    proj_z=self.to_bias,
                    proj_o=self.to_out,
                    inf=1e6,
                    num_heads=H,
                    c_h=D,
                )
            else:
                ns = SimpleNamespace(
                    n_heads=H,
                    head_width=D,
                    Wv=self.to_value,
                    Wgate=self.to_gate,
                    Wout=self.to_out,
                    compute_bias=(self.ln_pair, self.to_bias),
                    norm_single=self.ln_msa,
                )
            self._anthropic_ns_obj, self._anthropic_ns_key = ns, key
        return self._anthropic_ns_obj

    def _forward_anthropic(self, msa, pair, mask=None) -> torch.Tensor:
        if self.dead:  # inference only: a dead instance returns the residual input (as the reference does)
            return msa
        from miniworld_engine.modules import _anthropic

        route, _ = self._anthropic_route(msa, pair, mask)
        B, _, L, _ = msa.shape
        if mask is None:
            mask = torch.ones((B, L), dtype=torch.bool, device=msa.device)
        pair_mask = mask[:, None, :].expand(B, L, L)  # row i's keys j: the module's key mask
        if route == "af3":
            return self._anthropic_af3(msa, pair, mask)
        if route == "ef2_t13":
            return self._anthropic_ef2_t13(msa, pair, mask)
        if route == "pwa2":
            return self._anthropic_pwa2(msa, pair, pair_mask)
        ns = self._anthropic_ns(cast(str, route))  # a supported call has a route
        with master_autocast(msa, self):
            if route == "boltz2":
                kit = _anthropic.load("fpf_msa.boltz2", path="boltz2/opt/forward")
                return msa + kit.msa_pair_weighted_avg(ns, msa.contiguous(), pair.contiguous(), pair_mask.to(torch.float32), False)
            kit = _anthropic.load("ef2_msa_v2", path=_EF2_DRIVER)
            outs = []
            for b in range(B):  # the kit's token-major layout [1, L, M, c_m], batch 1; it returns m + pwa
                m_t = msa[b].transpose(0, 1).contiguous()[None]
                outs.append(kit.pwa_fused(ns, m_t, pair[b : b + 1].contiguous(), pair_mask[b : b + 1].contiguous())[0].transpose(0, 1))
            return torch.stack(outs, 0)

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
        8 / 16, or d_msa 256, d_pair 512, 8 heads x 32 (the BIG build); N <= 1024 training / 2048 inference): bf16 activations, batch 1, sm_90a, forward + backward
        (the residual and the row dropout in the kernels)."""
        why = self._common_refusal(msa, pair, mask)
        if why is not None:
            return why
        if torch.cuda.get_device_capability(msa.device) == (8, 0):  # A100: kernels.integrations.pwa_sm80 (forward + backward)
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
        if p_drop > 0:  # the keep-mask Dropout(broadcast_dim=1) draws, broadcast over the MSA rows
            keep = torch.rand((msa.shape[0], msa.shape[2], msa.shape[3]), device=msa.device, dtype=msa.dtype) > p_drop
        return triton_pair_weighted_averaging(
            msa,
            pair,
            mask,
            self.ln_msa.weight,
            self.ln_msa.bias,
            self.to_value.weight,
            self.to_gate.weight,
            self.ln_pair.weight,
            self.to_bias.weight,
            self.to_out.weight,
            eps_msa=float(self.ln_msa.eps),
            eps_pair=float(self.ln_pair.eps),
            keep=keep,
            p_drop=p_drop,
        )

    def make_inputs(self, *, length, samples, depth, dtype, device, generator, stream=None) -> dict:
        d_msa, d_pair = self.ln_msa.normalized_shape[0], self.ln_pair.normalized_shape[0]
        return {
            "msa": random_input((samples, depth or 1, length, d_msa), dtype=dtype, device=device, generator=generator),
            "pair": random_input((samples, length, length, d_pair), dtype=dtype, device=device, generator=generator),
            "mask": key_mask(samples, length, device=device),
        }
