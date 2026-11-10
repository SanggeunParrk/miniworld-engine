"""Transitions: ``Transition`` (LayerNorm + SwiGLU expand / squeeze + residual), ``ConditionedTransition`` (AlphaFold 3 Algorithm 25:
AdaLN + SwiGLU + sigmoid conditioning gate + residual) and Boltz-2's ``UpGatedConditionedTransition`` (a third, up-gate projection).

Each is a :class:`~miniworld_engine.modules.primitives.BackendModule`; ``_forward_pytorch`` is the reference.
"""

from typing import ClassVar

import torch
import torch.nn.functional as F

from miniworld_engine.modules.functional import swish_gate
from miniworld_engine.modules.norm import AdaptiveLayerNorm, LayerNorm, _as_rows3, a100, h100_unsupported, triton_unsupported
from miniworld_engine.modules.primitives import BackendModule, Linear, MasterWeights, master_linear, random_input, stream_shape

#: The OpenDDE kit's transition composition (``fpf_transition_odde``), a release directory for ``modules/_anthropic.load``.
_ODDE_ARMT = "opendde/opt/forward/fast_inference/levers/ARMT/third_party"


class Transition(BackendModule, MasterWeights):
    """Transition layer with SwiGLU activation::

        h = LN(x)                                  # ln_in
        y = x + W_s (silu(W_a h) * (W_b h))        # expand_a, expand_b, squeeze (no biases)

    Parameters
    ----------
    d_hidden : int
        Dimension of the input and output features.
    n : int
        Expansion factor.
    master_weights : bool
        Keep fp32 master parameters under a bf16 cast (:class:`MasterWeights`).
    backend : str
        See :class:`BackendModule`. ``"anthropic"``: the shared core's transition provider (``opt_core.kernels.transition``, the af3_torch
        kit's lever ``transition``) by its ``fast`` tier word -- LayerNorm + SwiGLU + output projection in its fused rows (the residual
        added after, as the reference does), bfloat16 / float32 cells, inference only, where the provider's measured row for the cell is one of its kernels.
        ``"cuda"`` (H100): the engine's sm_90a Transition kernels (``kernels/transition/cuda.transition_sm90a``: LayerNorm + SwiGLU expand +
        squeeze with the residual in its epilogue, forward and backward): ``fused_sm90a`` (d 128, n 2 / 4), ``fused_wide_sm90a`` (d 64 /
        256 / 384 / 512 with n 2 / 4, d 768 with n 2; whole 128-row tiles) and ``fused_generic_sm90a`` (d a multiple of 64 up to 1024,
        hidden a multiple of 128, any row count; fp32 activations on TF32 tensor cores, so only where torch allows TF32); fp32 at d 64 / 128
        (n 2 / 4), 256 / 512 (n 2) inference and d 128 / 256 / 512 (n 2) training runs ``fused_wide_sm90a.transition_f32_sm90a`` instead
        (the bf16 builds' fused forwards with fp16 products at TF32's rounding; the generic TF32 backward). bf16 / fp32
        parameters (fp32 masters cast inside, their gradients fp32). ``"triton"``: the Triton split Transition
        (``kernels/transition/triton/residual.transition_residual``: the LayerNorm that also hands back the residual, the expand-SwiGLU,
        the squeeze with the residual in its epilogue, and their backward), any width, bf16 / fp16 / fp32 (fp32 dots IEEE unless torch
        allows TF32), the parameters cast to the activation dtype.

    """
    version = 2   # human-managed: bump when a change alters this class's accuracy or speed results

    backends = ("pytorch", "anthropic", "cuda", "triton")
    backend_versions: ClassVar[dict[str, int]] = {"anthropic": 2, "cuda:h100_sm90": 4, "cuda:a100_sm80": 2}   # 2026-11-10: kits; H100 d384 cuBLAS; fp32 eval f32io; fp32 fused fwd d256 / d512 + train; A100 capture forms + fp32 (TF32) build

    #: A numerically dead instance of a checkpoint (set by a model's checkpoint loader): inference returns the residual input.
    dead: bool = False

    def __init__(
        self,
        d_hidden: int = 128,
        n: int = 4,
        master_weights: bool = False,
        *,
        backend: str = "pytorch",
    ) -> None:
        super().__init__()
        self.d_hidden = d_hidden
        self.n = n
        self.ln_in = LayerNorm(d_hidden, dtype=torch.bfloat16)
        self.expand_a = Linear(d_hidden, d_hidden * n, bias=False, init="relu", dtype=torch.bfloat16)
        self.expand_b = Linear(d_hidden, d_hidden * n, bias=False, init="relu", dtype=torch.bfloat16)
        self.squeeze = Linear(d_hidden * n, d_hidden, bias=False, init="zero", dtype=torch.bfloat16)
        if master_weights:
            self.set_master_weights()
        self.set_backend(backend)

    def _forward_pytorch(self, x: torch.Tensor) -> torch.Tensor:
        """ALWAYS returns the residual output ``y = x + transition(x)`` (the residual is this module's own input ``x``)."""
        if self.dead and not (self.training or torch.is_grad_enabled()):
            return x
        # The norm affine is fp32-pinned and the projections may be fp32 masters: every parameter is cast to the activation dtype.
        h = F.layer_norm(
            x,
            (self.d_hidden,),
            self.ln_in.weight.to(x.dtype),
            self.ln_in.bias.to(x.dtype),
            self.ln_in.eps,
        )
        a = F.linear(h, self.expand_a.weight.to(x.dtype))
        b = F.linear(h, self.expand_b.weight.to(x.dtype))
        h = swish_gate(a, b)
        return F.linear(h, self.squeeze.weight.to(x.dtype)) + x

    def make_inputs(self, *, length, samples, depth, dtype, device, generator, stream=None) -> dict:
        shape = stream_shape(stream, samples=samples, length=length, depth=depth, width=self.d_hidden)
        return {"x": random_input(shape, dtype=dtype, device=device, generator=generator)}

    # ------------------------------------------------------------------------------------------------------------------ anthropic
    #: The activation dtypes the provider has transition cells for (its cell table: bf16 | fp32).
    _ANTHROPIC_DTYPES = (torch.bfloat16, torch.float32)
    #: Provider cells that dump core on this H100 stack (driver 575 / CUDA 12.9, torch 2.13 + cu129), refused by name: pair_c256_n4 (its
    #: serving call engages the cell's esm_t16 cubin, built with nvrtc 13.0, at every size measured: L 128 and 384).
    _ANTHROPIC_EXCLUDED_CELLS: tuple[str, ...] = ("|pair_c256_n4|",)

    @staticmethod
    def _anthropic_family(x: torch.Tensor) -> str:
        """The provider's cell family of the call: ``pair`` ([B, L, L, c]: L^2 rows), ``single`` ([B, L, c]), ``rows`` (an MSA [B, S, L, c])."""
        if x.dim() == 4 and x.shape[1] == x.shape[2]:
            return "pair"
        return "single" if x.dim() == 3 else "rows"

    def _anthropic_select(self, x: torch.Tensor):
        from miniworld_engine.modules import _anthropic

        prov = _anthropic.load("opt_core.kernels.transition")
        return prov, _anthropic.kernel_selection(
            prov, "fast", c=self.d_hidden, hidden=self.d_hidden * self.n, n_tokens=int(x.shape[-2]), dtype={torch.bfloat16: "bf16", torch.float32: "fp32"}[x.dtype],
            family=self._anthropic_family(x), residual=False, cc="{}.{}".format(*torch.cuda.get_device_capability(x.device)), rows_count=x.numel() // self.d_hidden,
        )

    def _anthropic_core_refusal(self, x: torch.Tensor) -> str | None:
        sel, why = self._anthropic_select(x)[1]
        if sel is not None and any(c in str(sel.cell_key) for c in self._ANTHROPIC_EXCLUDED_CELLS):
            return f"the provider cell {sel.cell_key} dumps core on this H100 stack (driver 575 / CUDA 12.9): refused by name"
        return why

    #: Where the shared core's provider has no cell for the call, the kits' own transition kernels of that form (the module is model-agnostic:
    #: the kit of the models the row serves):
    #:   ``"ef2"``  an MSA stream of width 128, bf16 (ESMFold2 / OpenDDE / Protenix-v2 rows): the esmfold2 kit's ``ef2_msa_v2.mtr_fused``
    #:              (LN + W12 + silu(x1) * x2 + W3 + the residual in one Triton kernel per row block; hidden a multiple of 64)
    #:   ``"pf"``   otherwise: ``opt_core.attn.pair_fused.transition`` (impl fpf, LN in the kernel; the pair transition the Boltz-2 pairfuse and
    #:              OpenFold3 kits install), served where its cell table / safe settings take (C, n*C)
    def _anthropic_route(self, x: torch.Tensor) -> tuple[str | None, str | None]:
        from miniworld_engine.modules import _anthropic

        if (self._anthropic_family(x) == "pair" and (self.d_hidden, self.d_hidden * self.n) == (384, 1536)
                and x.dtype == torch.bfloat16 and not self.training):
            # the pair transition of OpenDDE (c_z 384 x 4; the row is OpenDDE's alone): its kit's own composition (fpf_transition_odde sep16:
            # the module's LN -> bf16, two cuBLAS GEMMs, the Triton silu(a) * b kernel, the out GEMM)
            why = _anthropic.missing("fpf_transition_odde", path=_ODDE_ARMT)
            if why is None:
                return "odde", None
        why_core = self._anthropic_core_refusal(x)
        if why_core is None:
            return "core", None
        if x.dtype == torch.bfloat16 and self._anthropic_family(x) == "rows" and self.d_hidden == 128 and (self.d_hidden * self.n) % 64 == 0:
            why = _anthropic.missing("ef2_msa_v2", path="esmfold2/opt/forward/fast_inference/driver")
            if why is None:
                return "ef2", None
            why_core += f"; esmfold2 mtr_fused: {why}"
        why = _anthropic.missing("opt_core.attn.pair_fused")
        if why is None:
            ok, why = _anthropic.load("opt_core.attn.pair_fused").supported_transition(x, self._anthropic_pf_weights(x), impl="fpf", ln="fused",
                                                                                        residual=False)
            if ok:
                return "pf", None
        return None, f"{why_core}; pair_fused.transition: {why}"

    def _anthropic_pf_weights(self, x: torch.Tensor):
        from miniworld_engine.modules import _anthropic

        params = (self.expand_a.weight, self.expand_b.weight, self.squeeze.weight, self.ln_in.weight, self.ln_in.bias)
        key = (x.device, *((p.data_ptr(), p._version, p.dtype) for p in params))
        if getattr(self, "_anthropic_pf_key", None) != key:
            self._anthropic_pf = _anthropic.load("opt_core.attn.pair_fused").pack_transition_weights(
                w_out=params[2], ln_w=params[3], ln_b=params[4], w_a=params[0], w_b=params[1], eps=self.ln_in.eps, device=x.device)
            self._anthropic_pf_key = key
        return self._anthropic_pf

    def _supports_anthropic(self, x: torch.Tensor) -> str | None:
        from miniworld_engine.modules import _anthropic

        why = _anthropic.unavailable("opt_core.kernels.transition", x.device)
        if why is not None:
            return why
        if x.dtype not in self._ANTHROPIC_DTYPES:
            return f"dtype {x.dtype}: the provider has bfloat16 / float32 cells only"
        if x.dim() < 2:
            return "x must be [..., L, d_hidden]"
        if torch.is_grad_enabled():
            if any(p.requires_grad for p in self.parameters()):
                return ("training with parameter gradients: the kits' differentiable transitions (opt_core transition_autograd rows esm_kd3*, "
                        "ef2inv LNLinearSwiGLUFrozen) are dX-only with frozen weights; freeze the parameters or call under torch.no_grad()")
            return self._anthropic_autograd_select(x)[1][1]
        return self._anthropic_route(x)[1]

    def _anthropic_autograd_select(self, x: torch.Tensor):
        """The provider's differentiable row (``transition_autograd``: esm_kd3 / esm_t15_kd3 / esm_t16_kd3, forward + dX backward, frozen
        weights) for a frozen module whose input takes a gradient."""
        from miniworld_engine.modules import _anthropic

        prov = _anthropic.load("opt_core.kernels.transition")
        return prov, _anthropic.kernel_selection(
            prov, "fast", c=self.d_hidden, hidden=self.d_hidden * self.n, n_tokens=int(x.shape[-2]),
            dtype={torch.bfloat16: "bf16", torch.float32: "fp32"}[x.dtype], direction="fwdbwd", family="esmpair", residual=True,
            cc="{}.{}".format(*torch.cuda.get_device_capability(x.device)), rows_count=x.numel() // self.d_hidden)

    def _forward_anthropic(self, x: torch.Tensor) -> torch.Tensor:
        if self.dead:   # inference only: a dead instance returns the residual input (as the reference does)
            return x
        if torch.is_grad_enabled():   # a frozen module, the input takes a gradient: the provider's differentiable row (dX backward)
            prov, (sel, _why) = self._anthropic_autograd_select(x)
            with torch.no_grad():
                pack = prov.pack(w_a=self.expand_a.weight.to(x.dtype), w_b=self.expand_b.weight.to(x.dtype), w_o=self.squeeze.weight.to(x.dtype),
                                 ln_w=self.ln_in.weight.to(x.dtype), ln_b=self.ln_in.bias.to(x.dtype), eps=self.ln_in.eps)
            return prov.transition_autograd(x.contiguous(), pack, word="fast", n_tokens=int(x.shape[-2]), _sel=sel)[0]
        route, _ = self._anthropic_route(x)
        if route == "odde":
            from types import SimpleNamespace

            from miniworld_engine.modules import _anthropic

            kit = _anthropic.load("fpf_transition_odde", path=_ODDE_ARMT)
            if getattr(self, "_anthropic_odde", None) is None:   # OpenDDE's attribute schema; the kit caches its bf16 weights by version
                self._anthropic_odde = SimpleNamespace(c_in=self.d_hidden, linear_no_bias_a=self.expand_a, linear_no_bias_b=self.expand_b,
                                                       linear_no_bias=self.squeeze, layernorm1=self.ln_in, training=False)
            return kit.forward_sep16(self._anthropic_odde, x.contiguous()).to(x.dtype) + x
        if route == "ef2":
            from types import SimpleNamespace

            from miniworld_engine.modules import _anthropic

            kit = _anthropic.load("ef2_msa_v2", path="esmfold2/opt/forward/fast_inference/driver")
            key = tuple((p.data_ptr(), p._version) for p in self.parameters())
            if getattr(self, "_anthropic_ef2_key", None) != key:   # the kit caches W12^T / W3^T on this object by the weights' identity
                with torch.no_grad():
                    w12 = torch.cat([self.expand_a.weight, self.expand_b.weight], 0)        # [x1 | x2] rows: x1 the silu branch
                self._anthropic_ef2 = SimpleNamespace(norm=self.ln_in, ffn=SimpleNamespace(w12=SimpleNamespace(weight=w12), w3=self.squeeze))
                self._anthropic_ef2_key = key
            return kit.mtr_fused(self._anthropic_ef2, x.contiguous())                        # x + transition(x), bf16
        if route == "pf":
            from miniworld_engine.modules import _anthropic

            PF = _anthropic.load("opt_core.attn.pair_fused")
            x = x.contiguous()
            return PF.transition(x, self._anthropic_pf_weights(x), residual=False, ln="fused", impl="fpf").to(x.dtype) + x
        prov, (sel, _why) = self._anthropic_select(x)
        if sel.row in ("engine_module", "compile"):   # the provider measured the caller's own module fastest for this cell: the release runs it
            return self._forward_pytorch(x)
        params = (self.expand_a.weight, self.expand_b.weight, self.squeeze.weight, self.ln_in.weight, self.ln_in.bias)
        key = (x.dtype, *((p.data_ptr(), p._version, p.dtype) for p in params))
        if getattr(self, "_anthropic_pack_key", None) != key:
            # the reference casts every parameter to the activation dtype (fp32 masters, the fp32-pinned LayerNorm affine): the same values here
            self._anthropic_pack = prov.pack(w_a=params[0].to(x.dtype), w_b=params[1].to(x.dtype), w_o=params[2].to(x.dtype),
                                             ln_w=params[3].to(x.dtype), ln_b=params[4].to(x.dtype), eps=self.ln_in.eps)
            self._anthropic_pack_key = key
        x = x.contiguous()
        y, served = prov.transition(x, self._anthropic_pack, word="fast", residual=False, n_tokens=int(x.shape[-2]),
                                    family=self._anthropic_family(x))
        if served.row != sel.row:
            from miniworld_engine.modules.primitives import BackendUnsupported

            raise BackendUnsupported(f"Transition backend 'anthropic': the provider served row {served.row!r}, its selection was {sel.row!r}")
        return y + x   # the update in the activation dtype, then the residual: the reference's two roundings

    # ------------------------------------------------------------------------------------------------------------- engine kernels
    def _supports_engine(self, x: torch.Tensor) -> str | None:
        if x.dim() < 2 or x.shape[-1] != self.d_hidden:
            return f"x must be [..., {self.d_hidden}]"
        if x.numel() == 0:
            return "empty input"
        for p in (self.ln_in.weight, self.ln_in.bias):
            if p is None:
                return "the kernels take the affine LayerNorm (weight and bias)"
        return None

    def _supports_cuda(self, x: torch.Tensor) -> str | None:
        if a100(x):
            return self._supports_cuda_sm80(x)
        from miniworld_engine.kernels.transition.cuda import fused_generic_sm90a, fused_sm90a, fused_wide_sm90a

        why = h100_unsupported(x) or self._supports_engine(x)
        if why is not None:
            return why
        if x.dtype not in (torch.bfloat16, torch.float32):
            return f"dtype {x.dtype}: the sm_90a kernels take bfloat16 (float32 on TF32) activations"
        matrices = (self.expand_a.weight, self.expand_b.weight, self.squeeze.weight)
        if any(p.dtype not in (torch.bfloat16, torch.float32) for p in matrices) or len({p.dtype for p in matrices}) != 1:
            return "the projections must share one dtype, bfloat16 or float32"
        wa, ws = self.expand_a.weight, self.squeeze.weight
        if any(k.supported(x, wa, ws) for k in (fused_sm90a, fused_wide_sm90a, fused_generic_sm90a)):
            return None
        if x.dtype is torch.float32 and not fused_generic_sm90a._tf32_allowed():
            return "float32 activations run on TF32 tensor cores only (torch does not allow TF32 here)"
        return (f"d_hidden {self.d_hidden}, hidden {wa.shape[0]}, {x.numel() // self.d_hidden} rows: no sm_90a build takes it (generic: d a "
                "multiple of 64 up to 1024, hidden a multiple of 128)")

    def _forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        if self.dead and not (self.training or torch.is_grad_enabled()):
            return x
        if a100(x):
            from miniworld_engine.kernels.transition.cuda import transition_sm80

            out = transition_sm80(x, self.ln_in.weight, self.ln_in.bias, self.expand_a.weight, self.expand_b.weight, self.squeeze.weight,
                                  self.ln_in.eps)
            if out is None:
                from miniworld_engine.modules.primitives import BackendUnsupported

                raise BackendUnsupported("Transition backend 'cuda': no sm_80 Transition build is available (the extension failed to build)")
            return out
        from miniworld_engine.kernels.transition.cuda import transition_sm90a

        out = transition_sm90a(x, self.ln_in.weight, self.ln_in.bias, self.expand_a.weight, self.expand_b.weight, self.squeeze.weight,
                               self.ln_in.eps, self.n)
        if out is None:   # every build that takes the call failed to compile: an environment fault, never another backend
            from miniworld_engine.modules.primitives import BackendUnsupported

            raise BackendUnsupported("Transition backend 'cuda': no sm_90a Transition build is available (the extension failed to build)")
        return out

    def _supports_cuda_sm80(self, x: torch.Tensor) -> str | None:
        """A100: ``kernels/transition/cuda.transition_sm80`` (``fused_sm80`` at D 128 / n 4 with whole 256-row tiles and enough rows,
        ``fused_wide_sm80`` at D 64 / 128 / 256 / 384 / 512 / 768 with a hidden width a multiple of 64, any row count): bf16 activations,
        bf16 or fp32-master parameters, forward and backward; fp32 activations where torch allows TF32 (``fused_wide_sm80.transition_f32_sm80``:
        fp16 products at TF32's rounding, a TF32 backward)."""
        from miniworld_engine.kernels.transition.cuda import fused_wide_sm80

        why = self._supports_engine(x)
        if why is not None:
            return why
        if x.dtype not in (torch.bfloat16, torch.float32):
            return f"dtype {x.dtype}: the sm_80 Transition kernels take bfloat16 (float32 on TF32) activations"
        matrices = (self.expand_a.weight, self.expand_b.weight, self.squeeze.weight)
        if any(p.dtype not in (torch.bfloat16, torch.float32) for p in matrices) or len({p.dtype for p in matrices}) != 1:
            return "the projections must share one dtype, bfloat16 or float32"
        wa, ws = self.expand_a.weight, self.squeeze.weight
        if x.dtype is torch.float32:
            from miniworld_engine.kernels.transition.cuda.fused_generic_sm90a import _tf32_allowed

            if not _tf32_allowed():
                return "float32 activations run on TF32 tensor cores only (torch does not allow TF32 here)"
            if not fused_wide_sm80.f32_supported(x, wa, ws):
                return (f"d_hidden {self.d_hidden}, hidden {wa.shape[0]}: no sm_80 fp32 build takes it (D in {fused_wide_sm80.F32_WIDTHS}, hidden a "
                        "multiple of 64)")
            if not fused_wide_sm80.f32_available(x, wa, ws):
                return "the sm_80 Transition extension failed to build"
            return None
        if not fused_wide_sm80.supported(x, wa, ws):
            return (f"d_hidden {self.d_hidden}, hidden {wa.shape[0]}: no sm_80 build takes it (D in {fused_wide_sm80.WIDTHS}, hidden a "
                    "multiple of 64)")
        if fused_wide_sm80.route(x, wa, ws) is None:
            return "the sm_80 Transition extension failed to build"
        return None

    def _supports_triton(self, x: torch.Tensor) -> str | None:
        why = triton_unsupported(x) or self._supports_engine(x)
        if why is not None:
            return why
        if x.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            return f"dtype {x.dtype}: the Triton kernels take bfloat16 / float16 / float32 activations"
        return None

    def _forward_triton(self, x: torch.Tensor) -> torch.Tensor:
        if self.dead and not (self.training or torch.is_grad_enabled()):
            return x
        from miniworld_engine.kernels.transition.triton.residual import transition_residual

        rows = _as_rows3(x)
        # the reference casts every projection to the activation dtype (fp32 masters, bf16 weights under fp32 activations): the casts stay
        # in autograd, so a master's gradient lands on it in fp32
        y = transition_residual(rows, self.ln_in.weight, self.ln_in.bias, self.expand_a.weight.to(x.dtype),
                                self.expand_b.weight.to(x.dtype), self.squeeze.weight.to(x.dtype), self.ln_in.eps)
        return y.reshape(x.shape)


class ConditionedTransition(BackendModule):
    """AdaLN + SwiGLU transition + sigmoid conditioning gate. Given the RAW activation ``x`` and conditioning ``cond``::

        h = ada_ln_in(x, cond)                 # AdaLN
        a = h @ Wa^T ; b = h @ Wb^T            # expand_a, expand_b (no bias)
        out = (silu(a) * b) @ Ws^T             # SwiGLU, squeeze (no bias)
        scale = cond @ Wsc^T + b_sc            # to_scale (with bias)
        return x + sigmoid(scale) * out        # output gate, residual

    Parameters
    ----------
    d_hidden : int
        Feature dimension of ``x`` (and of the output).
    d_cond : int
        Feature dimension of the conditioning input ``cond``.
    n : int
        SwiGLU expansion factor (hidden width ``n * d_hidden``).
    dtype : torch.dtype
        Dtype of the projections and of the AdaLN (fp32 by default).
    backend : str
        See :class:`BackendModule`.
    """

    backends = ("pytorch",)

    def __init__(
        self,
        d_hidden: int = 128,
        d_cond: int = 128,
        n: int = 2,
        dtype: torch.dtype = torch.float32,
        *,
        backend: str = "pytorch",
    ) -> None:
        super().__init__()
        self.d_hidden = d_hidden
        self.d_cond = d_cond
        self.n = n
        # Checkpoint keys: ada_ln_in.* + the flat tail.
        self.ada_ln_in = AdaptiveLayerNorm(d_hidden=d_hidden, d_cond=d_cond, dtype=dtype)
        self.dtype = dtype
        self.expand_a = Linear(d_hidden, d_hidden * n, bias=False, init="relu", dtype=dtype)
        self.expand_b = Linear(d_hidden, d_hidden * n, bias=False, init="relu", dtype=dtype)
        self.squeeze = Linear(d_hidden * n, d_hidden, bias=False, init="zero", dtype=dtype)
        self.to_scale = Linear(d_cond, d_hidden, bias=True, init="default", dtype=dtype)
        self.set_backend(backend)

    def _forward_pytorch(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """ALWAYS returns the residual output ``x + delta(x, cond)`` (the residual is this module's own input). A caller that composes the
        update some other way (a magnitude-preserving sum, a released checkpoint's own residual) takes :meth:`delta`."""
        return x + self.delta(x, cond)

    def delta(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """The update alone (no residual). ``x`` is the RAW residual-stream activation, ``cond`` the conditioning signal."""
        h = self.ada_ln_in(x, cond)
        out = master_linear(self.squeeze, self._hidden(h))
        scale = master_linear(self.to_scale, cond)
        return torch.sigmoid(scale) * out

    def _hidden(self, h: torch.Tensor) -> torch.Tensor:
        """The SwiGLU hidden ``silu(W_a h) * (W_b h)`` of the normalised ``h`` (``[..., n * d_hidden]``)."""
        a = master_linear(self.expand_a, h)
        b = master_linear(self.expand_b, h)
        return F.silu(a) * b


class UpGatedConditionedTransition(ConditionedTransition):
    """Boltz-2's conditioned transition: :class:`ConditionedTransition` whose SwiGLU hidden is gated once more by a third projection of the
    normalised input (``up``, no bias)::

        hidden = silu(W_a h) * (W_b h) * (W_u h)
    """

    def __init__(
        self,
        d_hidden: int = 128,
        d_cond: int = 128,
        n: int = 2,
        dtype: torch.dtype = torch.float32,
        *,
        backend: str = "pytorch",
    ) -> None:
        super().__init__(d_hidden, d_cond, n, dtype, backend=backend)
        self.up = Linear(d_hidden, d_hidden * n, bias=False, init="default", dtype=dtype)

    def _hidden(self, h: torch.Tensor) -> torch.Tensor:
        return super()._hidden(h) * master_linear(self.up, h)
