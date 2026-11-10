"""Transitions: ``Transition`` (LayerNorm + SwiGLU expand / squeeze + residual), ``ConditionedTransition`` (AlphaFold 3 Algorithm 25:
AdaLN + SwiGLU + sigmoid conditioning gate + residual) and Boltz-2's ``UpGatedConditionedTransition`` (a third, up-gate projection).

Each is a :class:`~miniworld_engine.modules.primitives.BackendModule`; ``_forward_pytorch`` is the reference.
"""

import torch
import torch.nn.functional as F

from miniworld_engine.modules.functional import swish_gate
from miniworld_engine.modules.norm import AdaptiveLayerNorm, LayerNorm, _as_rows3, a100, h100_unsupported, triton_unsupported
from miniworld_engine.modules.primitives import BackendModule, Linear, MasterWeights, master_linear, random_input, stream_shape


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
        hidden a multiple of 128, any row count; fp32 activations on TF32 tensor cores, so only where torch allows TF32). bf16 / fp32
        parameters (fp32 masters cast inside, their gradients fp32). ``"triton"``: the Triton split Transition
        (``kernels/transition/triton/residual.transition_residual``: the LayerNorm that also hands back the residual, the expand-SwiGLU,
        the squeeze with the residual in its epilogue, and their backward), any width, bf16 / fp16 / fp32 (fp32 dots IEEE unless torch
        allows TF32), the parameters cast to the activation dtype.

    """
    version = 1   # human-managed: bump when a change alters this class's accuracy or speed results

    backends = ("pytorch", "anthropic", "cuda", "triton")

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

    def _supports_anthropic(self, x: torch.Tensor) -> str | None:
        from miniworld_engine.modules import _anthropic

        why = _anthropic.unavailable("opt_core.kernels.transition", x.device)
        if why is not None:
            return why
        if torch.is_grad_enabled():
            return ("inference only: the provider's differentiable transition (transition_autograd) keeps the weights frozen (no weight "
                    "gradients); call under torch.no_grad() / inference_mode")
        if x.dtype not in self._ANTHROPIC_DTYPES:
            return f"dtype {x.dtype}: the provider has bfloat16 / float32 cells only"
        if x.dim() < 2:
            return "x must be [..., L, d_hidden]"
        sel, why = self._anthropic_select(x)[1]
        if sel is not None and any(c in str(sel.cell_key) for c in self._ANTHROPIC_EXCLUDED_CELLS):
            return f"the provider cell {sel.cell_key} dumps core on this H100 stack (driver 575 / CUDA 12.9): refused by name"
        return why

    def _forward_anthropic(self, x: torch.Tensor) -> torch.Tensor:
        if self.dead:   # inference only: a dead instance returns the residual input (as the reference does)
            return x
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
        bf16 or fp32-master parameters, forward and backward."""
        from miniworld_engine.kernels.transition.cuda import fused_wide_sm80

        why = self._supports_engine(x)
        if why is not None:
            return why
        if x.dtype is not torch.bfloat16:
            return f"dtype {x.dtype}: the sm_80 Transition kernels take bfloat16 activations (no fp32 / TF32 A100 build)"
        matrices = (self.expand_a.weight, self.expand_b.weight, self.squeeze.weight)
        if any(p.dtype not in (torch.bfloat16, torch.float32) for p in matrices) or len({p.dtype for p in matrices}) != 1:
            return "the projections must share one dtype, bfloat16 or float32"
        wa, ws = self.expand_a.weight, self.squeeze.weight
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
