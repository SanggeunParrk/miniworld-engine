"""The token DiT block variants of Boltz-2, RoseTTAFold3 and Chai-1 (FoldForge's dense ``DiffusionTransformer`` with their spec
flags). The math is in ``kernels/dit_variants/reference.py``; on H100 a bf16 call without a gradient runs
``kernels/dit_variants/interface.py`` (CUDA rows + cuBLAS + the token DiT's attention core), a call with a gradient (bf16 under autocast or fp32 on
TF32) ``kernels/dit_variants/cuda/train.py`` (CUDA row kernels forward and backward, cuBLAS, the sm_90a attention cores), anything else the
portable path or the reference.

The four models' options (AlphaFold 3's own token / atom stacks are the ``AF3`` / ``AF3_ATOM`` presets):

    AF3       sequential, sigmoid AdaLN, gating query, plain transition, d_cond 384, pair 128, one shared pair LayerNorm
    Boltz-2  sequential, sigmoid AdaLN, gating query, up-gate transition (silu(g) v u), d_cond 768, pair 128
    RF3       parallel, sigmoid AdaLN, gating query, kq-norm (LayerNorm over the flat q / k), d_cond 384, pair 128
    Chai-1    parallel, identity AdaLN ((W s + 1) LN(x) + W s), no gating query, d_cond 384, pair 256 (affine pair LN)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from miniworld_engine.kernels.dit_variants import reference
from miniworld_engine.modules.exceptions import ImplementationType
from miniworld_engine.modules.primitives import ImplementationSwitch

#: Each model's options (``DiTVariantStack(n_blocks, **BOLTZ2)`` ...).
BOLTZ2 = {"d_cond": 768, "d_pair": 128, "parallel": False, "identity_scale": False, "gating": True, "kq_norm": False,
          "up_gate": True, "pair_norm_bias": False}
ROSETTAFOLD3 = {"d_cond": 384, "d_pair": 128, "parallel": True, "identity_scale": False, "gating": True, "kq_norm": True,
                "up_gate": False, "pair_norm_bias": False}
CHAI1 = {"d_cond": 384, "d_pair": 256, "parallel": True, "identity_scale": True, "gating": False, "kq_norm": False,
         "up_gate": False, "pair_norm_bias": True}
#: AlphaFold 3's own token DiT (FoldForge's ``DiffusionTransformer`` with spec ``ALPHAFOLD3``): sequential, sigmoid AdaLN, gating query, a
#: plain SwiGLU transition (no up-gate), d_cond 384, pair 128, ONE scale-only pair LayerNorm whose projection is one matrix per super block of
#: four blocks (``from_released(..., super_block_size=4)`` slices it per block). The same kernels as Boltz-2's, with ``up_gate`` off.
AF3 = {"d_cond": 384, "d_pair": 128, "parallel": False, "identity_scale": False, "gating": True, "kq_norm": False,
       "up_gate": False, "pair_norm_bias": False}


class VariantAdaLN(nn.Module):
    """AdaLN's conditioning projections: sigmoid form (``cond_norm_weight``, ``to_scale`` with bias, ``to_shift``) or identity form
    (``to_scale`` / ``to_shift`` without bias)."""

    def __init__(self, d: int, d_cond: int, identity_scale: bool) -> None:
        super().__init__()
        self.identity_scale = identity_scale
        if not identity_scale:
            self.cond_norm_weight = nn.Parameter(torch.ones(d_cond))
        self.to_scale = nn.Linear(d_cond, d, bias=not identity_scale)
        self.to_shift = nn.Linear(d_cond, d, bias=False)


class DiTVariantBlock(nn.Module):
    def __init__(self, d_single: int = 768, d_cond: int = 384, d_pair: int = 128, n_head: int = 16, n: int = 2, *,
                 identity_scale: bool = False, gating: bool = True, kq_norm: bool = False, up_gate: bool = False,
                 pair_norm_bias: bool = False, adaln_eps: float = 1e-5, pair_eps: float = 1e-5) -> None:
        super().__init__()
        d = d_single
        self.n_head, self.gating, self.kq_norm, self.up_gate, self.adaln_eps = n_head, gating, kq_norm, up_gate, adaln_eps
        self.attn_adaln = VariantAdaLN(d, d_cond, identity_scale)
        self.to_q = nn.Linear(d, d)
        self.to_k = nn.Linear(d, d, bias=False)
        self.to_v = nn.Linear(d, d, bias=False)
        if gating:
            self.to_g = nn.Linear(d, d, bias=False)
        if kq_norm:
            self.q_norm, self.k_norm = nn.LayerNorm(d), nn.LayerNorm(d)
        self.to_out = nn.Linear(d, d, bias=False)
        self.attn_zero = nn.Linear(d_cond, d)
        self.pair_norm = nn.LayerNorm(d_pair, eps=pair_eps, bias=pair_norm_bias)
        self.to_pair_bias = nn.Linear(d_pair, n_head, bias=False)
        self.trans_adaln = VariantAdaLN(d, d_cond, identity_scale)
        self.expand = nn.Linear(d, 2 * n * d, bias=False)
        if up_gate:
            self.up = nn.Linear(d, n * d, bias=False)
        self.squeeze = nn.Linear(n * d, d, bias=False)
        self.trans_zero = nn.Linear(d_cond, d)


def serves_portable(stack: nn.Module, *inputs: torch.Tensor | None) -> bool:
    """Would ``stack(*inputs)`` run on the portable Triton path (``kernels/dit_variants/triton/portable.py``) on this card? The inputs
    are the ones of the stack's forward: ``(single, cond, pair, mask)`` for a :class:`DiTVariantStack`, ``(single, cond, pair, mask, uid)``
    for an :class:`AtomDiTVariantStack`. True on a card the engine holds a record for (``docs/dispatch.md``) or under
    ``settings.force_portable_kernels``; False on H100 / B200 by default, where the CUDA path (``kernels.dit_variants.interface.serves``)
    or the PyTorch reference runs. A caller's ``variant_serves`` seam asks this after the H100 question."""
    from miniworld_engine.kernels.dit_variants.triton import portable

    return bool(portable.atom_serves(stack, *inputs) if isinstance(stack, AtomDiTVariantStack) else portable.serves(stack, *inputs))


def serving(stack: nn.Module, *inputs: torch.Tensor | None) -> str:
    """Which implementation runs ``stack(*inputs)``: ``"cuda"`` (the H100 hand-CUDA rows), ``"portable"`` (the Triton path of a recorded
    card) or ``"pytorch"`` (``kernels/dit_variants/reference.py``: differentiable torch statements, any card, inference and training).
    A ``PYTORCH`` stack is always ``"pytorch"``: it SERVES every call (including a call with a gradient and a mask), it is not a
    refusal. Callers that count engine-served calls (FoldForge's ``variant_serves``) ask this instead of :func:`serves_portable`."""
    if stack.implementation == ImplementationType.PYTORCH:
        return "pytorch"
    from miniworld_engine.kernels.dit_variants import interface
    from miniworld_engine.kernels.dit_variants.cuda import infer32, train

    atom = isinstance(stack, AtomDiTVariantStack)
    for path, name in ((interface, "cuda"), (infer32, "cuda"), (train, "cuda")):
        if (path.atom_serves if atom else path.serves)(stack, *inputs):
            return name
    return "portable" if serves_portable(stack, *inputs) else "pytorch"


class DiTVariantStack(ImplementationSwitch):
    """``n_blocks`` variant blocks; ``forward(single [A, L, D], cond [L, c_s] (shared by the samples) or [A, L, c_s], pair [L, L, C],
    mask [L] bool or None)`` -> [A, L, D]."""

    def __init__(self, n_blocks: int, d_single: int = 768, d_cond: int = 384, d_pair: int = 128, n_head: int = 16, n: int = 2, *,
                 parallel: bool = False, implementation: ImplementationType = ImplementationType.MINIWORLD, **options) -> None:
        super().__init__()
        self.parallel, self.implementation = parallel, implementation
        self.blocks = nn.ModuleList(
            DiTVariantBlock(d_single, d_cond, d_pair, n_head, n, **options) for _ in range(n_blocks))

    @classmethod
    def from_released(cls, state_dict, *, n_head: int, parallel: bool, super_block_size: int = 4, prefix: str = "",
                      implementation: ImplementationType = ImplementationType.PYTORCH, dtype: torch.dtype | None = None,
                      adaln_eps: float = 1e-5, pair_eps: float = 1e-5, **options) -> DiTVariantStack:
        """The stack from a released token ``DiffusionTransformer`` state (``self_attention.i`` / ``transition_block.i`` /
        ``pair_input_layer_norm`` / ``pair_logits_projection``; Boltz-2, RoseTTAFold3, Chai-1 layouts, see ``modules.released``). The block
        options follow from the weights; ``options`` override them."""
        from miniworld_engine.modules import released

        converted, args = released.dit_variant_stack_from_released(state_dict, n_head=n_head, super_block_size=super_block_size, prefix=prefix)
        args = {**args, **options}
        stack = cls(args.pop("n_blocks"), args.pop("d_single"), args.pop("d_cond"), args.pop("d_pair"), args.pop("n_head"), args.pop("n"),
                    parallel=parallel, implementation=implementation, adaln_eps=adaln_eps, pair_eps=pair_eps, **args)
        return released.load(stack, converted, dtype=dtype)

    def serving(self, *inputs: torch.Tensor | None) -> str:
        """``"cuda"`` / ``"portable"`` / ``"pytorch"``: see :func:`serving`."""
        return serving(self, *inputs)

    def forward(self, single: torch.Tensor, cond: torch.Tensor, pair: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if self.implementation != ImplementationType.PYTORCH:
            from miniworld_engine.kernels.dit_variants import interface

            if interface.serves(self, single, cond, pair, mask):
                return interface.inference(self, single, cond, pair, mask)
            from miniworld_engine.kernels.dit_variants.cuda import infer32

            if infer32.serves(self, single, cond, pair, mask):
                return infer32.inference(self, single, cond, pair, mask)
            from miniworld_engine.kernels.dit_variants.cuda import train

            if train.serves(self, single, cond, pair, mask):
                return train.stack_train(self, single, cond, pair, mask)
            from miniworld_engine.kernels.dit_variants.triton import portable

            if portable.serves(self, single, cond, pair, mask):
                return portable.stack_forward(self, single, cond, pair, mask)
        return reference.stack_forward(self, single, cond, pair, mask)


#: The atom stacks' options (``AtomDiTVariantStack(n_blocks, **BOLTZ2_ATOM)`` ...): 128 wide, 4 heads x 32, pair 16, conditioning 128.
BOLTZ2_ATOM = {"parallel": False, "identity_scale": False, "gating": True, "project_output": True, "kq_norm": False, "up_gate": True,
               "per_block_pair_norm": False, "pair_norm_bias": False}
ROSETTAFOLD3_ATOM = {"parallel": True, "identity_scale": False, "gating": True, "project_output": True, "kq_norm": True,
                     "up_gate": False, "per_block_pair_norm": True, "pair_norm_bias": False}
CHAI1_ATOM = {"parallel": True, "identity_scale": True, "gating": False, "project_output": False, "kq_norm": False, "up_gate": False,
              "per_block_pair_norm": False, "pair_norm_bias": True}
#: AlphaFold 3's atom stack (``DiffusionCrossAttTransformer`` with spec ``ALPHAFOLD3``, ``key_window="slide"``): sequential, sigmoid AdaLN on
#: the queries and (separately, both of the block input) on the keys, gating query, output projection, plain SwiGLU transition, one shared pair
#: LayerNorm + ``Linear(d_pair, 4 * n_blocks)``.
AF3_ATOM = {"parallel": False, "identity_scale": False, "gating": True, "project_output": True, "kq_norm": False, "up_gate": False,
            "per_block_pair_norm": False, "pair_norm_bias": False}


class AtomDiTVariantBlock(nn.Module):
    def __init__(self, d: int = 128, d_cond: int = 128, d_pair: int = 16, n: int = 2, *, identity_scale: bool = False,
                 gating: bool = True, project_output: bool = True, kq_norm: bool = False, up_gate: bool = False,
                 per_block_pair_norm: bool = False, pair_norm_bias: bool = False, adaln_eps: float = 1e-5,
                 pair_eps: float = 1e-5) -> None:
        super().__init__()
        self.gating, self.project_output, self.kq_norm, self.up_gate, self.adaln_eps = (
            gating, project_output, kq_norm, up_gate, adaln_eps)
        self.q_adaln = VariantAdaLN(d, d_cond, identity_scale)
        self.k_adaln = VariantAdaLN(d, d_cond, identity_scale)
        self.to_q = nn.Linear(d, d)
        self.to_k = nn.Linear(d, d, bias=False)
        self.to_v = nn.Linear(d, d, bias=False)
        if gating:
            self.to_g = nn.Linear(d, d, bias=False)
        if project_output:
            self.to_out = nn.Linear(d, d, bias=False)
        if kq_norm:
            self.q_norm, self.k_norm = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn_zero = nn.Linear(d_cond, d)
        if per_block_pair_norm:
            self.pair_norm = nn.LayerNorm(d_pair, eps=pair_eps, bias=pair_norm_bias)
            self.to_pair_bias = nn.Linear(d_pair, 4, bias=False)
        self.trans_adaln = VariantAdaLN(d, d_cond, identity_scale)
        self.expand = nn.Linear(d, 2 * n * d, bias=False)
        if up_gate:
            self.up = nn.Linear(d, n * d, bias=False)
        self.squeeze = nn.Linear(n * d, d, bias=False)
        self.trans_zero = nn.Linear(d_cond, d)


class AtomDiTVariantStack(ImplementationSwitch):
    """``n_blocks`` atom variant blocks; ``forward(single [A, N, 128], cond [N, 128] (shared by the samples) or [A, N, 128], pair
    [N / 32, 32, 128, d_pair] (trunked), mask [N] bool (real atoms first) or None, uid [N] (Chai-1's same-conformer mask) or None)``.
    The shared pair projection's output channel ``4 b + h`` is block b's head h. ``key_window``: "slide" (FoldForge's fast mode) or
    "pad"."""

    def __init__(self, n_blocks: int = 3, d: int = 128, d_cond: int = 128, d_pair: int = 16, n: int = 2, *, parallel: bool = False,
                 per_block_pair_norm: bool = False, pair_norm_bias: bool = False, key_window: str = "slide",
                 implementation: ImplementationType = ImplementationType.MINIWORLD, **options) -> None:
        super().__init__()
        if key_window not in ("slide", "pad"):
            raise ValueError(f"key_window must be 'slide' or 'pad', got {key_window!r}")
        self.parallel, self.per_block_pair_norm, self.key_window, self.implementation = (
            parallel, per_block_pair_norm, key_window, implementation)
        pair_eps = options.get("pair_eps", 1e-5)
        if not per_block_pair_norm:
            self.pair_norm = nn.LayerNorm(d_pair, eps=pair_eps, bias=pair_norm_bias)
            self.to_pair_bias = nn.Linear(d_pair, 4 * n_blocks, bias=False)
        self.blocks = nn.ModuleList(
            AtomDiTVariantBlock(d, d_cond, d_pair, n, per_block_pair_norm=per_block_pair_norm, pair_norm_bias=pair_norm_bias,
                                **options) for _ in range(n_blocks))

    @classmethod
    def from_released(cls, state_dict, *, parallel: bool, key_window: str = "slide", prefix: str = "",
                      implementation: ImplementationType = ImplementationType.PYTORCH, dtype: torch.dtype | None = None,
                      adaln_eps: float = 1e-5, pair_eps: float = 1e-5, **options) -> AtomDiTVariantStack:
        """The stack from a released ``DiffusionCrossAttTransformer`` state (``cross_attention.i`` / ``transition_block.i`` /
        ``pair_input_layer_norm`` / ``pair_logits_projection``; see ``modules.released``)."""
        from miniworld_engine.modules import released

        converted, args = released.atom_dit_variant_stack_from_released(state_dict, prefix=prefix)
        args = {**args, **options}
        stack = cls(args.pop("n_blocks"), args.pop("d"), args.pop("d_cond"), args.pop("d_pair"), args.pop("n"), parallel=parallel,
                    per_block_pair_norm=args.pop("per_block_pair_norm"), pair_norm_bias=args.pop("pair_norm_bias"), key_window=key_window,
                    implementation=implementation, adaln_eps=adaln_eps, pair_eps=pair_eps, **args)
        return released.load(stack, converted, dtype=dtype)

    def serving(self, *inputs: torch.Tensor | None) -> str:
        """``"cuda"`` / ``"portable"`` / ``"pytorch"``: see :func:`serving`."""
        return serving(self, *inputs)

    def forward(self, single, cond, pair, mask=None, uid=None):
        if self.implementation != ImplementationType.PYTORCH:
            from miniworld_engine.kernels.dit_variants import interface

            if interface.atom_serves(self, single, cond, pair, mask, uid):
                return interface.atom_inference(self, single, cond, pair, mask, uid)
            from miniworld_engine.kernels.dit_variants.cuda import infer32

            if infer32.atom_serves(self, single, cond, pair, mask, uid):
                return infer32.atom_inference(self, single, cond, pair, mask, uid)
            from miniworld_engine.kernels.dit_variants.cuda import train

            if train.atom_serves(self, single, cond, pair, mask, uid):
                return train.atom_stack_train(self, single, cond, pair, mask, uid)
            from miniworld_engine.kernels.dit_variants.triton import portable

            if portable.atom_serves(self, single, cond, pair, mask, uid):
                return portable.atom_stack_forward(self, single, cond, pair, mask, uid)
        return reference.atom_stack_forward(self, single, cond, pair, mask, uid)
