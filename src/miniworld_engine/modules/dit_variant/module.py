"""The token DiT block variants of Boltz-2, RoseTTAFold3 and Chai-1 (FoldForge's dense ``DiffusionTransformer`` with their spec
flags), inference only. The math is in ``kernels/dit_variants/reference.py``; on H100 a bf16 call without a gradient runs
``kernels/dit_variants/interface.py`` (CUDA rows + cuBLAS + the token DiT's attention core), anything else the reference.

The three models' options:

    Boltz-2   sequential, sigmoid AdaLN, gating query, up-gate transition (silu(g) v u), d_cond 768, pair 128
    RF3       parallel, sigmoid AdaLN, gating query, kq-norm (LayerNorm over the flat q / k), d_cond 384, pair 128
    Chai-1    parallel, identity AdaLN ((W s + 1) LN(x) + W s), no gating query, d_cond 384, pair 256 (affine pair LN)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from miniworld_engine.kernels.dit_variants import reference
from miniworld_engine.modules.exceptions import ImplementationType

#: Each model's options (``DiTVariantStack(n_blocks, **BOLTZ2)`` ...).
BOLTZ2 = {"d_cond": 768, "d_pair": 128, "parallel": False, "identity_scale": False, "gating": True, "kq_norm": False,
          "up_gate": True, "pair_norm_bias": False}
ROSETTAFOLD3 = {"d_cond": 384, "d_pair": 128, "parallel": True, "identity_scale": False, "gating": True, "kq_norm": True,
                "up_gate": False, "pair_norm_bias": False}
CHAI1 = {"d_cond": 384, "d_pair": 256, "parallel": True, "identity_scale": True, "gating": False, "kq_norm": False,
         "up_gate": False, "pair_norm_bias": True}


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


class DiTVariantStack(nn.Module):
    """``n_blocks`` variant blocks; ``forward(single [A, L, D], cond [L, c_s] (shared by the samples) or [A, L, c_s], pair [L, L, C],
    mask [L] bool or None)`` -> [A, L, D]."""

    def __init__(self, n_blocks: int, d_single: int = 768, d_cond: int = 384, d_pair: int = 128, n_head: int = 16, n: int = 2, *,
                 parallel: bool = False, implementation: ImplementationType = ImplementationType.MINIWORLD, **options) -> None:
        super().__init__()
        self.parallel, self.implementation = parallel, implementation
        self.blocks = nn.ModuleList(
            DiTVariantBlock(d_single, d_cond, d_pair, n_head, n, **options) for _ in range(n_blocks))

    def forward(self, single: torch.Tensor, cond: torch.Tensor, pair: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if self.implementation != ImplementationType.PYTORCH:
            from miniworld_engine.kernels.dit_variants import interface

            if interface.serves(self, single, cond, pair, mask):
                return interface.inference(self, single, cond, pair, mask)
        return reference.stack_forward(self, single, cond, pair, mask)


#: The atom stacks' options (``AtomDiTVariantStack(n_blocks, **BOLTZ2_ATOM)`` ...): 128 wide, 4 heads x 32, pair 16, conditioning 128.
BOLTZ2_ATOM = {"parallel": False, "identity_scale": False, "gating": True, "project_output": True, "kq_norm": False, "up_gate": True,
               "per_block_pair_norm": False, "pair_norm_bias": False}
ROSETTAFOLD3_ATOM = {"parallel": True, "identity_scale": False, "gating": True, "project_output": True, "kq_norm": True,
                     "up_gate": False, "per_block_pair_norm": True, "pair_norm_bias": False}
CHAI1_ATOM = {"parallel": True, "identity_scale": True, "gating": False, "project_output": False, "kq_norm": False, "up_gate": False,
              "per_block_pair_norm": False, "pair_norm_bias": True}


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


class AtomDiTVariantStack(nn.Module):
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

    def forward(self, single, cond, pair, mask=None, uid=None):
        if self.implementation != ImplementationType.PYTORCH:
            from miniworld_engine.kernels.dit_variants import interface

            if interface.atom_serves(self, single, cond, pair, mask, uid):
                return interface.atom_inference(self, single, cond, pair, mask, uid)
        return reference.atom_stack_forward(self, single, cond, pair, mask, uid)
