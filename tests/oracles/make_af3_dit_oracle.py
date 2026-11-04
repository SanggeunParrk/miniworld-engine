"""Generate ``af3_dit_oracle.pt``: seeded weights, inputs and the expected outputs / input gradients of AlphaFold 3's own diffusion
transformer stacks, computed ONCE by FoldForge's dense classes (``foldforge.modules.dense.diffusion_transformer``, spec ``ALPHAFOLD3``:
shared pair LayerNorm, sequential schedule, sigmoid AdaLN, gating query, output projection, plain SwiGLU transition).

The engine's tests load the frozen file and never import FoldForge. This script is not a test (its name does not start with ``test_``); it
needs FoldForge on ``PYTHONPATH`` and runs on a CPU allocation (``docs/gpus/h100/h100.md``):

    PYTHONPATH=<FoldForge>/src:<engine>/src python tests/oracles/make_af3_dit_oracle.py tests/oracles/af3_dit_oracle.pt

Every tensor is rounded to bf16 first (stored as bf16, exact in fp32), so the file is half the size and the fp32 oracle is computed from
exactly the stored values (the atom pair gradient is stored in fp16 to stay under 1 MB). Small sizes, 16 heads (the dense token attention is built with 16, whatever num_head says): the atom stack is not the 128-wide production stack (the engine's variant stacks take every
width from the weights); the production widths are exercised by ``tests/gpu/sm90/test_dit_variant_af3_h100_gpu.py``.
"""

from __future__ import annotations

import sys

import torch
from foldforge.modules.dense.diffusion_transformer import (
    CrossAttention,
    DiffusionCrossAttTransformer,
    DiffusionTransformer,
)
from foldforge.modules.dense.spec import ALPHAFOLD3
from foldforge.utils import atom_layout

# token stack
D, C_COND, C_PAIR, HEADS, BLOCKS, SUPER, L, L_REAL, A = 48, 16, 8, 16, 4, 2, 40, 37, 2
# atom stack: 5 windows of 32 queries, 150 real atoms, 128 keys per window slid inside the real atoms
AD, A_PAIR, A_BLOCKS, N, N_REAL = 32, 4, 2, 160, 150


def bf16(t: torch.Tensor) -> torch.Tensor:
    return t.bfloat16().float()


def randomise(module: torch.nn.Module, seed: int) -> None:
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in module.named_parameters():
            if p.ndim >= 2:
                p.copy_(bf16(torch.randn(p.shape, generator=g) * p.shape[-1] ** -0.5))
            elif name.endswith("weight"):
                p.copy_(bf16(1 + 0.1 * torch.randn(p.shape, generator=g)))
            else:
                p.copy_(bf16(0.1 * torch.randn(p.shape, generator=g)))


def state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().bfloat16().clone() for k, v in module.state_dict().items()}


def token_case() -> dict:
    dense = DiffusionTransformer(c_act=D, c_single_cond=C_COND, c_pair_cond=C_PAIR, num_head=HEADS, num_blocks=BLOCKS,
                                 super_block_size=SUPER, spec=ALPHAFOLD3)
    randomise(dense, 1)
    g = torch.Generator().manual_seed(2)
    act = bf16(torch.randn(A, L, D, generator=g)).requires_grad_()
    cond = bf16(torch.randn(L, C_COND, generator=g)).requires_grad_()
    pair = bf16(torch.randn(L, L, C_PAIR, generator=g)).requires_grad_()
    dy = bf16(torch.randn(A, L, D, generator=g))
    mask = torch.arange(L) < L_REAL
    out = dense(act, mask.float(), cond, pair)
    (out[:, :L_REAL] * dy[:, :L_REAL]).sum().backward()
    return {"state": state(dense), "act": act.detach().bfloat16(), "cond": cond.detach().bfloat16(), "pair": pair.detach().bfloat16(),
            "dy": dy.bfloat16(), "mask": mask, "out": out.detach(), "grad_act": act.grad, "grad_cond": cond.grad, "grad_pair": pair.grad,
            "n_head": HEADS, "super_block_size": SUPER, "n_real": L_REAL}


def atom_case() -> dict:
    dense = DiffusionCrossAttTransformer(c_query=AD, c_single_cond=AD, c_pair_cond=A_PAIR, num_blocks=A_BLOCKS, num_head=4, spec=ALPHAFOLD3)
    # the dense stack builds its cross attentions at the 128-wide default; the same class at the small width
    dense.cross_attention = torch.nn.ModuleList(
        CrossAttention(key_dim=AD, value_dim=AD, c_single_cond=AD, num_head=4) for _ in range(A_BLOCKS))
    randomise(dense, 3)
    nwin = N // 32
    g = torch.Generator().manual_seed(4)
    act = bf16(torch.randn(A, nwin, 32, AD, generator=g)).requires_grad_()
    cond = bf16(torch.randn(N, AD, generator=g)).requires_grad_()
    pair = bf16(torch.randn(nwin, 32, 128, A_PAIR, generator=g)).requires_grad_()
    dy = bf16(torch.randn(A, N, AD, generator=g))
    start = (32 * torch.arange(nwin) - 48).clamp(0, N_REAL - 128)
    idx = start[:, None] + torch.arange(128)
    info = atom_layout.GatherInfo(gather_idxs=idx, gather_mask=torch.ones_like(idx, dtype=torch.bool), input_shape=(nwin, 32))
    queries_mask = (torch.arange(N) < N_REAL).view(nwin, 32)
    keys_cond = atom_layout.convert(info, cond.view(nwin, 32, AD), layout_axes=(0, 1))
    out = dense(act, queries_mask, info, torch.ones(nwin, 128, dtype=torch.bool), cond.view(nwin, 32, AD), keys_cond, pair)
    out = out.reshape(A, N, AD)
    (out[:, :N_REAL] * dy[:, :N_REAL]).sum().backward()
    return {"state": state(dense), "act": act.detach().reshape(A, N, AD).bfloat16(), "cond": cond.detach().bfloat16(),
            "pair": pair.detach().bfloat16(), "dy": dy.bfloat16(), "mask": torch.arange(N) < N_REAL, "out": out.detach(),
            "grad_act": act.grad.reshape(A, N, AD), "grad_cond": cond.grad, "grad_pair": pair.grad.half(), "n_real": N_REAL}


if __name__ == "__main__":
    torch.save({"token": token_case(), "atom": atom_case()}, sys.argv[1])
