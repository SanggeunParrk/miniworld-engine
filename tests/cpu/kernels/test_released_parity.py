# ruff: noqa: PT018, PT011, C408
"""PYTORCH engine modules built ``from_released`` against FoldForge's dense modules, the numerical oracle of the released checkpoints.

Skipped unless FoldForge is importable (``FOLDFORGE_SRC=<.../FoldForge/src>`` or ``foldforge`` installed): the engine does not depend on it.
Both sides are built from the same random weights; the oracle returns raw deltas, the engine modules the residual output.
"""
from __future__ import annotations

import os
import sys
from dataclasses import replace

import pytest
import torch

if os.environ.get("FOLDFORGE_SRC"):
    sys.path.insert(0, os.environ["FOLDFORGE_SRC"])
pytest.importorskip("foldforge.modules.dense.spec")

from foldforge.modules.dense.attention import (
    GridSelfAttention,
    MSAAttention,
)
from foldforge.modules.dense.pairformer import (
    PairformerBlock as DensePairformerBlock,
)
from foldforge.modules.dense.primitives import (
    OuterProductMean as DenseOPM,
)
from foldforge.modules.dense.primitives import (
    Transition as DenseTransition,
)
from foldforge.modules.dense.spec import ALPHAFOLD3
from foldforge.modules.dense.triangle_multiplication import (
    TriangleMultiplication as DenseTriMul,
)
from foldforge.modules.engine.routing import _length_divided_triangle

from miniworld_engine.modules.msa_pair_weighted_averaging import (
    MSAPairWeightedAveraging,
)
from miniworld_engine.modules.outer_product import OuterProductMean
from miniworld_engine.modules.pairformer import (
    PairformerBlock,
    PairformerConfig,
)
from miniworld_engine.modules.transition import Transition
from miniworld_engine.modules.triangle_attention import TriangleAttention
from miniworld_engine.modules.triangle_multiplication import (
    TriangleMultiplication,
)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
L, C = 16, 16
TOL = {"atol": 2e-4, "rtol": 2e-4}


def _rand(module, seed=0, std=0.3):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in module.parameters():
            p.copy_((torch.randn(p.shape, generator=g) * std).to(p))
    return module.to(DEV).float()


def _inputs(seed=1, masked=3):
    g = torch.Generator().manual_seed(seed)
    pair = torch.randn(L, L, C, generator=g).to(DEV)
    res = torch.ones(L, dtype=torch.bool)
    res[L - masked:] = False
    res = res.to(DEV)
    return pair, res, (res[:, None] & res[None, :]).float()


@pytest.mark.parametrize("outgoing", [True, False])
@pytest.mark.parametrize("divide", [False, True])
@pytest.mark.parametrize("hidden", [None, 24])
def test_trimul(outgoing, divide, hidden):
    pair, res, pair_mask = _inputs()
    dense = _rand(DenseTriMul(C, outgoing, hidden_dim=hidden, divide_by_length=divide))
    eng = TriangleMultiplication.from_released(dense.state_dict(), outgoing=outgoing, divide_by_length=divide).to(DEV)
    # the dense module's forward is the plain update; FoldForge applies divide_by_length in routing (the oracle of the RoseTTAFold3 form)
    delta = _length_divided_triangle(dense, pair, pair_mask) if divide else dense(pair, pair_mask)
    got = eng(pair[None], res[None])[0]
    torch.testing.assert_close(got, pair + delta, **TOL)


@pytest.mark.parametrize("kwargs", [
    {}, {"bias_after_norm": True}, {"clamped_norm": True}, {"projection_bias": True}, {"groups": 2},
    {"bias_after_norm": True, "clamped_norm": True, "projection_bias": True}])
def test_opm(kwargs):
    g = torch.Generator().manual_seed(4)
    msa = torch.randn(6, L, 24, generator=g).to(DEV)
    mask = (torch.rand(6, L, generator=g) > 0.3).float().to(DEV)
    mask[0] = 1
    dense = _rand(DenseOPM(24, C, 4, **kwargs), std=0.2)
    flags = {k: v for k, v in kwargs.items() if k in ("bias_after_norm", "clamped_norm")}
    eng = OuterProductMean.from_released(dense.state_dict(), **flags).to(DEV)
    torch.testing.assert_close(eng(msa[None], (mask > 0)[None])[0], dense(msa, mask), **TOL)


@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("bias_dir", [False, True])
@pytest.mark.parametrize("variant", ["plain", "biases", "untransposed", "dual"])
def test_triangle_attention(transpose, bias_dir, variant):
    spec = replace(ALPHAFOLD3, transposed_column_pair_bias=bias_dir,
                   triangle_attention_bias=variant in ("biases", "dual"),
                   untransposed_column_pair_output=variant == "untransposed")
    pair, res, pair_mask = _inputs()
    dense = _rand(GridSelfAttention(C, 2, transpose=transpose, spec=spec, dual_output=variant == "dual"))
    eng = TriangleAttention.from_released(
        dense.state_dict(), n_head=2, starting=not transpose, transposed_bias=bias_dir,
        untransposed_output=dense.untransposed_output).to(DEV)
    # padded query rows: the dense module masks keys per pair row (a fully masked row is uniform), the engine masks keys by residue: compare the real rows
    torch.testing.assert_close(eng(pair[None], res[None])[0][res][:, res], (pair + dense(pair, pair_mask))[res][:, res], **TOL)


@pytest.mark.parametrize("n", [4, 2])
def test_transition(n):
    pair, _, _ = _inputs()
    dense = _rand(DenseTransition(C, n))
    eng = Transition.from_released(dense.state_dict()).to(DEV)
    torch.testing.assert_close(eng(pair[None])[0], pair + dense(pair), **TOL)


def test_msa_pwa():
    g = torch.Generator().manual_seed(6)
    msa = torch.randn(5, L, 32, generator=g).to(DEV)
    pair, res, _ = _inputs()
    msa_mask = torch.ones(5, L).to(DEV)
    msa_mask[:, L - 3:] = 0
    dense = _rand(MSAAttention(32, C, num_head=4))
    eng = MSAPairWeightedAveraging.from_released(dense.state_dict(), n_head=4).to(DEV)
    torch.testing.assert_close(eng(msa[None], pair[None], res[None])[0], msa + dense(msa, msa_mask, pair), **TOL)


@pytest.mark.parametrize("flavour", ["sequential", "parallel", "rf3", "chai"])
def test_pairformer_block(flavour):
    spec = {
        "sequential": ALPHAFOLD3,
        "parallel": replace(ALPHAFOLD3, parallel_pairformer_block=True),
        "rf3": replace(ALPHAFOLD3, triangle_mul_divide_by_length=True, triangle_attention_bias=True, transposed_column_pair_bias=True),
        "chai": replace(ALPHAFOLD3, parallel_pairformer_block=True, single_attention_gate_bias=1.0, transposed_column_pair_bias=True),
    }[flavour]
    pair, res, pair_mask = _inputs()
    single = torch.randn(L, 32, generator=torch.Generator().manual_seed(8)).to(DEV)
    dense = _rand(DensePairformerBlock(n_heads=4, c_pair=C, c_single=32, c_hidden_mul=C, n_heads_pair=2, spec=spec), std=0.2)
    cfg = PairformerConfig(
        d_pair=C, d_hidden_tri_multi=C, d_hidden_tri_attention=C // 2, n_head_tri_attention=2, p_drop=0.0, d_single=32, n_head_single=4,
        parallel=spec.parallel_pairformer_block, divide_by_length=spec.triangle_mul_divide_by_length,
        transposed_bias=spec.transposed_column_pair_bias, triangle_attention_gate_bias=spec.triangle_attention_bias,
        triangle_attention_out_bias=spec.triangle_attention_bias, single_gate_bias=spec.single_attention_gate_bias)
    eng = PairformerBlock.from_released(dense.state_dict(), config=cfg).to(DEV)
    want_pair, want_single = dense(pair.clone(), pair_mask, single.clone(), res.float())
    got_pair, got_single = eng(pair[None], res[None], single[None])
    torch.testing.assert_close(got_pair[0][res][:, res], want_pair[res][:, res], **TOL)
    torch.testing.assert_close(got_single[0][res], want_single[res], **TOL)


# ------------------------------------------------------------------------------------------------------------------- DiT variant stacks
from foldforge.modules.dense.diffusion_transformer import (
    DiffusionCrossAttTransformer,
    DiffusionTransformer,
)

from miniworld_engine.modules.dit_variant import (
    AtomDiTVariantStack,
    DiTVariantStack,
)

TOKEN_FLAVOURS = {
    "boltz2": dict(per_block_pair_layer_norm=True, transition_up_gate=True),
    "boltz2_shared": dict(transition_up_gate=True),
    "rf3": dict(per_block_pair_layer_norm=True, attention_kq_norm=True, parallel_attention_transition=True),
    "chai1": dict(per_block_pair_layer_norm=True, adaptive_identity_scale=True, token_attention_gating_query=False,
                  parallel_attention_transition=True, affine_norms=frozenset({"pair_input_layer_norm"})),
}


@pytest.mark.parametrize("flavour", list(TOKEN_FLAVOURS))
@pytest.mark.parametrize("grad", [False, True])
def test_token_variant_stack(flavour, grad):
    spec = replace(ALPHAFOLD3, **TOKEN_FLAVOURS[flavour])
    g = torch.Generator().manual_seed(11)
    act = torch.randn(1, L, 64, generator=g).to(DEV)
    cond = torch.randn(L, 16, generator=g).to(DEV)
    pair = torch.randn(L, L, 8, generator=g).to(DEV)
    res = torch.ones(L, dtype=torch.bool, device=DEV)
    res[-3:] = False
    dense = _rand(DiffusionTransformer(c_act=64, c_single_cond=16, c_pair_cond=8, num_head=16, num_blocks=4, super_block_size=2, spec=spec), std=0.15)
    stack = DiTVariantStack.from_released(dense.state_dict(), n_head=16, parallel=spec.parallel_attention_transition, super_block_size=2).to(DEV)
    assert stack.serving(act, cond, pair, res) == "pytorch"
    a = act.clone().requires_grad_(grad)
    b = act.clone().requires_grad_(grad)
    want = dense(a[0], res.float(), cond, pair)[None]
    got = stack(b, cond, pair, res)
    torch.testing.assert_close(got[:, res], want[:, res], **TOL)
    if grad:
        want[:, res].square().sum().backward()
        got[:, res].square().sum().backward()
        torch.testing.assert_close(b.grad, a.grad, atol=2e-3, rtol=2e-3)


@pytest.mark.parametrize("flavour", ["boltz2", "rf3", "chai1"])
def test_atom_variant_stack_loads_strictly(flavour):
    kw = {"boltz2": dict(transition_up_gate=True), "rf3": dict(per_block_atom_pair_layer_norm=True, attention_kq_norm=True, parallel_attention_transition=True),
          "chai1": dict(adaptive_identity_scale=True, atom_attention_gating_query=False, atom_attention_project_output=False, parallel_attention_transition=True)}[flavour]
    spec = replace(ALPHAFOLD3, **kw)
    dense = _rand(DiffusionCrossAttTransformer(spec=spec), std=0.1)
    stack = AtomDiTVariantStack.from_released(dense.state_dict(), parallel=spec.parallel_attention_transition)
    assert stack.per_block_pair_norm == spec.per_block_atom_pair_layer_norm
    assert len(stack.blocks) == 3
