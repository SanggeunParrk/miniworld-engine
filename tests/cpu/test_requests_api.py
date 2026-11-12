"""FoldForge API requests (docs: modules.md): R4 the ``protenix_v2`` triangle key, R5 separate triangle keys of a PairformerBlock, R10 the DiT
stacks and ``from_shared_pair``, R12 ``parallel_residual``, R14 the documented invariants (shared K / V AdaLN, RoseTTAFold3's bias-free
query), R18 the multi-fold contract of the DiT blocks (one call over B folds == B calls of one fold) -- on the ``pytorch`` reference, CPU."""

from __future__ import annotations

import typing

import pytest
import torch

from miniworld_engine.modules import (
    AtomDiT,
    AtomDiTBoltz2,
    AtomDiTChai1,
    AtomDiTRoseTTAFold3,
    AtomDiTStack,
    PairformerBlock,
    PairformerConfig,
    ParallelResidual,
    TokenDiT,
    TokenDiTRoseTTAFold3,
    TokenDiTStack,
    TriangleAttention,
    TriangleMultiplication,
    TriangleMultiplicationRF3,
    from_shared_pair,
    parallel_residual,
)
from miniworld_engine.modules.attention import FlatQKNormAugmentedAttention
from miniworld_engine.modules.primitives import BackendUnsupported
from miniworld_engine.modules.triangle import TriangleAttentionRF3


def _gen(seed: int = 0) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


# ------------------------------------------------------------------------------------------------------------------------------ R4
def test_protenix_v2_is_an_alias_of_protenix():
    assert type(TriangleAttention(64, 2, model="protenix_v2")) is type(TriangleAttention(64, 2, model="protenix"))
    assert type(TriangleMultiplication(64, model="protenix_v2")) is type(TriangleMultiplication(64, model="protenix"))
    torch.manual_seed(0)
    a = TriangleAttention(32, 2, model="protenix_v2")
    b = TriangleAttention(32, 2, model="protenix")
    assert a.state_dict().keys() == b.state_dict().keys()


# ------------------------------------------------------------------------------------------------------------------------------ R5
def test_pairformer_separate_triangle_keys():
    cfg = PairformerConfig(d_pair=32, d_hidden_tri_multi=32, d_hidden_tri_attention=8, n_head_tri_attention=2, model="rf3")
    default = PairformerBlock(cfg)
    assert type(default.tri_atten_starting) is TriangleAttentionRF3
    assert type(default.tri_multi_outgoing) is TriangleMultiplicationRF3  # today's behaviour: one key for both
    fast = PairformerBlock(PairformerConfig(**{**cfg.__dict__, "trimul_model": "miniworld"}))
    assert type(fast.tri_atten_starting) is TriangleAttentionRF3 and type(fast.tri_atten_ending) is TriangleAttentionRF3
    assert type(fast.tri_multi_outgoing) is TriangleMultiplication and type(fast.tri_multi_incoming) is TriangleMultiplication
    other = PairformerBlock(PairformerConfig(**{**cfg.__dict__, "model": "miniworld", "triattn_model": "rf3"}))
    assert type(other.tri_atten_starting) is TriangleAttentionRF3 and type(other.tri_multi_outgoing) is TriangleMultiplication


# ----------------------------------------------------------------------------------------------------------------------------- R10
def test_token_stack_equals_its_blocks():
    torch.manual_seed(0)
    stack = TokenDiTStack("protenix", 2, d_single=64, d_cond=32, d_pair=16, n_head=4)
    assert [type(b).__name__ for b in stack.blocks] == ["TokenDiTProtenix"] * 2
    g = _gen()
    single, cond = torch.randn(2, 1, 8, 64, generator=g), torch.randn(2, 1, 8, 32, generator=g)
    pair, mask = torch.randn(1, 8, 8, 16, generator=g), torch.ones(1, 8, dtype=torch.bool)
    with torch.no_grad():
        ref = stack.blocks[1](stack.blocks[0](single, cond, pair, mask), cond, pair, mask)
        assert torch.equal(stack(single, cond, pair, mask), ref)
    assert set(stack.state_dict()) == {f"blocks.{i}.{k}" for i in range(2) for k in stack.blocks[0].state_dict()}
    assert stack.set_backend("triton").backend == "triton"


def test_atom_stack_equals_its_blocks_and_refuses_pair_free_models():
    torch.manual_seed(0)
    stack = AtomDiTStack("chai1", 2)
    g = _gen()
    n = 64
    single, cond = torch.randn(2, 1, n, 128, generator=g), torch.randn(2, 1, n, 128, generator=g)
    pair = torch.randn(1, n // 32, 32, 128, 16, generator=g)
    mask = torch.ones(1, n, dtype=torch.bool)
    uid = (torch.arange(n) // 20)[None]
    with torch.no_grad():
        ref = stack.blocks[1](stack.blocks[0](single, cond, pair, mask, uid=uid), cond, pair, mask, uid=uid)
        assert torch.equal(stack(single, cond, pair, mask, uid), ref)
    with pytest.raises(ValueError, match="no atom pair"):
        AtomDiTStack("esmfold2", 2)


def test_from_shared_pair_slices_the_projection():
    n_blocks, n_head, d_pair = 3, 4, 16
    w = torch.randn(d_pair)
    proj = torch.randn(n_blocks * n_head, d_pair)
    states = from_shared_pair(w, proj, n_blocks=n_blocks, n_head=n_head, norm_bias=torch.randn(d_pair))
    assert len(states) == n_blocks
    for b, s in enumerate(states):
        assert torch.equal(s["attention.ln_pair.weight"], w)
        assert torch.equal(s["attention.to_bias.weight"], proj[b * n_head : (b + 1) * n_head])
    # super-block projections (AF3's token DiT: one per four blocks) are their row-concatenation
    split = from_shared_pair(w, [proj[:8], proj[8:]], n_blocks=n_blocks, n_head=n_head)
    assert all(torch.equal(a["attention.to_bias.weight"], b["attention.to_bias.weight"]) for a, b in zip(states, split, strict=True))
    stack = AtomDiTStack("protenix", n_blocks).load_shared_pair(w, proj)
    for b, blk in enumerate(stack.blocks):
        block = typing.cast("typing.Any", blk)
        assert torch.equal(block.attention.to_bias.weight, proj[b * n_head : (b + 1) * n_head])
        assert torch.equal(block.attention.ln_pair.weight, w)
    with pytest.raises(ValueError, match="projection"):
        from_shared_pair(w, proj[:-1], n_blocks=n_blocks, n_head=n_head)


def test_shared_pair_norm_offset_is_cancelled():
    """The dropped LayerNorm offset adds one constant per head to every logit of a block: the block's output is unchanged."""
    torch.manual_seed(0)
    block = typing.cast("typing.Any", TokenDiT("protenix", 64, 32, 16, 4, 2))
    g = _gen(1)
    single, cond, pair = torch.randn(1, 1, 8, 64, generator=g), torch.randn(1, 1, 8, 32, generator=g), torch.randn(1, 8, 8, 16, generator=g)
    beta = torch.randn(16, generator=g)
    with torch.no_grad():
        ref = block(single, cond, pair)
        # the offset's contribution to the bias: W beta per head, constant over (query, key)
        shift = block.attention.to_bias.weight @ beta
        block.attention.to_bias.bias = torch.nn.Parameter(shift)
        torch.testing.assert_close(block.attention.to_bias(torch.zeros(16)), shift)  # the projection adds it
        out = block(single, cond, pair)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


# ----------------------------------------------------------------------------------------------------------------------------- R12
def test_parallel_residual_reference():
    g = _gen()
    x = torch.randn(2, 5, 7, generator=g)
    ys = [x + 0.1 * torch.randn(2, 5, 7, generator=g) for _ in range(3)]
    want = x + sum(y - x for y in ys)
    torch.testing.assert_close(parallel_residual(x, *ys), want)
    m = ParallelResidual(d=7, branches=3)
    torch.testing.assert_close(m(x, *ys), want)
    with pytest.raises(BackendUnsupported, match="CUDA"):
        parallel_residual(x, *ys, backend="triton")
    with pytest.raises(BackendUnsupported, match="CUDA"):
        parallel_residual(x, *ys, backend="cuda")
    with pytest.raises(ValueError, match="at least one"):
        parallel_residual(x)


def test_parallel_residual_inputs_follow_the_stream():
    m = ParallelResidual(d=8, branches=2)
    inp = m.make_inputs(length=4, samples=1, depth=3, dtype=torch.float32, device="cpu", generator=_gen(), stream="msa_token")
    assert set(inp) == {"x", "y0", "y1"} and inp["x"].shape == (1, 3, 4, 8)
    inp = m.make_inputs(length=4, samples=1, depth=None, dtype=torch.float32, device="cpu", generator=_gen(), stream="token_pair")
    assert inp["y1"].shape == (1, 4, 4, 8)


# ----------------------------------------------------------------------------------------------------------------------------- R14
@pytest.mark.parametrize("cls", [AtomDiTBoltz2, AtomDiTRoseTTAFold3, AtomDiTChai1])
def test_shared_kv_adaln_invariant(cls):
    block = cls()
    assert block.kv_norm == "shared"
    assert not hasattr(block.attention, "ada_ln_kv")
    assert not any(".ada_ln_kv." in k or k.startswith("attention.ada_ln_kv") for k in block.state_dict())
    assert "Invariant" in (cls.__doc__ or "")


@pytest.mark.parametrize("block", [TokenDiTRoseTTAFold3(), AtomDiTRoseTTAFold3()], ids=["token", "atom"])
def test_rosettafold3_has_no_query_bias(block):
    assert isinstance(block.attention, FlatQKNormAugmentedAttention)
    assert block.attention.to_query.bias is None
    assert "attention.to_query.bias" not in block.state_dict()
    assert "Invariant" in (FlatQKNormAugmentedAttention.__doc__ or "")


# ----------------------------------------------------------------------------------------------------------------------------- R18
def _fold_inputs(block, stream, length, folds, samples):
    return block.make_inputs(
        length=length, samples=samples, depth=folds, dtype=torch.float32, device="cpu", generator=_gen(), stream=stream
    )


@pytest.mark.parametrize(
    ("make", "stream", "length"),
    [
        (lambda: TokenDiT("protenix", 64, 32, 16, 4, 2), "token_folds", 8),
        (lambda: AtomDiT("protenix"), "atom_folds", 96),
        (lambda: AtomDiT("af3", key_window="slide"), "atom_folds", 96),
    ],
    ids=["token", "atom-pad", "atom-slide"],
)
def test_multi_fold_call_equals_one_call_per_fold(make, stream, length):
    torch.manual_seed(0)
    block = make()
    folds, samples = 3, 2
    inp = _fold_inputs(block, stream, length, folds, samples)
    assert inp["single"].shape[:2] == (samples, folds) and inp["pair"].shape[0] == folds
    assert inp["mask"].shape[0] == folds and inp["mask"].stride(0) == 0  # one input's mask over its folds
    assert inp["cond"].stride(0) == 0  # one conditioning per fold, shared by its samples
    with torch.no_grad():
        out = block(**inp)
        for b in range(folds):
            one = block(inp["single"][:, b : b + 1], inp["cond"][:, b : b + 1], inp["pair"][b : b + 1], inp["mask"][b : b + 1])
            torch.testing.assert_close(out[:, b : b + 1], one, rtol=1e-5, atol=1e-5)
