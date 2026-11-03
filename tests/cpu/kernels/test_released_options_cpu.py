# ruff: noqa: PT018, PT011, C408
"""The released-checkpoint options of the engine modules (CPU, no oracle needed): what each option means, that every default is
unchanged, and that the released-layout adapters are exact tensor reshuffles.

``divide_by_length`` (TriangleMultiplication), ``count_offset`` (OuterProductMean), ``transposed_bias`` (TriangleAttention, already there),
the ``from_released`` adapters (``modules/released.py``), the single-track / ``parallel`` PairformerBlock and ``serving`` of the DiT variant stacks.
Against FoldForge's dense modules (the oracle of the released equations): ``test_released_parity.py``.
"""
from __future__ import annotations

import pytest
import torch

from miniworld_engine.modules import released
from miniworld_engine.modules.dit_variant import DiTVariantStack, serving
from miniworld_engine.modules.exceptions import ImplementationType as IT
from miniworld_engine.modules.outer_product import OuterProductMean
from miniworld_engine.modules.pairformer import PairformerBlock, PairformerConfig
from miniworld_engine.modules.transition import Transition
from miniworld_engine.modules.triangle_attention import TriangleAttention
from miniworld_engine.modules.triangle_multiplication import TriangleMultiplication
from miniworld_engine.modules.triangle_multiplication.reference import (
    TriangleMultiplicationReference,
    inverse_length,
)

L, C = 12, 16


def _randomise(module: torch.nn.Module, seed: int = 0, std: float = 0.3) -> None:
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in module.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * std)


def _data(b: int = 2, n_masked: int = 3, seed: int = 1):
    g = torch.Generator().manual_seed(seed)
    pair = torch.randn(b, L, L, C, generator=g)
    mask = torch.ones(b, L, dtype=torch.bool)
    mask[0, L - n_masked:] = False
    return pair, mask


# ---------------------------------------------------------------------------------------------------------------- TriMul
@pytest.mark.parametrize("outgoing", [True, False])
def test_trimul_default_is_unchanged(outgoing):
    """divide_by_length=False is the old function (same module, same weights, bit equal to the reference module)."""
    pair, mask = _data()
    mod = TriangleMultiplication(C, outgoing=outgoing, p_drop=0.0).float()
    ref = TriangleMultiplicationReference(C, outgoing=outgoing)
    _randomise(mod)
    ref.load_state_dict(mod.state_dict())
    assert mod.divide_by_length is False and ref.divide_by_length is False
    torch.testing.assert_close(mod(pair, mask), pair + ref(pair, mask), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("outgoing", [True, False])
@pytest.mark.parametrize("masked", [True, False])
def test_trimul_divide_by_length_semantics(outgoing, masked):
    """The contraction is scaled by 1 / (unmasked residues of the sample) before ln_out: equal to dividing a hand-made contraction."""
    pair, mask = _data()
    if not masked:
        mask = None
    mod = TriangleMultiplication(C, outgoing=outgoing, p_drop=0.0, divide_by_length=True).float()
    _randomise(mod)
    plain = TriangleMultiplication(C, outgoing=outgoing, p_drop=0.0).float()
    plain.load_state_dict(mod.state_dict())
    out, base = mod(pair, mask), plain(pair, mask)
    assert not torch.allclose(out, base, atol=1e-3)  # the option does something
    # hand-made reference of the released form
    x = mod.ln_pair(pair)
    left = torch.sigmoid(mod.to_left_gate(x)) * mod.to_left(x)
    right = torch.sigmoid(mod.to_right_gate(x)) * mod.to_right(x)
    if mask is not None:
        m2 = (mask[:, :, None] & mask[:, None, :])[..., None]
        left, right = left * m2, right * m2
    y = torch.einsum("bikd,bjkd->bijd", left, right) if outgoing else torch.einsum("bkid,bkjd->bijd", left, right)
    count = (mask.sum(-1) if mask is not None else torch.full((pair.shape[0],), L)).clamp_min(1).float()
    y = y / count.view(-1, 1, 1, 1)
    want = pair + torch.sigmoid(mod.to_gate(x)) * mod.to_out(mod.ln_out(y))
    torch.testing.assert_close(out, want, atol=1e-5, rtol=1e-5)
    # the reference module agrees
    ref = TriangleMultiplicationReference(C, outgoing=outgoing, divide_by_length=True)
    ref.load_state_dict(mod.state_dict())
    torch.testing.assert_close(out, pair + ref(pair, mask), atol=1e-5, rtol=1e-5)


def test_inverse_length_counts_per_sample():
    pair, mask = _data()
    got = inverse_length(pair, mask).flatten()
    torch.testing.assert_close(got, torch.tensor([1 / (L - 3), 1 / L]))
    torch.testing.assert_close(inverse_length(pair, None).flatten(), torch.full((2,), 1 / L))


# ---------------------------------------------------------------------------------------------------------------- OPM
def _opm_data(b=2, s=5):
    g = torch.Generator().manual_seed(3)
    msa = torch.randn(b, s, L, 24, generator=g)
    mask = torch.rand(b, s, L, generator=g) > 0.3
    mask[:, 0] = True
    return msa, mask


def test_opm_default_is_unchanged():
    msa, mask = _opm_data()
    opm = OuterProductMean(24, C, 8)
    assert opm.count_offset is None
    _randomise(opm)
    out = opm(msa, mask)
    # clamp(count, 1) divisor, bias scaled with the product (normalize_before_proj=True: W(out / n) + b)
    left, right = opm.to_left(opm.ln_msa(msa)), opm.to_right(opm.ln_msa(msa))
    left, right = left * mask[..., None], right * mask[..., None]
    prod = torch.einsum("bmid,bmje->bijde", left, right).flatten(-2)
    count = torch.einsum("bmi,bmj->bij", mask.float(), mask.float()).clamp(min=1)[..., None]
    torch.testing.assert_close(out, opm.to_out(prod / count), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("before", [True, False])
def test_opm_count_offset_semantics(before):
    """count + offset replaces clamp(count, 1); AF3 is (W out + b) / (count + 1e-3) (normalize_before_proj=False)."""
    msa, mask = _opm_data()
    opm = OuterProductMean(24, C, 8, count_offset=1e-3, normalize_before_proj=before)
    _randomise(opm)
    out = opm(msa, mask)
    left, right = opm.to_left(opm.ln_msa(msa)) * mask[..., None], opm.to_right(opm.ln_msa(msa)) * mask[..., None]
    prod = torch.einsum("bmid,bmje->bijde", left, right).flatten(-2)
    count = torch.einsum("bmi,bmj->bij", mask.float(), mask.float())[..., None] + 1e-3
    want = opm.to_out(prod / count) if before else opm.to_out(prod) / count
    torch.testing.assert_close(out, want, atol=1e-4, rtol=1e-4)
    # a pair with no valid row is zero-divided by the offset, not by 1
    plain = OuterProductMean(24, C, 8, normalize_before_proj=before)
    plain.load_state_dict(opm.state_dict())
    assert not torch.allclose(out, plain(msa, mask), atol=1e-6)


def test_opm_count_offset_refuses_groups():
    with pytest.raises(ValueError):
        OuterProductMean(24, C, 8, groups=2, count_offset=1e-3)


@pytest.mark.parametrize("before", [True, False])
@pytest.mark.parametrize("option", [{"count_offset": 1e-3}, {}])
@pytest.mark.parametrize("bias", [False, True])
def test_opm_kernel_reference_and_divisor_params_agree_with_the_module(before, option, bias):
    """``kernels.outer_product_mean.reference`` (the oracle of the GPU tests) and ``opm_train.divisor_params`` (the kernels' two scalars,
    divisor = max(count, dmin) + doff) say what the module says."""
    from miniworld_engine.kernels.integrations.opm_train import divisor_params
    from miniworld_engine.kernels.outer_product_mean.reference import outer_product_mean_reference

    msa, mask = _opm_data()
    opm = OuterProductMean(24, C, 8, normalize_before_proj=before, projection_bias=bias, **option)
    _randomise(opm)
    got = opm(msa, mask)
    ref = outer_product_mean_reference(
        msa, mask, opm.ln_msa.weight, opm.ln_msa.bias, opm.to_left.weight, opm.to_right.weight, opm.to_out.weight, opm.to_out.bias,
        eps=float(getattr(opm.ln_msa, "eps", 1e-5)), normalize_before_proj=before, b_left=opm.to_left.bias, b_right=opm.to_right.bias, **option)
    torch.testing.assert_close(got, ref, atol=1e-4, rtol=1e-4)
    # the scalars: the module's divisor table is max(count, dmin) + doff
    dmin, doff = divisor_params(opm, msa.shape[1])
    count = torch.einsum("bmi,bmj->bij", mask.float(), mask.float())
    table = count.clamp(min=dmin) + doff
    if "count_offset" in option:
        assert (dmin, doff) == (0.0, 1e-3) and torch.equal(table, count + 1e-3)
    else:
        assert (dmin, doff) == (1.0, 0.0) and torch.equal(table, count.clamp(min=1))


def test_opm_divisor_survives_a_fully_masked_token():
    """A token no row covers has count 0: the offset divides by 1e-3 (not by 1)."""
    msa, mask = _opm_data()
    mask = mask.clone()
    mask[..., 0] = False
    opm = OuterProductMean(24, C, 8, count_offset=1e-3, normalize_before_proj=False)
    _randomise(opm)
    out = opm(msa, mask)
    torch.testing.assert_close(out[:, 0, 0], opm.to_out.bias / 1e-3 * torch.ones_like(out[:, 0, 0]), atol=1e-3, rtol=1e-4)


# ---------------------------------------------------------------------------------------------------------------- adapters
def _released_trimul_state(c=C, h=24, seed=0):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g) * 0.3
    return {"left_norm_input.weight": r(c) + 1, "left_norm_input.bias": r(c), "projection.weight": r(2 * h, c), "gate.weight": r(2 * h, c),
            "center_norm.weight": r(h) + 1, "center_norm.bias": r(h), "output_projection.weight": r(c, h), "gating_linear.weight": r(c, c)}


@pytest.mark.parametrize("outgoing", [True, False])
def test_trimul_interleaved_layout(outgoing):
    """The interleaved projection: row 2k is channel k of branch a, row 2k+1 of branch b; incoming swaps the branches."""
    state = _released_trimul_state()
    conv = released.triangle_multiplication_from_released(state, outgoing=outgoing)
    a, b = state["projection.weight"][0::2], state["projection.weight"][1::2]
    ga, gb = state["gate.weight"][0::2], state["gate.weight"][1::2]
    left, right = (a, b) if outgoing else (b, a)
    gl, gr = (ga, gb) if outgoing else (gb, ga)
    assert torch.equal(conv["to_left.weight"], left) and torch.equal(conv["to_right.weight"], right)
    assert torch.equal(conv["to_left_gate.weight"], gl) and torch.equal(conv["to_right_gate.weight"], gr)
    mod = TriangleMultiplication.from_released(state, outgoing=outgoing)
    assert mod.d_hidden == 24 and mod.to_left.weight.shape == (24, C) and mod.p_drop == 0.0


def test_trimul_from_released_matches_the_released_equation():
    """The released function written out on the interleaved matrix (xfold / FoldForge ``triangle_update``), the engine module from the same weights."""
    pair, mask = _data()
    state = _released_trimul_state()
    for outgoing in (True, False):
        mod = TriangleMultiplication.from_released(state, outgoing=outgoing, divide_by_length=True)
        x = torch.nn.functional.layer_norm(pair, (C,), state["left_norm_input.weight"], state["left_norm_input.bias"])
        proj = (x @ state["projection.weight"].T).movedim(-1, -3)  # [b, 2h, i, j]
        proj = proj * (mask[:, None, :, None] & mask[:, None, None, :])
        proj = proj * torch.sigmoid((x @ state["gate.weight"].T).movedim(-1, -3))
        a, b = proj.unflatten(-3, (-1, 2)).unbind(-3)
        y = torch.einsum("...cik,...cjk->...cij" if outgoing else "...ckj,...cki->...cij", a, b).movedim(-3, -1)
        y = y / mask.sum(-1).view(-1, 1, 1, 1)
        y = torch.nn.functional.layer_norm(y, (24,), state["center_norm.weight"], state["center_norm.bias"])
        want = pair + torch.sigmoid(x @ state["gating_linear.weight"].T) * (y @ state["output_projection.weight"].T)
        torch.testing.assert_close(mod(pair, mask), want, atol=2e-5, rtol=2e-5)


def test_transition_from_released():
    g = torch.Generator().manual_seed(0)
    state = {"input_layer_norm.weight": torch.randn(C, generator=g), "input_layer_norm.bias": torch.randn(C, generator=g),
             "transition1.weight": torch.randn(2 * 3 * C, C, generator=g) * 0.3, "transition2.weight": torch.randn(C, 3 * C, generator=g) * 0.3}
    mod = Transition.from_released(state)
    assert mod.n == 3 and mod.expand_a.weight.dtype == torch.float32
    x = torch.randn(2, L, C, generator=g)
    n = torch.nn.functional.layer_norm(x, (C,), state["input_layer_norm.weight"], state["input_layer_norm.bias"])
    a, b = (n @ state["transition1.weight"].T).chunk(2, dim=-1)
    want = x + (torch.nn.functional.silu(a) * b) @ state["transition2.weight"].T
    torch.testing.assert_close(mod(x), want, atol=1e-5, rtol=1e-5)


def test_released_missing_key_is_named():
    with pytest.raises(KeyError, match="gating_linear"):
        released.triangle_multiplication_from_released({k: v for k, v in _released_trimul_state().items() if k != "gating_linear.weight"}, outgoing=True)


def test_prefix_is_stripped():
    state = {f"blocks.3.tm.{k}": v for k, v in _released_trimul_state().items()}
    mod = TriangleMultiplication.from_released(state, outgoing=True, prefix="blocks.3.tm")
    assert mod.to_left.weight.shape == (24, C)


# ---------------------------------------------------------------------------------------------------------------- TriangleAttention
def test_triangle_attention_bias_orientation_option():
    """AF3's ending node projects the bias from the pair as it is (transposed_bias=False): the output differs from the SI form, the starting node is indifferent."""
    pair, mask = _data()
    outs = {}
    for starting in (True, False):
        for tb in (True, False):
            m = TriangleAttention(C, 2, starting=starting, transposed_bias=tb, p_drop=0.0, gate_bias=True, out_bias=True)
            _randomise(m, seed=5)
            outs[starting, tb] = m(pair, mask)
    assert torch.allclose(outs[True, True], outs[True, False])
    assert not torch.allclose(outs[False, True], outs[False, False], atol=1e-4)


# ---------------------------------------------------------------------------------------------------------------- Pairformer
def _cfg(**kw):
    base = {"d_pair": C, "d_hidden_tri_multi": C, "d_hidden_tri_attention": 8, "n_head_tri_attention": 2, "p_drop": 0.0}
    return PairformerConfig(**{**base, **kw})


def test_pairformer_defaults_unchanged():
    cfg = PairformerConfig()
    assert (cfg.d_single, cfg.parallel, cfg.divide_by_length, cfg.transposed_bias) == (0, False, False, True)
    pair, mask = _data()
    blk = PairformerBlock(_cfg())
    _randomise(blk)
    out = blk(pair, mask)
    assert isinstance(out, torch.Tensor)
    with pytest.raises(ValueError):
        blk(pair, mask, torch.zeros(2, L, 8))


@pytest.mark.parametrize("parallel", [False, True])
def test_pairformer_single_track_schedules(parallel):
    pair, mask = _data()
    single = torch.randn(2, L, 32, generator=torch.Generator().manual_seed(9))
    blk = PairformerBlock(_cfg(d_single=32, n_head_single=4, parallel=parallel, single_gate_bias=1.0))
    _randomise(blk, seed=2, std=0.2)
    p_out, s_out = blk(pair, mask, single)
    # the schedule by hand
    delta = lambda m, x: m(x, mask) - x
    if parallel:
        upd = sum(delta(m, pair) for m in (blk.tri_multi_outgoing, blk.tri_multi_incoming, blk.tri_atten_starting, blk.tri_atten_ending))
        want_pair = pair + upd + (blk.transition_pair(pair) - pair)
        want_single = single + (blk.single_attention(single, pair, mask) - single) + (blk.single_transition(single) - single)
    else:
        x = pair
        for m in (blk.tri_multi_outgoing, blk.tri_multi_incoming, blk.tri_atten_starting, blk.tri_atten_ending):
            x = m(x, mask)
        want_pair = blk.transition_pair(x)
        want_single = blk.single_transition(blk.single_attention(single, want_pair, mask))
    torch.testing.assert_close(p_out, want_pair, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(s_out, want_single, atol=1e-5, rtol=1e-5)
    seq_pair = PairformerBlock(_cfg(d_single=32, n_head_single=4, parallel=not parallel, single_gate_bias=1.0))
    seq_pair.load_state_dict(blk.state_dict())
    assert not torch.allclose(seq_pair(pair, mask, single)[0], p_out, atol=1e-4)  # the schedules differ


def test_pairformer_missing_single_is_refused():
    pair, mask = _data()
    with pytest.raises(ValueError):
        PairformerBlock(_cfg(d_single=32, n_head_single=4))(pair, mask)


# ---------------------------------------------------------------------------------------------------------------- DiT variant stacks
def test_variant_stack_pytorch_serves_and_trains():
    """PYTORCH is a serving implementation: serving() says so, the call runs with a mask, and it differentiates."""
    stack = DiTVariantStack(2, d_single=32, d_cond=16, d_pair=8, n_head=4, n=2, parallel=True, implementation=IT.PYTORCH, kq_norm=True)
    _randomise(stack, std=0.1)
    x = torch.randn(2, 10, 32, requires_grad=True)
    cond, pair = torch.randn(10, 16), torch.randn(10, 10, 8)
    mask = torch.ones(10, dtype=torch.bool)
    mask[-2:] = False
    assert serving(stack, x, cond, pair, mask) == stack.serving(x, cond, pair, mask) == "pytorch"
    out = stack(x, cond, pair, mask)
    out.square().sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(p.grad is not None for p in stack.parameters())
