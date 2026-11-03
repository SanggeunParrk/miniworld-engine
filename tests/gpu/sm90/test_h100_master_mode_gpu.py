# ruff: noqa: PT018
"""fp32-master / bf16-compute mode (``set_master_weights`` / ``master_weights=True``) and ``set_implementation`` on H100, registry-declared shapes only.

A module whose parameters are the fp32 master, called with bf16 activations, against the same module cast to bf16 (the layout the registry rows are
recorded for): the bf16 matrices of the cast are the roundings of the master, so the two run the same kernels on the same values and must agree
within bf16 bands, forward and backward (input and parameter gradients; the master's gradients are fp32). A module whose master an optimizer step
changed must run on the NEW weights (the cached bf16 packs are invalidated by the version counter). A block switched with ``set_implementation``
computes what a block built with that implementation computes.

Shapes (``registry_module.csv``): TriMul / bidirectional d_pair = d_hidden = 128; TriangleAttention d_pair 64, d_hidden 64, 4 heads; Transition d_hidden 128,
n 4 (token_pair); OuterProductMean 64 / 128 / 32; MSAPairWeightedAveraging 64 / 128 / 8 heads; AttentionPairBias 384 / 128 / 8 heads; PairformerBlock 128 /
128 / 32 per head; all at length 128 (the registry's smallest rung), bf16.
"""
import copy

import pytest
import torch
from portable_helpers import randomize, relative

from miniworld_engine import settings
from miniworld_engine.modules.attention_pair_bias import AttentionPairBias
from miniworld_engine.modules.exceptions import ImplementationType as IT
from miniworld_engine.modules.msa_pair_weighted_averaging import (
    MSAPairWeightedAveraging,
)
from miniworld_engine.modules.outer_product import OuterProductMean
from miniworld_engine.modules.pairformer import PairformerBlock, PairformerConfig
from miniworld_engine.modules.transition import Transition
from miniworld_engine.modules.triangle_attention import TriangleAttention
from miniworld_engine.modules.triangle_multiplication import TriangleMultiplication
from miniworld_engine.modules.triangle_multiplication.bidirectional import (
    BidirectionalTriangleMultiplication,
)

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
        reason="Hopper (sm_90) required",
    ),
]

L = 128
BF = torch.bfloat16
MW = IT.MINIWORLD
CFG = PairformerConfig(d_pair=128, d_hidden_tri_multi=128, d_hidden_tri_attention=32, p_drop=0.0, n_block=1)


@pytest.fixture(autouse=True)
def policy():
    old = settings.configure(force_portable_kernels=False)
    try:
        yield
    finally:
        settings.configure(**vars(old))


def _pair(g):
    return torch.randn(1, L, L, 128, device="cuda", generator=g).to(BF)


# name -> (factory(master: bool), inputs(generator) -> tuple of tensors, mask-free call(module, *inputs))
CASES = {
    "trimul": (lambda **kw: TriangleMultiplication(128, d_hidden=128, implementation=MW, p_drop=0.0, **kw),
               lambda g: (_pair(g),), lambda m, x: m(x)),
    "trimul_bidirectional": (lambda **kw: BidirectionalTriangleMultiplication(128, 128, implementation=MW, p_drop=0.0, **kw),
                             lambda g: (_pair(g),), lambda m, x: m(x)),
    "triangle_attention": (lambda **kw: TriangleAttention(64, 4, d_hidden=64, implementation=MW, p_drop=0.0, **kw),
                           lambda g: (torch.randn(1, L, L, 64, device="cuda", generator=g).to(BF),), lambda m, x: m(x)),
    "transition": (lambda **kw: Transition(128, 4, implementation=MW, **kw),
                   lambda g: (_pair(g),), lambda m, x: m(x)),
    "outer_product_mean": (lambda **kw: OuterProductMean(64, 128, 32, implementation=MW, **kw),
                           lambda g: (torch.randn(1, 256, L, 64, device="cuda", generator=g).to(BF), _pair(g)),
                           lambda m, msa, pair: m(msa, residual=pair)),
    "msa_pair_weighted_averaging": (lambda **kw: MSAPairWeightedAveraging(64, 128, 8, implementation=MW, **kw),
                                    lambda g: (torch.randn(1, 256, L, 64, device="cuda", generator=g).to(BF), _pair(g)),
                                    lambda m, msa, pair: m(msa, pair)),
    "attention_pair_bias": (lambda **kw: AttentionPairBias(384, 128, 8, implementation=MW, **kw),
                            lambda g: (torch.randn(1, L, 384, device="cuda", generator=g).to(BF), _pair(g)),
                            lambda m, single, pair: m(single, pair)),
    "pairformer_block": (lambda **kw: PairformerBlock(CFG, implementation=MW, **kw),
                         lambda g: (_pair(g),), lambda m, x: m(x)),
}


def _bf16_copy(master):
    """The pure-bf16 module: a copy of ``master`` with the pin lifted and its matrices rounded (norm affines stay fp32, as in the registry layout)."""
    plain = copy.deepcopy(master)
    plain.set_master_weights(False)
    plain.bfloat16()
    assert not plain.master_weights and any(p.dtype == BF for p in plain.parameters())
    return plain


def _pair_modules(name, seed=0):
    """(master module: fp32 parameters, pure-bf16 module with the master's values rounded)."""
    torch.manual_seed(seed)
    make = CASES[name][0]
    master = randomize(make(master_weights=True).cuda())
    assert all(p.dtype == torch.float32 for p in master.parameters())
    return master, _bf16_copy(master)


def _run(name, module, inputs, train):
    call = CASES[name][2]
    ins = [t.detach().clone().requires_grad_(train) for t in inputs]
    module.zero_grad(set_to_none=True)
    with torch.set_grad_enabled(train):
        out = call(module, *ins)
        if train:
            g = torch.Generator(device="cuda").manual_seed(7)
            out.backward((torch.randn(out.shape, device="cuda", generator=g) * 0.1).to(out.dtype))
    grads = {n: p.grad.detach().clone() for n, p in module.named_parameters() if p.grad is not None}
    return out.detach(), [t.grad for t in ins if t.grad is not None], grads


@pytest.mark.parametrize("train", [False, True], ids=["eval", "train"])
@pytest.mark.parametrize("name", sorted(CASES))
def test_master_with_bf16_activations_matches_the_bf16_module(name, train):
    master, plain = _pair_modules(name)
    master.train(train)
    plain.train(train)
    inputs = CASES[name][1](torch.Generator(device="cuda").manual_seed(1))
    out_m, dx_m, g_m = _run(name, master, inputs, train)
    out_p, dx_p, g_p = _run(name, plain, inputs, train)
    assert out_m.dtype == BF
    assert relative(out_m, out_p) < 1.5e-2, (name, relative(out_m, out_p))
    if not train:
        return
    for a, b in zip(dx_m, dx_p, strict=True):
        assert relative(a, b) < 3e-2, (name, "dx", relative(a, b))
    assert set(g_m) == set(g_p)
    for n in g_m:
        assert master.get_parameter(n).dtype == torch.float32
        if g_m[n].dtype != torch.float32:
            raise AssertionError((name, n, "the master's gradient must be fp32", g_m[n].dtype))
        assert relative(g_m[n], g_p[n]) < 6e-2, (name, n, relative(g_m[n], g_p[n]))


@pytest.mark.parametrize("name", ["attention_pair_bias", "triangle_attention", "transition", "trimul"])
def test_an_optimizer_step_invalidates_the_bf16_packs(name):
    """The output after ``SGD.step()`` is the NEW weights': a stale cached pack would reproduce the first output."""
    master, plain = _pair_modules(name, seed=3)
    master.eval()
    plain.eval()
    inputs = CASES[name][1](torch.Generator(device="cuda").manual_seed(2))
    with torch.no_grad():
        before = CASES[name][2](master, *inputs).clone()
        before_ref = CASES[name][2](plain, *inputs).clone()
        assert torch.equal(before, CASES[name][2](master, *inputs)), "an unchanged master must give the same output (cached packs)"
    master.train()
    _run(name, master, inputs, True)
    groups = [{"params": [p], "lr": float(0.2 * p.detach().norm() / p.grad.norm().clamp_min(1e-12))} for p in master.parameters()
              if p.grad is not None]
    torch.optim.SGD(groups, lr=0.1).step()
    master.eval()
    with torch.no_grad():
        after = CASES[name][2](master, *inputs)
        after_ref = CASES[name][2](_bf16_copy(master).eval(), *inputs)
    assert relative(after_ref, before_ref) > 2e-2, "the step must change the output (test precondition)"
    assert relative(after, before) > 2e-2, "the packs were not invalidated: the call ran on the weights before the step"
    assert relative(after, after_ref) < 1.5e-2, (name, relative(after, after_ref))


def test_set_implementation_on_a_live_h100_block():
    """PYTORCH <-> MINIWORLD on one block object: each state computes what a block built with that implementation computes, bit for bit."""
    torch.manual_seed(5)
    live = randomize(PairformerBlock(CFG, implementation=IT.PYTORCH).cuda().bfloat16()).eval()
    built = {i: PairformerBlock(CFG, implementation=i).cuda().bfloat16().eval() for i in (IT.PYTORCH, MW)}
    for b in built.values():
        b.load_state_dict(live.state_dict())
    x = _pair(torch.Generator(device="cuda").manual_seed(4))
    with torch.no_grad():
        a = live(x)
        assert torch.equal(a, built[IT.PYTORCH](x))
        assert live.set_implementation(MW) is live
        b = live(x)
        assert torch.equal(b, built[MW](x))
        assert relative(b, a) < 5e-2
        live.set_implementation(IT.PYTORCH)
        assert torch.equal(live(x), a)
