"""``architecture.checkpoints.dead_modules``: residual modules whose parameters decayed to ~1e-37 (Protenix v2's 41 instances) are
flagged and return their residual input at inference -- the same bits the full forward gives (``x + update == x``), as the
Protenix-v2 kit's deadskip probe checks.  Training and grad-enabled calls still run them."""

import pytest
import torch

from miniworld_engine.architecture.checkpoints.dead_modules import (
    clear_dead_modules,
    is_dead,
    mark_dead_modules,
)
from miniworld_engine.modules import (
    AttentionPairBias,
    MSAPairWeightedAveraging,
    OuterProductMean,
    Transition,
    TriangleAttention,
)
from miniworld_engine.modules.exceptions import ImplementationType
from miniworld_engine.modules.triangle_multiplication.module import (
    TriangleMultiplication,
)

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]

ENGINE = ImplementationType.MINIWORLD
B, L, M, C, CS, CM = 1, 128, 16, 128, 384, 64


def _inputs():
    g = torch.Generator(device="cuda").manual_seed(0)

    def randn(*shape):
        return torch.randn(*shape, device="cuda", generator=g).to(torch.bfloat16)

    return {"pair": randn(B, L, L, C), "single": randn(B, L, CS), "msa": randn(B, M, L, CM),
            "mask": torch.ones(B, L, device="cuda", dtype=torch.bool)}


#: name -> (module, call(module, inputs) -> output, the residual input it returns when dead)
CASES = {
    "tri_att_start": (lambda: TriangleAttention(C, 4, implementation=ENGINE, p_drop=0.0),
                      lambda m, x: m(x["pair"], x["mask"]), "pair"),
    "tri_att_end": (lambda: TriangleAttention(C, 4, starting=False, implementation=ENGINE, p_drop=0.0),
                    lambda m, x: m(x["pair"], x["mask"]), "pair"),
    "tri_mul_out": (lambda: TriangleMultiplication(C, implementation=ENGINE, p_drop=0.0),
                    lambda m, x: m(x["pair"], x["mask"]), "pair"),
    "pair_transition": (lambda: Transition(C, 4, implementation=ENGINE), lambda m, x: m(x["pair"]), "pair"),
    "single_transition": (lambda: Transition(CS, 4, implementation=ENGINE), lambda m, x: m(x["single"]), "single"),
    "attention_pair_bias": (lambda: AttentionPairBias(CS, C, 16, implementation=ENGINE),
                            lambda m, x: m(x["single"], x["pair"], x["mask"]), "single"),
    "msa_pwa": (lambda: MSAPairWeightedAveraging(CM, C, 8, implementation=ENGINE),
                lambda m, x: m(x["msa"], x["pair"], x["mask"]), "msa"),
    "opm": (lambda: OuterProductMean(CM, C, implementation=ENGINE),
            lambda m, x: m(x["msa"], residual=x["pair"]), "pair"),
}


def _decayed(make):
    """A module whose every parameter is ~5e-37 (Protenix v2's dead instances), bf16 matrices and fp32 norms as loaded."""
    m = make().cuda().eval()
    with torch.no_grad():
        for p in m.parameters():
            p.fill_(4.9e-37)
    return m.to(torch.bfloat16)


@pytest.mark.parametrize("name", sorted(CASES))
def test_a_dead_module_returns_its_residual_input_bitwise(name):
    make, call, residual = CASES[name]
    m = _decayed(make)
    x = _inputs()
    assert is_dead(m)
    with torch.no_grad():
        full = call(m, x)
        assert mark_dead_modules(m) == [""]
        skipped = call(m, x)
    assert skipped is x[residual]
    assert torch.equal(full, skipped), f"{name}: the full forward is not x + 0 (max |diff| {(full.float() - skipped.float()).abs().max()})"


@pytest.mark.parametrize("name", ["tri_att_start", "pair_transition"])
def test_grad_and_training_calls_run_the_module(name):
    make, call, residual = CASES[name]
    m = _decayed(make)
    mark_dead_modules(m)
    x = _inputs()
    with torch.no_grad():
        m.train()
        assert call(m, x) is not x[residual]
        m.eval()
    assert call(m, x) is not x[residual]
    clear_dead_modules(m)
    assert not m.dead


def test_live_modules_are_not_flagged():
    model = torch.nn.ModuleDict({"live": TriangleAttention(C, 4, implementation=ENGINE).cuda(),
                                 "dead": _decayed(CASES["tri_mul_out"][0])})
    assert mark_dead_modules(model) == ["dead"]
    assert not model["live"].dead
