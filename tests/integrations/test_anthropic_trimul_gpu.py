"""The wiring against a real payload: what it computes, that the auto option finds it, and that training never takes it.

Needs an H100 and a payload (`TRIMUL_NATIVE_BUILD_DIR`, built by `experiments/trimul_k1k3_inference/build_payload.py` with both
`tmn90_z128_h128` and `tmn90_z128_h256`); skips without either. The reference is the module's own fp32 pytorch path, and the bound is the
payload's tolerance class, not bitwise equality -- the release rounds differently from our kernels by design.
"""
import copy
import math

import pytest
import torch

from miniworld_engine.integrations import anthropic_trimul as native
from miniworld_engine.modules import (
    BidirectionalTriangleMultiplication,
    TriangleMultiplication,
)

pytestmark = pytest.mark.gpu

L, C = 128, 128


def _init(m):
    m = m.to("cuda").eval()
    for name, t in m.named_parameters():
        if t.ndim >= 2:
            t.data = t.data.to(torch.bfloat16)
            t.data.normal_(std=1 / math.sqrt(t.shape[-1]))
        else:
            t.data = t.data.float()
            t.data.normal_(1.0, 0.05) if name.endswith("weight") else t.data.normal_(0.0, 0.05)
    return m


def _like(ref, make, impl):
    m = make(impl).to("cuda").eval()
    m.load_state_dict(ref.state_dict())
    for _n, prm in m.named_parameters():                       # load_state_dict keeps the destination dtype
        prm.data = prm.data.to(torch.bfloat16 if prm.ndim >= 2 else torch.float32)
    return m


CASES = {
    "outgoing": lambda impl: TriangleMultiplication(C, outgoing=True, implementation=impl, p_drop=0.0),
    "incoming": lambda impl: TriangleMultiplication(C, outgoing=False, implementation=impl, p_drop=0.0),
    "bidirectional": lambda impl: BidirectionalTriangleMultiplication(C, implementation=impl, p_drop=0.0),
}


@pytest.fixture(scope="module")
def fixture():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("the payload units are sm_90a")
    if not native.payload_dir():
        pytest.skip(f"{native.ENV} names no payload")
    torch.manual_seed(4103)
    torch.backends.cuda.matmul.allow_tf32 = False
    x = torch.randn(1, L, L, C, device="cuda", dtype=torch.bfloat16)
    mask = torch.ones(1, L, device="cuda", dtype=torch.bool)
    mask[:, ::7] = False
    return x, mask


@pytest.mark.parametrize("case", list(CASES))
def test_the_payload_computes_the_module(fixture, case):
    x, mask = fixture
    ref_mod = _init(CASES[case]("pytorch"))
    with torch.no_grad():
        ref = copy.deepcopy(ref_mod).float()(x.float(), mask)
        y = _like(ref_mod, CASES[case], "anthropic")(x, mask)
    d = (y.float() - ref).square().mean().sqrt() / ref.square().mean().sqrt()
    assert torch.isfinite(y).all(), case
    assert d < 5e-3, (case, float(d))


@pytest.mark.parametrize("case", list(CASES))
def test_the_auto_option_finds_it_and_agrees_bit_for_bit(fixture, case):
    x, mask = fixture
    ref_mod = _init(CASES[case]("pytorch"))
    with torch.no_grad():
        named = _like(ref_mod, CASES[case], "anthropic")(x, mask)
        auto = _like(ref_mod, CASES[case], "miniworld")(x, mask)
    assert torch.equal(named, auto)


@pytest.mark.parametrize("case", list(CASES))
def test_a_forward_under_grad_never_takes_it(fixture, case):
    x, mask = fixture
    m = _like(_init(CASES[case]("pytorch")), CASES[case], "miniworld")
    out = m(x.clone().requires_grad_(), mask)
    assert out.requires_grad, case
    assert getattr(m, "native_selection", None) is None, case
