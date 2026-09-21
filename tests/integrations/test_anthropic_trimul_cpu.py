"""What the Anthropic TriMul wiring promises when no payload is present (the CI machine's situation).

The payload itself needs an H100 and TRIMUL_NATIVE_BUILD_DIR; these cover the contract around it: a named-but-absent payload refuses with
its reason instead of quietly running something else, the auto option falls back, and no other op accepts the name.
"""
import pytest
import torch

from miniworld_engine.integrations import anthropic_trimul as native
from miniworld_engine.modules import (
    BidirectionalTriangleMultiplication,
    TriangleMultiplication,
    dispatch,
)
from miniworld_engine.modules.exceptions import ImplementationType

PAIR = torch.zeros(1, 8, 8, 128)                    # fp32: also exercises the dtype refusal
BF16 = torch.zeros(1, 8, 8, 128, dtype=torch.bfloat16)


@pytest.fixture(autouse=True)
def _no_payload(monkeypatch):
    monkeypatch.delenv(native.ENV, raising=False)


def test_the_option_resolves_only_for_trimul():
    assert dispatch.resolve("triangle_multiplication", ImplementationType.ANTHROPIC) is dispatch.KernelBackend.ANTHROPIC
    for op in ("layernorm", "transition", "triangle_attention"):
        with pytest.raises(ValueError, match="triangle_multiplication only"):
            dispatch.resolve(op, ImplementationType.ANTHROPIC)


def test_only_the_named_and_the_auto_option_ask_for_it(monkeypatch):
    assert native.wanted(ImplementationType.ANTHROPIC)
    assert not native.wanted(ImplementationType.PYTORCH)
    assert not native.wanted(ImplementationType.MINIWORLD)          # no payload named
    monkeypatch.setenv(native.ENV, "/nonexistent/build")
    assert native.wanted(ImplementationType.MINIWORLD)


def test_the_refusal_says_why(monkeypatch):
    assert native.refusal(BF16, 128, 128, grad=False, dropout=False) == f"{native.ENV} is not set"
    monkeypatch.setenv(native.ENV, "/nonexistent/build")
    assert "forward-only" in native.refusal(BF16, 128, 128, grad=True, dropout=False)
    assert "dropout" in native.refusal(BF16, 128, 128, grad=False, dropout=True)
    assert "bf16" in native.refusal(PAIR, 128, 128, grad=False, dropout=False)
    assert "CUDA" in native.refusal(BF16, 128, 128, grad=False, dropout=False)   # the cheap checks come first
    assert not native.serves(BF16, 128, 128, grad=False, dropout=False)


def test_an_unusable_payload_is_a_refusal_not_a_crash(monkeypatch):
    monkeypatch.setenv(native.ENV, "/nonexistent/build")
    with pytest.raises(native.PayloadUnavailable, match="trimul_native"):
        native._load()


@pytest.mark.parametrize("cls", [TriangleMultiplication, BidirectionalTriangleMultiplication])
def test_the_named_option_refuses_rather_than_reroute(cls):
    with pytest.raises(native.PayloadUnavailable, match="cannot serve this call"):
        cls(128, implementation="anthropic")(PAIR)


@pytest.mark.parametrize("cls", [TriangleMultiplication, BidirectionalTriangleMultiplication])
def test_the_module_still_builds_its_own_primitives_under_the_option(cls):
    m = cls(128, implementation="anthropic")
    assert m.implementation is ImplementationType.ANTHROPIC          # the public option survives construction
    assert m.ln_pair.implementation is not ImplementationType.ANTHROPIC   # the payload is a TriMul one, not a LayerNorm one


def test_the_mask_element_type_follows_the_payload():
    """A payload without the templated mask has only the fp32 kernel, and it reads whatever buffer it is given AS fp32:
    handing that one a bool mask is a four-times-too-long read (wrong numbers at L384, an illegal access at L768)."""
    class Templated:
        MASK_NATIVE_DTYPES = (torch.float32, torch.bfloat16, torch.bool, torch.uint8)

    class Upstream:
        pass

    mask = torch.ones(1, 8, dtype=torch.bool)
    mask[:, ::3] = False
    assert native._pair_mask(PAIR, mask, Templated()).dtype is torch.bool
    assert native._pair_mask(PAIR, mask, Upstream()).dtype is torch.float32
    assert native._pair_mask(PAIR, None, Upstream()) is None


def test_the_auto_option_falls_back_when_nothing_is_named():
    out = TriangleMultiplication(128, implementation="pytorch")(PAIR)
    assert out.shape == PAIR.shape
