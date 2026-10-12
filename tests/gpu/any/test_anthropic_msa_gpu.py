"""The Anthropic MSA paths against this engine's own, on the machine they are built for.

Needs an H100 and OPT_CORE_DIR; skipped everywhere else. The engine's own path is the SAME module with the
payload hidden -- building a second module with implementation="miniworld" would compare the fused path with
itself, because naming a payload opts that path in too.
"""
import os

import pytest
import torch

from miniworld_engine.integrations import anthropic_msa as msa
from miniworld_engine.modules.exceptions import ImplementationType
from miniworld_engine.modules.msa_pair_weighted_averaging import (
    MSAPairWeightedAveraging,
)
from miniworld_engine.modules.outer_product import OuterProductMean

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU"),
    pytest.mark.skipif(torch.cuda.is_available() and torch.cuda.get_device_capability() != (9, 0),
                       reason="the epilogue is built for sm_90a"),
    pytest.mark.skipif(not os.environ.get(msa.ENV), reason=f"{msa.ENV} is not set"),
]

D_MSA, D_PAIR, D_HID, HEADS, L, S = 64, 128, 32, 8, 384, 256
DEV, DT = "cuda", torch.bfloat16


@pytest.fixture(scope="module")
def inputs():
    torch.manual_seed(0)
    return {
        "msa": (torch.randn(1, S, L, D_MSA, device=DEV, dtype=DT) * 0.5).contiguous(),
        "pair": (torch.randn(1, L, L, D_PAIR, device=DEV, dtype=DT) * 0.5).contiguous(),
        "full": torch.ones(1, S, L, device=DEV, dtype=torch.bool),
        "ragged": ((torch.rand(1, S, 1, device=DEV) < 0.7) & (torch.rand(1, 1, L, device=DEV) < 0.85)),
    }


class hidden:
    """The engine's own path: the same module with the payload hidden for the duration."""

    def __enter__(self):
        self.saved = os.environ.pop(msa.ENV, None)

    def __exit__(self, *a):
        if self.saved is not None:
            os.environ[msa.ENV] = self.saved


def rel(a, b):
    return (torch.linalg.vector_norm(a.float() - b.float()) / torch.linalg.vector_norm(b.float())).item()


def _opm_pair():
    own = OuterProductMean(D_MSA, D_PAIR, D_HID, implementation=ImplementationType.MINIWORLD).to(DEV).to(DT).eval()
    with torch.no_grad():                       # the output projection is zero-initialised: give it something to say
        own.to_out.weight.normal_(0, 0.2)
        own.to_out.bias.normal_(0, 0.2)
    fused = OuterProductMean(D_MSA, D_PAIR, D_HID, implementation=ImplementationType.ANTHROPIC).to(DEV).to(DT).eval()
    fused.load_state_dict(own.state_dict())
    return own, fused


@pytest.mark.parametrize("mask_key", ["full", "ragged"])
def test_opm_matches_the_engines_own_path(inputs, mask_key):
    own, fused = _opm_pair()
    m, mask = inputs["msa"], inputs[mask_key]
    with torch.inference_mode():
        with hidden():
            ref = own(m, mask)
        assert rel(fused(m, mask), ref) < 5e-3          # one bf16 ulp is 3.9e-3


def test_opm_is_no_further_from_the_exact_answer_than_the_engines_own_path(inputs):
    """The fused path divides by the mask count AFTER the projection -- the same function with one rounding
    fewer -- so it must not be the worse of the two against an fp32 reference of the same statements."""
    own, fused = _opm_pair()
    m, mask = inputs["msa"], inputs["ragged"]
    with torch.inference_mode():
        y = torch.nn.functional.layer_norm(m.float(), (D_MSA,), own.ln_msa.weight.float(),
                                           own.ln_msa.bias.float(), own.ln_msa.eps)
        a = (y @ own.to_left.weight.float().t()) * mask[..., None].float()
        b = (y @ own.to_right.weight.float().t()) * mask[..., None].float()
        mf = mask[0].float()
        o = torch.einsum("bmid,bmje->bijde", a, b).flatten(-2) / (mf.t() @ mf).clamp(min=1)[None, ..., None]
        exact = o @ own.to_out.weight.float().t() + own.to_out.bias.float()
        with hidden():
            engine = own(m, mask)
        assert rel(fused(m, mask), exact) <= rel(engine, exact)


def test_opm_residual_is_added_by_the_module_not_the_path(inputs):
    _own, fused = _opm_pair()
    m, mask = inputs["msa"], inputs["full"]
    z = torch.randn(1, L, L, D_PAIR, device=DEV, dtype=DT)
    with torch.inference_mode():
        assert rel(fused(m, mask, None, z) - z, fused(m, mask)) < 5e-3


def test_pwa_matches_the_engines_own_path(inputs):
    own = MSAPairWeightedAveraging(D_MSA, D_PAIR, HEADS, D_HID, p_drop=0.0,
                                   implementation=ImplementationType.MINIWORLD).to(DEV).to(DT).eval()
    with torch.no_grad():
        own.to_out.weight.normal_(0, 0.2)
    fused = MSAPairWeightedAveraging(D_MSA, D_PAIR, HEADS, D_HID, p_drop=0.0,
                                     implementation=ImplementationType.ANTHROPIC).to(DEV).to(DT).eval()
    fused.load_state_dict(own.state_dict())
    m, pair = inputs["msa"], inputs["pair"]
    for mask in (None, inputs["ragged"][:, 0, :].contiguous()):
        with torch.inference_mode():
            with hidden():
                ref = own(m, pair, mask)
            assert rel(fused(m, pair, mask), ref) < 5e-3


def test_a_grad_enabled_call_is_refused_not_run(inputs):
    _, fused = _opm_pair()
    with pytest.raises(msa.PayloadUnavailable, match="forward-only"):
        fused(inputs["msa"].clone().requires_grad_(True), inputs["full"])


def test_the_auto_option_takes_the_path_when_a_payload_is_named(inputs):
    """`miniworld` uses it where it fits (the payload is named) and its own statements where it does not."""
    own, _ = _opm_pair()
    m = inputs["msa"]
    with torch.inference_mode():
        assert msa.wanted(own.implementation)
        assert msa.serves_opm(m, D_HID, D_PAIR, grad=False, interchain=False)
    assert not msa.serves_opm(m.float(), D_HID, D_PAIR, grad=False, interchain=False)
