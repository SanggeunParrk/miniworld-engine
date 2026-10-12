"""What the Anthropic MSA wiring promises when no payload is present (the CI machine's situation).

The paths themselves need an H100 and OPT_CORE_DIR; these cover the contract around them: a named-but-absent
payload refuses with its reason instead of quietly running something else, the auto option falls back, and the
refusals that do not need a GPU (dtype, batch, the epilogue's fixed dims) name what they refused.
"""
import pytest
import torch

from miniworld_engine.integrations import anthropic_msa as msa
from miniworld_engine.modules.exceptions import ImplementationType
from miniworld_engine.modules.msa_pair_weighted_averaging import (
    MSAPairWeightedAveraging,
)
from miniworld_engine.modules.outer_product import OuterProductMean

BF16 = torch.zeros(1, 8, 16, 64, dtype=torch.bfloat16)


@pytest.fixture(autouse=True)
def _no_payload(monkeypatch):
    monkeypatch.delenv(msa.ENV, raising=False)


def test_only_the_named_and_the_auto_option_ask_for_it(monkeypatch):
    assert msa.wanted(ImplementationType.ANTHROPIC)
    assert not msa.wanted(ImplementationType.PYTORCH)
    assert not msa.wanted(ImplementationType.MINIWORLD)             # no payload named
    monkeypatch.setenv(msa.ENV, "/nonexistent/opt_core")
    assert msa.wanted(ImplementationType.MINIWORLD)


def test_an_absent_payload_is_a_reason_not_a_crash(monkeypatch):
    reason = msa.opm_refusal(BF16, 32, 128, grad=False, interchain=False)
    assert reason is not None
    assert msa.ENV in reason
    reason = msa.pwa_refusal(BF16, 64, 128, 8, 32, grad=False, dropout=False)
    assert reason is not None
    assert msa.ENV in reason
    # a directory that is not a checkout is named in the reason too (the loader states it; on a CPU box the
    # device refusal comes first, which is why this asks the loader rather than the refusal)
    monkeypatch.setenv(msa.ENV, "/nonexistent/opt_core")
    with pytest.raises(msa.PayloadUnavailable, match="no opt_core/ops/msa_opm"):
        msa._load()


@pytest.mark.parametrize(
    ("kwargs", "dims", "word"),
    [
        ({"grad": True, "interchain": False}, (32, 128), "forward-only"),
        ({"grad": False, "interchain": True}, (32, 128), "interchain"),
        ({"grad": False, "interchain": False}, (32, 192), "d_pair=128"),
    ],
)
def test_the_opm_refusals_name_what_they_refused(monkeypatch, kwargs, dims, word):
    monkeypatch.setenv(msa.ENV, "/nonexistent/opt_core")           # past the "not set" check
    reason = msa.opm_refusal(BF16, *dims, **kwargs)
    assert reason is not None
    assert word in reason


def test_the_pwa_refusals_name_what_they_refused(monkeypatch):
    monkeypatch.setenv(msa.ENV, "/nonexistent/opt_core")
    reason = msa.pwa_refusal(BF16, 64, 128, 8, 32, grad=True, dropout=False)
    assert reason is not None
    assert "forward-only" in reason
    reason = msa.pwa_refusal(BF16, 64, 128, 8, 32, grad=False, dropout=True)
    assert reason is not None
    assert "row-dropout" in reason
    reason = msa.pwa_refusal(BF16, 64, 128, 4, 32, grad=False, dropout=False)
    assert reason is not None
    assert "n_head=8" in reason
    reason = msa.pwa_refusal(BF16.float(), 64, 128, 8, 32, grad=False, dropout=False)
    assert reason is not None
    assert "bf16" in reason


def test_the_explicit_option_raises_instead_of_rerouting():
    with pytest.raises(msa.PayloadUnavailable, match=msa.ENV):
        msa.require_opm(BF16, 32, 128, grad=False, interchain=False)
    with pytest.raises(msa.PayloadUnavailable, match=msa.ENV):
        msa.require_pwa(BF16, 64, 128, 8, 32, grad=False, dropout=False)


def test_the_modules_build_under_the_option_and_keep_their_own_primitives():
    """The option names an MSA payload, not a LayerNorm one, so the primitives take the engine's auto choice."""
    opm = OuterProductMean(64, 128, 32, implementation=ImplementationType.ANTHROPIC)
    pwa = MSAPairWeightedAveraging(64, 128, 8, 32, implementation=ImplementationType.ANTHROPIC)
    assert opm.implementation is ImplementationType.ANTHROPIC
    assert pwa.implementation is ImplementationType.ANTHROPIC


def test_a_cpu_call_under_the_option_refuses_rather_than_run():
    opm = OuterProductMean(64, 128, 32, implementation=ImplementationType.ANTHROPIC)
    with pytest.raises(msa.PayloadUnavailable), torch.inference_mode():
        opm(BF16, torch.ones(1, 8, 16, dtype=torch.bool))
