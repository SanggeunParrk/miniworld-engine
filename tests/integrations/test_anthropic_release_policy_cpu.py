"""Release routing and conditioning broadcasts must retain their original contracts."""
from collections import namedtuple
from types import SimpleNamespace

import pytest
import torch

from miniworld_engine.integrations import anthropic as A
from miniworld_engine.integrations.anthropic_modules import _periodic_rows


@pytest.mark.parametrize(("source", "target", "period"), [
    ((1, 1, 5, 7), (3, 2, 5, 7), 5),
    ((1, 2, 5, 7), (3, 2, 5, 7), 10),
    ((3, 1, 5, 7), (3, 2, 5, 7), 30),
    ((1, 1, 1, 7), (3, 2, 5, 7), 1),
    ((5, 7), (3, 2, 5, 7), 5),
    ((3, 2, 5, 7), (3, 2, 5, 7), 30),
])
def test_periodic_conditioning_matches_broadcast(source, target, period):
    x = torch.randn(source)
    rows, actual = _periodic_rows(x, target)
    assert actual == period
    want = x.expand(target).reshape(-1, target[-1])
    torch.testing.assert_close(rows.repeat(want.shape[0] // period, 1), want, rtol=0, atol=0)


@pytest.mark.parametrize("registered", [True, False])
@pytest.mark.parametrize("timing", ["eager", "graph"])
def test_module_calls_release_selector_and_preserves_refusal(monkeypatch, registered, timing):
    calls = []
    selection_type = namedtuple("Selection", "row capture_safe")
    selection = selection_type("fpf_apb" if registered else "apb_attn", True)

    def select(*args, **kwargs):
        calls.append((args, kwargs))
        return selection

    def serve(*args, **kwargs):
        assert kwargs["selection"] is selection
        raise RuntimeError("named upstream refusal")

    face = SimpleNamespace(cell_word=lambda kind, **kw: "dit_h16d48" if registered else None,
                           select=select, pair_bias_attention=serve)
    monkeypatch.setattr(A, "provider", lambda name: face)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (8, 0))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    q = torch.empty(2, 128, 16, 48)
    previous = A._TIMING.get()
    with torch.no_grad(), A.execution_policy(timing=timing), pytest.raises(RuntimeError, match="named upstream refusal"):
        A.module_pair_bias_attention(q, q, q, None)
    assert A._TIMING.get() == previous
    args, kw = calls[0]
    assert args[2] == ("dit_h16d48" if registered else None)
    assert kw["word"] == ("fast" if registered else "apb_attn")
    assert kw["timing"] == timing
    assert kw["capture"] == (timing == "graph")


def test_triangle_default_uses_release_tier_and_keeps_explicit_rows():
    from miniworld_engine.modules import TriangleAttention
    assert TriangleAttention(128).anthropic_row == "block:fast"
    assert TriangleAttention(128, use_qk_norm=True).anthropic_row == "fast"
    assert TriangleAttention(128, anthropic_row="k2b").anthropic_row == "k2b"


@pytest.mark.parametrize("outcome", ["pass", "mismatch", "empty"])
def test_rebuilt_trimul_requires_original_vector_outputs(monkeypatch, outcome):
    from miniworld_engine.integrations import anthropic_trimul as T
    calls = []
    report = {"passed": 31 if outcome != "empty" else 0,
              "failed": ["wrong bytes"] if outcome == "mismatch" else []}

    def replay(**kwargs):
        calls.append(kwargs)
        return report

    monkeypatch.setattr(T, "_CHECKED", set())
    monkeypatch.setattr(T, "_BYTE_CHECKS", {})
    monkeypatch.setattr(T, "_load", lambda: {"face": SimpleNamespace(check=lambda **kw: None)})
    monkeypatch.setattr(T.importlib, "import_module", lambda name: SimpleNamespace(replay=replay))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (8, 0))
    if outcome == "pass":
        T._checked(torch.device("cuda:0"))
        T._checked(torch.device("cuda:0"))
        assert T._BYTE_CHECKS[0] is report
    else:
        with pytest.raises(T.PayloadUnavailable, match="upstream byte vectors"):
            T._checked(torch.device("cuda:0"))
        assert not T._CHECKED
    assert len(calls) == 1
    assert calls[0]["ignore_build"] is True
    assert calls[0]["raise_on_fail"] is True
