"""A named upstream block must never time pair_fused's fallback as that row."""
from collections import namedtuple
from types import SimpleNamespace

import pytest
import torch

from miniworld_engine.integrations import anthropic
from miniworld_engine.modules import TriangleAttention


@pytest.mark.parametrize("failure", ["resolve", "serve", None])
def test_block_propagates_named_refusal(monkeypatch, failure):
    class Refusal(RuntimeError):
        pass

    selection = namedtuple("Selection", "row")("triattn_native")
    calls = []

    def resolve(*args, **kwargs):
        if failure == "resolve":
            raise Refusal("native loadcheck failed")
        return selection

    def serve(q, k, v, bias, mask, scale, **kwargs):
        calls.append(kwargs)
        if failure == "serve":
            raise Refusal("native shape refused")
        return torch.ones_like(q)

    def block(pair, weights, mask, *, core, **kwargs):
        assert callable(core), "tier:<word> would silently substitute flash on refusal"
        return core(pair, pair, pair, None, mask, 1.0)

    pf = SimpleNamespace(pack_triattn_weights=lambda **kw: None,
                         _plan_triattn=lambda *a, **kw: {"impl": "fpf"},
                         resolve_tier_core=resolve, _call_class=lambda q, k: ((8, 0), "bf16", 32, 4, 2),
                         tri_attn_block=block)
    monkeypatch.setattr(anthropic, "_ROOT", "configured")
    monkeypatch.setattr(anthropic, "import_module", lambda name: pf)
    monkeypatch.setattr(anthropic, "provider", lambda name: SimpleNamespace(FWD="fwd", triangle_attention=serve))
    module = TriangleAttention(128, 4, implementation="pytorch", anthropic_row="block:triattn_native").eval()
    pair = torch.randn(1, 2, 2, 128)
    with torch.no_grad():
        if failure:
            with pytest.raises(Refusal, match="native"):
                anthropic.module_triangle_attention(module, pair, None)
        else:
            out = anthropic.module_triangle_attention(module, pair, None)
            torch.testing.assert_close(out, pair + 1)
            assert module.anthropic_selection["core"]["row"] == "triattn_native"
            assert calls[0]["selection"] is selection
            assert calls[0]["word"] == "triattn_native"
