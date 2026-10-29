"""``kernels._capture`` under ``torch.compile``: a compiled module that consults the weight-form / hoisted-table caches must trace
through a no-grad call (a recycle) and then a grad call (the step) as ONE graph each, with no graph break, and return what eager returns.

The cache logic (storage addresses, version counters, weak references, the driver's capture state) is not graph material: Dynamo
raised on it (``data_ptr`` is data-dependent; on the B200 MiniWorld model, ``InternalTorchDynamoError: NameError: name 'torch' is not
defined`` on the weak-reference comprehension of ``param_forms``, reached from the Transition's no-grad path at the first no-grad
recycle). Traced, every entry point returns ``build()``: the forms are graph ops and the stores are not touched. CPU only."""
from __future__ import annotations

import pytest
import torch
import torch._dynamo as dynamo

from miniworld_engine.kernels import _capture


class _Block(torch.nn.Module):
    """A Transition-shaped block whose inference branch reads its weights through every cache entry point, as the engine's no-grad
    paths do (``param_forms`` for the weight casts, ``lookup`` for a pack, ``hoisted_table`` / ``lookup_inputs`` for a per-input table)."""

    def __init__(self, d=16, h=32):
        super().__init__()
        self.ln = torch.nn.LayerNorm(d)
        self.wa = torch.nn.Parameter(torch.randn(h, d) * d ** -0.5)
        self.ws = torch.nn.Parameter(torch.randn(d, h) * h ** -0.5)
        self.store: dict = {}
        self.tables: dict = {}

    def forward(self, x, cond):
        if torch.is_grad_enabled() and any(t.requires_grad for t in (x, self.wa, self.ws)):
            wa, ws, g = self.wa.to(torch.bfloat16), self.ws.to(torch.bfloat16), self.ln.weight.float()
            gate = torch.sigmoid(cond.float().mean(-1, keepdim=True))
        else:
            params = (self.ln.weight, self.wa, self.ws)
            g, wa, ws = _capture.param_forms(("capture_compile_test",), params,
                                             lambda: (self.ln.weight.float(), self.wa.to(torch.bfloat16), self.ws.to(torch.bfloat16)))
            wa = _capture.lookup(self.store, ("pack", self.wa.data_ptr(), self.wa._version), lambda: wa.contiguous())
            gate = _capture.hoisted_table(self.tables, cond, ("gate",), lambda: torch.sigmoid(cond.float().mean(-1, keepdim=True)))
            gate = _capture.lookup_inputs(self.tables, ("gate2", cond.data_ptr()), lambda: gate)
        xn = torch.nn.functional.layer_norm(x.float(), x.shape[-1:], g, self.ln.bias.float())
        h = torch.nn.functional.silu(xn.to(torch.bfloat16) @ wa.t())
        return x + gate * (h @ ws.t()).float()




def test_no_grad_then_grad_call_compile_fullgraph():
    torch.manual_seed(0)
    m = _Block()
    x, cond = torch.randn(2, 8, 16), torch.randn(2, 8, 4)
    with torch.no_grad():
        want_ng = m(x, cond)
    want_g = m(x, cond)
    eager_entries = (len(_capture._FORMS), len(m.store), len(m.tables))

    dynamo.reset()
    graphs = []

    def backend(gm, example_inputs):
        graphs.append(gm)
        return gm.forward

    cm = torch.compile(m, backend=backend, fullgraph=True, dynamic=False)
    _capture._FORMS.clear()
    m.store.clear()
    m.tables.clear()
    with torch.no_grad():                                  # the recycles: the inference branch, the caches consulted
        got_ng = cm(x, cond)
    got_g = cm(x, cond)                                    # the step: the training branch
    got_g.float().pow(2).mean().backward()
    assert m.wa.grad is not None and m.ws.grad is not None
    torch.testing.assert_close(got_ng, want_ng)
    torch.testing.assert_close(got_g, want_g)
    assert len(graphs) == 2, f"expected one graph per grad mode, got {len(graphs)}"
    # traced, nothing was cached: the forms are graph ops (eager calls did fill the stores)
    assert eager_entries[0] >= 1 and eager_entries[1] == 1 and eager_entries[2] >= 1
    assert not _capture._FORMS and not m.store and not m.tables
    # a weight update between compiled no-grad calls is seen (no stale cache in the graph)
    with torch.no_grad():
        m.wa.add_(0.1)
        again = cm(x, cond)
        torch.testing.assert_close(again, m.__class__.forward(m, x, cond))


def test_no_grad_path_has_no_graph_break():
    torch.manual_seed(1)
    m = _Block()
    x, cond = torch.randn(2, 8, 16), torch.randn(2, 8, 4)

    def recycle(x, cond):
        with torch.no_grad():
            return m(x, cond)

    dynamo.reset()
    explanation = dynamo.explain(recycle)(x, cond)
    assert explanation.graph_break_count == 0, [str(getattr(r, "reason", r))[:200] for r in explanation.break_reasons]
    assert sum(len(g.graph.nodes) for g in explanation.graphs) > 5


@pytest.mark.parametrize("name", ["param_forms", "lookup", "lookup_inputs", "hoisted_table"])
def test_each_entry_point_returns_build_when_traced(name):
    w = torch.nn.Parameter(torch.randn(4, 4))
    store: dict = {}
    calls = {
        "param_forms": lambda b: _capture.param_forms(("t", name), (w,), b),
        "lookup": lambda b: _capture.lookup(store, ("k", w.data_ptr(), w._version), b),
        "lookup_inputs": lambda b: _capture.lookup_inputs(store, ("k", w.data_ptr()), b),
        "hoisted_table": lambda b: _capture.hoisted_table(store, w, ("k",), b),
    }

    def f(x):
        with torch.no_grad():
            return x @ calls[name](lambda: w.to(torch.bfloat16)).float()

    dynamo.reset()
    x = torch.randn(3, 4)
    got = torch.compile(f, backend="eager", fullgraph=True)(x)
    torch.testing.assert_close(got, x @ w.detach().to(torch.bfloat16).float())
    assert not store
