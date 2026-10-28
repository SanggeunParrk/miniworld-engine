"""A weight-pack entry a capture under ``static_weights()`` read is never evicted by a store's count limit: the replay reads its memory.

Evicting it freed memory a captured graph still read (the Transition's 64-entry ``Ws^T`` cache faulted Boltz-2's trunk graph on the next
fold); ``param_forms`` keeps an entry while its parameters live and ``lookup`` pins what a static capture touched."""

import pytest
import torch

from miniworld_engine.kernels import _capture

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


def test_a_static_capture_s_entry_survives_the_count_limit():
    store: dict = {}
    w = torch.randn(64, 64, device="cuda")
    x = torch.randn(64, 64, device="cuda")
    with torch.no_grad(), _capture.static_weights():
        y = x @ _capture.lookup(store, "w", lambda: w * 2, limit=2)        # eager warmup: fills the entry, creates the cuBLAS handle
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(stream), torch.cuda.graph(graph):
            y = x @ _capture.lookup(store, "w", lambda: w * 2, limit=2)  # the capture reads it
        torch.cuda.current_stream().wait_stream(stream)
    entry = store[(None, "w")]
    for i in range(10):                                                # eager traffic past the limit
        _capture.lookup(store, f"other{i}", lambda: torch.zeros(1, device="cuda"), limit=2)
    assert store[(None, "w")] is entry
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(y, x @ (w * 2))


def test_param_forms_keeps_a_live_parameter_s_form_and_drops_a_dead_one():
    _capture._FORMS.clear()
    keep = [torch.randn(8, 8, device="cuda") for _ in range(100)]
    forms = [_capture.param_forms("t", (p,), lambda p=p: p.t().contiguous()) for p in keep]
    assert all(_capture.param_forms("t", (p,), lambda: None) is f for p, f in zip(keep, forms, strict=True))
    dead = torch.randn(8, 8, device="cuda")
    _capture.param_forms("t", (dead,), lambda: dead.t().contiguous())
    del dead
    _capture.param_forms("t", (torch.randn(8, 8, device="cuda"),), lambda: None)  # a miss prunes the dead entry
    assert len(_capture._FORMS) == 101
