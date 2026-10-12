"""The weight-pack cache of the SWA atom DiT sm_100a block (``kernels/swa_dit/cuda/sm100._cached``).

It serves the packed forms of a block's weights (``torch.cat`` of the qkvg weights, transposes) while the very same tensor objects are
alive and unmodified. These tests pin the three ways a stale pack used to be served or recorded: a new tensor at a freed one's
address, an in-place weight update, and a CUDA-graph capture. The pack logic needs no GPU; the capture state is faked except in the
last test.
"""

from __future__ import annotations

import gc

import pytest
import torch

from miniworld_engine.kernels.swa_dit.cuda import sm100


@pytest.fixture(autouse=True)
def _fresh_cache(monkeypatch, request):
    sm100._PACKS.clear()
    if request.node.get_closest_marker("gpu") is None:  # the graph test below needs the real capture state
        monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    yield
    sm100._PACKS.clear()


def _counting(fn):
    calls = []

    def wrapped(*ts):
        calls.append(1)
        return fn(*ts)

    wrapped.__name__ = fn.__name__
    return wrapped, calls


def test_unchanged_weights_are_packed_once():
    w = torch.randn(8, 4)
    pack, calls = _counting(sm100._t)
    first = sm100._cached(pack, w)
    assert sm100._cached(pack, w) is first
    assert len(calls) == 1


def test_an_in_place_update_refreshes_the_pack():
    w = torch.randn(8, 4)
    pack, calls = _counting(sm100._t)
    before = sm100._cached(pack, w)
    w.add_(1.0)
    after = sm100._cached(pack, w)
    assert len(calls) == 2
    assert torch.equal(after, w.t())
    assert not torch.equal(after, before)


def test_a_new_tensor_object_over_changed_data_is_not_served_the_old_pack():
    """What a freed address that the allocator hands to a new weight looks like to a key made of (data_ptr, version, shape): the same
    pointer, the same version, other numbers. detach() shares storage and version counter; the data change bypasses the version."""
    a = torch.randn(8, 4)
    pack, calls = _counting(sm100._t)
    sm100._cached(pack, a)
    a.data.mul_(2.0)
    b = a.detach()
    assert b.data_ptr() == a.data_ptr()
    assert b._version == a._version
    assert torch.equal(sm100._cached(pack, b), b.t())
    assert len(calls) == 2


def test_a_different_tensor_with_the_same_key_is_not_served_the_old_pack(monkeypatch):
    """The old key was (data_ptr, version, shape): a new weight the allocator placed at a freed one's address -- a fresh bf16 cast,
    the next test's weights -- got the old tensor's packed copy. Every tensor is forced onto one key here to make that collision
    deterministic; the cache must then notice that the cached entry belongs to another object."""
    monkeypatch.setattr(sm100, "id", lambda _t: 0, raising=False)
    a, b = torch.randn(8, 4), torch.randn(8, 4)
    pack, calls = _counting(sm100._t)
    pack_a = sm100._cached(pack, a)
    pack_b = sm100._cached(pack, b)
    assert len(calls) == 2
    assert torch.equal(pack_a, a.t())
    assert torch.equal(pack_b, b.t())


def test_a_freed_weight_is_not_resurrected():
    pack, calls = _counting(sm100._t)
    w = torch.randn(8, 4)
    sm100._cached(pack, w)
    del w
    gc.collect()
    sm100._cached(pack, torch.randn(8, 4))
    assert len(calls) == 2


def test_inference_tensors_are_never_cached():
    """They carry no version counter, so nothing could invalidate an entry."""
    pack, calls = _counting(sm100._t)
    with torch.inference_mode():
        w = torch.randn(8, 4)
    first, second = sm100._cached(pack, w), sm100._cached(pack, w)
    assert len(calls) == 2
    assert torch.equal(first, second)
    assert not sm100._PACKS


def test_nothing_is_cached_while_a_graph_is_captured(monkeypatch):
    """A hit during capture would record no pack kernel, and every replay would keep reading the packed weights of the capture-time
    step. Inside a capture the pack always runs, and nothing is stored for later eager calls."""
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    w = torch.randn(8, 4)
    pack, calls = _counting(sm100._t)
    sm100._cached(pack, w)
    sm100._cached(pack, w)
    assert len(calls) == 2
    assert not sm100._PACKS


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph capture")
def test_a_graph_recomputes_the_pack_on_every_replay():
    w = torch.randn(64, 32, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        sm100._cached(sm100._t, w)                         # warm-up: an eager entry exists
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        out = sm100._cached(sm100._t, w)
    w.mul_(2.0)                                            # what an optimizer step does between replays
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, w.t())
