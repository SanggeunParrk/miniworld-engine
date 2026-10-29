"""``_capture.lookup_inputs`` across trajectories that a sampler captures one by one (FoldForge: an eager warm-up under
``static_inputs()``, then the capture reads the warm-up's entries and pins them). The pinned entries of earlier trajectories must not
crowd out the next trajectory's: they used to fill the count limit, so the new entries were evicted as they were made and the capture
recorded their rebuild into every replay (a Protenix-v2 sampler's pair bias, +41 % at 512 tokens from its second call). An entry whose
input tensor is gone leaves the store and the pin set."""
import gc
import weakref

import pytest
import torch

from miniworld_engine.kernels import _capture


def _trajectory(monkeypatch, store, pair, blocks, limit):
    """warm-up (eager, builds) then the capture (must hit every block); returns how many tables the warm-up built."""
    built = []
    monkeypatch.setattr(_capture, "_graph_capture_id", lambda: None)
    for b in range(blocks):
        _capture.lookup_inputs(store, (pair.data_ptr(), b), lambda: built.append(1) or (object(), weakref.ref(pair)), limit=limit,
                               valid=lambda e: e[1]() is pair, alive=lambda e: e[1]() is not None)
    monkeypatch.setattr(_capture, "_graph_capture_id", lambda: 7)
    for b in range(blocks):
        _capture.lookup_inputs(store, (pair.data_ptr(), b), lambda: pytest.fail("the capture rebuilt a table its warm-up made"),
                               limit=limit, valid=lambda e: e[1]() is pair, alive=lambda e: e[1]() is not None)
    return len(built)


def test_pinned_tables_of_live_graphs_do_not_evict_the_next_trajectory(monkeypatch):
    store, pairs = {}, []
    with _capture.static_inputs():
        for _ in range(4):                   # four trajectories whose graphs (and pairs) all stay alive: 4 x 3 pinned > limit 4
            pairs.append(torch.zeros(8))
            assert _trajectory(monkeypatch, store, pairs[-1], blocks=3, limit=4) == 3
    assert len(store) == 12


def test_a_table_goes_with_its_input(monkeypatch):
    store = {}
    pair = torch.zeros(8)
    with _capture.static_inputs():
        _trajectory(monkeypatch, store, pair, blocks=3, limit=4)
    pinned = _capture._PINNED[id(store)]
    assert len(store) == 3 and len(pinned) == 3
    del pair
    gc.collect()
    with _capture.static_inputs():
        _trajectory(monkeypatch, store, torch.zeros(8), blocks=1, limit=4)
    assert len(store) == 1 and len(pinned) == 1, "the dead trajectory's tables (and pins) stayed"
