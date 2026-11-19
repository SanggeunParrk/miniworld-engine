"""``GatedInferenceCore.runs`` bookkeeping (``BoundedRuns``) without a card: an LRU bound by count and by the scratch bytes the runs own,
a hit returns the cached run without calling ``make``, and a cached run does not own its caller's tensors."""

from __future__ import annotations

import gc
import weakref

import torch

from miniworld_engine.kernels.augmented_attention.cuda.sm100 import BoundedRuns


def _run(nbytes: int = 0):
    def run(x=None):
        return None

    run.nbytes = nbytes  # type: ignore[attr-defined]
    return run


def test_hit_reuses_run_and_never_remakes() -> None:
    runs, made = BoundedRuns(), []
    for _ in range(5):
        runs.bind(("k", 1), lambda: made.append(1) or _run())
    assert made == [1] and len(runs) == 1


def test_fresh_keys_every_step_stay_bounded_and_lru_order() -> None:
    runs = BoundedRuns()
    for step in range(200):  # 200 'steps', each with a new address
        runs.bind(("a", step), _run)
        runs.bind(("static",), _run)  # touched every step: never the oldest
    assert len(runs) <= runs.cap
    assert ("static",) in runs and ("a", 199) in runs and ("a", 0) not in runs


def test_scratch_budget_evicts_oldest_but_keeps_newest() -> None:
    runs = BoundedRuns()
    big = runs.scratch_budget // 2 + 1
    for step in range(10):
        runs.bind(step, lambda: _run(big))
    assert len(runs) == 1 and 9 in runs
    for step in range(10):
        runs.bind(("s", step), lambda: _run(1))
    assert len(runs) <= runs.cap


def test_run_does_not_pin_a_tensor_it_was_made_from() -> None:
    runs = BoundedRuns()
    t = torch.zeros(4)
    ref = weakref.ref(t)
    runs.bind("x", lambda: _run())  # make() closes over nothing of the caller
    del t
    gc.collect()
    assert ref() is None and len(runs) == 1
