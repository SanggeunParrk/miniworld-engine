"""A precompile round that runs late keeps the work that finished, and says which configs it was.

The round used to be `map_async(...).get(timeout=budget)`: wait for every chunk, and on the
timeout raise -- taking the chunks that had already finished with it. Measured on the A6000
rebuild, one unit of `adaln_gemm_gate_triton` at token L=384 D=384:

    round 1: 2417 configs (463 already compiled) on 14 workers
             -> 0 compiled, 0 failed, 11827s at 0% occupancy

Three and a quarter hours of compiles by fourteen workers, reported as nothing, because the round
ran 56 minutes past a budget of 8,460 s. Nothing was marked settled, so the serial pass re-checked
all 1,954 configs one at a time -- and the unit sat at 0% GPU for thirteen hours doing it. Four
units were in that state at once.

The budget was the inner guard's OWN worst case:

    _compile_chunk bounds every config at _COMPILE_BUDGET_S (60 s) with its own SIGKILL
        worst case for the round = 1954 configs x 60 s / 14 workers = 8,374 s
        the outer budget         = 60 x (1954 // 14 + 2)            = 8,460 s

Eighty-six seconds of slack, two configs' worth. So the outer timeout could only fire when the
inner guard was working as designed -- a round with many register-spill configs, each legitimately
killed at 60 s -- and never on the failure it was written for.

Two properties below. Neither is about the deadline being right; a deadline can always be wrong.
They are about what a wrong one costs.
"""
from __future__ import annotations

import inspect

from miniworld_engine.autotune import capture as X


def test_a_finished_chunk_is_kept_even_if_the_round_runs_late() -> None:
    """The round collects chunks as they finish. `map_async(...).get(timeout=)` cannot: it is one
    call that either returns everything or raises."""
    src = inspect.getsource(X)
    assert "imap_unordered(_worker_compile_keyed" in src, (
        "the round waits for all chunks again -- a late round will discard the ones that finished")
    assert "map_async(_worker_compile" not in src


def test_a_kept_result_names_the_config_it_is_about() -> None:
    """Out-of-order collection needs self-labelled rows. Zipping results against a `sent` list in
    chunk order only works when EVERY chunk comes back, which is the case a partial harvest is
    not."""
    src = inspect.getsource(X._worker_compile_keyed)
    assert "for pl, _ in chunk" in src, "the keyed worker no longer takes (payload, key) pairs"
    assert "zip(keys" in src
    settle = inspect.getsource(X)
    assert "_mark_settled((d[0], bool(d[1])) for d in done if d)" in settle, (
        "settled configs are being derived by position again")


def test_the_deadline_is_not_the_inner_guard_s_own_worst_case() -> None:
    """`_compile_chunk` already bounds a round; the outer deadline is for a worker that dies
    without answering. Set at the worst case it fires on healthy rounds instead."""
    assert X._PRECOMPILE_DEADLINE_X > 1, (
        "the outer deadline is back at the inner guard's worst case, where it fires exactly when "
        "the inner guard is doing its job")


def test_every_config_is_still_bounded_on_its_own() -> None:
    """The property that makes the outer deadline a last resort rather than a schedule: a stalled
    config is SIGKILLed, recorded failed, and the untouched remainder of its chunk is retried."""
    src = inspect.getsource(X._compile_chunk)
    assert "SIGKILL" in src
    assert "_COMPILE_BUDGET_S" in src
    assert "_compile_chunk(rest)" in src, (
        "a killed config now condemns the rest of its chunk, which is what the outer timeout was "
        "insuring against -- put one back before relying on it")
