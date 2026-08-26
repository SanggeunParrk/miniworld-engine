"""The precompile pool forks once per CHUNK; the 60 s budget must still be per CONFIG.

Measured on one unit (1944 configs, warm cache, idle A6000): a fork per config cost 1966
worker-seconds against 15 worker-seconds for the compiles themselves -- forking a worker holding
torch + triton + the kernel module is ~1 s of page-table work to guard 7.7 ms of compiling. The
guard cannot simply be dropped: make_llir and ptxas block in native code where no Python signal
reaches, so a register-spill monster can only be stopped by killing the process inside it, and a
config that compiles only because it was never given its own budget would be kept by a
pre-compiled build and dropped by a serial one -- the same inputs, two different caches.

So it is amortised instead. These tests pin the three properties that makes safe:
  1. a chunk reports per-config outcomes, not one verdict for all of them;
  2. a stall is charged to the config that stalled, not to the chunk;
  3. configs after the stall are retried, not condemned with it.
"""
from __future__ import annotations

import time

import pytest

from miniworld_engine.autotune import capture


def _payload(tag):
    """Shape-compatible with the real payload tuple; only the tag is read by the fakes."""
    return (tag, "fn", {}, {}, {}, ("cuda", 86, 32), {})


@pytest.fixture
def fast_budget(monkeypatch):
    monkeypatch.setattr(capture, "_COMPILE_BUDGET_S", 2)


def test_a_chunk_reports_one_outcome_per_config(monkeypatch):
    monkeypatch.setattr(capture, "_resolve_jit", lambda *a: None)
    monkeypatch.setattr(capture, "_compile_payload",
                        lambda p, pre=None: (_ for _ in ()).throw(RuntimeError())
                        if p[0] == "bad" else None)
    got = capture._worker_compile([_payload("ok1"), _payload("bad"), _payload("ok2")])
    assert [g[0] for g in got] == [True, False, True], (
        "one bad config must not fail the two good ones sharing its child")


def test_a_stall_is_charged_to_the_config_that_stalled(monkeypatch, fast_budget):
    """The whole point of the per-config budget: the monster is identified, not the chunk."""
    monkeypatch.setattr(capture, "_resolve_jit", lambda *a: None)

    def _compile(p, pre=None):
        if p[0] == "monster":
            time.sleep(3600)

    monkeypatch.setattr(capture, "_compile_payload", _compile)
    got = capture._worker_compile([_payload("a"), _payload("monster"), _payload("b")])
    assert [g[0] for g in got] == [True, False, True], (
        "the config after the monster was never attempted; it must be retried, not condemned")


def test_two_stalls_in_one_chunk_are_both_isolated(monkeypatch, fast_budget):
    monkeypatch.setattr(capture, "_resolve_jit", lambda *a: None)

    def _compile(p, pre=None):
        if p[0].startswith("monster"):
            time.sleep(3600)

    monkeypatch.setattr(capture, "_compile_payload", _compile)
    got = capture._worker_compile(
        [_payload("a"), _payload("monster1"), _payload("b"), _payload("monster2"), _payload("c")])
    assert [g[0] for g in got] == [True, False, True, False, True]


def test_the_budget_is_per_config_not_per_chunk(monkeypatch, fast_budget):
    """A chunk of slow-but-finishing configs must not be killed for taking chunk_len x budget."""
    monkeypatch.setattr(capture, "_resolve_jit", lambda *a: None)
    # each takes 1.2 s, under the 2 s budget; 4 of them is 4.8 s, well over it
    monkeypatch.setattr(capture, "_compile_payload", lambda p, pre=None: time.sleep(1.2))
    got = capture._worker_compile([_payload(str(i)) for i in range(4)])
    assert [g[0] for g in got] == [True] * 4, "the deadline must reset on each config's progress"


def test_an_empty_chunk_is_not_a_fork():
    assert capture._worker_compile([]) == []


def test_each_config_is_timed_on_its_own(monkeypatch):
    """The tail is the whole story of a build's compile cost. The A6000 rebuild spent 54% of its
    429 compile CPU-hours on the 1.6% of configs that ran the full budget and were killed, and
    nothing said so, because the only number kept per config was the chunk mean -- 0.83 s."""
    monkeypatch.setattr(capture, "_resolve_jit", lambda *a: None)
    monkeypatch.setattr(capture, "_compile_payload",
                        lambda p, pre=None: time.sleep(0.6) if p[0] == "slow" else None)
    got = capture._worker_compile([_payload("fast"), _payload("slow"), _payload("fast2")])
    slow = got[1][2]
    assert slow > 0.4, f"the slow config was charged {slow:.2f}s, not its own time"
    assert sum(g[2] for g in got) > 0.5, "the per-config times must still sum to the chunk's cost"


def test_a_killed_config_is_charged_the_budget(monkeypatch, fast_budget):
    """It really did cost that; a build that reports it as free cannot choose a budget."""
    monkeypatch.setattr(capture, "_resolve_jit", lambda *a: None)
    monkeypatch.setattr(capture, "_compile_payload",
                        lambda p, pre=None: time.sleep(3600) if p[0] == "monster" else None)
    got = capture._worker_compile([_payload("a"), _payload("monster"), _payload("b")])
    assert got[1][2] >= capture._COMPILE_BUDGET_S


def test_a_killed_config_is_named_not_just_counted(monkeypatch, tmp_path, fast_budget):
    """13,875 of the A6000 rebuild's 869,844 configs were killed by the budget and they took 54%
    of its compile CPU. Deciding whether that is predictable -- the way shared-memory overflow
    now is -- needs to know WHICH configs they were, and a count says nothing."""
    from miniworld_engine.autotune import smem_log
    log = tmp_path / "u.smem"
    monkeypatch.setenv("MINIWORLD_SMEM_LOG", str(log))
    monkeypatch.setattr(capture, "_resolve_jit", lambda *a: None)
    monkeypatch.setattr(capture, "_compile_payload",
                        lambda p, pre=None: time.sleep(3600) if p[0] == "monster" else None)

    def _payload_named(tag, sig):
        return (tag, "the_kernel", {}, {}, {}, ("cuda", 86, 32), {}, sig)

    capture._worker_compile([_payload_named("a", "BLOCK_M=32"),
                             _payload_named("monster", "BLOCK_M=256,num_stages=8")])
    assert smem_log.killed(tmp_path) == {"the_kernel": {"BLOCK_M=256,num_stages=8"}}
    assert smem_log.read(tmp_path) == {}, (
        "a killed config has no shared-memory reading; recording the budget as one would put a "
        "60 among values in the hundreds of thousands")
