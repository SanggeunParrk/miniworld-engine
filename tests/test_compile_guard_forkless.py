"""The compile guard must not re-prove what the precompile pool already proved.

Measured on one unit (cProfile, 1944 configs, idle A6000): `_fork_compile` was 348 s of the
366 s the unit took -- posix.read 186 s blocking on the child's pipe, select.poll 79 s,
time.sleep 47 s across 43,936 polls, posix.fork 28 s -- to guard a `triton.compile` whose own
cumulative cost was 6.8 s, because every one of those compiles was a warm on-disk cache hit the
pool had just produced. The guard exists for compile monsters; a config the pool compiled under
the same SIGKILL budget is by construction not one.

The two directions both matter. Skipping the fork for a settled config is the speedup; still
forking for an unsettled one is what keeps the guard a guard.
"""
from __future__ import annotations

import pytest

from miniworld_engine.autotune import capture


@pytest.fixture(autouse=True)
def _clean():
    for s in (capture._COMPILE_OK, capture._COMPILE_BAD, capture._COMPILED):
        s.clear()
    capture._COMPILED_FILE.clear()
    capture._CURRENT_CFG.clear()
    for k, v in list(capture._COMPILE_T.items()):
        capture._COMPILE_T[k] = type(v)()
    yield
    for s in (capture._COMPILE_OK, capture._COMPILE_BAD, capture._COMPILED):
        s.clear()
    capture._COMPILED_FILE.clear()
    capture._CURRENT_CFG.clear()


class _Cfg:
    def __init__(self, bm, warps=4, stages=2):
        self.kwargs = {"BLOCK_M1": bm}
        self.num_warps = warps
        self.num_stages = stages


def _arm(cfg):
    """Put the guard in the state a bench iteration puts it in for `cfg`."""
    capture._CURRENT_CFG.clear()
    capture._CURRENT_CFG.update(cfg.kwargs)
    capture._CURRENT_CFG["num_warps"] = cfg.num_warps
    capture._CURRENT_CFG["num_stages"] = cfg.num_stages


def test_a_pool_settled_config_is_recorded_with_its_outcome(tmp_path):
    capture._COMPILED_FILE.append(tmp_path / "u.compiled")
    good, bad = _Cfg(32), _Cfg(64)
    capture._mark_outcome("k", [(capture._cfg_sig(good), True),
                                (capture._cfg_sig(bad), False)])
    assert f"k\t{capture._cfg_sig(good)}" in capture._COMPILE_OK
    assert f"k\t{capture._cfg_sig(bad)}" in capture._COMPILE_BAD


def test_the_outcome_survives_a_restart(tmp_path):
    """A unit that dies to a node failure must not go back to forking 1944 children."""
    shard = tmp_path / "u.json"
    capture.load_probe_state(shard)
    cfg = _Cfg(32)
    capture._mark_outcome("k", [(capture._cfg_sig(cfg), True)])
    capture._COMPILE_OK.clear()
    capture._COMPILED.clear()
    capture._COMPILED_FILE.clear()
    capture.load_probe_state(shard)                     # the restart
    assert f"k\t{capture._cfg_sig(cfg)}" in capture._COMPILE_OK
    assert capture._cfg_sig(cfg) in capture._COMPILED, "the ROUND skip must still see it"


def test_a_legacy_bare_line_does_not_claim_an_outcome(tmp_path):
    """`.compiled` files written before this change say only 'settled'. Reading one as 'compiled
    fine' would skip the fork for a config that in fact failed."""
    shard = tmp_path / "u.json"
    (tmp_path / "u.compiled").write_text("BLOCK_M1=32,num_warps=4,num_stages=2\n")
    capture.load_probe_state(shard)
    assert capture._COMPILED, "legacy line still suppresses recompiling in the round"
    assert not capture._COMPILE_OK
    assert not capture._COMPILE_BAD


def test_load_is_idempotent_across_repeated_appends(tmp_path):
    """`_mark_outcome` appends; a resumed unit re-marking the same configs must not grow the file
    without bound -- 527 units x 1944 configs is where that turns into real I/O."""
    shard = tmp_path / "u.json"
    capture.load_probe_state(shard)
    cfg = _Cfg(32)
    for _ in range(5):
        capture._mark_outcome("k", [(capture._cfg_sig(cfg), True)])
    assert len((tmp_path / "u.compiled").read_text().strip().splitlines()) == 1
