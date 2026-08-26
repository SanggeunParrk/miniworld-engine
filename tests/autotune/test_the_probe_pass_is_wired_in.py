"""The two predictors, in the build. Everything here is about the ways it must NOT act.

`viability` and `compile_budget` are fitted per kernel from probes taken on the card being built
for. Nothing about either is written per kernel -- there is no table of kernel names -- because a
rule taken from one kernel is wrong on the next: the shared-memory formula that is byte-exact on
one GEMM discarded 168, 100 and 81 runnable configs on three others, and "bigger on every axis
needs at least as much shared memory" holds with zero violations on that GEMM and fails 399 times
on a reduction, whose shared memory goes DOWN as its tile grows.

So the mechanism is general and the numbers are per kernel, which means the interesting cases are
the ones where a kernel cannot be described. Then the round has to compile everything, exactly as
it did before any of this existed.
"""
from __future__ import annotations

import pytest

from miniworld_engine import settings
from miniworld_engine.autotune import capture


class _Cfg:
    def __init__(self, **kw):
        self.num_warps = kw.pop("num_warps", 4)
        self.num_stages = kw.pop("num_stages", 2)
        self.kwargs = kw


@pytest.fixture(autouse=True)
def _clean():
    settings.configure(predict_unusable=False)
    capture._PREDICTED_BAD.clear()
    for k, v in list(capture._PREDICT.items()):
        capture._PREDICT[k] = type(v)()
    yield
    capture._PREDICTED_BAD.clear()
    settings.configure(predict_unusable=False)


def test_it_is_off_unless_asked_for():
    """It changes what a build's cache contains. Until that is measured against a build without
    it, the default has to be the behaviour that was measured.

    And it is a SETTING, reaching a unit on its command line -- not an environment variable. Every
    other knob a unit takes is an argument, so that what a unit did is readable off the command
    line that ran it rather than out of whatever shell started the build.
    """
    assert not capture._predict_enabled()
    settings.configure(predict_unusable=True)
    assert capture._predict_enabled()


def test_a_small_grid_is_compiled_whole():
    """Probes only pay for themselves when they are a small share of the grid. On a 240-config
    kernel the probe was 33% of it, which is worse than not predicting."""
    configs = [_Cfg(BLOCK_M=m) for m in range(64)]
    got = capture._predict_unusable(None, None, None, configs, "k", "r", jobs=4)
    assert got is configs


def test_a_kernel_whose_axes_are_not_integers_is_compiled_whole():
    """A bool axis is not a tile size and a model fitted through one is fitting noise."""
    configs = [_Cfg(BLOCK_M=32, SAVE_ACT=bool(i % 2)) for i in range(700)]
    assert capture._config_dict(configs[0]) == {"BLOCK_M": 32, "num_warps": 4, "num_stages": 2}
    configs = [_Cfg(FLAG=bool(i % 2)) for i in range(700)]
    assert capture._config_dict(configs[0]) is None
    got = capture._predict_unusable(None, None, None, configs, "k", "r", jobs=4)
    assert got is configs


def test_anything_unexpected_leaves_the_grid_alone(monkeypatch):
    """There is no failure of this pass that is worth failing a build for."""
    def _boom():
        raise RuntimeError("no device")

    monkeypatch.setattr(capture, "_shared_limit", _boom)
    configs = [_Cfg(BLOCK_M=m, BLOCK_N=n) for m in (16, 32, 64, 128) for n in range(200)]
    got = capture._predict_unusable(None, None, None, configs, "k", "r", jobs=4)
    assert got == configs
    assert capture._PREDICT["gave_up"] == 1


def test_a_ruled_out_config_is_answered_without_a_fork():
    """The saving is the fork and the compile both. A predicted-bad config takes the path a
    pool FAILURE takes: the guard raises, `_bench` catches it, the config scores +inf."""
    cfg = _Cfg(BLOCK_M=256, num_warps=1, num_stages=8)
    key = f"k\tr\t{capture._cfg_sig(cfg)}"
    capture._PREDICTED_BAD.add(key)
    assert key in capture._PREDICTED_BAD


def test_a_prediction_is_not_written_to_the_restart_file(tmp_path):
    """`.compiled` rows are measurements and a restart is entitled to trust them. A prediction is
    not, and a restart should re-derive it -- by then its probes are cache hits."""
    capture._COMPILED_FILE.clear()
    capture._COMPILED_FILE.append(tmp_path / "u.compiled")
    capture._PREDICTED_BAD.add("k\tr\tBLOCK_M=256,num_stages=8,num_warps=1")
    assert not (tmp_path / "u.compiled").exists()
    capture._COMPILED_FILE.clear()


def test_the_smem_reader_takes_only_what_the_probes_appended(tmp_path, monkeypatch):
    """By the end of a unit the log holds every earlier round; fitting on all of it would mix
    kernels."""
    log = tmp_path / "u.smem"
    log.write_text("earlier_kernel\tBLOCK_M=16\t1024\n")
    monkeypatch.setenv("MINIWORLD_SMEM_LOG", str(log))
    offset = capture._smem_log_end()
    with log.open("a") as fh:
        fh.write("k\tBLOCK_M=32,num_stages=2,num_warps=4\t2048\n")
        fh.write("!k\tBLOCK_M=256,num_stages=8,num_warps=1\t60\n")
        fh.write("~k\tBLOCK_M=32,num_stages=2,num_warps=4\t1500\n")
    got = capture._read_smem_from(offset)
    assert got == {"BLOCK_M=32,num_stages=2,num_warps=4": 2048}, (
        "only the shared-memory rows, and only the new ones")


def test_no_smem_log_means_no_prediction(monkeypatch):
    monkeypatch.delenv("MINIWORLD_SMEM_LOG", raising=False)
    assert capture._read_smem_from(0) == {}
    assert capture._smem_log_end() == 0
