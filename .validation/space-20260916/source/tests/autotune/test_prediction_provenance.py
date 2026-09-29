"""Prediction skips remain distinguishable when old and new build workers share rounds."""
from __future__ import annotations

import dataclasses
import json
import math
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import triton
from triton.runtime.autotuner import Autotuner

from miniworld_engine import settings
from miniworld_engine.autotune import cache, capture, round_cache

IDENTITY = ("GPU", "op", "source", "env", cache.KEY_SCHEME, 1, "float32|128")


def test_sidecar_preserves_numeric_format_and_distinguishes_exclusions(tmp_path):
    outcomes = {}
    with round_cache.transaction(str(tmp_path), IDENTITY, outcomes=outcomes) as data:
        data.update(winner=1.0, predicted=float("inf"), failed=float("inf"), unknown=float("inf"))
        outcomes.update(predicted="predicted_skip", failed="observed_failure")
    raw = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert all(isinstance(ms, (int, float)) for ms in raw.values())
    loaded = {}
    with round_cache.transaction(str(tmp_path), IDENTITY, outcomes=loaded) as data:
        reused, retry = round_cache.reusable_timings(data, loaded, predict=False)
        assert reused == {"winner": 1.0, "failed": float("inf")}
        assert retry == {"predicted", "unknown"}
        reused, retry = round_cache.reusable_timings(data, loaded, predict=True)
        assert reused == raw
        assert not retry


def test_old_writer_in_another_process_invalidates_sidecar_without_losing_finite(tmp_path):
    outcomes = {}
    with round_cache.transaction(str(tmp_path), IDENTITY, outcomes=outcomes) as data:
        data.update(winner=1.0, failed=float("inf"))
        outcomes["failed"] = "observed_failure"
    # The old API writes only the numeric mapping, under the same file lock.
    script = (
        "import json, sys\n"
        "from miniworld_engine.autotune.round_cache import transaction\n"
        "with transaction(sys.argv[1], tuple(json.loads(sys.argv[2]))) as data:\n"
        "    data['new_measurement'] = 0.5\n"
    )
    subprocess.run([sys.executable, "-c", script, str(tmp_path), json.dumps(IDENTITY)],
                   check=True, capture_output=True, text=True, timeout=30)
    loaded = {}
    with round_cache.transaction(str(tmp_path), IDENTITY, outcomes=loaded) as data:
        assert not loaded
        reused, retry = round_cache.reusable_timings(data, loaded, predict=False)
        assert reused == {"winner": 1.0, "new_measurement": 0.5}
        assert retry == {"failed"}


def test_failed_transaction_does_not_publish_timings_or_provenance(tmp_path):
    outcomes = {}

    def fail():
        with round_cache.transaction(str(tmp_path), IDENTITY, outcomes=outcomes) as data:
            data["candidate"] = float("inf")
            outcomes["candidate"] = "predicted_skip"
            raise RuntimeError("module failed")

    with pytest.raises(RuntimeError, match="module failed"):
        fail()
    assert not list(tmp_path.glob("*.json"))
    assert not list(tmp_path.glob("*.provenance"))


@pytest.fixture
def instrumented_round(monkeypatch, tmp_path):
    configs = [triton.Config({"BLOCK": n}) for n in (16, 32, 64, 128)]
    control = {"predict": True, "cache_hit": False, "prune_error": False, "fatal": ""}
    calls = []

    def bench(self, *args, config, **kwargs):
        block = config.kwargs["BLOCK"]
        calls.append(block)
        if block == 32 and control["fatal"]:
            raise RuntimeError(control["fatal"])
        if block == 32 and control["predict"]:
            raise capture._PredictedSkip("model exclusion")
        if block == 64:
            raise RuntimeError("observed compile failure")
        return [1.0 if block == 16 else 0.5] * 3

    def prune(self, kwargs):
        # Model the engine's cache wrapper around an original shape/resource prune.
        bypass = cache._BYPASS_CACHED_SUBSET.get()
        if bypass and control["prune_error"]:
            raise RuntimeError("base prune failed")
        return self.configs[:1] if settings.current().fill_gaps and control["cache_hit"] and not bypass \
            else self.configs[:3]  # BLOCK=128 is ineligible, even with cache pruning disabled.

    def run(self, *args, **kwargs):
        self.nargs = {}
        try:
            return {capture._sig(c): self._bench(config=c, **kwargs)
                    for c in self.prune_configs(kwargs)}
        finally:
            self.nargs = None

    assert capture._orig_bench is None
    monkeypatch.setattr(Autotuner, "_bench", bench)
    monkeypatch.setattr(Autotuner, "prune_configs", prune)
    monkeypatch.setattr(Autotuner, "run", run)
    monkeypatch.setattr(capture, "_install_launch_probes", lambda: None)
    monkeypatch.setattr(capture, "_bench_lock_acquire", lambda: None)
    monkeypatch.setattr(capture, "_bench_lock_release", lambda: None)
    monkeypatch.setattr(capture, "_op_name", lambda t: "op")
    monkeypatch.setattr(capture, "gpu_key", lambda: "GPU")
    monkeypatch.setattr(capture, "op_identity", lambda t: "source")
    monkeypatch.setattr(capture, "_entry_key", lambda *a: "float32|128")
    monkeypatch.setattr(capture, "_entry_parts", lambda *a: ("float32", "128"))
    monkeypatch.setattr(capture, "_known_timings", lambda *a: {})
    monkeypatch.setattr(capture, "configs_to_bench", lambda op, gpu, cfgs, **kw: list(cfgs))
    monkeypatch.setattr(capture, "_predict_enabled", lambda: control["predict"])
    monkeypatch.setattr(capture, "_INCREMENTAL", True)
    monkeypatch.setattr(capture, "_ROUND_CACHE_DIR", str(tmp_path))
    for name in ("_CAPTURE", "_UNUSABLE", "_SKIPPED", "_REUSED_TIMINGS", "_ROUND_OUTCOMES",
                 "_ROUND_RETRY", "_ROUND", "_ROUND_LEFT", "_ROUND_ID", "_ROUND_FASTEST",
                 "_CURRENT", "_CURRENT_CFG"):
        monkeypatch.setattr(capture, name, {})
    monkeypatch.setattr(cache, "env_identity", lambda: "env")
    monkeypatch.setattr(cache, "build_rev", lambda op: 1)
    monkeypatch.setattr(settings, "_ACTIVE", dataclasses.replace(
        settings.current(), run_autotune=True, fill_gaps=True))
    tuner = object.__new__(Autotuner)
    tuner.configs = configs
    tuner.arg_names = []
    tuner.keys = []
    capture.install()
    try:
        yield SimpleNamespace(tuner=tuner, configs=configs, calls=calls, control=control,
                              directory=tmp_path)
    finally:
        capture.uninstall()


def test_predicted_candidate_is_not_searched_and_no_predict_verifies_only_it(
        instrumented_round, monkeypatch):
    test = instrumented_round
    a, b, c, _ = test.configs
    test.tuner.run()
    assert test.calls == [16, 32, 64]
    searched = capture._CAPTURE["op"]["searched"][("float32", "128")]
    assert searched == {capture._sig(a), capture._sig(c)}
    outcomes = {}
    with round_cache.transaction(str(test.directory), IDENTITY, outcomes=outcomes) as data:
        assert data[repr(capture._sig(a))] == 1.0
        assert math.isinf(data[repr(capture._sig(b))])
        assert outcomes[repr(capture._sig(b))] == "predicted_skip"
        assert outcomes[repr(capture._sig(c))] == "observed_failure"

    # An old committed cache claims all configs were searched and exposes only its old top-K.
    monkeypatch.setattr(capture, "configs_to_bench", lambda *a, **kw: [])
    test.control["cache_hit"] = True
    test.calls.clear()
    test.tuner.run()
    assert not test.calls
    test.control["predict"] = False
    test.tuner.run()
    assert test.calls == [32]
    assert settings.current().fill_gaps
    assert settings.current().run_autotune
    with round_cache.transaction(str(test.directory), IDENTITY, outcomes=outcomes) as data:
        assert data[repr(capture._sig(a))] == 1.0
        assert data[repr(capture._sig(b))] == 0.5
        assert repr(capture._sig(b)) not in outcomes


def test_legacy_unknown_inf_overrides_committed_searched_but_keeps_resource_prune(
        instrumented_round, monkeypatch):
    test = instrumented_round
    a, b, _, d = test.configs
    with round_cache.transaction(str(test.directory), IDENTITY) as data:
        data.update({repr(capture._sig(a)): 1.0,
                     repr(capture._sig(b)): float("inf"),
                     repr(capture._sig(d)): float("inf")})
    monkeypatch.setattr(capture, "configs_to_bench", lambda *a, **kw: [])
    test.control.update(predict=False, cache_hit=True)
    test.tuner.run()
    assert test.calls == [32]
    assert capture._sig(d) not in capture._CAPTURE["op"]["searched"][("float32", "128")]


def test_finite_committed_winner_survives_unknown_shared_inf(instrumented_round, monkeypatch):
    test = instrumented_round
    a, b, _, _ = test.configs
    with round_cache.transaction(str(test.directory), IDENTITY) as data:
        data.update({repr(capture._sig(a)): 1.0, repr(capture._sig(b)): float("inf")})
    monkeypatch.setattr(capture, "_known_timings", lambda *a: {repr(capture._sig(b)): 0.25})
    monkeypatch.setattr(capture, "configs_to_bench", lambda *a, **kw: [])
    test.control.update(predict=False, cache_hit=True)
    test.tuner.run()
    assert not test.calls
    with round_cache.transaction(str(test.directory), IDENTITY) as data:
        assert data[repr(capture._sig(b))] == 0.25


def test_prune_error_restores_global_settings_and_does_not_publish(instrumented_round):
    test = instrumented_round
    _, b, _, _ = test.configs
    with round_cache.transaction(str(test.directory), IDENTITY) as data:
        data[repr(capture._sig(b))] = float("inf")
    before = settings.current()
    test.control.update(predict=False, cache_hit=True, prune_error=True)
    with pytest.raises(RuntimeError, match="base prune failed"):
        test.tuner.run()
    assert settings.current() == before
    assert not cache._BYPASS_CACHED_SUBSET.get()
    assert not test.calls
    assert not capture._ROUND_RETRY
    assert not capture._ROUND_OUTCOMES
    assert not list(test.directory.glob("*.provenance"))


def test_no_predict_ignores_an_old_in_process_prediction(instrumented_round, monkeypatch):
    import triton.compiler.compiler as compiler

    test = instrumented_round
    cfg = test.configs[1]
    mark = f"kernel\t128\t{capture._cfg_sig(cfg)}"
    monkeypatch.setattr(capture, "_COMPILE_BUDGET_S", 60)
    monkeypatch.setattr(capture, "_PREDICTED_BAD", {mark})
    monkeypatch.setattr(capture, "_COMPILE_OK", {mark})
    monkeypatch.setattr(capture, "_COMPILE_BAD", set())
    capture._CURRENT.update(id=id(test.tuner), round="128")
    capture._CURRENT_CFG.update(cfg.kwargs, num_warps=cfg.num_warps, num_stages=cfg.num_stages)
    src = SimpleNamespace(fn=SimpleNamespace(__name__="kernel"))
    # Restore this hook before the fixture uninstalls capture's compiler wrapper.
    with monkeypatch.context() as local:
        local.setattr(capture, "_orig_compile", lambda *a, **kw: "compiled")
        with pytest.raises(capture._PredictedSkip, match="probe pass"):
            compiler.compile(src)
        test.control["predict"] = False
        assert compiler.compile(src) == "compiled"


@pytest.mark.parametrize("message", [
    "CUDA error: an illegal memory access was encountered",
    "CUDA error: device-side assert triggered",
    "CUDA error: misaligned address",
    "CUDA_ERROR_ILLEGAL_ADDRESS",
    "CUDA error: unspecified launch failure",
])
def test_fatal_cuda_error_stops_the_round_without_publishing_failures(
        instrumented_round, message):
    test = instrumented_round
    a, _, _, _ = test.configs
    # Preserve an older finite result byte-for-byte even when the verification round fails.
    with round_cache.transaction(str(test.directory), IDENTITY) as data:
        data[repr(capture._sig(a))] = 1.0
    path, = test.directory.glob("*.json")
    before = path.read_bytes()
    test.control.update(predict=False, fatal=message)
    with pytest.raises(RuntimeError, match="CUDA"):
        test.tuner.run()
    assert test.calls == [32]  # No attempt to benchmark later candidates on the poisoned context.
    assert path.read_bytes() == before
    assert not list(test.directory.glob("*.provenance"))
    assert not capture._ROUND_OUTCOMES


@pytest.mark.parametrize("message", [
    "out of resource: shared memory, Required: 153600, Hardware limit: 101376",
    "triton compile exceeded 60s (register-spill config); skipped",
    "CUDA out of memory",
    "compiler rejected a misaligned address expression",
])
def test_recoverable_failures_are_not_mistaken_for_a_poisoned_context(message):
    assert not capture._fatal_cuda_error(RuntimeError(message))


def test_verification_cache_bypass_is_isolated_from_other_threads():
    before = settings.current()
    with ThreadPoolExecutor(max_workers=1) as pool, cache.without_cached_subset():
        assert cache._BYPASS_CACHED_SUBSET.get()
        assert not pool.submit(cache._BYPASS_CACHED_SUBSET.get).result(timeout=5)
        with cache.without_cached_subset():
            assert cache._BYPASS_CACHED_SUBSET.get()
        assert cache._BYPASS_CACHED_SUBSET.get()
    assert not cache._BYPASS_CACHED_SUBSET.get()
    assert settings.current() is before
