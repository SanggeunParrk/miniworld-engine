"""The judge's version logic (``judge.differences`` / ``verdict``, ``measure.config.versions``) on synthetic tokens: what a change owes,
what it asks a person about, and what it leaves alone."""

from __future__ import annotations

import copy
from typing import cast

import pytest

from miniworld_engine import judge
from miniworld_engine.measure import config as cfgmod

SETTINGS = {"seed": 0, "fixed": {"bfloat16": 0.008, "float32": 1e-05, "master": 0.008}, "speed": {}}
ENV = {"capability": [9, 0], "torch": "2.13.0+cu129", "cuda": "12.9", "triton": "3.7.1"}


def _versions(**over) -> dict:
    v = {"module": "1", "kernels": {"k_a": "1"}, "tables": {"fam": "1"}, "backends": {}, "tf32": {"pytorch": "1", "cuda": "2"}}
    v.update(over)
    return v


class FakeNow:
    """Duck-typed :class:`judge.Now`: the current versions, settings and environment, all given."""

    def __init__(self, versions: dict | None = None, gpu: str = "h100_sm90", **over):
        self.gpu = gpu
        self._v = versions or _versions()
        self.measurer = {"accuracy": "3", "speed": "3"}
        self.config_version = "1"
        self.prebuild_version = "1"
        self.settings = {"accuracy": SETTINGS, "speed": SETTINGS}
        self.env = dict(ENV)
        self.__dict__.update(over)

    def versions(self, module: str, mode: str) -> dict:
        return self._v

    def env_for(self, backend: str) -> dict:
        return judge.Now.env_for(cast(judge.Now, self), backend)


def _token(backend: str = "cuda", dtype: str = "master", versions: dict | None = None, **over) -> dict:
    tok = {
        "case": {"module": "Transition", "mode": "train", "dtype": dtype, "backend": backend},
        "versions": versions or _versions(),
        "settings": SETTINGS,
        "measurer": "3",
        "config_version": "1",
        "prebuild_version": "1",
        "env": dict(ENV),
    }
    tok.update(over)
    return tok


def _diff(kind: str, tok: dict, now: FakeNow) -> list:
    return judge.differences(kind, tok, cast(judge.Now, now))


def _parts(diffs) -> list[tuple[str, str]]:
    return [(cls, part) for cls, part, _, _ in diffs]


def test_current_token_is_carried():
    now = FakeNow()
    diffs = _diff("accuracy", _token(), now)
    assert diffs == []
    assert judge.verdict(diffs, {}) == ("carried", [])


def test_module_version_owes():
    now = FakeNow(_versions(module="2"))
    assert _parts(_diff("accuracy", _token(), now)) == [("owed", "module:Transition")]


def test_adapter_card_entry_wins_over_the_plain_backend():
    """``backend_versions``: "<backend>:<card>" (this card's adapter) before "<backend>"; a token made before has "1"."""
    old = _versions(backends={"cuda": "1"})
    # this card's own entry moved: owed
    now = FakeNow(_versions(backends={"cuda": "1", "cuda:h100_sm90": "2"}))
    assert ("owed", "adapter:cuda") in _parts(_diff("accuracy", _token(versions=old), now))
    # another card's entry moved: nothing owed here
    now = FakeNow(_versions(backends={"cuda": "1", "cuda:a100_sm80": "4"}))
    assert _diff("accuracy", _token(versions=old), now) == []
    # the same, judged for the A100: owed there
    now = FakeNow(_versions(backends={"cuda": "1", "cuda:a100_sm80": "4"}), gpu="a100_sm80")
    assert ("owed", "adapter:cuda") in _parts(_diff("accuracy", _token(versions=old), now))
    # a token without adapters (older format) is version "1"
    now = FakeNow(_versions(backends={"cuda:h100_sm90": "1"}))
    assert _diff("accuracy", _token(versions=_versions(backends={})), now) == []


def test_adapter_of_another_backend_is_ignored():
    now = FakeNow(_versions(backends={"anthropic": "3"}))
    assert _diff("accuracy", _token(backend="cuda"), now) == []


@pytest.mark.parametrize(("dtype", "owes"), [("float32", True), ("master", False), ("bfloat16", False)])
def test_tf32_rule_only_touches_float32(dtype: str, owes: bool):
    now = FakeNow(_versions(tf32={"pytorch": "1", "cuda": "3"}))
    parts = _parts(_diff("accuracy", _token(dtype=dtype), now))
    assert (("owed", "tf32:cuda") in parts) is owes
    assert parts == ([("owed", "tf32:cuda")] if owes else [])


def test_kernel_version_change_owes_engine_backends_only():
    now = FakeNow(_versions(kernels={"k_a": "2"}))
    assert _parts(_diff("accuracy", _token(backend="cuda"), now)) == [("owed", "kernel:k_a")]
    for backend in ("pytorch", *cfgmod.EXTERNAL):  # they run none of the engine's kernels
        assert _diff("accuracy", _token(backend=backend, env=now.env_for(backend)), now) == []  # made on its stack


def test_a_new_kernel_row_alone_owes_nothing():
    """A kernel entering (or leaving) the list is a correction of requirements_kernel.csv, not a changed result."""
    now = FakeNow(_versions(kernels={"k_a": "1", "k_new": "1"}))
    assert _diff("accuracy", _token(), now) == []
    now = FakeNow(_versions(kernels={}))
    assert _diff("accuracy", _token(), now) == []
    # ... but a version change of a kernel listed both then and now does
    now = FakeNow(_versions(kernels={"k_a": "3", "k_new": "1"}))
    assert _parts(_diff("accuracy", _token(), now)) == [("owed", "kernel:k_a")]


def test_table_versions():
    now = FakeNow(_versions(tables={"fam": "2"}))
    assert _parts(_diff("speed", _token(), now)) == [("owed", "table:fam")]
    # a token made before table versions counts every current family as "1"
    old = _versions()
    del old["tables"]
    assert _diff("speed", _token(versions=old), FakeNow()) == []
    assert _parts(_diff("speed", _token(versions=old), now)) == [("owed", "table:fam")]
    # a family entering the list owes nothing
    now = FakeNow(_versions(tables={"fam": "1", "other": "5"}))
    assert _diff("speed", _token(), now) == []


def test_settings_owe_and_measurer_config_prebuild_env_ask():
    tok = _token(settings={**SETTINGS, "seed": 1})
    assert _parts(_diff("accuracy", tok, FakeNow())) == [("owed", "settings")]
    for field, part in (("measurer", "measurer:accuracy"), ("config_version", "config_version"), ("prebuild_version", "prebuild_version")):
        tok = {**_token(), field: "0"}
        assert _parts(_diff("accuracy", tok, FakeNow())) == [("ask", part)]
    tok = _token(env={**ENV, "torch": "2.12.0"})
    assert _parts(_diff("accuracy", tok, FakeNow())) == [("ask", "env:torch")]
    tok = _token()
    del tok["prebuild_version"]  # tokens made before the prebuild version: "1"
    assert _diff("accuracy", tok, FakeNow()) == []


def test_verdict_answers():
    ask = [("ask", "measurer:accuracy", "2", "3")]
    assert judge.verdict(ask, {}) == ("pending", [("measurer:accuracy", "2", "3")])
    assert judge.verdict(ask, {("measurer:accuracy", "2", "3"): "keep"}) == ("carried", [])
    assert judge.verdict(ask, {("measurer:accuracy", "2", "3"): "remeasure"})[0] == "owed"
    assert judge.verdict(ask, {("measurer:accuracy", "1", "3"): "keep"})[0] == "pending"  # an answer to another change
    both = [*ask, ("owed", "kernel:k_a", "1", "2")]
    assert judge.verdict(both, {("measurer:accuracy", "2", "3"): "keep"}) == ("owed", [("kernel:k_a", "1", "2")])


def test_card_of():
    assert judge.card_of({"gpu": "NVIDIA H100 80GB HBM3", "capability": [9, 0]}) == "h100_sm90"
    assert judge.card_of({"gpu": "NVIDIA A100-SXM4-80GB", "capability": [8, 0]}) == "a100_sm80"
    assert set(judge.GPUS) == {"h100_sm90", "a100_sm80"}


def test_case_key_is_readable_and_ordered():
    rec = {
        "module": "LayerNorm",
        "model": "",
        "stream": "token_pair",
        "kwargs": {"normalized_shape": 128, "bias": True},
        "length": 128,
        "depth": None,
        "samples": 1,
        "dtype": "master",
        "mode": "eval",
        "backend": "cuda",
        "timing": "graph",
    }
    assert judge.case_key("speed", rec) == (
        "speed | LayerNorm | - | token_pair | bias=True;normalized_shape=128 | L128 | DNone | S1 | master | eval | cuda | graph"
    )
    assert judge.case_key("accuracy", {**rec, "timing": None}).endswith("| cuda")


# ------------------------------------------------------------------------------------------- measure.config.versions on a real class
@pytest.fixture
def kernel_table(tmp_path, monkeypatch):
    """A synthetic requirements_kernel.csv for LayerNorm: generic, sm80-only and sm90a-only kernels of each role."""
    path = tmp_path / "requirements_kernel.csv"
    path.write_text(
        "name,family,modules,role,version\n"
        "ln_fwd,layernorm,LayerNorm,forward,1\n"
        "ln_fwd_infer_sm80,layernorm,LayerNorm,forward_infer,2\n"
        "ln_fwd_infer_sm90a,layernorm,LayerNorm,forward_infer,3\n"
        "ln_bwd_sm90,layernorm,LayerNorm|RMSNorm,backward,4\n"
        "ln_fwd_train,layernorm,LayerNorm,forward_train,5\n"
        "other_fwd,transition,Transition,forward,6\n"
    )
    monkeypatch.setattr(cfgmod, "KERNELS", path)
    cfgmod._table.cache_clear()
    yield path
    cfgmod._table.cache_clear()


def test_kernel_versions_only_for_the_cards_own_kernels(kernel_table):
    h100_eval = cfgmod.versions("LayerNorm", "eval", "h100_sm90")["kernels"]
    assert h100_eval == {"ln_fwd": "1", "ln_fwd_infer_sm90a": "3"}
    h100_train = cfgmod.versions("LayerNorm", "train", "h100_sm90")["kernels"]
    assert h100_train == {"ln_bwd_sm90": "4", "ln_fwd": "1", "ln_fwd_train": "5"}
    a100_eval = cfgmod.versions("LayerNorm", "eval", "a100_sm80")["kernels"]
    assert a100_eval == {"ln_fwd": "1", "ln_fwd_infer_sm80": "2"}
    a100_train = cfgmod.versions("LayerNorm", "train", "a100_sm80")["kernels"]
    assert a100_train == {"ln_fwd": "1", "ln_fwd_train": "5"}


def test_versions_shape(kernel_table):
    v = cfgmod.versions("LayerNorm", "eval", "h100_sm90")
    assert v["module"] == str(cfgmod.module_class("LayerNorm").version)
    assert v["tables"] == {"layernorm": cfgmod.table_version("layernorm", "h100_sm90")}
    assert v["tf32"] == cfgmod.TF32_RULE
    assert v["tf32"] is not cfgmod.TF32_RULE  # a copy: a record can never alias the rule


def test_differences_does_not_mutate_its_inputs():
    tok, now = _token(backend="pytorch"), FakeNow(_versions(kernels={"k_a": "9"}))
    before = copy.deepcopy(tok)
    _diff("accuracy", tok, now)
    assert tok == before
    assert now.versions("Transition", "train")["kernels"] == {"k_a": "9"}


def test_kit_stack_backend_compares_with_the_kit_environment():
    """H100 anthropic results are measured on the kits' CUDA-13 stack (measure.config.KIT_STACK_BACKENDS): a token from that stack is
    current, one from the engine's CUDA-12.9 stack asks a person (env), and the engine backends keep comparing with this process's stack."""
    kit_env = {**ENV, **cfgmod.KIT_STACK_ENV}
    now = FakeNow()
    assert _diff("speed", _token(backend="anthropic", env=kit_env), now) == []
    assert ("ask", "env:cuda") in _parts(_diff("speed", _token(backend="anthropic"), now))
    assert _diff("speed", _token(backend="cuda"), now) == []
    assert ("ask", "env:cuda") in _parts(_diff("speed", _token(backend="cuda", env=kit_env), now))
    # an A100 kit stack is the engine's environment under a CUDA-13 driver library: the env keys are this process's
    assert _diff("speed", _token(backend="anthropic"), FakeNow(gpu="a100_sm80")) == []


def test_kit_stack_records_are_told_by_the_cuda_13_torch_or_driver_library():
    """A record is a kit-stack record when its torch is a CUDA-13 build (the H100 container) or its process loaded a CUDA-13 driver
    library (``cuda_driver``: the forward-compatible libcuda of the A100 wrapper); both cards measure anthropic there, and only an
    adapter keyed by the card (``anthropic:a100_sm80``) makes the A100's anthropic tokens owed without touching the H100's."""
    assert cfgmod.on_kit_stack({"cuda": "13.0"})
    assert cfgmod.on_kit_stack({"cuda": "12.9", "cuda_driver": "13.0"})
    assert not cfgmod.on_kit_stack({"cuda": "12.9", "cuda_driver": "12.8"})
    assert not cfgmod.on_kit_stack({"cuda": "12.9"})  # a record made before ``cuda_driver`` was recorded
    assert cfgmod.kit_stack("a100_sm80", "anthropic") and cfgmod.kit_stack("h100_sm90", "anthropic")
    assert not cfgmod.kit_stack("a100_sm80", "cuda")
    assert cfgmod.kit_stack_env("a100_sm80") == {} and cfgmod.kit_stack_env("h100_sm90") == cfgmod.KIT_STACK_ENV
    old = _token(backend="anthropic", versions=_versions(backends={"anthropic": "3"}))
    cur = _versions(backends={"anthropic": "3", "anthropic:a100_sm80": "4"})
    assert ("owed", "adapter:anthropic") in _parts(_diff("speed", old, FakeNow(cur, gpu="a100_sm80")))
    kit_env = {**ENV, **cfgmod.KIT_STACK_ENV}
    assert _diff("speed", _token(backend="anthropic", versions=_versions(backends={"anthropic": "3"}), env=kit_env), FakeNow(cur)) == []


def test_plan_puts_kit_stack_cases_into_k_groups(tmp_path):
    import collections

    def entry(backends):
        return {
            "rows": {1},
            "lengths": {128},
            "backends": set(backends),
            "dtypes": {"master"},
            "modes": {"eval"},
            "n": 2,
            "cells": collections.Counter({(1, 128): 2}),
        }

    todo = {("speed", "LayerNorm", ""): entry(["cuda"]), ("speed", "LayerNorm", "kit"): entry(["anthropic"])}
    judge.write_plan(str(tmp_path), todo, n_groups=2)
    g = (tmp_path / "G1").read_text() + (tmp_path / "G2").read_text()
    k = (tmp_path / "K1").read_text()
    assert "LayerNorm " in g and ".kit" not in g
    assert "speed LayerNorm.kit " in k
    import yaml

    assert yaml.safe_load((tmp_path / "cfg" / "speed.LayerNorm.kit.yaml").read_text())["backends"] == ["anthropic", "pytorch"]
