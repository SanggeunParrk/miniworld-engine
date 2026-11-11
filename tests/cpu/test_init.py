# ruff: noqa: NPY002  -- numpy's legacy global state is what construction draws from (rng: legacy)
"""Parameter initialisation from presets / config (miniworld_engine.modules.init, docs/init.md) over every requirements module class."""

from __future__ import annotations

import csv
import inspect
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn as nn

from miniworld_engine.measure import config as cfgmod
from miniworld_engine.modules import apply_init, describe_init
from miniworld_engine.modules.init import PRESET_DIR, load_preset, preset_names
from miniworld_engine.modules.primitives import Linear
from miniworld_engine.modules.token_dit import TokenDiTBoltz2, TokenDiTProtenix
from miniworld_engine.modules.triangle import BidirectionalTriangleAttention

with open(cfgmod.REQUIREMENTS, newline="") as _fh:
    _ROWS = list(csv.DictReader(_fh))
FIRST: dict[str, dict] = {}
for _r in _ROWS:
    FIRST.setdefault(_r["module"], _r)
CLASSES = sorted(FIRST)
#: Classes whose construction draws random numbers that no managed parameter of the finished model holds (modules built and then
#: replaced), so `reinit: all` + legacy cannot replay construction's draw sequence.
NOT_REPLAYABLE = {
    "AtomDiTChai1": "replaces both AdaLNs after construction (the discarded ones drew numbers)",
    "TokenDiTChai1": "replaces both AdaLNs after construction (the discarded ones drew numbers)",
}


def _kwargs(module: str, backend: str | None = None) -> dict:
    kw = cfgmod._variants(FIRST[module]["dims"], FIRST[module]["options"])[0]
    return kw if backend is None else {**kw, "backend": backend}


def _build(module: str, seed: int = 0, backend: str | None = None) -> Any:
    np.random.seed(seed)
    torch.manual_seed(seed)
    return cfgmod.construct(cfgmod.module_class(module), _kwargs(module, backend))


def _state(m: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in m.named_parameters()}


def _same(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor], only=None) -> list[str]:
    """The parameter names whose values differ (bitwise)."""
    names = a.keys() if only is None else only
    return [k for k in names if not torch.equal(a[k], b[k])]


def _managed(m: nn.Module) -> list[str]:
    return [r["parameter"] for r in describe_init(m, rows=True) if r["source"] != "unmanaged"]  # pyright: ignore[reportIndexIssue]


# --------------------------------------------------------------------------------------------------------------------- the presets
def test_presets_ship_as_package_data():
    assert preset_names() == ["af3", "engine", "miniworld"]
    assert PRESET_DIR.resolve() == Path(__file__).resolve().parents[2] / "src" / "miniworld_engine" / "modules" / "init_presets"
    pyproject = (Path(__file__).resolve().parents[2] / "pyproject.toml").read_text()
    globs = next(line for line in pyproject.splitlines() if line.startswith("miniworld_engine = [") and ".cu" in line)
    assert '"**/*.yaml"' in globs  # [tool.setuptools.package-data]: init_presets/*.yaml ship in the wheel
    assert load_preset("engine") == ({}, "legacy")


def test_construction_keeps_the_conditioning_gate_bias():
    """``Linear(..., bias_init=-2.0)`` (was a fill_ after construction) gives the same parameters."""
    m = _build("TokenDiTProtenix")
    assert torch.all(m.attention.to_scale.bias == -2.0)
    assert m.attention.to_scale.code_init() == ("default", -2.0)
    assert m.transition.to_scale.code_init() == ("default", "default")


# ------------------------------------------------------------------------------------------------------------- engine = the code
@pytest.mark.parametrize("module", CLASSES)
@pytest.mark.parametrize("preset", ["engine", "miniworld"])
def test_engine_and_miniworld_presets_are_bit_identical(module: str, preset: str):
    """``engine`` (no entries) and ``miniworld`` (today's code defaults written out) leave every engine class as construction drew it."""
    ref = _state(_build(module, seed=3))
    m = _build(module, seed=3)
    np.random.seed(99)
    torch.manual_seed(99)
    rows = describe_init(m, {"preset": preset, "rng": "legacy"}, rows=True)
    assert [r for r in rows if r["parameter"] is not None and r["code"] != r["effective"]] == []  # pyright: ignore[reportIndexIssue,reportArgumentType,reportCallIssue]
    apply_init(m, {"preset": preset, "rng": "legacy"})
    assert _same(ref, _state(m)) == []
    apply_init(m)  # None = the engine preset
    assert _same(ref, _state(m)) == []


@pytest.mark.parametrize("module", CLASSES)
def test_reinit_all_legacy_replays_construction(module: str):
    """``reinit: all`` + legacy under construction's seeds redraws every managed parameter exactly as construction did."""
    m = _build(module, seed=5)
    ref = _state(m)
    managed = _managed(m)
    with torch.no_grad():
        for name, p in m.named_parameters():
            if name in managed:
                p.fill_(0.123)
    np.random.seed(5)
    torch.manual_seed(5)
    apply_init(m, {"preset": "engine", "rng": "legacy", "reinit": "all"})
    diff = _same(ref, _state(m), managed)
    if module in NOT_REPLAYABLE:
        pytest.skip(f"{module}: {NOT_REPLAYABLE[module]} ({len(diff)} of {len(managed)} managed parameters differ)")
    assert diff == []
    # with a seed: the same draws without touching the global states (numpy RandomState(seed) == np.random.seed(seed))
    m2 = _build(module, seed=11)
    state = torch.random.get_rng_state()
    apply_init(m2, {"preset": "engine", "rng": "legacy", "reinit": "all", "seed": 5})
    assert torch.equal(state, torch.random.get_rng_state())
    assert _same(ref, _state(m2), managed) == []


# --------------------------------------------------------------------------------------------------------------------------- af3
def _check_value(p: torch.Tensor, effective: str, fan_in: int) -> None:
    word = effective.removeprefix("bias ")
    if word == "zero":
        assert torch.all(p == 0)
    elif word == "one":
        assert torch.all(p == 1)
    elif word == "default" and effective.startswith("bias"):
        assert p.abs().max() <= 1 / math.sqrt(fan_in) + 1e-6
    elif word == "default":
        assert p.abs().max() > 0 and p.abs().max() <= 2.0 / math.sqrt(fan_in) / 0.8796 + 1e-4
    else:
        assert torch.all(p == float(word))


@pytest.mark.parametrize("module", CLASSES)
def test_af3_changes_exactly_its_entries(module: str):
    m = _build(module)
    before = _state(m)
    rows = {r["parameter"]: r for r in describe_init(m, "af3", rows=True) if r["parameter"] is not None}  # pyright: ignore[reportIndexIssue,reportCallIssue,reportArgumentType]
    apply_init(m, {"preset": "af3"})
    after = _state(m)
    mods = dict(m.named_modules())
    for name, r in rows.items():
        if r["source"] in ("code", "unmanaged") or r["effective"] == r["code"]:
            assert torch.equal(before[name], after[name]), f"{name} changed but af3 does not set it ({r})"
        else:
            assert r["source"] == "preset:af3"
            owner = mods[name.rpartition(".")[0]]
            _check_value(after[name], r["effective"], getattr(owner, "in_features", 1))


def test_af3_entries_all_match_and_cover_the_audited_list():
    """Every af3 entry sets at least one parameter of some requirements class (no dead or misspelt entry), and the audited changes land."""
    entries, _ = load_preset("af3")
    hit: set[str] = set()
    seen: dict[tuple[str, str], str] = {}
    models = [(m, _build(m)) for m in CLASSES]
    models.append(("BidirectionalTriangleAttention", BidirectionalTriangleAttention(32, 4, use_self_attention=True)))  # no row has q / k
    for module, model in models:
        for r in describe_init(model, "af3", rows=True):  # pyright: ignore[reportGeneralTypeIssues]
            if r["source"] == "preset:af3":  # pyright: ignore[reportIndexIssue,reportArgumentType,reportCallIssue]
                hit.add(r["target"])  # pyright: ignore[reportIndexIssue,reportArgumentType,reportCallIssue]
                seen[(module, r["parameter"])] = r["effective"]  # pyright: ignore[reportIndexIssue,reportArgumentType,reportCallIssue]
    assert set(entries) == hit, f"af3 entries matching nothing: {sorted(set(entries) - hit)}"
    assert seen[("TokenDiTProtenix", "attention.ada_ln_in.to_scale.weight")] == "zero"
    assert seen[("TokenDiTProtenix", "attention.ada_ln_in.to_scale.bias")] == "bias zero"
    assert seen[("TokenDiTProtenix", "attention.to_scale.weight")] == "zero"
    assert seen[("TokenDiTProtenix", "transition.to_scale.bias")] == "bias -2.0"
    assert seen[("TokenDiTProtenix", "transition.squeeze.weight")] == "default"
    assert seen[("TokenDiTMiniWorldBiasOnly", "attention.to_bias.weight")] == "default"
    assert seen[("AttentionPairBias", "to_query.bias")] == "bias zero"
    assert seen[("MSAPairWeightedAveraging", "to_value.weight")] == "default"


# ------------------------------------------------------------------------------------------------------------ overrides / precedence
class _Two(nn.Module):
    def __init__(self, cls=TokenDiTProtenix) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([cls(d_single=64, d_cond=32, d_pair=16, n_head=4), cls(d_single=64, d_cond=32, d_pair=16, n_head=4)])


def _rows(model, cfg) -> dict[str, dict]:
    return {r["parameter"]: r for r in describe_init(model, cfg, rows=True)}  # pyright: ignore[reportIndexIssue,reportCallIssue,reportArgumentType]


def test_precedence_instance_over_pattern_over_preset():
    torch.manual_seed(0)
    m: Any = _Two()
    cfg = {
        "preset": "af3",
        "overrides": {
            "ConditionedTransition.to_scale": {"bias": -1.0},  # refines af3's {init: zero, bias: -2.0} key by key
            "blocks.1.transition.to_scale": {"bias": 3.0},  # instance path beats the pattern
            "blocks.1.attention.to_out": {"init": "zero"},  # back to the code default over af3
        },
    }
    r = _rows(m, cfg)
    assert (r["blocks.0.transition.to_scale.weight"]["effective"], r["blocks.0.transition.to_scale.weight"]["source"]) == (
        "zero",
        "preset:af3",
    )
    assert (r["blocks.0.transition.to_scale.bias"]["effective"], r["blocks.0.transition.to_scale.bias"]["source"]) == (
        "bias -1.0",
        "override",
    )
    assert r["blocks.1.transition.to_scale.bias"]["effective"] == "bias 3.0"
    assert r["blocks.1.transition.to_scale.bias"]["target"] == "blocks.1.transition.to_scale"
    assert r["blocks.0.attention.to_out.weight"]["effective"] == "default"
    assert r["blocks.1.attention.to_out.weight"]["effective"] == "zero"
    apply_init(m, cfg)
    assert torch.all(m.blocks[0].transition.to_scale.bias == -1.0) and torch.all(m.blocks[0].transition.to_scale.weight == 0)
    assert torch.all(m.blocks[1].transition.to_scale.bias == 3.0)
    assert torch.all(m.blocks[1].attention.to_out.weight == 0) and m.blocks[0].attention.to_out.weight.abs().sum() > 0


def test_most_specific_pattern_wins():
    torch.manual_seed(0)
    m = _Two(TokenDiTBoltz2)
    r = _rows(
        m,
        {
            "overrides": {
                "ConditionedTransition.squeeze": {"init": "relu"},
                "TokenDiTBoltz2.transition.squeeze": {"init": "glorot"},  # longer attribute path
                "ConditionedTransition.to_scale": {"init": "relu"},
                "UpGatedConditionedTransition.to_scale": {"init": "glorot"},  # nearer class in the MRO
                "TokenDiT*.attention.to_out": {"init": "relu"},
                "TokenDiTBoltz2.attention.to_out": {"init": "default"},  # no wildcard
                "AdaptiveLayerNorm.ln_cond": {"init": "zero"},  # a LayerNorm target
                "AdaptiveLayerNorm.to_bias": {"init": "relu"},
                "miniworld_engine.modules.norm:AdaptiveLayerNorm.to_bias": {"init": "glorot"},  # module-qualified beats the bare name
            }
        },
    )
    assert r["blocks.0.transition.squeeze.weight"]["effective"] == "glorot"
    assert r["blocks.0.transition.to_scale.weight"]["effective"] == "glorot"
    assert r["blocks.0.attention.to_out.weight"]["effective"] == "default"
    assert r["blocks.0.attention.ada_ln_in.ln_cond.weight"]["effective"] == "zero"
    assert r["blocks.0.attention.ada_ln_in.to_bias.weight"]["effective"] == "glorot"


def test_equal_specificity_with_different_specs_is_an_error():
    m: Any = _Two()
    with pytest.raises(ValueError, match="equally specifically"):
        describe_init(m, {"overrides": {"TokenDiT*.attention.to_out": {"init": "zero"}, "*Protenix.attention.to_out": {"init": "relu"}}})


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"ConditionedTransition.to_scal": {"init": "zero"}}, "match.*no parameter.*Did you mean: ConditionedTransition.to_scale"),
        ({"ConditionedTransitio.to_scale": {"init": "zero"}}, "match.*no parameter"),
        ({"blocks.0.transition.to_scal": {"init": "zero"}}, "neither an instance path"),
        ({"blocks.0.transition": {"init": "zero"}}, "neither an instance path"),
        ({"ConditionedTransition.to_scale": {"init": "lecun"}}, "init 'lecun' is not one of"),
        ({"ConditionedTransition.to_scale": {"bias": "uniform"}}, "bias 'uniform' is not one of"),
        ({"ConditionedTransition.to_scale": {"inti": "zero"}}, "unknown key"),
        ({"AdaptiveLayerNorm.ln_cond": {"init": "relu"}}, "init must be one of"),
        ({"blocks.0.attention.to_out": {"bias": 1.0}}, "sets a bias, but"),
    ],
)
def test_typos_raise(overrides, match):
    with pytest.raises(ValueError, match=match):
        apply_init(_Two(), {"overrides": overrides})


def test_bad_config_and_preset_keys_raise(tmp_path):
    with pytest.raises(ValueError, match="unknown key"):
        apply_init(_Two(), {"preset": "af3", "overide": {}})
    with pytest.raises(ValueError, match="rng"):
        apply_init(_Two(), {"rng": "numpy"})
    with pytest.raises(ValueError, match="neither a shipped preset"):
        apply_init(_Two(), {"preset": "af2"})
    bad = tmp_path / "bad.yaml"
    bad.write_text("entries:\n  ConditionedTransiton.to_scale: {init: zero}\n")  # a preset entry matching nothing: skipped, listed
    r = describe_init(_Two(), {"preset": str(bad)}, rows=True)
    assert r[-1] == {
        "parameter": None,
        "module": "-",
        "target": "ConditionedTransiton.to_scale",
        "code": "-",
        "effective": "-",  # pyright: ignore[reportIndexIssue,reportCallIssue,reportArgumentType]
        "source": "not present",
    }
    assert "not present" in str(describe_init(_Two(), {"preset": str(bad)}))
    bad.write_text("entrys: {}\n")
    with pytest.raises(ValueError, match="unknown key"):
        apply_init(_Two(), {"preset": str(bad)})
    bad.write_text("entries:\n  blocks.0.transition.to_scale: {init: zero}\n")
    with pytest.raises(ValueError, match="names classes, not instances"):
        apply_init(_Two(), {"preset": str(bad)})


def test_extends_and_yaml_config(tmp_path):
    mine = tmp_path / "mine.yaml"
    mine.write_text(
        "extends: af3\nrng: torch\nentries:\n"
        "  ConditionedTransition.squeeze: {init: zero}\n"  # replaces af3's entry
        "  TokenDiTProtenix.transition.expand_a: {init: default}\n"
    )
    entries, rng = load_preset(mine)
    assert rng == "torch" and str(entries["ConditionedTransition.squeeze"]) == "{init: zero}"
    assert "AdaptiveLayerNorm.to_scale" in entries
    cfg = tmp_path / "model.yaml"
    cfg.write_text(f"model: {{d: 1}}\ninit:\n  preset: {mine}\n  seed: 0\n")
    r = _rows(_Two(), cfg)
    assert r["blocks.0.transition.squeeze.weight"]["effective"] == "zero"
    assert r["blocks.0.transition.expand_a.weight"]["effective"] == "default"
    assert r["blocks.0.attention.to_out.weight"]["source"] == "preset:mine"
    (tmp_path / "a.yaml").write_text("extends: b.yaml\n")
    (tmp_path / "b.yaml").write_text("extends: a.yaml\n")
    with pytest.raises(ValueError, match="cycle"):
        load_preset(tmp_path / "a.yaml")


# ------------------------------------------------------------------------------------------------------------------------------- rng
@pytest.mark.parametrize("module", ["TokenDiTProtenix", "Transition", "AttentionPairBias", "TriangleAttention", "AtomDiTESMFold2"])
def test_rng_torch_depends_on_the_torch_seed_only(module: str):
    states = []
    for numpy_seed in (1, 2):
        torch.manual_seed(0)
        np.random.seed(numpy_seed)
        m = cfgmod.construct(cfgmod.module_class(module), _kwargs(module))
        np.random.seed(numpy_seed + 10)
        torch.manual_seed(7)
        apply_init(m, {"preset": "af3", "rng": "torch", "reinit": "all"})
        states.append(_state(m))
    assert _same(states[0], states[1]) == []
    # a seed in the config: independent of the global torch state
    seeded = []
    for global_seed in (3, 4):
        m = _build(module, seed=0)
        torch.manual_seed(global_seed)
        apply_init(m, {"preset": "af3", "rng": "torch", "reinit": "all", "seed": 7})
        seeded.append(_state(m))
    assert _same(seeded[0], seeded[1]) == []
    # legacy draws from numpy: different numpy states, different weights
    legacy = []
    for numpy_seed in (1, 2):
        m = _build(module, seed=0)
        np.random.seed(numpy_seed)
        apply_init(m, {"rng": "legacy", "reinit": "all"})
        legacy.append(_state(m))
    if any(isinstance(x, Linear) and x.init in ("default", "relu") for x in m.modules()):
        assert _same(legacy[0], legacy[1]) != []


def test_rng_torch_distribution():
    torch.manual_seed(0)
    lin = Linear(256, 4096, init="relu")
    apply_init(lin, {"rng": "torch", "reinit": "all", "seed": 1})
    std_target = math.sqrt(2.0 / 256)
    assert abs(lin.weight.std().item() / std_target - 1) < 0.01
    assert lin.weight.abs().max().item() <= 2 * std_target / 0.8796 + 1e-6  # truncation at +-2 sigma of the untruncated normal
    assert lin.bias is not None and lin.bias.abs().max().item() <= 1 / 16 + 1e-6


# ----------------------------------------------------------------------------------------------------------- describe / backends
@pytest.mark.parametrize("module", CLASSES)
def test_describe_covers_every_parameter(module: str):
    m = _build(module)
    rows = describe_init(m, "af3", rows=True)
    assert [r["parameter"] for r in rows if r["parameter"] is not None] == [n for n, _ in m.named_parameters()]  # pyright: ignore[reportIndexIssue,reportArgumentType,reportCallIssue]
    text = describe_init(m, {"preset": "af3"})
    assert isinstance(text, str) and text.startswith("# init: preset af3")
    assert len(text.splitlines()) == len(rows) + 2


@pytest.mark.parametrize("module", CLASSES)
def test_init_is_backend_independent(module: str):
    """Every backend's object (constructed on CPU: construction is device-agnostic) has the same targets and gets the same values."""
    cls = cfgmod.module_class(module)
    if "backend" not in inspect.signature(cls.__init__).parameters:
        pytest.skip("the constructor takes no backend")
    ref = _build(module, seed=2)
    apply_init(ref, "af3")
    want_rows, want = describe_init(ref, "af3", rows=True), _state(ref)
    for backend in cls.backends:
        m = _build(module, seed=2, backend=backend)
        apply_init(m, "af3")
        assert describe_init(m, "af3", rows=True) == want_rows
        assert _same(want, _state(m)) == []
