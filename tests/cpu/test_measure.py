"""The measurement config (``measure.config``): parsing, case expansion, and the accuracy light rule (``measure.accuracy``) on plain
numbers. Nothing here runs a module on a GPU."""

from __future__ import annotations

import math

import pytest

from miniworld_engine.measure import accuracy
from miniworld_engine.measure import config as cfgmod

CSV_HEADER = "module,model,stream,lengths,depths,dims,options,dtypes,modes,eval_samples,train_samples,source\n"


def test_from_dict_defaults_and_reference_first():
    cfg = cfgmod.from_dict({"modules": "LayerNorm", "backends": ["cuda"]})
    assert cfg.modules == ("LayerNorm",)
    assert cfg.backends == ("pytorch", "cuda")   # the reference is always measured
    assert cfg.fixed == cfgmod.FIXED
    assert cfg.lengths == "min" and cfg.depths == "min" and cfg.samples == "all"


def test_from_dict_parsing():
    cfg = cfgmod.from_dict({"modules": ["A", "B"], "rows": [3, "4"], "lengths": [256, "128"], "depths": "all", "dtypes": "master",
                            "fixed": {"float32": "2e-5"}, "timing": ["graph"], "warmup": 3, "speed": {"iterations": 10}})
    assert cfg.rows == (3, 4)
    assert cfg.lengths == (256, 128)
    assert cfg.depths == "all"
    assert cfg.dtypes == ("master",)
    assert cfg.fixed == {**cfgmod.FIXED, "float32": 2e-5}
    assert cfg.speed == {"iterations": 10, "timing": ["graph"], "warmup": 3}   # speed keys may stand at the top level
    assert cfgmod.speed_settings(cfg) == cfg.speed


def test_unknown_key_is_an_error():
    with pytest.raises(ValueError, match="unknown config keys"):
        cfgmod.from_dict({"modules": "all", "lenghts": "all"})


def test_replace_keeps_parsing():
    cfg = cfgmod.replace(cfgmod.from_dict({"backends": ["cuda"]}), backends=["triton"], lengths=None)
    assert cfg.backends == ("pytorch", "triton")
    assert cfg.lengths == "min"   # None = keep


@pytest.mark.parametrize("name", ["accuracy.yaml", "speed.yaml", "all.yaml"])
def test_canonical_configs_load(name: str):
    path = cfgmod.REQUIREMENTS.parents[1] / "measure" / "configs" / name
    cfg = cfgmod.load(path)
    assert "pytorch" in cfg.backends
    assert cfgmod.settings(cfg)["seed"] == cfg.seed


def test_yaml_and_toml_load_the_same(tmp_path):
    (tmp_path / "c.yaml").write_text("modules: [LayerNorm]\nbackends: [cuda]\nlengths: [128, 256]\nseed: 3\n")
    (tmp_path / "c.toml").write_text('modules = ["LayerNorm"]\nbackends = ["cuda"]\nlengths = [128, 256]\nseed = 3\n')
    assert cfgmod.load(tmp_path / "c.yaml") == cfgmod.load(tmp_path / "c.toml")


def test_settings_leave_out_selection_keys():
    a = cfgmod.from_dict({"modules": "LayerNorm", "rows": [1], "lengths": "all"})
    b = cfgmod.from_dict({"modules": "all"})
    assert cfgmod.settings(a) == cfgmod.settings(b)   # a subset run measures each case exactly as the full run


def test_variants_and_values():
    assert cfgmod._variants("d=64;n=2|4", "bias=False|True;name=x;p=0.25;z=None") == [
        {"d": 64, "n": n, "bias": b, "name": "x", "p": 0.25, "z": None} for n in (2, 4) for b in (False, True)]
    assert cfgmod._variants("", "") == [{}]


def test_pick():
    assert cfgmod._pick([256, 128, 512], "min") == [128]
    assert cfgmod._pick([256, 128, 512], "all") == [128, 256, 512]
    assert cfgmod._pick([256, 128, 512], (512, 128, 999)) == [128, 512]
    assert cfgmod._pick([], "min") == [None]
    with pytest.raises(ValueError):
        cfgmod._pick([1], "most")


def test_cases_expand_and_filter(tmp_path):
    req = tmp_path / "req.csv"
    req.write_text(CSV_HEADER
                   + "LayerNorm,,token_pair,128|256,,normalized_shape=64|128,bias=True,float32|master,eval|train,1|5,1,x\n"
                   + "Transition,,token_pair,128,,d_hidden=64,n=4,master,eval,1,,y\n")
    cfg = cfgmod.from_dict({"modules": ["LayerNorm"], "requirements": str(req), "lengths": "all", "samples": "all"})
    cases = list(cfgmod.cases(cfg))
    # 2 widths x 2 lengths x (2 dtypes x (eval: 2 samples + train: 1 sample))
    assert len(cases) == 2 * 2 * 2 * 3
    assert {c.row for c in cases} == {1}
    assert {c.samples for c in cases if c.mode == "eval"} == {1, 5}
    cfg = cfgmod.replace(cfg, dtypes=["master"], modes=["train"], lengths="min")
    cases = list(cfgmod.cases(cfg))
    assert [(c.kwargs["normalized_shape"], c.length, c.dtype, c.mode, c.samples) for c in cases] == [
        (64, 128, "master", "train", 1), (128, 128, "master", "train", 1)]
    assert cases[0].label == "r1 LayerNorm(normalized_shape=64;bias=True) L128 S1 master train"
    every = cfgmod.from_dict({"modules": "all", "requirements": str(req), "rows": [2]})
    assert [(c.module, c.row) for c in cfgmod.cases(every)] == [("Transition", 2)]


def test_backends_of_keeps_the_classes_backends_reference_first():
    cfg = cfgmod.from_dict({"backends": ["no-such", "cuda", "triton"]})
    case = cfgmod.Case(row=1, module="LayerNorm", model="", stream="token_pair", kwargs={"normalized_shape": 64}, length=128,
                       depth=None, samples=1, dtype="master", mode="eval", seed=0)
    got = cfgmod.backends_of(cfg, case)
    assert got[0] == "pytorch"
    assert "no-such" not in got
    assert set(got) <= {"pytorch", *case.cls.backends}


def test_module_class_lookup():
    assert cfgmod.module_class("LayerNorm").__name__ == "LayerNorm"
    assert cfgmod.module_class("ProteinMPNN").__name__ == "ProteinMPNN"   # a subpackage class
    with pytest.raises(ValueError, match="no BackendModule class"):
        cfgmod.module_class("NoSuchModule")


# ---------------------------------------------------------------------------------------------------------------- the light rule
@pytest.mark.parametrize(("err", "tol", "light"), [(0.0, 1e-5, "GREEN"), (1e-5, 1e-5, "GREEN"), (2e-5, 1e-5, "YELLOW"),
                                                   (5e-5, 1e-5, "YELLOW"), (5.1e-5, 1e-5, "RED"), (math.nan, 1e-5, "RED"),
                                                   (math.inf, 1.0, "RED")])
def test_light(err: float, tol: float, light: str):
    assert accuracy._light(err, tol) == light


def test_verdict_takes_the_worst_tensor():
    import torch

    gold = {"out": torch.tensor([1.0, 2.0], dtype=torch.float64), "grad:x": torch.tensor([4.0, 0.0], dtype=torch.float64)}
    ref = {"out": torch.tensor([1.0, 2.0]), "grad:x": torch.tensor([4.0, 0.0])}
    be = {"out": torch.tensor([1.0, 2.0 + 2e-2]), "grad:x": torch.tensor([4.0, 0.0])}   # out: 1e-2 relative, tol 8e-3: YELLOW
    rec = accuracy._verdict(be, ref, gold, fixed=8e-3)
    assert rec["tensor"] == "out" and rec["light"] == "YELLOW" and rec["err_pytorch"] == 0.0
    be["out"] = torch.tensor([1.0, 3.0])
    assert accuracy._verdict(be, ref, gold, fixed=8e-3)["light"] == "RED"
    assert accuracy._verdict({}, {}, {}, fixed=8e-3)["status"] == "error"


def test_rel_error_against_zero_gold():
    import torch

    assert accuracy._rel(torch.tensor([0.5]), torch.zeros(1, dtype=torch.float64)) == 0.5
    assert accuracy._rel(None, None) == 0.0
