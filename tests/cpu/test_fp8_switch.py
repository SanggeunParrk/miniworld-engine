"""The one FP8 switch (``miniworld_engine.fp8``): ``MINIWORLD_FP8=<preset>`` / ``fp8.enable(preset)`` map to the per-route switches;
an explicitly set per-route variable wins; unset = every route off (the default route). TriAttn FP8 only for a model in
``fp8.TRIATTN_MODELS`` (``MINIWORLD_FP8_MODEL`` / ``enable(model=)``); FP8 is outside the judge (no registry dtype, no judge / measure path)."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from miniworld_engine import fp8

_STATUS = "import json; from miniworld_engine import fp8; print(json.dumps(fp8.status()))"
_ROUTE_VARS = ("MINIWORLD_FP8", *fp8.PRESETS["safe"])  # "MINIWORLD_FP8" also covers MINIWORLD_FP8_MODEL
_PKG = Path(fp8.__file__).resolve().parent


def _clean_env() -> dict:
    return {k: v for k, v in os.environ.items() if not k.startswith(_ROUTE_VARS)}


def _status_in_process(**env) -> dict:
    out = subprocess.run([sys.executable, "-c", _STATUS], env={**_clean_env(), **env}, capture_output=True, text=True, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


NO_TRIATTN = {"pv": [], "qk": False}
OFF = {
    "triattn": NO_TRIATTN,
    "trimul": None,
    "opm": None,
    "pwa": False,
    "transition": None,
    "token_dit": [],
    "atom_dit": [],
}
#: ``safe`` without a model: every route but TriAttn
SAFE = {
    "triattn": NO_TRIATTN,
    "trimul": "calib",
    "opm": "bound",
    "pwa": True,
    "transition": "calib",
    "token_dit": ["expand"],
    "atom_dit": ["post", "pre"],
}
#: the TriAttn part per validated model (af3: PV fused + unfused and q / k; protenix1 / opendde: PV only, on their checked kinds)
TRIATTN_OF = {
    "af3": {"pv": ["fused", "unfused"], "qk": True},
    "protenix1": {"pv": ["fused"], "qk": False},
    "opendde": {"pv": ["unfused"], "qk": False},
}


def _routes(status: dict) -> dict:
    return {k: v for k, v in status.items() if k != "guard"}


def test_env_preset(monkeypatch):
    for k in (*_ROUTE_VARS, "MINIWORLD_FP8_MODEL"):
        monkeypatch.delenv(k, raising=False)
    assert fp8.preset() is None and fp8.env("MINIWORLD_TRIMUL_FP8") is None and fp8.model() is None
    monkeypatch.setenv("MINIWORLD_FP8", "safe")
    assert fp8.preset() == "safe" and fp8.env("MINIWORLD_TRIMUL_FP8") == "calib"
    monkeypatch.setenv("MINIWORLD_TRIMUL_FP8", "")  # explicitly off wins over the preset
    assert fp8.env("MINIWORLD_TRIMUL_FP8") == ""
    monkeypatch.setenv("MINIWORLD_FP8_MODEL", " AF3 ")
    assert fp8.model() == "af3"
    monkeypatch.setenv("MINIWORLD_FP8", "fastest")
    with pytest.raises(ValueError):
        fp8.preset()


def test_triattn_whitelist():
    assert set(fp8.TRIATTN_MODELS) == {"af3", "protenix1", "opendde"}  # extend only after an end-level check
    assert "boltz2" not in fp8.TRIATTN_MODELS and "openfold3" not in fp8.TRIATTN_MODELS
    assert [m for m, (_k, qk) in fp8.TRIATTN_MODELS.items() if qk] == ["af3"]  # e4m3 q / k validated end to end on AF3 only
    assert fp8.triattn_for(None) == (frozenset(), False) and fp8.triattn_for("boltz2") == (frozenset(), False)
    assert fp8.triattn_for("AF3") == fp8.TRIATTN_MODELS["af3"]


def test_default_is_off():
    assert _routes(_status_in_process()) == OFF
    assert _routes(_status_in_process(MINIWORLD_FP8_MODEL="af3")) == OFF  # the model alone switches nothing on


def test_env_safe_without_model_keeps_triattn_off():
    assert _routes(_status_in_process(MINIWORLD_FP8="safe")) == SAFE
    st = _status_in_process(MINIWORLD_FP8="safe", MINIWORLD_TOKEN_DIT_FP8="", MINIWORLD_PWA_FP8_GUARD="lazy")
    assert _routes(st) == {**SAFE, "token_dit": []} and st["guard"]["pwa"] == "lazy"


@pytest.mark.parametrize("model", ["af3", "protenix1", "opendde", "boltz2", "openfold3"])
def test_env_safe_with_model(model):
    st = _status_in_process(MINIWORLD_FP8="safe", MINIWORLD_FP8_MODEL=model)
    assert _routes(st) == {**SAFE, "triattn": TRIATTN_OF.get(model, NO_TRIATTN)}


def test_enable_disable_api():
    code = (
        "import json; from miniworld_engine import fp8; out = []\n"
        "fp8.enable('safe', guard='lazy'); out.append(fp8.status())\n"
        "for m in ('af3', 'protenix1', 'opendde', 'boltz2', 'openfold3'):\n"
        "    fp8.enable('safe', model=m); out.append(fp8.status())\n"
        "fp8.enable('safe'); out.append(fp8.status())\n"
        "fp8.disable(); out.append(fp8.status()); print(json.dumps(out))"
    )
    # MINIWORLD_FP8_MODEL is not read by enable(): set it to show that
    out = subprocess.run(
        [sys.executable, "-c", code], env={**_clean_env(), "MINIWORLD_FP8_MODEL": "af3"}, capture_output=True, text=True, check=True
    )
    a, *per_model, again, off = json.loads(out.stdout.strip().splitlines()[-1])
    assert _routes(a) == SAFE and set(a["guard"].values()) == {"lazy"}
    for m, st in zip(("af3", "protenix1", "opendde", "boltz2", "openfold3"), per_model, strict=True):
        assert _routes(st) == {**SAFE, "triattn": TRIATTN_OF.get(m, NO_TRIATTN)}, m
    assert _routes(again) == SAFE  # enable() without model= turns the TriAttn part off again
    assert _routes(off) == OFF


def test_pv_whitelist_is_derived():
    from miniworld_engine.kernels.triangle_attention.cuda import h100_sm90 as ta

    derived = {m: kinds for m, (kinds, _qk) in fp8.TRIATTN_MODELS.items()}
    assert derived == ta.FP8_PV_WHITELIST


def test_fp8_is_outside_the_judge():
    """No registry dtype / row for FP8, and no judge / measure / requirements code path that switches or names an FP8 route."""
    for name in ("requirements_module.csv",):
        with (_PKG / "requirements" / name).open() as fh:
            dtypes = {d for row in csv.DictReader(fh) for d in row["dtypes"].split("|") if d}
        assert dtypes <= {"float32", "master"}, dtypes
    for sub in ("judge", "measure", "requirements"):
        for f in (_PKG / sub).rglob("*"):
            if f.suffix in (".py", ".csv", ".json", ".yaml", ".yml", ".toml"):
                text = f.read_text(errors="replace").lower()
                assert "fp8" not in text and "e4m3" not in text and "float8" not in text, f
