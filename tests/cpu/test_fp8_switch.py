"""The one FP8 switch (``miniworld_engine.fp8``): ``MINIWORLD_FP8=<preset>`` / ``fp8.enable(preset)`` map to the per-route switches;
an explicitly set per-route variable wins; unset = every route off (the default route)."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from miniworld_engine import fp8

_STATUS = "import json; from miniworld_engine import fp8; print(json.dumps(fp8.status()))"
_ROUTE_VARS = ("MINIWORLD_FP8", *fp8.PRESETS["safe"])


def _status_in_process(**env) -> dict:
    clean = {k: v for k, v in os.environ.items() if not k.startswith(_ROUTE_VARS)}
    out = subprocess.run([sys.executable, "-c", _STATUS], env={**clean, **env}, capture_output=True, text=True, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


OFF = {
    "triattn": {"pv": [], "qk": False},
    "trimul": None,
    "opm": None,
    "pwa": False,
    "transition": None,
    "token_dit": [],
    "atom_dit": [],
}
SAFE = {
    "triattn": {"pv": ["fused", "unfused"], "qk": True},
    "trimul": "calib",
    "opm": "bound",
    "pwa": True,
    "transition": "calib",
    "token_dit": ["expand"],
    "atom_dit": ["post", "pre"],
}


def _routes(status: dict) -> dict:
    return {k: v for k, v in status.items() if k != "guard"}


def test_env_preset(monkeypatch):
    for k in _ROUTE_VARS:
        monkeypatch.delenv(k, raising=False)
    assert fp8.preset() is None and fp8.env("MINIWORLD_TRIMUL_FP8") is None
    monkeypatch.setenv("MINIWORLD_FP8", "safe")
    assert fp8.preset() == "safe" and fp8.env("MINIWORLD_TRIMUL_FP8") == "calib"
    monkeypatch.setenv("MINIWORLD_TRIMUL_FP8", "")  # explicitly off wins over the preset
    assert fp8.env("MINIWORLD_TRIMUL_FP8") == ""
    monkeypatch.setenv("MINIWORLD_FP8", "fastest")
    with pytest.raises(ValueError):
        fp8.preset()


def test_default_is_off():
    assert _routes(_status_in_process()) == OFF


def test_env_safe_sets_every_route():
    assert _routes(_status_in_process(MINIWORLD_FP8="safe")) == SAFE
    st = _status_in_process(MINIWORLD_FP8="safe", MINIWORLD_TOKEN_DIT_FP8="", MINIWORLD_PWA_FP8_GUARD="lazy")
    assert _routes(st) == {**SAFE, "token_dit": []} and st["guard"]["pwa"] == "lazy"


def test_enable_disable_api():
    code = (
        "import json; from miniworld_engine import fp8; fp8.enable('safe', guard='lazy'); a = fp8.status(); "
        "fp8.enable('safe', model='boltz2'); b = fp8.status(); fp8.disable(); print(json.dumps([a, b, fp8.status()]))"
    )
    clean = {k: v for k, v in os.environ.items() if not k.startswith(_ROUTE_VARS)}
    out = subprocess.run([sys.executable, "-c", code], env=clean, capture_output=True, text=True, check=True)
    a, b, c = json.loads(out.stdout.strip().splitlines()[-1])
    assert _routes(a) == SAFE and set(a["guard"].values()) == {"lazy"}
    assert _routes(b) == {**SAFE, "triattn": {"pv": [], "qk": False}}  # boltz2 is not on the TriAttn whitelist
    assert _routes(c) == OFF
