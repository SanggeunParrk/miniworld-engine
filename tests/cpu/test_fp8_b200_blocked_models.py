"""fp8.B200_BLOCKED_MODELS: on a B200 (capability (10, 0)) FP8 stays off for a blocked model, through the environment preset and through
fp8.enable(); other models and other devices are unchanged. The device is faked (fp8.arch), no GPU needed. The list is empty since
2026-11-24 (Chai-1 finite with the B200 PWA e4m3 route off): the mechanism is tested with a stand-in model."""

from types import MappingProxyType

import pytest

from foldjet_engine import fp8

BLOCKED = "blockedmodel"


@pytest.fixture
def b200(monkeypatch):
    monkeypatch.setattr(fp8, "arch", lambda: "sm100")
    monkeypatch.setattr(fp8, "B200_BLOCKED_MODELS", MappingProxyType({BLOCKED: "test stand-in"}))


def test_chai1_no_longer_blocked():
    assert "chai1" not in fp8.B200_BLOCKED_MODELS


def test_preset_off_for_blocked_model_on_b200(b200, monkeypatch):
    monkeypatch.setenv("MINIWORLD_FP8", "safe")
    monkeypatch.setenv("MINIWORLD_FP8_MODEL", BLOCKED)
    assert fp8.preset() is None
    assert fp8.env("MINIWORLD_TRIMUL_FP8") is None
    monkeypatch.setenv("MINIWORLD_FP8_MODEL", BLOCKED.capitalize() + " ")
    assert fp8.preset() is None


def test_preset_kept_for_other_models_on_b200(b200, monkeypatch):
    monkeypatch.setenv("MINIWORLD_FP8", "safe")
    for m in ("af3", "protenix1", "boltz2", "chai1", ""):
        monkeypatch.setenv("MINIWORLD_FP8_MODEL", m)
        assert fp8.preset() == "safe"


def test_preset_kept_off_b200(monkeypatch):
    monkeypatch.setattr(fp8, "arch", lambda: "sm90")
    monkeypatch.setattr(fp8, "B200_BLOCKED_MODELS", MappingProxyType({BLOCKED: "test stand-in"}))
    monkeypatch.setenv("MINIWORLD_FP8", "safe")
    monkeypatch.setenv("MINIWORLD_FP8_MODEL", BLOCKED)
    assert fp8.preset() == "safe"


def test_enable_acts_as_disable_for_blocked_model_on_b200(b200, monkeypatch):
    calls = []
    monkeypatch.setattr(fp8, "disable", lambda: calls.append("disable"))
    monkeypatch.setattr(fp8, "_modules", lambda: pytest.fail("enable() must not switch routes on for a blocked model"))
    fp8.enable("safe", model=BLOCKED)
    assert calls == ["disable"]
    fp8.set_model(BLOCKED)
    try:
        fp8.enable("safe")
    finally:
        fp8.set_model(None)
    assert calls == ["disable", "disable"]
