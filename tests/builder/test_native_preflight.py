"""Native import/header failures must be found before full-grid tuning."""
from types import SimpleNamespace

import pytest

from miniworld_engine.autotune import preflight
from miniworld_engine.kernels import _nvcc
from miniworld_engine.modules.swa_atom_attention import module as swa


def test_ampere_does_not_require_mathdx_or_cute(monkeypatch):
    monkeypatch.setattr(swa, "_flash_backend", lambda: "fa2")
    seen = []
    def load(name):
        seen.append(name)
        return SimpleNamespace(flash_attn_varlen_func=lambda: None)
    monkeypatch.setattr(preflight.importlib, "import_module", load)
    monkeypatch.setattr(_nvcc, "mathdx_includes", lambda: pytest.fail("Ampere requested mathdx"))
    preflight.native_dependencies("sm86")
    assert seen == ["flash_attn.flash_attn_interface"]


def test_foreign_native_failures_are_reported_together(monkeypatch):
    monkeypatch.setattr(swa, "_flash_backend", lambda: "fa4")
    def load(name):
        if name == "flash_attn.cute":
            raise ImportError("incompatible CUTLASS")
        return SimpleNamespace()
    def missing():
        raise RuntimeError("cuBLASDx headers missing")
    monkeypatch.setattr(preflight.importlib, "import_module", load)
    monkeypatch.setattr(_nvcc, "mathdx_includes", missing)
    with pytest.raises(ValueError, match="before tuning") as exc:
        preflight.native_dependencies("sm100")
    assert "incompatible CUTLASS" in str(exc.value)
    assert "cuBLASDx headers missing" in str(exc.value)


def test_backend_without_varlen_entry_is_rejected(monkeypatch):
    monkeypatch.setattr(swa, "_flash_backend", lambda: "fa2")
    monkeypatch.setattr(preflight.importlib, "import_module", lambda name: SimpleNamespace())
    with pytest.raises(ValueError, match="flash_attn_varlen_func"):
        preflight.native_dependencies("sm86")


def test_full_build_stops_before_plan_or_tuning_on_dependency_failure(monkeypatch):
    from miniworld_engine import cli
    from miniworld_engine.autotune import builder, plan
    monkeypatch.setattr(builder, "device_sm", lambda: "sm90")
    monkeypatch.setattr(cli, "apply_config_dir", lambda *args: 0)
    def fail(arch):
        raise ValueError("native dependencies are unavailable")
    def forbidden(*args, **kwargs):
        pytest.fail("work started despite failed dependency preflight")
    monkeypatch.setattr(preflight, "native_dependencies", fail)
    monkeypatch.setattr(plan, "ensure", forbidden)
    monkeypatch.setattr(builder, "build_all", forbidden)
    assert cli.main(["build", "all"]) == 1
