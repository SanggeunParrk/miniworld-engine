"""Optional providers preserve package identity and named refusal contracts."""
from types import SimpleNamespace

import pytest

from miniworld_engine.integrations import anthropic as A


def test_carried_provider_keeps_package_context(monkeypatch):
    calls = []
    module = object()
    registry = SimpleNamespace(route=lambda name: calls.append(("route", name)))

    def importing(name):
        calls.append(("import", name))
        return registry if name == "opt_core.kernels" else module

    monkeypatch.setattr(A, "_ROOT", "configured")
    monkeypatch.setattr(A, "import_module", importing)
    assert A.carried_kernel("pallas") is module
    assert calls[-1] == ("import", "opt_core.kernels.pallas")
    assert ("route", "pallas") in calls


def test_pallas_named_refusal_is_strict_by_default(monkeypatch):
    def attention(*args, **kwargs):
        assert kwargs["strict"] is True
        assert kwargs["word"] == "named_row"
        raise RuntimeError("unsupported named row")

    monkeypatch.setattr(A, "_ROOT", "configured")
    monkeypatch.setattr(A, "import_module", lambda name: SimpleNamespace(attention=attention))
    with pytest.raises(RuntimeError, match="unsupported named row"):
        A.pallas_call("attention", word="named_row")


def test_catalog_does_not_load_optional_frameworks(monkeypatch):
    registry = SimpleNamespace(names=lambda: ("pallas", "trimul_xla"), sums=lambda name: {"name": name})

    def importing(name):
        assert name == "opt_core.kernels"
        return registry

    monkeypatch.setattr(A, "_ROOT", "configured")
    monkeypatch.setattr(A, "import_module", importing)
    assert set(A.catalog()) == {"pallas", "trimul_xla"}
