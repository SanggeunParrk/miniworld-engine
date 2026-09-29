"""A read-only install fails before spending GPU time, without changing cache semantics."""
import os

import pytest

from miniworld_engine.autotune import cache, preflight


def test_writable_cache_is_only_checked(monkeypatch, tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    evidence = root / "sentinel.json"
    evidence.write_bytes(b"existing cache")
    monkeypatch.setattr(cache, "_CACHE_ROOT", root)
    preflight.cache_write_access()
    assert list(root.iterdir()) == [evidence]
    assert evidence.read_bytes() == b"existing cache"


def test_missing_cache_uses_parent_permissions_without_creating_it(monkeypatch, tmp_path):
    root = tmp_path / "absent" / "data"
    monkeypatch.setattr(cache, "_CACHE_ROOT", root)
    preflight.cache_write_access()
    assert not root.parent.exists()
    monkeypatch.setattr(os, "access", lambda path, mode: False)
    with pytest.raises(ValueError, match="writable MiniWorld installation or source checkout"):
        preflight.cache_write_access()
    assert not root.parent.exists()


@pytest.mark.parametrize("target", ["all", "transition"])
def test_cli_rejects_read_only_cache_before_config_plan_or_gpu_work(monkeypatch, tmp_path, capsys, target):
    from miniworld_engine import cli
    from miniworld_engine.autotune import builder, plan

    root = tmp_path / "data"
    root.mkdir()
    monkeypatch.setattr(cache, "_CACHE_ROOT", root)
    monkeypatch.setattr(os, "access", lambda path, mode: False)
    def forbidden(*args, **kwargs):
        pytest.fail("build work started despite a read-only cache destination")
    monkeypatch.setattr(cli, "apply_config_dir", forbidden)
    monkeypatch.setattr(builder, "device_sm", forbidden)
    monkeypatch.setattr(builder, "build_all", forbidden)
    monkeypatch.setattr(plan, "ensure", forbidden)
    assert cli.main(["build", target]) == 1
    error = capsys.readouterr().err
    assert str(root) in error
    assert "writable MiniWorld installation or source checkout" in error
    assert not list(root.iterdir())
