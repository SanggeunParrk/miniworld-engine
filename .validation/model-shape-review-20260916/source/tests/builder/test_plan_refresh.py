"""Never start expensive tuning against missing, stale or foreign architecture evidence."""
import json
import subprocess
from pathlib import Path

import pytest

from miniworld_engine.autotune import derive, plan


def setup_plan(monkeypatch, tmp_path):
    monkeypatch.setenv("MINIWORLD_PLAN_CACHE_DIR", str(tmp_path / "user-cache"))
    monkeypatch.setattr(plan, "ROOT", tmp_path / "package")
    monkeypatch.setattr(plan, "source_identity", lambda: "revision1")
    monkeypatch.setattr(derive, "REGISTRY_KERNEL", tmp_path / "shipped.csv")


def publish(path, arch="sm86"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("kernel,arch\nexample," + arch + "\n")
    path.with_suffix(".units.json").write_text(json.dumps({
        "arch": arch, "complete": True, "errors": [], "units": {}}))
    plan.stamp(path.with_suffix(".units.json"), path, plan.source_identity())


@pytest.mark.parametrize("state", ["missing", "stale", "foreign"])
def test_refresh_is_isolated_and_verified(monkeypatch, tmp_path, state):
    setup_plan(monkeypatch, tmp_path)
    if state != "missing":
        publish(derive.REGISTRY_KERNEL, "sm90" if state == "foreign" else "sm86")
        if state == "stale":
            derive.REGISTRY_KERNEL.write_text("changed")
    calls = []
    def run(command, *, env, check):
        calls.append(command)
        assert env["MINIWORLD_COMPILE_WRAP"] == "disable"
        publish(Path(command[command.index("--out") + 1]))
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(subprocess, "run", run)
    got = plan.ensure("sm_86", workers=2)
    assert plan.load("sm86", got)["complete"]
    assert derive.registry_path("sm86") == got
    assert plan.ensure("sm86") == got
    assert len(calls) == 1


def test_failed_refresh_does_not_publish(monkeypatch, tmp_path):
    setup_plan(monkeypatch, tmp_path)
    publish(derive.REGISTRY_KERNEL, "sm90")
    before = derive.REGISTRY_KERNEL.read_bytes()
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 1))
    with pytest.raises(ValueError, match="no tuning started"):
        plan.ensure("sm86")
    assert not plan.generated_registry("sm86").exists()
    assert derive.REGISTRY_KERNEL.read_bytes() == before


def test_architectures_and_revisions_do_not_overwrite_each_other(monkeypatch, tmp_path):
    setup_plan(monkeypatch, tmp_path)
    first = plan.generated_registry("sm86")
    assert first != plan.generated_registry("sm90")
    monkeypatch.setattr(plan, "source_identity", lambda: "revision2")
    assert first != plan.generated_registry("sm86")


def test_build_stops_before_module_or_driver_work_if_plan_fails(monkeypatch):
    from miniworld_engine import cli
    from miniworld_engine.autotune import builder
    monkeypatch.setattr(builder, "device_sm", lambda: "sm_86")
    monkeypatch.setattr(cli, "apply_config_dir", lambda *a: 0)
    def fail(*a, **k):
        raise ValueError("cannot derive")
    monkeypatch.setattr(plan, "ensure", fail)
    def forbidden(*a, **k):
        pytest.fail("work started before verifying the plan")
    monkeypatch.setattr(builder, "cases", forbidden)
    monkeypatch.setattr(builder, "op_units", forbidden)
    monkeypatch.setattr(builder, "build_all", forbidden)
    assert cli.main(["build", "all"]) == 1


@pytest.mark.parametrize("location", ["override", "xdg", "home", "relative_xdg"])
def test_generated_plans_use_user_writable_storage(monkeypatch, tmp_path, location):
    monkeypatch.setattr(plan, "ROOT", tmp_path / "read-only-site-packages")
    monkeypatch.setattr(plan, "source_identity", lambda: "revision1")
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    monkeypatch.delenv("MINIWORLD_PLAN_CACHE_DIR", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    expected = tmp_path / "home" / ".cache" / "miniworld-engine" / "plans"
    if location == "override":
        expected = tmp_path / "explicit-plans"
        monkeypatch.setenv("MINIWORLD_PLAN_CACHE_DIR", str(expected))
    elif location == "xdg":
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
        expected = tmp_path / "xdg" / "miniworld-engine" / "plans"
    elif location == "relative_xdg":
        monkeypatch.setenv("XDG_CACHE_HOME", "relative-path-is-invalid")
    assert plan.generated_registry("sm_86") == (
        expected / "sm86" / "revision1" / "registry_kernel.csv")


def test_packaged_verified_plan_needs_no_writable_install(monkeypatch, tmp_path):
    setup_plan(monkeypatch, tmp_path)
    packaged = plan.packaged_registry("sm86")
    publish(packaged)
    # An interrupted user-cache publication must not shadow valid packaged evidence.
    generated = plan.generated_registry("sm86")
    generated.parent.mkdir(parents=True)
    generated.write_text("partial")
    def no_write(*args, **kwargs):
        pytest.fail("reading a packaged verified plan attempted to create a directory")
    monkeypatch.setattr(Path, "mkdir", no_write)
    assert plan.ensure("sm86") == packaged
    assert derive.registry_path("sm86") == packaged


def test_source_and_wheel_dispatch_identity_agree(monkeypatch, tmp_path):
    import shutil

    source = tmp_path / "source"
    paths = ("settings.py", "kernels/registry_module.csv", "kernels/registry.csv",
             "modules/example.py", "kernels/example/triton/main.py",
             "autotune/builder.py", "autotune/checkpoint_cases.py", "autotune/derive.py", "autotune/module_registry.py",
             "autotune/shape_key.py", "build/gpu_to_kernels/sm86.csv")
    for relative in paths:
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative)
    notes = source / "kernels" / "example" / "notes" / "v1" / "experiment.py"
    notes.parent.mkdir(parents=True)
    notes.write_text("local experimental implementation")
    wheel = tmp_path / "wheel"
    shutil.copytree(source, wheel, ignore=shutil.ignore_patterns("notes"))
    monkeypatch.setattr(plan, "ROOT", source)
    expected = plan.source_identity()
    notes.write_text("an unrelated historical edit")
    assert plan.source_identity() == expected
    monkeypatch.setattr(plan, "ROOT", wheel)
    assert plan.source_identity() == expected
    (wheel / "modules" / "example.py").write_text("changed runtime dispatch")
    assert plan.source_identity() != expected
