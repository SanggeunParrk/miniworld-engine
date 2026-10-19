"""A JIT build says when it starts and when it ends; a cache hit says nothing.

Every extension and cubin here is built with ``verbose=False``, so the first call after an install
or a source change could sit in nvcc for minutes printing nothing -- indistinguishable from a hang.
``kernels/_nvcc.building`` (raw cubins, the TriMul cubin cache) and the torch loaders
(``load_extension`` / ``load_inline_extension``) print one ``[miniworld-engine build]`` line when a
compile starts and one when it ends, and keep the list ``built_this_process()``. A run that prints
none of these lines reused every artifact, which is what a prebuild step checks.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
from settings_helpers import override_setting

from miniworld_engine.kernels import _nvcc

PREFIX = "[miniworld-engine build]"


@pytest.fixture(autouse=True)
def _fresh_counter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_nvcc, "_BUILT", [])


def test_a_build_prints_a_start_and_an_end_line(capsys: pytest.CaptureFixture[str]) -> None:
    with _nvcc.building("augattn_sm100/attn_fwd", "cubin"):
        pass
    with _nvcc.building("token_pair_init_sm100/ri4", "cubin"):
        pass
    lines = capsys.readouterr().err.splitlines()
    assert lines[0] == f"{PREFIX} compiling augattn_sm100/attn_fwd (cubin) ..."
    assert lines[1].startswith(f"{PREFIX} built augattn_sm100/attn_fwd in ")
    assert lines[1].endswith(" s; 1 built in this process")
    assert lines[3].endswith(" s; 2 built in this process")
    assert len(lines) == 4
    assert [b.name for b in _nvcc.built_this_process()] == [
        "augattn_sm100/attn_fwd", "token_pair_init_sm100/ri4"]
    assert all(b.kind == "cubin" for b in _nvcc.built_this_process())


def test_a_failed_build_says_so_and_is_not_counted(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(RuntimeError, match="nvcc"), _nvcc.building("x", "cubin"):
        raise RuntimeError("nvcc failed")
    err = capsys.readouterr().err
    assert f"{PREFIX} failed to build x after " in err
    assert _nvcc.built_this_process() == ()


def test_the_setting_silences_it(monkeypatch: pytest.MonkeyPatch,
                                 capsys: pytest.CaptureFixture[str]) -> None:
    override_setting(monkeypatch, "jit_build_log", False)
    with _nvcc.building("x", "cubin"):
        pass
    assert capsys.readouterr().err == ""
    assert len(_nvcc.built_this_process()) == 1      # still recorded


def _fake_build(build: Path, name: str, write: bool):
    """A stand-in for torch's load: (re)links ``<build>/<name>.so`` when ``write``."""
    def call():
        if write:
            so = build / f"{name}.so"
            so.write_bytes(b"x")
            st = so.stat()
            os.utime(so, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))  # a distinct mtime on coarse filesystems
        return "module"
    return call


def test_a_torch_load_that_links_logs_both_lines(tmp_path: Path,
                                                 capsys: pytest.CaptureFixture[str]) -> None:
    """The library is missing: the compile is known up front."""
    out = _nvcc._guarded_load("ext_a", "torch extension", str(tmp_path),
                              _fake_build(tmp_path, "ext_a", write=True))
    assert out == "module"
    lines = capsys.readouterr().err.splitlines()
    assert lines[0] == f"{PREFIX} compiling ext_a (torch extension) ..."
    assert lines[1].startswith(f"{PREFIX} built ext_a in ")
    assert len(lines) == 2


def test_a_cached_torch_load_prints_nothing(tmp_path: Path,
                                            capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "ext_b.so").write_bytes(b"x")
    _nvcc._guarded_load("ext_b", "torch extension", str(tmp_path),
                        _fake_build(tmp_path, "ext_b", write=False))
    assert capsys.readouterr().err == ""
    assert _nvcc.built_this_process() == ()


def test_a_relink_over_an_existing_library_logs_the_build(tmp_path: Path,
                                                          capsys: pytest.CaptureFixture[str]) -> None:
    """A source edit under an existing library: torch's ninja rebuilds, and that is a build."""
    (tmp_path / "ext_c.so").write_bytes(b"x")
    _nvcc._guarded_load("ext_c", "torch extension", str(tmp_path),
                        _fake_build(tmp_path, "ext_c", write=True))
    lines = capsys.readouterr().err.splitlines()
    assert len(lines) == 1
    assert lines[0].startswith(f"{PREFIX} built ext_c in ")


def _cached_build(tmp_path: Path, name: str, source_age: float) -> tuple[Path, Path]:
    """A built extension in ``tmp_path / 'build'``: its library, a build.ninja naming ``src.cu``, and that
    source ``source_age`` seconds older than the library (negative: newer)."""
    build = tmp_path / "build"
    build.mkdir()
    src = tmp_path / "src.cu"
    src.write_text("// kernel")
    so = build / f"{name}.so"
    so.write_bytes(b"x")
    (build / "build.ninja").write_text(f"build src.cuda.o: cuda_compile {src}\n")
    t = so.stat().st_mtime
    os.utime(src, (t - source_age, t - source_age))
    return build, src


def test_an_up_to_date_extension_with_known_sources_prints_nothing(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    build, src = _cached_build(tmp_path, "ext_e", source_age=100.0)
    _nvcc._guarded_load("ext_e", "torch extension", str(build),
                        _fake_build(build, "ext_e", write=False), sources=[str(src)])
    assert capsys.readouterr().err == ""


def test_a_source_newer_than_the_library_is_announced(tmp_path: Path,
                                                      capsys: pytest.CaptureFixture[str]) -> None:
    """An edited source (or a pin bump) rebuilds an existing library: say so before the minutes in nvcc."""
    build, src = _cached_build(tmp_path, "ext_f", source_age=-100.0)
    _nvcc._guarded_load("ext_f", "torch extension", str(build),
                        _fake_build(build, "ext_f", write=True), sources=[str(src)])
    lines = capsys.readouterr().err.splitlines()
    assert lines[0] == f"{PREFIX} compiling ext_f (torch extension) ..."
    assert lines[1].startswith(f"{PREFIX} built ext_f in ")


def test_a_cache_built_by_another_checkout_is_announced(tmp_path: Path,
                                                        capsys: pytest.CaptureFixture[str]) -> None:
    """Extension names do not depend on the checkout: a build.ninja naming other source paths rebuilds."""
    build, _ = _cached_build(tmp_path, "ext_g", source_age=100.0)
    other = tmp_path / "other_checkout" / "src.cu"
    other.parent.mkdir()
    other.write_text("// kernel")
    os.utime(other, (0, 0))
    _nvcc._guarded_load("ext_g", "torch extension", str(build),
                        _fake_build(build, "ext_g", write=False), sources=[str(other)])
    lines = capsys.readouterr().err.splitlines()
    assert lines == [f"{PREFIX} compiling ext_g (torch extension) ...", f"{PREFIX} ext_g needed no rebuild"]


def test_a_failed_torch_load_drops_its_lock(tmp_path: Path) -> None:
    def call():
        (tmp_path / "lock").touch()
        raise RuntimeError("Error building extension")
    with pytest.raises(RuntimeError):
        _nvcc._guarded_load("ext_d", "torch extension", str(tmp_path), call)
    assert not (tmp_path / "lock").exists()


def test_cache_roots_lists_the_existing_directories(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    jit = tmp_path / "jit"
    torch_ext = tmp_path / "torch_ext"
    jit.mkdir()
    override_setting(monkeypatch, "engine_jit_root", str(jit))
    monkeypatch.setenv("TORCH_EXTENSIONS_DIR", str(torch_ext))
    assert _nvcc.cache_roots() == [jit]             # torch's does not exist yet
    torch_ext.mkdir()
    assert _nvcc.cache_roots() == [jit, torch_ext]
    monkeypatch.setenv("TORCH_EXTENSIONS_DIR", str(jit))
    assert _nvcc.cache_roots() == [jit]             # one directory is listed once


def test_every_raw_nvcc_build_is_logged() -> None:
    """A cubin cache that runs nvcc outside ``building`` is a silent first call again."""
    import ast

    root = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file()) / "src" / "miniworld_engine" / "kernels"
    offenders = []
    for path in sorted(root.rglob("*.py")):
        if path.name == "_nvcc.py" or "/notes/" in path.as_posix() or path.name.startswith("build_"):
            continue
        text = path.read_text()
        tree = ast.parse(text)
        for fn in ast.walk(tree):
            if not isinstance(fn, ast.FunctionDef):
                continue
            src = ast.get_source_segment(text, fn) or ""
            if ("-cubin" in src or "--cubin" in src or "_NVCC_FLAGS" in src) and "subprocess.run(" in src \
                    and "building(" not in src and "_probe" not in fn.name and "_host_for" not in fn.name:
                offenders.append(f"{path.relative_to(root)}:{fn.name}")
    assert not offenders, f"nvcc runs here without `_nvcc.building`: {offenders}"

