"""Content keys of compiled kernel artifacts (kernels/_codegraph.py) and the prebuilt-bundle lookup (kernels/prebuilt.py).

A key must be the same on every machine and checkout (the bundle builder and a user process compute it independently),
must not move for an edit the compiler cannot see (a comment, whitespace, an unrelated header next to the source), and
must move for every edit it can (a reached header, a define, a flag). docs/kernels/prebuilt-kernels.md.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from settings_helpers import override_setting

from miniworld_engine.kernels import _codegraph as cg
from miniworld_engine.kernels import _nvcc, prebuilt

VERSIONS = {"torch": "2.13.0+cu129", "cuda": "12.9"}


# --------------------------------------------------------------------------------------------------- normalization
def test_comments_whitespace_and_line_breaks_do_not_change_the_digest() -> None:
    a = '#include "x.cuh"\n__global__ void k(float* p) { p[0] = 1.0f + 2.0f; }\n'
    b = ('// a new comment\n#  include "x.cuh"   /* trailing */\n\n'
         '__global__ void k(float *p)\n{\n    p[0] = 1.0f+2.0f;  // inline\n}\n')
    assert cg.parse_text(a).digest == cg.parse_text(b).digest
    assert cg.parse_text(a).includes == cg.parse_text(b).includes == (("x.cuh", True),)


def test_code_and_literal_edits_change_the_digest() -> None:
    base = 'void k(int* p) { p[0] = 1; const char* s = "a  b"; }'
    assert cg.parse_text(base).digest != cg.parse_text(base.replace("= 1", "= 2")).digest
    assert cg.parse_text(base).digest != cg.parse_text(base.replace('"a  b"', '"a b"')).digest  # literals kept
    assert cg.parse_text("x = a - -b;").digest != cg.parse_text("x = a --b;").digest


def test_if0_blocks_are_dropped_and_their_includes_not_followed() -> None:
    a = '#if 0\n#include "gone.cuh"\nint old;\n#endif\nint x;\n'
    assert cg.parse_text(a).digest == cg.parse_text("int x;\n").digest
    assert cg.parse_text(a).includes == ()
    kept = cg.parse_text('#if 0\nint old;\n#else\n#include "y.cuh"\n#endif\nint x;\n')
    assert kept.includes == (("y.cuh", True),)


def test_function_like_and_object_like_macros_differ() -> None:
    assert cg.parse_text("#define F(x) x\n").digest != cg.parse_text("#define F (x) x\n").digest


# --------------------------------------------------------------------------------------------------- graph
def _tree(root: Path) -> Path:
    """A tiny engine-like package: kern/a.cu includes "local.cuh" (next to it) and <shared.cuh> (-I inc1, then inc2)."""
    pkg = root / "pkg"
    (pkg / "kern").mkdir(parents=True)
    (pkg / "inc1").mkdir()
    (pkg / "inc2").mkdir()
    (pkg / "kern" / "a.cu").write_text('#include "local.cuh"\n#include <shared.cuh>\n#include <cuda_bf16.h>\n'
                                       "__global__ void a() {}\n")
    (pkg / "kern" / "local.cuh").write_text('#include "../inc2/deep.cuh"\nint local;\n')
    (pkg / "kern" / "unrelated.cuh").write_text("int unrelated;\n")
    (pkg / "inc1" / "shared.cuh").write_text("int shared_first;\n")
    (pkg / "inc2" / "shared.cuh").write_text("int shared_second;\n")
    (pkg / "inc2" / "deep.cuh").write_text("int deep;\n")
    return pkg


def _key(pkg: Path, flags: list[str] | None = None) -> tuple[str, cg.Graph]:
    ctx = cg.Context(str(pkg), None, "/nonexistent-prefix", None)
    flags = flags if flags is not None else ["-O3", f"-I{pkg / 'inc1'}", f"-I{pkg / 'inc2'}", "-DX=1"]
    graph = cg.closure([pkg / "kern" / "a.cu"], cg.include_dirs_of(flags), ctx)
    material = cg.key_material("cubin", "fam/a", graph, flags, arch="sm90", ctx=ctx, versions=VERSIONS)
    return cg.key_of(material), graph


def test_the_closure_follows_the_compilers_search_order(tmp_path: Path) -> None:
    pkg = _tree(tmp_path)
    _, graph = _key(pkg)
    labels = [n.label for n in graph.nodes]
    assert labels == ["pkg:inc1/shared.cuh", "pkg:inc2/deep.cuh", "pkg:kern/a.cu", "pkg:kern/local.cuh"]
    assert graph.external == ("cuda_bf16.h",)          # not found anywhere: represented by the CUDA version
    _, swapped = _key(pkg, ["-O3", f"-I{pkg / 'inc2'}", f"-I{pkg / 'inc1'}", "-DX=1"])
    assert "pkg:inc2/shared.cuh" in [n.label for n in swapped.nodes]   # -I order decides


def test_a_quoted_include_looks_next_to_the_including_file_first(tmp_path: Path) -> None:
    pkg = _tree(tmp_path)
    (pkg / "kern" / "shared.cuh").write_text("int shared_local;\n")
    _, graph = _key(pkg)
    assert "pkg:kern/shared.cuh" not in [n.label for n in graph.nodes]  # <shared.cuh> is an angle include
    (pkg / "kern" / "a.cu").write_text('#include "shared.cuh"\n')
    _, graph = _key(pkg)
    assert [n.label for n in graph.nodes] == ["pkg:kern/a.cu", "pkg:kern/shared.cuh"]


def test_unreached_and_cosmetic_edits_keep_the_key(tmp_path: Path) -> None:
    pkg = _tree(tmp_path)
    key, _ = _key(pkg)
    (pkg / "kern" / "unrelated.cuh").write_text("int changed;\n")      # same directory, never included
    (pkg / "inc2" / "shared.cuh").write_text("int shadowed_changed;\n")  # shadowed by inc1
    (pkg / "kern" / "local.cuh").write_text('/* doc */\n#include "../inc2/deep.cuh"\n\nint   local ;  // x\n')
    assert _key(pkg)[0] == key


@pytest.mark.parametrize("edit", ["header", "define", "flag", "arch"])
def test_reached_header_define_flag_and_arch_edits_change_the_key(tmp_path: Path, edit: str) -> None:
    pkg = _tree(tmp_path)
    key, _ = _key(pkg)
    flags = ["-O3", f"-I{pkg / 'inc1'}", f"-I{pkg / 'inc2'}", "-DX=1"]
    if edit == "header":
        (pkg / "inc2" / "deep.cuh").write_text("int deep = 1;\n")
    elif edit == "define":
        flags[-1] = "-DX=2"
    elif edit == "flag":
        flags[0] = "-O2"
    if edit == "arch":
        ctx = cg.Context(str(pkg), None, "/nonexistent-prefix", None)
        graph = cg.closure([pkg / "kern" / "a.cu"], cg.include_dirs_of(flags), ctx)
        other = cg.key_of(cg.key_material("cubin", "fam/a", graph, flags, arch="sm100", ctx=ctx, versions=VERSIONS))
        assert other != key
        return
    assert _key(pkg, flags)[0] != key


def test_keys_are_identical_across_checkouts_at_different_paths(tmp_path: Path) -> None:
    a = _tree(tmp_path / "checkout_a")
    b = tmp_path / "somewhere" / "else" / "pkg"
    shutil.copytree(a, b)
    assert _key(a)[0] == _key(b)[0]


def test_host_compiler_and_machine_paths_are_not_part_of_the_key(tmp_path: Path) -> None:
    ctx = cg.Context(str(tmp_path / "pkg"), str(tmp_path / "jit"), "/opt/env", "/opt/env")
    norm = cg.normalize_flags([cg.HOST_MARKER, "-ccbin", "/opt/gcc/bin/g++", f"-I{tmp_path / 'pkg' / 'k'}",
                               f"-I{tmp_path / 'jit' / 'trimul_h100' / 'hdr_0123'}", "-I/opt/env/include",
                               "-DROOT=/opt/env/lib"], ctx)
    assert norm == ["-I<pkg:k>", "-I<gen>", "-I<cuda:include>", "-DROOT=<cuda>/lib"]


def test_a_cutlass_directory_is_keyed_by_its_version_present_or_not(tmp_path: Path) -> None:
    ctx = cg.Context(str(tmp_path / "pkg"), None, "/opt/env", None)
    cut = tmp_path / "cutlass-4.2" / "include"
    (cut / "cutlass").mkdir(parents=True)
    (cut / "cutlass" / "version.h").write_text("#define CUTLASS_MAJOR 4\n#define CUTLASS_MINOR 2\n#define CUTLASS_PATCH 0\n")
    assert cg.classify_dir(str(cut), ctx) == ("cutlass@4.2.0", True)
    assert cg.classify_dir("/not/here/cutlass/include", ctx) == (f"cutlass@{cg.CUTLASS_PINNED}", True)


def test_generated_headers_are_read_from_the_overlay_and_not_written(tmp_path: Path) -> None:
    gen = tmp_path / "jit" / "trimul_h100" / "headers_abc"
    cg.publish(gen / "g.cuh", "int generated;\n")
    ctx = cg.Context(str(tmp_path / "pkg"), str(tmp_path / "jit"), "/opt/env", None)
    root = cg.TextRoot("text:t", '#include "g.cuh"\n', str(tmp_path / "jit" / "trimul_h100"))
    graph = cg.closure([root], [str(gen)], ctx)
    assert [n.label for n in graph.nodes] == ["inc:g.cuh", "text:t"]
    assert not gen.exists()
    # the directory's name (a digest of unrelated sources) is not part of the key
    other = tmp_path / "jit" / "trimul_h100" / "headers_def"
    cg.publish(other / "g.cuh", "int generated;\n")
    graph2 = cg.closure([root], [str(other)], ctx)
    flags = [f"-I{gen}"]
    k1 = cg.key_of(cg.key_material("cubin", "t", graph, flags, arch="sm90", ctx=ctx, versions=VERSIONS))
    k2 = cg.key_of(cg.key_material("cubin", "t", graph2, [f"-I{other}"], arch="sm90", ctx=ctx, versions=VERSIONS))
    assert k1 == k2
    cg.materialize(graph.paths)
    assert (gen / "g.cuh").read_text() == "int generated;\n"


def test_gencode_and_host_placeholders_expand_only_for_a_build(monkeypatch: pytest.MonkeyPatch) -> None:
    flags = [*_nvcc.host_flags(), "-O3", *_nvcc.gencodes("80", "90a", "100", ptx=("100",))]
    assert all("/" not in f for f in flags)
    monkeypatch.setattr(_nvcc, "real_host_flags", lambda: ["-ccbin", "/x/g++"])
    monkeypatch.setattr(_nvcc, "real_gencodes", lambda *a, ptx=(): [f"G{x}" for x in a] + [f"P{x}" for x in ptx])
    assert _nvcc.expand_flags(flags) == ["-ccbin", "/x/g++", "-O3", "G80", "G90a", "G100", "P100"]
    monkeypatch.setenv(_nvcc.BUNDLE_ARCH_ENV, "sm90")
    assert _nvcc.expand_flags(flags) == ["-ccbin", "/x/g++", "-O3", "G90a"]


# --------------------------------------------------------------------------------------------------- lookup
@pytest.fixture
def jit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "jit"
    override_setting(monkeypatch, "engine_jit_root", str(root))
    monkeypatch.setenv("MINIWORLD_ENGINE_TARGET_ARCH", "sm90")
    monkeypatch.delenv(prebuilt.BUNDLE_ENV, raising=False)
    monkeypatch.delenv(prebuilt.RECORD_ENV, raising=False)
    monkeypatch.setattr(prebuilt, "PREBUILT_DIR", tmp_path / "no_prebuilt")
    prebuilt.bundles.cache_clear()
    yield root
    prebuilt.bundles.cache_clear()


def _no_compiler(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse():
        raise AssertionError("a compiler was asked for")
    monkeypatch.setattr(_nvcc, "TOOLCHAINS", dict.fromkeys(_nvcc.TOOLCHAINS, refuse))


def _bundle(tmp_path: Path, entries: dict) -> Path:
    root = tmp_path / "bundle"
    root.mkdir()
    v = cg._versions()
    manifest = {"format": prebuilt.MANIFEST_FORMAT, "key_scheme": cg.KEY_SCHEME, "arch": "sm90",
                "engine": prebuilt.engine_version(), "torch": v["torch"], "cuda": v["cuda"], "python": cg.python_abi(),
                "artifacts": {}}
    for key, (name, data) in entries.items():
        rel = f"artifacts/cubin/{key[:16]}.cubin"
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(data)
        manifest["artifacts"][key] = {"kind": "cubin", "name": name, "path": rel}
    (root / prebuilt.MANIFEST).write_text(json.dumps(manifest))
    return root


def _source(tmp_path: Path) -> Path:
    src = tmp_path / "src" / "k.cu"
    src.parent.mkdir()
    src.write_text("__global__ void k() {}\n")
    return src


def test_a_bundle_hit_loads_without_a_compiler_and_writes_nothing(tmp_path: Path, jit: Path,
                                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    src = _source(tmp_path)
    flags = ["-O3", "-arch=sm_90a", "-cubin"]
    graph = cg.closure([str(src)], [])
    key = cg.key_of(cg.key_material("cubin", "fam/k", graph, flags))
    root = _bundle(tmp_path, {key: ("fam/k", b"CUBIN")})
    monkeypatch.setenv(prebuilt.BUNDLE_ENV, str(root))
    prebuilt.bundles.cache_clear()
    _no_compiler(monkeypatch)
    path = _nvcc.cubin("fam/k", flags, source=src)
    assert path.read_bytes() == b"CUBIN"
    assert path.resolve().is_relative_to(root.resolve())
    assert not jit.exists()


def test_a_miss_with_jit_fallback_off_raises(tmp_path: Path, jit: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _no_compiler(monkeypatch)
    override_setting(monkeypatch, "jit_fallback", False)
    with pytest.raises(prebuilt.PrebuiltMissing, match="fam/k"):
        _nvcc.cubin("fam/k", ["-O3"], source=_source(tmp_path))


def test_a_bundle_for_another_torch_or_arch_is_ignored(tmp_path: Path, jit: Path,
                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    root = _bundle(tmp_path, {})
    manifest = json.loads((root / prebuilt.MANIFEST).read_text())
    assert prebuilt.mismatch(manifest) is None
    assert "torch" in prebuilt.mismatch({**manifest, "torch": "0.0"})
    assert "arch" in prebuilt.mismatch({**manifest, "arch": "sm100"})
    assert prebuilt.mismatch({**manifest, "engine": "0.0.1"}) is None  # keys cover the code; the version is not compared


def test_a_local_cache_hit_is_keyed_by_content(tmp_path: Path, jit: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    src = _source(tmp_path)
    calls = []

    def fake_toolchain():
        calls.append(1)
        return "/bin/true-nvcc", []

    def fake_run(cmd, **kw):
        Path(cmd[cmd.index("-o") + 1]).write_bytes(b"BUILT")
        return type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(_nvcc, "TOOLCHAINS", {"torch": fake_toolchain})
    monkeypatch.setattr(_nvcc.subprocess, "run", fake_run)
    first = _nvcc.cubin("fam/k", ["-O3"], source=src)
    assert _nvcc.cubin("fam/k", ["-O3"], source=src) == first
    assert len(calls) == 1
    src.write_text("// a comment only\n__global__ void k() {}\n")
    assert _nvcc.cubin("fam/k", ["-O3"], source=src) == first
    assert len(calls) == 1
    src.write_text("__global__ void k() { int x; }\n")
    assert _nvcc.cubin("fam/k", ["-O3"], source=src) != first
    assert len(calls) == 2


def _record(tmp_path: Path, name: str, key: str, data: bytes) -> dict:
    art = tmp_path / "cache" / f"{name}-{key[:8]}.cubin"
    art.parent.mkdir(parents=True, exist_ok=True)
    art.write_bytes(data)
    art.with_suffix(".ptxas.log").write_text("ptxas info")
    return {"kind": "cubin", "name": f"fam/{name}", "key": key, "path": str(art), "origin": "built",
            "material": {"nodes": [], "python": "-", "flags": []}, "sidecars": [".ptxas.log"]}


def test_a_refresh_reuses_unchanged_keys_adds_new_ones_and_drops_unreached(tmp_path: Path, jit: Path) -> None:
    from miniworld_engine.build import kernel_bundle

    root = tmp_path / "_prebuilt"
    a, b = "a" * 64, "b" * 64
    rec_a = _record(tmp_path, "a", a, b"A" * 10)
    target = kernel_bundle.update_prebuilt({a: rec_a}, root, "sm90")
    manifest = json.loads((target / "manifest.json").read_text())
    entry = manifest["artifacts"][a]
    assert entry["size"] == 10
    assert entry["sidecars"] == [".ptxas.log"]
    assert (target / entry["path"]).with_suffix(".ptxas.log").is_file()
    assert prebuilt.mismatch(manifest, arch="sm90") is None
    # the next run finds a in the bundle (a prebuilt hit, recorded at its bundle path) and builds b
    reused = dict(rec_a, path=str(target / entry["path"]), origin="prebuilt")
    kernel_bundle.update_prebuilt({a: reused, b: _record(tmp_path, "b", b, b"B")}, root, "sm90")
    manifest = json.loads((target / "manifest.json").read_text())
    assert set(manifest["artifacts"]) == {a, b}
    assert (target / manifest["artifacts"][a]["path"]).read_bytes() == b"A" * 10
    # a run that no longer reaches a drops it (and its file)
    kernel_bundle.update_prebuilt({b: dict(_record(tmp_path, "b", b, b"B"))}, root, "sm90")
    manifest = json.loads((target / "manifest.json").read_text())
    assert set(manifest["artifacts"]) == {b}
    assert not (target / entry["path"]).exists()


def test_a_mathdx_artifact_is_never_bundled(tmp_path: Path, jit: Path) -> None:
    from miniworld_engine.build import kernel_bundle

    rec = _record(tmp_path, "m", "c" * 64, b"M")
    rec["material"]["flags"] = ["-I<mathdx>"]
    target = kernel_bundle.update_prebuilt({"c" * 64: rec}, tmp_path / "_prebuilt", "sm90")
    assert json.loads((target / "manifest.json").read_text())["artifacts"] == {}


def test_the_binary_check_names_what_manylinux_does_not_allow() -> None:
    from miniworld_engine.build import kernel_bundle

    assert kernel_bundle.ALLOWED_LIBS.match("libc10.so")
    assert kernel_bundle.ALLOWED_LIBS.match("libcudart.so.12")
    assert kernel_bundle.ALLOWED_LIBS.match("libstdc++.so.6")
    assert not kernel_bundle.ALLOWED_LIBS.match("libcutlass.so")
    assert kernel_bundle._version_tuple("3.4.30") > kernel_bundle.MAX_SYMBOL_VERSIONS["GLIBCXX"]


def test_code_graph_affected_lists_the_artifacts_a_file_reaches(tmp_path: Path, jit: Path,
                                                                 capsys: pytest.CaptureFixture[str]) -> None:
    from miniworld_engine.cli import main

    header = cg.PKG_ROOT / "kernels" / "augmented_attention" / "cuda" / "sm100" / "sm100.cuh"
    current = cg.parse_file(str(header)).digest
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"artifacts": {
        "a" * 64: {"name": "augattn_sm100/attn_fwd2", "material": {"nodes": [["pkg:kernels/augmented_attention/cuda/sm100/sm100.cuh", current]]}},
        "b" * 64: {"name": "augattn_sm100/old", "material": {"nodes": [["pkg:kernels/augmented_attention/cuda/sm100/sm100.cuh", "0" * 64]]}},
        "c" * 64: {"name": "other/unrelated", "material": {"nodes": [["pkg:kernels/other.cu", "1" * 64]]}},
    }}))
    assert main(["dev", "code-graph", "--affected", str(header), "--manifest", str(manifest)]) == 0
    out = capsys.readouterr().out
    assert "reached by 2 of 3" in out
    assert "augattn_sm100/old" in out
    assert "other/unrelated" not in out
    assert out.count("rebuilds") == 1


def test_toolkit_and_cutlass_stand_ins_key_like_the_real_directories(tmp_path: Path) -> None:
    """A process that only loads prebuilt artifacts may have no CUDA toolkit and no CUTLASS checkout: the include paths
    it computes then are stand-ins, and they must key exactly like the real ones of the build machine."""
    real = cg.Context(str(tmp_path / "pkg"), None, "/opt/env", "/opt/env")
    bare = cg.Context(str(tmp_path / "pkg"), None, "/usr", None)
    assert cg.classify_dir("/opt/env/targets/x86_64-linux/include", real) == \
        cg.classify_dir(f"{cg.CUDA_HOME_STANDIN}/targets/x86_64-linux/include", bare)
    assert cg.classify_dir(str(_nvcc.CUTLASS_STANDIN), bare)[0] == f"cutlass@{cg.CUTLASS_PINNED}"


def test_each_cuda_build_gets_its_own_directory_and_only_the_matching_one_is_used(tmp_path: Path, jit: Path,
                                                                                   capsys: pytest.CaptureFixture[str]) -> None:
    from miniworld_engine.build import kernel_bundle

    assert kernel_bundle.variant_name("sm90", "12.9") == "sm90-cu129"
    v = cg._versions()
    base = {"format": prebuilt.MANIFEST_FORMAT, "key_scheme": cg.KEY_SCHEME, "arch": "sm90", "artifacts": {}}
    for name, cuda in (("sm90-cuXX", "0.0"), ("sm90-mine", v["cuda"])):
        (tmp_path / "pb" / name).mkdir(parents=True)
        (tmp_path / "pb" / name / prebuilt.MANIFEST).write_text(json.dumps({**base, "torch": v["torch"], "cuda": cuda}))
    import os

    os.environ[prebuilt.BUNDLE_ENV] = os.pathsep.join(str(p) for p in sorted((tmp_path / "pb").iterdir()))
    try:
        prebuilt.bundles.cache_clear()
        assert [b.root.name for b in prebuilt.bundles()] == ["sm90-mine"]
        assert "no prebuilt kernels" not in capsys.readouterr().err  # the other CUDA build is skipped silently
    finally:
        del os.environ[prebuilt.BUNDLE_ENV]
        prebuilt.bundles.cache_clear()


def test_a_replay_compiles_its_own_checkout_and_jit_root(tmp_path: Path) -> None:
    """Records made in one checkout replay in another (a clean release tree): the recipe's source / include paths and
    generated-file paths must move to the replaying tree, or it compiles the recording tree's code under other keys."""
    from miniworld_engine.build import kernel_bundle

    old = "/old/checkout/src/miniworld_engine"
    recipe = {"fn": "cubin", "source": f"{old}/kernels/x/k.cu", "flags": [f"-I{old}/kernels/x", "-I/old/jit/trimul_h100/h"],
              "overlay": {"/old/jit/trimul_h100/h/a.cuh": "int a;"}, "jit": "/old/jit"}
    new = kernel_bundle.rebase_recipe(recipe, None, "/new/jit", package=tmp_path / "miniworld_engine")
    assert new["source"] == f"{tmp_path}/miniworld_engine/kernels/x/k.cu"
    assert new["flags"] == [f"-I{tmp_path}/miniworld_engine/kernels/x", "-I/new/jit/trimul_h100/h"]
    assert list(new["overlay"]) == ["/new/jit/trimul_h100/h/a.cuh"]
    text_only = {"fn": "cubin", "source": None, "text": '#include "k.cuh"', "flags": [f"-I{old}/kernels/x"], "jit": "/old/jit"}
    assert kernel_bundle.rebase_recipe(text_only, None, "/new/jit", package=tmp_path / "miniworld_engine")["flags"] == \
        [f"-I{tmp_path}/miniworld_engine/kernels/x"]


def test_a_replay_links_with_the_flags_the_key_was_computed_from() -> None:
    """torch's load appends its link flags to `extra_ldflags` in place; a record holding that list keyed its replay
    differently from the artifact a model asks for (2.3.1: the SWA atom DiT and the token DiT's owned GEMM)."""
    from miniworld_engine.build import kernel_bundle

    polluted = ["-lcublas", "-L/env/torch/lib", "-lc10", "-ltorch", "-L/cuda/lib", "-lcudart"]
    entry = {"recipe": {"fn": "ext", "name": "k", "kwargs": {"extra_cflags": ["-O3"], "extra_ldflags": polluted}},
             "material": {"flags": {"ldflags": ["-lcublas"]}}}
    assert kernel_bundle.keyed_recipe(entry)["kwargs"] == {"extra_cflags": ["-O3"], "extra_ldflags": ["-lcublas"]}
    assert entry["recipe"]["kwargs"]["extra_ldflags"] is polluted  # the record itself is left alone
    cubin = {"recipe": {"fn": "cubin", "name": "c"}, "material": {}}
    assert kernel_bundle.keyed_recipe(cubin) is cubin["recipe"]
