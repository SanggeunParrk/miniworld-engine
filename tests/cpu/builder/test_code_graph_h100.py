"""The code graph, made precise for the H100 (sm90): symbol-level test selection, string-dispatch edges, the ``.cu`` card rule, and the Triton-cache
policy. CPU only.

The test selection must never under-select; ``test_historic_fix_is_selected`` replays real fix commits (it needs the git history and skips when a
commit is not available, e.g. in a wheel or sdist checkout).
"""
from __future__ import annotations

import subprocess
import textwrap
from pathlib import Path

import pytest

from miniworld_engine.build import cards as cm
from miniworld_engine.build import repo_graph as rg
from miniworld_engine.build import symbols as sy

SRC = "src/miniworld_engine"
PY310 = "tests/cpu/layout/test_the_source_parses_on_python_310.py"
KERNEL = f"{SRC}/kernels/adaln/triton/inference.py"
TRITON_CACHE = f"{SRC}/autotune/data/adaln_gemm_gate_triton/NVIDIA H100 80GB HBM3 (sm90).json"


@pytest.fixture(scope="module")
def graph() -> rg.Graph:
    return rg.build()


# ------------------------------------------------------------------------------------------------------ symbols (synthetic files)
def _mini(files: dict[str, str]) -> rg.Graph:
    """A graph of synthetic python files: nodes, imports as bindings, no cards."""
    g = rg.Graph()
    g.nodes.update({p: rg.kind_of(p) for p in files})
    resolve = rg._Resolver(list(files))
    g.resolve = resolve
    for p, text in files.items():
        g.syms[p] = sy.analyze(p, text, resolve)
    for p, fs in g.syms.items():
        lazy_of: dict[str, bool] = {}
        for b in fs.binds:
            if b.target:
                lazy_of[b.target] = lazy_of.get(b.target, True) and b.lazy
                g.pair_binds[(p, b.target)].append(b)
        for dep, lazy in lazy_of.items():
            g.edge(p, dep, "lazy" if lazy else "import")
    return g


A, B, C = f"{SRC}/a.py", f"{SRC}/b.py", f"{SRC}/c.py"
T1, T2, T3 = "tests/cpu/test_one.py", "tests/cpu/test_two.py", "tests/cpu/test_three.py"
MINI = {
    A: textwrap.dedent("""
        LIMIT = 3

        def helper(x):
            return x + LIMIT

        def f(x):
            return helper(x)

        def g(x):
            return x
    """),
    B: "from miniworld_engine.a import f\nfrom miniworld_engine import a\n\ndef use_f():\n    return f(1)\n\ndef use_g():\n    return a.g(2)\n",
    C: "import miniworld_engine.a\n\ndef anything():\n    return miniworld_engine.a.g(1)\n",
    T1: "from miniworld_engine.b import use_f\n\ndef test_f():\n    assert use_f()\n",
    T2: "from miniworld_engine.b import use_g\n\ndef test_g():\n    assert use_g()\n",
    T3: "from miniworld_engine.c import anything\n\ndef test_c():\n    assert anything()\n",
}


def _tests(g, seeds):
    return {n for n in rg.reach_symbolic(g, seeds) if n.startswith("tests/")}


def test_an_import_of_a_name_depends_on_that_definition_and_its_callees_only() -> None:
    g = _mini(MINI)
    assert _tests(g, {A: {"g"}}) == {T2, T3}                       # b.use_g reads a.g by attribute; c imports the module (conservative)
    assert _tests(g, {A: {"helper"}}) == {T1, T3}                  # f calls helper, so use_f and its test are reached; g is untouched
    assert _tests(g, {A: {"LIMIT"}}) == {T1, T3}                   # a constant: its readers (helper -> f), the same closure
    assert _tests(g, {A: {sy.ALL}}) == {T1, T2, T3}               # the whole file: every importer
    assert _tests(g, {A: {"unused_name"}}) == {T3}                 # nobody imports it: only the plain ``import x`` stays conservative


def test_a_module_scope_use_of_a_changed_name_is_the_whole_file() -> None:
    fs = sy.analyze(A, "def f():\n    return 1\n\nREG = {}\nREG['x'] = f\n", rg._Resolver([A]))
    assert sy.closure(fs, {"f"}) == {sy.ALL}                       # module-level code runs on import and uses f
    fs = sy.analyze(A, "def f():\n    return 1\n\ndef g():\n    return f()\n\nif __name__ == '__main__':\n    g()\n", rg._Resolver([A]))
    assert sy.closure(fs, {"f"}) == {"f", "g", sy.MAIN}            # the __main__ block never runs on import: it is a unit, not the module scope


def test_the_mapping_falls_back_to_the_whole_file_when_it_is_not_certain() -> None:
    r = rg._Resolver([A])

    def changed(old: str, new: str) -> set[str]:
        return sy.changed_symbols(sy.analyze(A, old, r), sy.analyze(A, new, r))

    base = "import os\nX = 1\n\n@deco\ndef f():\n    return 1\n\ndef g():\n    return 2\n"
    assert changed(base, base.replace("return 2", "return 3")) == {"g"}                 # a function body only
    assert changed(base, base.replace("return 2", "return 2  # why")) == set()            # a comment: no change at all
    assert changed(base, base.replace("X = 1", "X = 2")) == {"X"}                       # a simple constant is a unit
    assert changed(base, base.replace("@deco", "@other")) == {sy.ALL}                    # a decorator
    assert changed(base, base.replace("import os", "import sys")) == {sy.ALL}            # an import
    assert changed(base, base + "print('x')\n") == {sy.ALL}                              # module level
    assert changed(base, "def f(:\n") == {sy.ALL}                                        # a syntax error
    assert changed(base, base + "def __getattr__(n):\n    return 1\n") == {sy.ALL}      # a module __getattr__: any name may be reached


def test_fixtures_are_requested_by_parameter_name() -> None:
    conf = "import pytest\n\n@pytest.fixture\ndef force():\n    return 1\n\n@pytest.fixture(autouse=True)\ndef always():\n    return 2\n\ndef pytest_configure(c):\n    pass\n"
    fs = sy.analyze("tests/conftest.py", conf, rg._Resolver([]))
    assert fs.units["force"].fixture == "fixture" and fs.units["always"].fixture == "autouse" and fs.units["pytest_configure"].fixture == ""
    test = sy.analyze("tests/cpu/test_x.py", "def test_a(force):\n    pass\n\ndef test_b():\n    pass\n", rg._Resolver([]))
    assert "force" in test.names["test_a"] and "force" not in test.names["test_b"]


# ------------------------------------------------------------------------------------------------------- the real graph
def test_a_registry_row_names_a_kernel_file_it_does_not_read_it(graph: rg.Graph) -> None:
    csv = f"{SRC}/kernels/registry/registry.csv"
    assert graph.deps[csv].get(KERNEL) == "names"
    assert graph.deps[csv].get(f"{SRC}/kernels/drivers/adaln.py") == "names"
    assert csv not in rg.reach(graph, [KERNEL], lazy=True)         # a kernel edit is not an edit of the registry
    assert f"{SRC}/settings.py" not in rg.reach(graph, [KERNEL])                     # the hub the old graph walked through (kernel -> registry.csv -> settings -> everything)


def test_a_comment_or_a_docstring_naming_a_file_is_not_a_dependency_on_it() -> None:
    text = textwrap.dedent('''
        """See kernels/adaln/triton/inference.py."""
        # and kernels/layernorm/cuda/__init__.py in a comment
        def f():
            """also docs/gpus/h100/dispatch.md"""
            return "kernels/_tiles.py"
    ''')
    seen = rg._py_strings(text)
    assert "kernels/_tiles.py" in seen                                  # a string the code can use
    assert "inference.py" not in seen and "layernorm" not in seen and "dispatch.md" not in seen


def test_the_symbol_level_reach_is_a_subset_and_narrower_than_the_whole_file(graph: rg.Graph) -> None:
    primitives = f"{SRC}/modules/primitives.py"
    whole = rg.reach_symbolic(graph, {primitives: {sy.ALL}})
    one = rg.reach_symbolic(graph, {primitives: {"Dropout"}})
    assert set(one) <= set(whole)
    assert set(one) <= set(rg.reach(graph, [primitives], lazy=True))
    t = lambda r: {n for n in r if graph.nodes[n] in ("test", "test-gpu")}
    assert len(t(one)) < len(t(whole))


def test_a_registry_kernel_name_in_a_test_is_an_edge_to_the_kernel_file(graph: rg.Graph) -> None:
    import csv

    rows = list(csv.DictReader((rg.ROOT / rg.REGISTRY_CSV).read_text(encoding="utf-8").splitlines()))
    by_name = {r["kernel"]: f"src/{r['file']}" for r in rows if r.get("file")}
    hits = [(u, d) for (u, d), v in graph.dispatch.items() if u.startswith("tests/") and d in by_name.values() and v]
    assert hits, "no test names a registry kernel: the dispatch index is empty"
    user, dep = hits[0]
    assert dep in graph.deps[user] and graph.deps[user][dep] in ("dispatch", "import", "lazy", "ref")


def test_a_fixture_edge_goes_to_the_fixture_not_to_every_test_below_the_conftest(graph: rg.Graph) -> None:
    conftest = "tests/conftest.py"
    users = [t for t in graph.users[conftest] if any(sym == "force_portable" for _, sym in graph.dispatch.get((t, conftest), ()))]
    assert users
    reached = rg.reach_symbolic(graph, {conftest: {"force_portable"}})
    tests = {n for n in reached if n.startswith("tests/") and n != conftest}
    assert set(users) <= tests
    assert len(tests) < len([n for n in graph.nodes if n.startswith("tests/") and n.endswith(".py")])      # not every test below it
    assert PY310 not in tests                                       # it does not ask for the fixture
    assert PY310 in rg.reach_symbolic(graph, {conftest: {sy.ALL}})  # the module level of a conftest is for every test


def test_the_python_310_test_is_selected_for_any_source_file(graph: rg.Graph) -> None:
    srcs = sorted(p for p in graph.nodes if p.startswith(f"{SRC}/") and p.endswith(".py"))
    for p in srcs[:: max(1, len(srcs) // 25)]:
        assert PY310 in rg.reach_symbolic(graph, {p: {sy.ALL}}), p
        first = next(iter(graph.syms[p].units), None)
        if first is not None:
            assert PY310 in rg.reach_symbolic(graph, {p: {first}}), p


def test_a_python_helper_imported_from_tests_is_an_import_edge(graph: rg.Graph) -> None:
    assert graph.deps["tests/cpu/builder/test_repo_graph.py"].get(f"{SRC}/build/repo_graph.py") == "import"
    helper_users = [u for u, k in graph.users["tests/paths.py"].items() if u.startswith("tests/") and k in ("import", "lazy")]
    assert helper_users                                             # ``from tests.paths import ...`` resolves to the helper file


# ------------------------------------------------------------------------------------------- the .cu card rule (builder arches)
def test_an_unhinted_cu_takes_the_arches_of_the_nvcc_call_that_builds_it(tmp_path: Path) -> None:
    py = tmp_path / "build.py"
    py.write_text(textwrap.dedent("""
        def ext():
            return load_extension(name="a", sources=[str(_dir / "only90.cu")], extra_cuda_cflags=[*host_flags(), *gencodes("90a")])

        def other():
            return load_extension(name="b", sources=[str(_dir / "both.cu")], extra_cuda_cflags=[*gencodes("80", "90", "100", ptx=("100",))])

        def flags():
            return load_extension(name="c", sources=[str(_dir / "flag100.cu")], extra_cuda_cflags=["-arch=sm_100a"])
    """), encoding="utf-8")
    assert cm.builder_cards(tmp_path, "build.py", "k/only90.cu") == {"sm90"}
    assert cm.builder_cards(tmp_path, "build.py", "k/both.cu") == {"sm80", "sm90", "sm100"}
    assert cm.builder_cards(tmp_path, "build.py", "k/flag100.cu") == {"sm100"}
    assert cm.builder_cards(tmp_path, "build.py", "k/unnamed.cu") == {"sm80", "sm90", "sm100"}      # the file's own arches when no call names it


def test_unhinted_cuda_sources_are_sm90_only_when_only_the_h100_builds_them(graph: rg.Graph) -> None:
    p = lambda s: f"{SRC}/kernels/{s}"
    assert graph.cards[p("adaln/cuda/adaln_rows.cu")] == {"sm90"}                  # gencodes("90a") in adaln/cuda/__init__.py
    assert graph.cards[p("bias_only_dit/cuda/bo_psb.cu")] == {"sm90"}
    assert graph.cards[p("layernorm/cuda/general/ln_fwd.cu")] == {"sm90", "sm100"}  # gencodes("80", "90", "100"): both restricted cards build it
    for path, c in graph.cards.items():
        if path.endswith(".cu") and path.startswith(f"{SRC}/kernels/"):
            assert c and c != cm.ALL and c <= graph.known_cards, path


# --------------------------------------------------------------------------------------------- Triton caches on sm90
def test_the_sm90_policy_is_a_documented_set_of_exempt_node_kinds(graph: rg.Graph) -> None:
    assert {"sm90": frozenset({rg.TRITON_CACHE})} == cm.EXEMPT_KINDS
    assert cm.exempt_kinds("sm90") == {"triton-cache"} and cm.exempt_kinds("sm80") == frozenset()
    assert "H100 keeps no Triton" in (cm.__doc__ or "") + Path(cm.__file__).read_text(encoding="utf-8")
    assert graph.nodes[TRITON_CACHE] == rg.TRITON_CACHE
    native = f"{SRC}/autotune/data/trimul_fwd_sm90_cuda/NVIDIA H100 80GB HBM3 (sm90).json"
    assert graph.nodes[native] == "cache"                           # a native cache is H100 evidence: not exempt


class _Row:
    def __init__(self, op: str, gpu: str, stale: bool = True) -> None:
        self.op, self.gpu, self.stale, self.reason = op, gpu, stale, "profiled kernel dependency changed (implementation)"


@pytest.fixture
def triton_result(graph: rg.Graph, monkeypatch):
    from miniworld_engine.autotune import plan

    monkeypatch.setattr(plan, "status", lambda path=None: {"units": 0, "stale": {}, "schema": 2})
    monkeypatch.setattr(rg, "derivation_plans", dict)
    seeds = [KERNEL]
    return {"graph": graph, "changed": seeds, "cosmetic": [], "deleted": [], "moved": {}, "rows": [],
            "reached": rg.reach(graph, seeds), "reached_lazy": rg.reach(graph, seeds, lazy=True),
            "card_reached": rg.reach_cards(graph, seeds), "card_reached_lazy": rg.reach_cards(graph, seeds, lazy=True)}


def test_stale_triton_caches_are_neither_pending_on_sm90_nor_hidden(graph, triton_result, monkeypatch) -> None:
    from miniworld_engine.autotune import cache_status

    rows = [_Row("adaln_gemm_gate_triton", "NVIDIA H100 80GB HBM3 (sm90)"), _Row("adaln_fwd_triton", "NVIDIA H100 80GB HBM3 (sm90)"),
            _Row("adaln_fwd_triton", "NVIDIA RTX A6000 (sm86)")]
    monkeypatch.setattr(cache_status, "scan", lambda: rows)
    assert TRITON_CACHE in rg.view(triton_result, "sm90")
    ev = rg.card_evidence(triton_result, "sm90", rows=rows)
    assert ev["stale"] == [] and ev["tuned_reached"] == 0           # not pending ...
    assert ev["triton"] == {"ops": ev["triton"]["ops"], "rows": 2, "stale": 2, "exempt": True, "exempt_cards": ["sm90"]}      # ... and counted
    assert ev["triton"]["ops"] >= 1
    assert rg.pending({"evidence": ev}) == []
    lines = "\n".join(rg._evidence_lines("sm90", ev, True, True, False))
    assert "EXEMPT on sm90 by policy" in lines and "2 committed row(s) named sm90, 2 stale by digest" in lines
    # under a fallback card the same rows are UNVERIFIABLE-HERE (they are that card's evidence), not exempt
    fb = rg.card_evidence(triton_result, "sm86", rows=rows)
    assert fb["triton"]["exempt"] is False and fb["triton"]["rows"] == 2
    assert [r.gpu for r in fb["stale"]] == ["NVIDIA RTX A6000 (sm86)"]                                 # its own row is judged on its own
    text = "\n".join(rg._evidence_lines("sm86", fb, False, True, False))
    assert "UNVERIFIABLE-HERE (owed on sm86, a fallback card): 2 committed row(s) kept under the sm90 file names" in text
    assert TRITON_CACHE in rg.view(triton_result, "sm86") or any(graph.nodes[n] == rg.TRITON_CACHE for n in rg.view(triton_result, "sm86"))
    owed = " | ".join(rg.actions(triton_result, "sm90"))
    assert "EXEMPT on sm90" in owed and "tuned caches reached" not in owed.split("EXEMPT on sm90")[0]
    assert "tuned caches reached" in " | ".join(rg.actions(triton_result, "sm86", accessible=False))


def test_check_on_sm90_does_not_fail_on_stale_triton_caches(graph, triton_result, monkeypatch) -> None:
    from miniworld_engine.autotune import cache_status

    rows = [_Row(op, "NVIDIA H100 80GB HBM3 (sm90)") for op in ("adaln_gemm_gate_triton", "adaln_fwd_triton", "adaln_epilogue_triton")]
    monkeypatch.setattr(cache_status, "scan", lambda: rows)
    monkeypatch.setattr(rg, "_rebuild_triples", lambda result, manifests: [])
    monkeypatch.setattr(rg, "changes", lambda base: triton_result)
    rep = rg.evaluate("HEAD", accessible="sm90")
    assert rep.blocking == [] and rep.code(check=True) == 0
    assert "EXEMPT on sm90 by policy" in rep.text                   # shown, not dropped
    assert "UNVERIFIABLE-HERE (owed on sm80, a fallback card)" in rep.text and "UNVERIFIABLE-HERE (owed on sm86, a fallback card)" in rep.text
    # a native H100 cache is still judged: the exemption is for the Triton kind only
    native = [_Row("trimul_fwd_sm90_cuda", "NVIDIA H100 80GB HBM3 (sm90)")]
    native[0].reason = "native implementation changed (op_identity)"
    monkeypatch.setattr(cache_status, "scan", lambda: native)
    ev = rg.card_evidence(triton_result, "sm90", rows=native)
    assert ev["native_stale"] == ["trimul_fwd_sm90_cuda"]


# ----------------------------------------------------------------------------------------------- the back-test (never under-select)
def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(rg.ROOT), *args], check=True, capture_output=True, text=True).stdout


def _has(commit: str) -> bool:
    try:
        subprocess.run(["git", "-C", str(rg.ROOT), "cat-file", "-e", f"{commit}^{{commit}}"], check=True, capture_output=True)
        subprocess.run(["git", "-C", str(rg.ROOT), "cat-file", "-e", f"{commit}^^{{commit}}"], check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, OSError):
        return False


#: commit -> why it is in the back-test, and the test(s) that catch its bug (a commit's own added / changed tests are always asserted too)
FIXES: dict[str, tuple[str, tuple[str, ...]]] = {
    "608a88fad": ("OPM projection bias counted twice (RF3 shallow MSA)", ("tests/gpu/sm90/test_opm_variants_gpu.py",)),
    "0a58dc716": ("python 3.10: nested same-quote f-strings", (PY310,)),
    "de295bf04": ("SWA SwiGLUFFN wired to the hand-CUDA kernel", ("tests/gpu/sm90/test_swa_swiglu_ffn_cuda_gpu.py",)),
    "2d5829e9f": ("MPNN backward bands (registry.csv rtol)", ("tests/gpu/any/test_mpnn_encoder_node_recompute.py",)),
    "53d858827": ("APB extra_pair_bias", ("tests/cpu/kernels/test_attention_pair_bias_extra_pair_bias.py",
                                          "tests/gpu/sm90/test_apb_extra_pair_bias_h100_gpu.py")),
    "27bc2b54a": ("OPM count_offset on the H100 CUDA paths", ()),
    "c2890c31e": ("Transition fp32 activation rounding on the host", ()),
    "4fb1f6129": ("generic Transition keeps fp32 masters unrounded", ()),
    "cad850743": ("TriAttn H100 forward softmax offset", ()),
    "08e5e4bce": ("SWA DiT: every H100 cell takes the native stages", ()),
    "87759fdc1": ("SWA DiT: global attention always native on H100", ()),
    "1b91a01cb": ("PyTorch legs keep fp32-master gradients unrounded", ()),
    "caa5bf684": ("identity / derive / plan back to main's (per-op identity tests)", ()),
    "122ef9716": ("pre-existing H100 failures (SWA fp32 master, bias-only DiT TF32)", ()),
    "475e14a5c": ("native H100 attention paths under fullgraph", ()),
    "9f222c8f2": ("TriMul H100 D128 one direction", ()),
    "8d074bc93": ("bias-only DiT per-sample pair", ()),
    "44faa4503": ("LayerNorm CUDA backward bit-reproducible", ()),
    "23eb738ff": ("token_dit_train La-Proteina widths", ()),
    "204bd2e27": ("TriMul H100 D256 configurations", ()),
}


def _own_tests(commit: str) -> list[str]:
    out = _git("diff-tree", "--no-commit-id", "--name-only", "-r", "--diff-filter=AMR", commit).split()
    return [p for p in out if p.startswith("tests/") and Path(p).name.startswith("test_") and p.endswith(".py")]


@pytest.mark.parametrize("commit", sorted(FIXES))
def test_historic_fix_is_selected(graph: rg.Graph, commit: str) -> None:
    if not _has(commit):
        pytest.skip(f"{commit} is not in this checkout's history")
    why, catching = FIXES[commit]
    res = rg.changes(f"{commit}^", commit, graph=graph, same_tree=False, skip_tests=True)
    if not res["sym_seeds"]:
        pytest.skip(f"no file of {commit} is in today's graph")
    runnable, needs = rg.plan_tests(res, "all", frozenset({"sm90"}))
    selected = set(runnable) | {t for v in needs.values() for t in v}
    expected = {p for p in (*catching, *_own_tests(commit)) if p in graph.nodes}
    if not expected:
        pytest.skip(f"the tests of {commit} are not in today's tree")
    missing = sorted(expected - selected)
    assert not missing, f"{commit} ({why}): not selected {missing}"
    # a registry row edit selects every test that names the row's kernel
    rows = rg.registry_rows_changed(f"{commit}^", commit) if rg.REGISTRY_CSV in _git("diff", "--name-only", f"{commit}^", commit).split() else []
    for t in (n for n, k in graph.nodes.items() if k in ("test", "test-gpu") and Path(n).name.startswith("test_") and (rg.ROOT / n).is_file()):
        text = (rg.ROOT / t).read_text(encoding="utf-8", errors="replace")
        if any(f'"{r}"' in text or f"'{r}'" in text for r in rows):
            assert t in selected, f"{commit}: {t} names a changed registry row"
    # the symbol-level selection is a subset of the whole-file one
    assert selected <= set(rg.tests({**res, "sym_reached": None}, "all")) | {t for v in needs.values() for t in v}
