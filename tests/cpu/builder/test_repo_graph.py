"""The whole-repository graph (build/repo_graph.py): every file a node, edges read from the files, a change walked backwards."""
from __future__ import annotations

import pytest

from miniworld_engine.build import repo_graph as rg

KERNEL = "src/miniworld_engine/kernels/adaln/triton/inference.py"


@pytest.fixture(scope="module")
def graph() -> rg.Graph:
    return rg.build()


def test_every_tracked_file_is_a_node(graph: rg.Graph) -> None:
    tracked = set(rg.files())
    assert tracked
    assert tracked <= set(graph.nodes)
    assert rg.DERIVATION in graph.nodes
    assert f"{rg.NATIVE_PREFIX}trimul_fwd_sm90_cuda" in graph.nodes


def test_python_imports_and_registry_names_are_edges(graph: rg.Graph) -> None:
    assert graph.deps[KERNEL].get("src/miniworld_engine/kernels/_tiles.py") == "import"
    registry_csv = "src/miniworld_engine/kernels/registry/registry.csv"
    assert graph.deps[registry_csv].get(KERNEL) == "ref"                      # a registry row names the kernel's file
    assert graph.deps[registry_csv].get("src/miniworld_engine/kernels/drivers/adaln.py") == "ref"   # ... and its driver


def test_a_tuned_cache_depends_on_its_kernel_helpers_driver_and_grid(graph: rg.Graph) -> None:
    cache = "src/miniworld_engine/autotune/data/adaln_gemm_gate_triton/NVIDIA H100 80GB HBM3 (sm90).json"
    deps = graph.deps[cache]
    assert {KERNEL, "src/miniworld_engine/kernels/_tiles.py", "src/miniworld_engine/kernels/drivers/adaln.py",
            "src/miniworld_engine/autotune/configs/grid/adaln_gemm_gate_triton.csv"} <= set(deps)


def test_a_native_cache_hangs_on_its_own_op_identity_and_that_on_the_ops_graph(graph: rg.Graph) -> None:
    op = "trimul_fwd_sm90_cuda"
    cache = f"src/miniworld_engine/autotune/data/{op}/NVIDIA H100 80GB HBM3 (sm90).json"
    ident = f"{rg.NATIVE_PREFIX}{op}"
    assert graph.deps[cache].get(ident) == "identity"
    assert cache in graph.users[ident]                                      # the walk ends at the identity
    other = f"{rg.NATIVE_PREFIX}layernorm_fwd_cuda"
    assert other not in graph.deps[cache]                                   # no longer one identity for all the native caches
    assert "src/miniworld_engine/kernels/trimul_inproj/cuda/h100_inference.py" in graph.deps[ident]


def test_an_unrelated_kernel_does_not_move_a_native_identity(graph: rg.Graph) -> None:
    unrelated = "src/miniworld_engine/kernels/triangle_attention/triton/main.py"
    if unrelated not in graph.nodes:
        pytest.skip("layout without that file")
    for op in ("trimul_fwd_sm90_cuda", "layernorm_fwd_cuda"):
        assert unrelated not in graph.deps[f"{rg.NATIVE_PREFIX}{op}"], op


def test_an_edit_to_a_kernel_reaches_its_cache_and_its_tests_and_moves_only_the_identities_that_hold_it(graph: rg.Graph) -> None:
    reached = rg.reach(graph, [KERNEL])
    assert {"cache", "test"} <= {graph.nodes[n] for n in reached}
    holders = [i for i in graph.nodes if rg.is_identity(i) and KERNEL in graph.deps[i]]
    assert rg.DERIVATION in holders or not graph.deps[rg.DERIVATION]       # a schema-1 derivation records no files


def test_the_actions_name_what_the_edit_owes(graph: rg.Graph) -> None:
    ident = f"{rg.NATIVE_PREFIX}layernorm_fwd_cuda"
    moved = {ident: [KERNEL], rg.DERIVATION: [KERNEL]}
    result = {"graph": graph, "changed": [KERNEL], "cosmetic": [], "deleted": [], "moved": moved, "reached": rg.reach(graph, [KERNEL])}
    owed = " | ".join(rg.actions(result))
    for phrase in ("restamp 1 of", "dev derive", "reuses the rest", "tuned caches reached", "CPU tests reached"):
        assert phrase in owed, phrase


def test_a_cache_hangs_on_its_kernels_registry_row_not_on_the_whole_file(graph: rg.Graph) -> None:
    cache = "src/miniworld_engine/autotune/data/adaln_gemm_gate_triton/NVIDIA H100 80GB HBM3 (sm90).json"
    assert graph.deps[cache].get("registry-row:adaln_gemm_gate_triton") == "cache"
    assert rg.REGISTRY_CSV not in graph.deps[cache]
    reached = rg.reach(graph, ["registry-row:adaln_gemm_gate_triton"])
    assert cache in reached
    other = "src/miniworld_engine/autotune/data/adaln_fwd_triton/NVIDIA H100 80GB HBM3 (sm90).json"
    assert other not in reached                                             # a different kernel's row does not reach it


def test_registry_rows_compare_by_kernel() -> None:
    old = rg._rows("kernel,dtypes\na,bf16\nb,bf16\n")
    new = rg._rows("kernel,dtypes\na,bf16\nb,bf16|fp32\nc,bf16\n")
    assert sorted(k for k in old.keys() | new.keys() if old.get(k) != new.get(k)) == ["b", "c"]


def test_the_check_fails_on_stale_evidence_only() -> None:
    class Row:
        op, gpu, reason = "k", "H100", "kernel source/key changed (op_identity)"

    assert rg.pending({"evidence": {"stale": [], "native_stale": [], "units_stale": {}}}) == []
    owed = rg.pending({"evidence": {"stale": [Row()], "native_stale": ["x_cuda"], "units_stale": {"u": ["f.py"]}, "units_total": 9}})
    assert len(owed) == 3
    assert any("dev restamp-native" in o for o in owed)
    assert any("1 of 9 units stale" in o and "dev derive" in o for o in owed)


def test_new_only_keeps_what_the_diff_causes_and_drops_what_was_already_stale(monkeypatch, graph: rg.Graph) -> None:
    from miniworld_engine.autotune import cache_status, identity, plan

    class Row:
        gpu, stale, reason = "H100", True, "native implementation changed"

        def __init__(self, op):
            self.op = op

    monkeypatch.setattr(cache_status, "scan", lambda: [Row("layernorm_fwd_cuda"), Row("trimul_fwd_sm90_cuda")])
    monkeypatch.setattr(plan, "status", lambda path=None: {"units": 3, "schema": 2, "stale": {"a": ["kernels/x.py"], "b": ["kernels/y.py"]}})
    touched = next(p for p in identity.native_files("layernorm_fwd_cuda") if p.startswith("kernels/layernorm/"))
    result = {"graph": graph, "changed": [f"src/miniworld_engine/{touched}"], "cosmetic": [], "reached": {}, "rows": []}
    everything = rg.live_evidence(result)
    assert everything["native_stale"] == ["layernorm_fwd_cuda", "trimul_fwd_sm90_cuda"]
    assert set(everything["units_stale"]) == {"a", "b"}
    new = rg.live_evidence(result, new_only=True)
    assert "layernorm_fwd_cuda" in new["native_stale"]
    result["changed"] = ["src/miniworld_engine/kernels/x.py"]
    new = rg.live_evidence(result, new_only=True)
    assert set(new["units_stale"]) == {"a"}
    assert "layernorm_fwd_cuda" not in new["native_stale"] or "kernels/x.py" in identity.native_files("layernorm_fwd_cuda")


def test_tests_lists_reached_test_files_one_path_each(graph: rg.Graph) -> None:
    result = {"graph": graph, "reached_lazy": rg.reach(graph, [KERNEL], lazy=True)}
    cpu = rg.tests(result, "cpu")
    assert cpu
    assert all(p.startswith("tests/") and p.endswith(".py") and "gpu" not in p.split("/")[:3] for p in cpu)
    assert set(rg.tests(result, "all")) == set(cpu) | set(rg.tests(result, "gpu"))


def test_the_derivation_is_judged_on_the_card_that_was_derived_not_on_another_cards_plan(monkeypatch, graph: rg.Graph, tmp_path) -> None:
    """``dev derive --arch sm90`` re-records the sm90 plan only. The check used to read the default (sm86) plan whatever was derived, so a
    change to a file every unit lists (the safety set) read as "1156 of 1156 stale" right after a successful derive. Now the plan of
    ``DERIVED_ARCH`` (or ``--gpu``) is judged, and the other cards' plans are only counted in the report."""
    from miniworld_engine.autotune import cache_status, plan

    plans = {"sm86": tmp_path / "sm86.json", "sm90": tmp_path / "sm90.json"}
    states = {plans["sm86"]: {"units": 4, "schema": 2, "stale": {f"u{i}": ["settings.py"] for i in range(4)}},
              plans["sm90"]: {"units": 4, "schema": 2, "stale": {}}}
    monkeypatch.setattr(rg, "derivation_plans", lambda: plans)
    monkeypatch.setattr(plan, "status", lambda path=None: states[path] if path else states[plans["sm86"]])
    monkeypatch.setattr(cache_status, "scan", lambda: [])
    result = {"graph": graph, "changed": ["src/miniworld_engine/settings.py"], "cosmetic": [], "reached": {}, "rows": []}

    after_sm90_derive = rg.live_evidence(result, new_only=True)
    assert rg.DERIVED_ARCH == "sm90"
    assert after_sm90_derive["units_arch"] == "sm90" and after_sm90_derive["units_stale"] == {}
    assert after_sm90_derive["units_other"] == {"sm86": 4}
    assert rg.pending({"evidence": after_sm90_derive}) == []

    sm86 = rg.live_evidence(result, new_only=True, gpu="sm86")
    assert sm86["units_arch"] == "sm86" and len(sm86["units_stale"]) == 4
    assert sm86["units_other"] == {"sm90": 0}
    assert any("4 of 4 units stale" in o for o in rg.pending({"evidence": sm86}))
