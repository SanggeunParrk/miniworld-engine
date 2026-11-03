"""One code graph, per-card evidence (build/cards.py, repo_graph.reach_cards / evaluate).

Every node has a ``cards`` set (``ALL``: card-independent); a change set maps to the cards it reaches; a card this environment cannot run on is
UNVERIFIABLE-HERE (never fresh, never dropped); ``--check`` fails on stale evidence of an accessible card only (``--strict-cards``: on
unverifiable ones too). CPU only.
"""
from __future__ import annotations

import pytest

from miniworld_engine.build import cards as cm
from miniworld_engine.build import repo_graph as rg

SRC = "src/miniworld_engine"
H100_CU = f"{SRC}/kernels/trimul_inproj/cuda/h100_sources/b1/b1_fused.cu"
SM100_CU = f"{SRC}/kernels/swa_dit/cuda/sm100/attn_dq.cu"
TRITON = f"{SRC}/kernels/adaln/triton/inference.py"
SM100_TEST = "tests/gpu/sm100/test_apb_b200_gpu.py"


@pytest.fixture(scope="module")
def graph() -> rg.Graph:
    return rg.build()


def _result(graph, seeds, moved=None):
    return {"graph": graph, "changed": [s for s in seeds if not s.startswith("identity:")], "cosmetic": [], "deleted": [], "moved": moved or {},
            "rows": [], "reached": rg.reach(graph, seeds), "reached_lazy": rg.reach(graph, seeds, lazy=True),
            "card_reached": rg.reach_cards(graph, seeds), "card_reached_lazy": rg.reach_cards(graph, seeds, lazy=True)}


@pytest.fixture
def quiet(monkeypatch):
    """No GPU, no committed caches: the CPU-only part of the evaluation, with the change set injected."""
    from miniworld_engine.autotune import cache_status, plan

    monkeypatch.setattr(cache_status, "scan", lambda: [])
    monkeypatch.setattr(rg, "_rebuild_triples", lambda result, manifests: [])
    monkeypatch.setattr(plan, "status", lambda path=None: {"units": 0, "stale": {}, "schema": 2})
    monkeypatch.setattr(rg, "derivation_plans", lambda: {})

    def evaluate(graph, seeds, **kw):
        moved = kw.pop("moved", None)
        monkeypatch.setattr(rg, "changes", lambda base: _result(graph, seeds, moved))
        return rg.evaluate("HEAD", **kw)

    return evaluate


# ----------------------------------------------------------------------------------------------------- the cards attribute
def test_a_core_python_file_is_card_independent(graph) -> None:
    for p in (f"{SRC}/settings.py", f"{SRC}/modules/dispatch.py", f"{SRC}/kernels/_tiles.py", "docs/CHANGELOG.md", "tests/cpu/builder/test_repo_graph.py"):
        assert graph.cards[p] == cm.ALL, p


def test_hand_cuda_takes_the_card_its_registry_row_or_path_names(graph) -> None:
    assert graph.cards[H100_CU] == {"sm90"}
    assert graph.cards[SM100_CU] == {"sm100"}
    assert graph.cards[f"{SRC}/kernels/transition/cuda/transition_b2b_kernel.cu"] == {"sm90"}        # no card in the name: the registry row's arch
    assert graph.cards["registry-row:transition_fwd_b2b_sm90_cuda"] == {"sm90"}
    assert graph.cards[next(p for p in graph.nodes if "/triangle_attention/cuda/h100_sources/" in p and p.endswith(".cu"))] == {"sm90"}
    for p, c in graph.cards.items():
        if p.endswith(".cu") and p.startswith(f"{SRC}/kernels/"):
            assert c != cm.ALL and c <= graph.known_cards, p      # a CUDA source is never card-independent


def test_a_triton_kernel_belongs_to_the_fallback_cards_not_h100_or_b200(graph) -> None:
    fallback = graph.cards[TRITON]
    assert fallback == graph.fallback_cards
    assert "sm90" not in fallback and "sm100" not in fallback
    assert {"sm80", "sm86"} <= fallback
    assert graph.restricted_cards == {"sm90", "sm100"}                     # read from modules/dispatch.restricted_card


def test_plans_caches_docs_and_gpu_tests_take_their_own_card(graph) -> None:
    assert graph.cards[f"{SRC}/autotune/plans/sm90/registry_kernel.units.json"] == {"sm90"}
    assert graph.cards[f"{SRC}/autotune/plans/sm86/registry_kernel.csv"] == {"sm86"}
    assert graph.cards[f"{SRC}/autotune/data/adaln_gemm_gate_triton/NVIDIA H100 80GB HBM3 (sm90).json"] == {"sm90"}
    assert graph.cards[f"{SRC}/autotune/data/gated_projection_bwd_gate_dropres_triton/NVIDIA B200 (sm100).json"] == {"sm100"}
    assert graph.cards["docs/gpus/b200/b200.md"] == {"sm100"}
    assert graph.cards["docs/gpus/h100/h100.md"] == {"sm90"}
    assert graph.cards["docs/gpus/ampere-workstation/ampere-workstation.md"] == {"sm86"}
    assert graph.cards[SM100_TEST] == {"sm100"}
    assert graph.cards[f"{rg.NATIVE_PREFIX}trimul_fwd_sm90_cuda"] == {"sm90"}
    assert graph.cards[rg.DERIVATION] == cm.ALL                          # judged per card through each card's plan
    assert {"sm80", "sm86", "sm90", "sm100"} <= graph.known_cards


def test_an_edge_into_a_card_specific_kernel_carries_that_card(graph) -> None:
    got = rg.reach_cards(graph, [SM100_CU])
    assert got[SM100_CU] == {"sm100"}
    assert all(c == {"sm100"} for c in got.values())                       # importers are ALL but reached for sm100 only
    assert not any("sm90" in c for c in rg.reach_cards(graph, [SM100_TEST]).values())
    core = rg.reach_cards(graph, [f"{SRC}/settings.py"])
    assert core[f"{SRC}/settings.py"] == graph.known_cards
    # a tuned H100 cache of a Triton kernel goes stale with the kernel, though the kernel serves other cards today
    cache = f"{SRC}/autotune/data/adaln_gemm_gate_triton/NVIDIA H100 80GB HBM3 (sm90).json"
    assert "sm90" in rg.reach_cards(graph, [TRITON]).get(cache, ())


# --------------------------------------------------------------------------------------------------------- accessibility
def test_accessible_cards_come_from_the_flag_then_the_environment_then_the_device_then_the_repo_default(monkeypatch) -> None:
    monkeypatch.delenv(cm.ENV_ACCESSIBLE, raising=False)
    monkeypatch.setattr(cm, "visible_card", lambda: [])
    assert cm.accessible("sm90, sm100") == ({"sm90", "sm100"}, "--accessible")
    assert cm.accessible()[0] == {"sm90"} and "default" in cm.accessible()[1]
    monkeypatch.setattr(cm, "visible_card", lambda: ["sm100"])
    assert cm.accessible() == ({"sm100"}, "visible device")
    monkeypatch.setenv(cm.ENV_ACCESSIBLE, "sm86")
    assert cm.accessible() == ({"sm86"}, cm.ENV_ACCESSIBLE)
    with pytest.raises(ValueError):
        cm.parse_cards("h100")


# --------------------------------------------------------------------------------------------------- the report and --check
def test_a_change_reaching_an_sm100_node_lists_it_unverifiable_and_does_not_fail_check_unless_strict(graph, quiet) -> None:
    rep = quiet(graph, [SM100_CU], accessible="sm90")
    assert "== [sm100] UNVERIFIABLE-HERE (owed on sm100) ==" in rep.text
    assert "hardware evidence : UNVERIFIABLE-HERE (owed on sm100)" in rep.text
    assert "[sm100] owed, cannot run here:" in rep.text
    assert "== [sm90]" not in rep.text                                    # an sm100 kernel is not an H100 change
    assert rep.unverifiable == ["sm100"] and rep.blocking == []
    assert "unverifiable here: 1 card(s) (sm100)" in rep.text
    assert rep.code(check=True) == 0
    assert rep.code(check=True, strict=True) != 0
    assert rep.code(check=False, strict=True) == 0
    assert "fresh" not in "\n".join(line for line in rep.text.splitlines() if "hardware evidence" in line)


def test_stale_evidence_of_an_accessible_card_fails_check_and_another_cards_stale_does_not(graph, quiet, monkeypatch, tmp_path) -> None:
    from miniworld_engine.autotune import plan

    plans = {"sm86": tmp_path / "sm86.json", "sm90": tmp_path / "sm90.json"}
    stale = {f"u{i}": ["settings.py"] for i in range(3)}
    monkeypatch.setattr(plan, "status", lambda path=None: {"units": 5, "stale": stale if path == plans["sm90"] or path == plans["sm86"] else {}, "schema": 2})
    monkeypatch.setattr(rg, "derivation_plans", lambda: plans)
    seeds = [f"{SRC}/settings.py", rg.DERIVATION]
    moved = {rg.DERIVATION: [f"{SRC}/settings.py"]}
    monkeypatch.setattr(rg, "changes", lambda base: _result(graph, seeds, moved))
    rep = rg.evaluate("HEAD", accessible="sm90")
    assert rep.blocking and all(b.startswith("[sm90]") for b in rep.blocking)
    assert any("3 of 5 units stale" in b for b in rep.blocking)
    assert rep.code(check=True) == 2
    assert "derivation plan   : stale 3 of 5 units (computed from file digests)" in rep.text
    assert "[sm86] owed, cannot run here: miniworld-engine derive (arch sm86" in rep.text
    assert "[sm90] miniworld-engine derive (arch sm90" in rep.text
    # the same staleness on sm86 alone: sm86 is not accessible, so it is reported (computed) but does not block
    only86 = rg.evaluate("HEAD", accessible="sm100")
    assert only86.blocking == [] and only86.code(check=True) == 0
    assert "derivation plan   : stale 3 of 5 units" in only86.text and "UNVERIFIABLE-HERE (owed on sm86)" in only86.text


def test_a_card_independent_change_lists_every_card_with_its_evidence_kind(graph, quiet) -> None:
    rep = quiet(graph, [f"{SRC}/settings.py"], accessible="sm90")
    for card in graph.known_cards:
        assert f"== [{card}]" in rep.text, card
    assert "== [sm90] accessible here ==" in rep.text
    assert "hardware evidence : run the owed commands" in rep.text           # the accessible card
    assert set(rep.unverifiable) == graph.known_cards - {"sm90"}             # every other card is owed, none dropped


def test_a_doc_edit_is_not_a_hardware_obligation_on_any_card(graph, quiet) -> None:
    rep = quiet(graph, ["docs/CHANGELOG.md"], accessible="sm90")
    assert rep.unverifiable == [] and rep.code(check=True, strict=True) == 0
    assert "not hardware-bound" in rep.text
    assert "UNVERIFIABLE" not in rep.text.replace("unverifiable here: 0 card(s)", "")


def test_gpu_alias_selects_the_accessible_card(graph, quiet) -> None:
    rep = quiet(graph, [SM100_CU], gpu="sm100")
    assert "== [sm100] accessible here ==" in rep.text and rep.unverifiable == []


def test_tests_that_need_another_card_are_listed_apart_and_not_claimed(graph) -> None:
    result = _result(graph, [SM100_TEST])
    runnable, needs = rg.plan_tests(result, "all", frozenset({"sm90"}))
    assert SM100_TEST not in runnable and needs == {"sm100": [SM100_TEST]}
    runnable, needs = rg.plan_tests(result, "all", frozenset({"sm90", "sm100"}))
    assert SM100_TEST in runnable and needs == {}
    core, needs = rg.plan_tests(_result(graph, [f"{SRC}/kernels/_tiles.py"]), "all", frozenset({"sm90"}))
    assert not any(t.startswith("tests/gpu/sm100/") for t in core)         # an sm100-bound GPU test is not an H100 obligation
    assert all(t.startswith("tests/gpu/sm100/") or t.startswith("tests/gpu/") for c in needs.values() for t in c)


def test_the_code_graph_change_itself_reaches_no_kernel_identity(graph) -> None:
    """build/ is in no kernel key: an edit to it reaches no CUDA source, no native identity, no tuned cache."""
    got = rg.reach_cards(graph, [f"{SRC}/build/cards.py", f"{SRC}/build/repo_graph.py"])
    kinds = {graph.nodes[n] for n in got}
    assert not kinds & {"cuda-source", "cache", "triton-cache", "identity"}
