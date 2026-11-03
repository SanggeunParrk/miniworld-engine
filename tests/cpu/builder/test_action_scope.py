"""What each release action does, derived from a (synthetic) code-graph answer -- no GPU, no graph build (``build.action_scope``): the scope of
every action, an empty scope, the ledger (what already passed at identical inputs is not run again), the unit -> coverage-case mapping, and
the full run that only ``--all --reason`` reaches."""
from __future__ import annotations

from miniworld_engine.build import action_scope as sc

READ = {"src/k.py": "v1", "src/other.py": "o1", "tests/cpu/t_a.py": "ta", "tests/cpu/t_b.py": "tb", "tests/gpu/sm90/t_g.py": "tg"}


def read(p: str) -> str:
    return READ.get(p, "absent")


def dims() -> dict[int, dict]:
    return {1: {"n_head": 4, "d_hidden": 128}, 2: {"d_hidden": 256}}


def call(row=1, module="transition", impl="miniworld", dtype="bfloat16", length=128, mode="eval", compute="", option=None, direction=""):
    return {"row": row, "module": module, "stream": "token_pair", "length": length, "impl": impl, "dtype": dtype, "compute": compute, "mode": mode,
            "option": option, "augmentation": 1, "direction": direction}


def case(**kw) -> dict:
    return {"kind": "registry_module", "row": kw.get("row", 1), "call": call(**kw), "heavy": False}


def facts(**kw) -> sc.Facts:
    base = {"arch": "sm90", "base": "v2.4.3", "changed": ["src/k.py"], "unit_files": {}, "tests_cpu": [], "tests_gpu": [],
                "reaching": {}, "universe": {"tests_cpu": ["tests/cpu/t_a.py", "tests/cpu/t_b.py", "tests/cpu/t_z.py"], "tests_gpu": ["tests/gpu/sm90/t_g.py"],
                                       "bench_ops": ["transition", "dit"]}}
    base.update(kw)
    return sc.Facts(**base)


# ------------------------------------------------------------------------------------------------------------------ test
def test_the_test_scope_is_exactly_the_files_the_graph_names():
    f = facts(tests_cpu=["tests/cpu/t_a.py"], tests_gpu=["tests/gpu/sm90/t_g.py"])
    s = sc.test_scope(f)
    assert s.names() == ["tests/cpu/t_a.py", "tests/gpu/sm90/t_g.py"]
    assert s.payload == {"cpu": ["tests/cpu/t_a.py"], "gpu": ["tests/gpu/sm90/t_g.py"]}
    assert "tests/cpu/t_z.py" not in s.items


def test_nothing_named_is_an_empty_scope_that_says_so():
    s = sc.test_scope(facts())
    assert s.empty
    assert "nothing to do" in s.block()


def test_what_the_last_run_left_unfinished_is_owed_whatever_the_diff_says():
    # facts are taken against the last run's snapshot: an unchanged tree names nothing, the failed file of that run is still owed
    s = sc.test_scope(facts(), owed=["tests/cpu/t_b.py", "tests/cpu/gone.py"])
    assert s.names() == ["tests/cpu/t_b.py"]                                     # a file that no longer exists is not owed
    assert s.items["tests/cpu/t_b.py"] == "owed"
    s = sc.test_scope(facts(tests_cpu=["tests/cpu/t_a.py"]), owed=["tests/cpu/t_a.py"])
    assert s.items == {"tests/cpu/t_a.py": "graph"}


def test_cards_this_machine_cannot_run_are_owed_not_done():
    s = sc.test_scope(facts(owed={"sm100": ["103 GPU test file(s)"]}))
    assert s.owed == {"sm100": ["103 GPU test file(s)"]}
    assert "UNVERIFIABLE-HERE (owed on sm100)" in s.block()


def test_all_is_the_whole_universe_and_records_its_reason():
    s = sc.test_scope(facts(), all_reason="torch upgrade")
    assert s.names() == ["tests/cpu/t_a.py", "tests/cpu/t_b.py", "tests/cpu/t_z.py", "tests/gpu/sm90/t_g.py"]
    assert s.payload["gpu"] == ["tests/gpu/sm90/t_g.py"]
    assert "FULL RUN (--all): reason: torch upgrade" in s.block()


# ------------------------------------------------------------------------------------------------------------------ derive / bench
def test_derive_is_the_stale_units_only():
    assert sc.derive_scope(facts()).empty
    s = sc.derive_scope(facts(stale_units={"u1": ["src/k.py"]}, unit_files={"u1": ["src/k.py"], "u2": ["src/o.py"]}))
    assert s.names() == ["u1"]
    assert sc.derive_scope(facts(unit_files={"u1": [], "u2": []}), "why").names() == ["u1", "u2"]


def test_bench_is_the_reached_modules_plus_the_unmeasured_ones():
    assert sc.bench_scope(facts(bench_ops=["transition"])).names() == ["transition"]
    assert sc.bench_scope(facts()).empty
    assert sc.bench_scope(facts(), owed=["dit"]).names() == ["dit"]
    assert sc.bench_scope(facts(), all_reason="r").names() == ["dit", "transition"]


def test_bench_op_of_reads_the_module_directory():
    assert sc.bench_op_of("benchmarks/modules/transition/configs/bench.yaml") == "transition"
    assert sc.bench_op_of("benchmarks/runners/bench.py") is None


# ------------------------------------------------------------------------------------------------------------------ bundle: the bundle against the tree
def test_a_unit_label_is_the_derive_unit_label():
    from miniworld_engine.autotune.derive import DeriveUnit

    c = call(compute="float32", option=("p_drop", "0.25"), mode="train")
    unit = DeriveUnit(c["module"], c["stream"], c["length"], tuple(dims()[1].items()), c["impl"], c["dtype"], c["compute"], c["mode"], tuple(c["option"]), c["augmentation"])
    assert sc.unit_label(c, dims()[1]) == unit.label


TREE = {"pkg:a.cu": "A1", "pkg:b.cu": "B1", "pkg:gone.cu": "absent"}


def digest_of(label: str):
    return TREE.get(label) if label.startswith("pkg:") else None


def art(name: str, *nodes: tuple[str, str]) -> dict:
    return {"name": name, "material": {"nodes": [list(n) for n in nodes]}}


def rows(recorded=None, missing=None) -> sc.Rows:
    return sc.Rows(keys={1: "transition/token_pair/n_head=4;d_hidden=128", 2: "transition/token_pair/d_hidden=256"},
                   text={"transition/token_pair/n_head=4;d_hidden=128": "t1", "transition/token_pair/d_hidden=256": "t2"},
                   recorded={"transition/token_pair/n_head=4;d_hidden=128": {"text": "t1", "artifacts": ["ka"]},
                             "transition/token_pair/d_hidden=256": {"text": "t2", "artifacts": ["kb"]}} if recorded is None else recorded,
                   reported_missing=missing or {})


K1, K2 = "transition/token_pair/n_head=4;d_hidden=128", "transition/token_pair/d_hidden=256"


def test_stale_is_judged_against_the_tree_not_the_base():
    arts = [art("ka", ("pkg:a.cu", "A1")), art("kb", ("pkg:b.cu", "B0"), ("file:cute.h", "x")), art("kg", ("pkg:gone.cu", "G"))]
    assert sc.stale_artifacts(arts, digest_of) == {"kb": ["pkg:b.cu"], "kg": ["pkg:gone.cu"]}     # an external header is not the tree's


def test_a_fresh_bundle_owes_nothing_whatever_the_base_diff_says():
    f = facts(changed=["src/hub.py"] * 50, unit_files={"anything": ["src/hub.py"]})
    s = sc.bundle_scope(f, [case(row=1), case(row=2)], rows(), {"ka", "kb"}, {}, {}, digest_of)
    assert s.empty
    assert "nothing to do" in s.block()


def test_an_owed_row_runs_although_its_artifacts_are_fresh():
    # a failing row under --retry-failing (or one an earlier run did not reach) runs its coverage again; nothing stale names it
    s = sc.bundle_scope(facts(), [case(row=1), case(row=2)], rows(), {"ka", "kb"}, {}, {}, digest_of, owed=[f"row:{K2}"])
    assert set(s.items) == {f"row:{K2}"}


def test_only_the_rows_served_by_a_stale_or_missing_artifact_run():
    cases = [case(row=1, length=128), case(row=1, length=256), case(row=2)]
    s = sc.bundle_scope(facts(), cases, rows(), {"ka", "kb"}, {"kb": ["pkg:b.cu"]}, {}, digest_of)
    assert s.names() == [f"row:{K2}"]
    assert s.payload["cases"] == [cases[2]]
    assert "stale" in s.payload["why"][f"row:{K2}"]
    s = sc.bundle_scope(facts(), cases, rows(), {"kb"}, {}, {}, digest_of)
    assert s.names() == [f"row:{K1}"]
    assert len(s.payload["cases"]) == 2
    assert "not in the bundle: ka" in s.payload["why"][f"row:{K1}"]


def test_a_new_an_edited_and_a_reported_row_are_owed():
    r = rows(recorded={K1: {"text": "old", "artifacts": ["ka"]}})
    s = sc.bundle_scope(facts(), [case(row=1), case(row=2)], r, {"ka"}, {}, {}, digest_of)
    assert s.payload["why"] == {f"row:{K1}": "row edited since its verify record", f"row:{K2}": "new row (no verify record)"}
    s = sc.bundle_scope(facts(), [case(row=1), case(row=2)], rows(missing={K1: ["kz"]}), {"ka", "kb"}, {}, {}, digest_of)
    assert s.names() == [f"row:{K1}"]
    assert "kz" in s.payload["why"][f"row:{K1}"]


def test_the_same_owed_build_is_not_run_twice_and_a_new_edit_runs_again():
    s = sc.bundle_scope(facts(), [case(row=2)], rows(), {"ka", "kb"}, {"kb": ["pkg:b.cu"]}, {}, digest_of)
    ledger = dict(s.items)
    assert sc.bundle_scope(facts(), [case(row=2)], rows(), {"ka", "kb"}, {"kb": ["pkg:b.cu"]}, ledger, digest_of).empty
    TREE["pkg:b.cu"] = "B2"
    try:
        assert not sc.bundle_scope(facts(), [case(row=2)], rows(), {"ka", "kb"}, {"kb": ["pkg:b.cu"]}, ledger, digest_of).empty
    finally:
        TREE["pkg:b.cu"] = "B1"


def test_a_stale_artifact_no_row_produces_is_a_gap_never_widened():
    s = sc.bundle_scope(facts(), [case(row=1)], rows(), {"ka", "kb", "orphan"}, {"orphan": ["pkg:a.cu"]}, {}, digest_of)
    assert s.empty
    assert s.payload["orphans"] == ["orphan"]
    assert "orphan" in s.gaps[0]


def test_kernel_rows_follow_the_graph_and_the_ledger():
    k = {"kind": "registry_kernel", "kernel": "opm_gather", "arch": "sm90", "heavy": False}
    f = facts(kernel_inputs={"opm_gather": ["src/k.py"]})
    s = sc.bundle_scope(f, [k, case()], rows(), {"ka", "kb"}, {}, {}, digest_of, read)
    assert s.names() == ["kernel:opm_gather"]
    assert s.payload["cases"] == [k]
    assert sc.bundle_scope(f, [k, case()], rows(), {"ka", "kb"}, {}, dict(s.items), digest_of, read).empty


def test_all_bundles_every_case():
    cases = [case(row=1), case(row=2)]
    s = sc.bundle_scope(facts(), cases, rows(), set(), {}, {}, digest_of, read, all_reason="r")
    assert len(s.items) == 2
    assert s.payload["cases"] == cases


# ------------------------------------------------------------------------------------------------------------------ verify
def recorded_rows(**over) -> sc.Rows:
    r = rows(recorded={K1: {"text": "t1", "artifacts": ["ka"], "served": "fp1", "inputs": "i1"},
                       K2: {"text": "t2", "artifacts": ["kb"], "served": "fp1", "inputs": "i2"}})
    r.inputs = {K1: "i1", K2: "i2"}
    for k, v in over.items():
        setattr(r, k, v)
    return r


def test_verify_is_judged_against_each_rows_own_record_not_the_base():
    f = facts(changed=["src/hub.py"] * 50, unit_files={"anything": ["src/hub.py"]})        # a wide base diff alone owes nothing
    assert sc.verify_scope(f, [case(row=1), case(row=2)], recorded_rows(), {}, fingerprint=lambda k: "fp1").empty


def test_verify_owes_a_row_whose_inputs_changed_whose_artifact_was_rebuilt_or_that_is_new():
    cases = [case(row=1), case(row=2)]
    s = sc.verify_scope(facts(), cases, recorded_rows(inputs={K1: "i1-edited", K2: "i2"}), {}, fingerprint=lambda k: "fp1")
    assert s.names() == [f"row:{K1}"]
    assert s.payload["rows"] == [1]
    assert "changed" in s.payload["why"][f"row:{K1}"]
    s = sc.verify_scope(facts(), cases, recorded_rows(), {}, fingerprint=lambda k: "fp2" if k == K2 else "fp1")
    assert s.names() == [f"row:{K2}"]
    assert "rebuilt" in s.payload["why"][f"row:{K2}"]
    s = sc.verify_scope(facts(), cases, rows(recorded={K1: {"text": "t1", "artifacts": ["ka"]}}), {})
    assert s.names() == [f"row:{K2}"]
    assert s.payload["why"][f"row:{K2}"].startswith("new row")


def test_kernel_rows_follow_the_graph_and_all_is_everything():
    k = {"kind": "registry_kernel", "kernel": "opm_gather", "arch": "sm90", "heavy": False}
    f = facts(kernel_inputs={"opm_gather": ["src/k.py"]})
    s = sc.verify_scope(f, [k, case()], recorded_rows(), {}, fingerprint=lambda key: "fp1", read=read)
    assert s.names() == ["kernel:opm_gather"]
    assert s.payload["kernels"] == ["opm_gather"]
    assert sc.verify_scope(f, [k, case()], recorded_rows(), dict(s.items), fingerprint=lambda key: "fp1", read=read).empty
    full = sc.verify_scope(facts(), [k, case()], recorded_rows(), {}, all_reason="why")
    assert full.names() == sorted(["kernel:opm_gather", f"row:{K1}", f"row:{K2}"])
    assert full.all_reason == "why"


def test_row_inputs_are_the_files_of_the_rows_derivation_units():
    c1, c2 = call(row=1, length=128), call(row=1, length=256)
    units = {sc.unit_label(c1, dims()[1]): ["src/a.py", "src/b.py"], sc.unit_label(c2, dims()[1]): ["src/b.py", "src/c.py"]}
    got = sc.row_inputs({K1: [c1, c2], K2: [call(row=2)]}, {K1: dims()[1], K2: dims()[2]}, units, lambda files: ",".join(files))
    assert got == {K1: "src/a.py,src/b.py,src/c.py"}                                         # K2 has no derivation unit: not judged on inputs


def test_each_cuda_build_and_python_owes_the_names_another_holds():
    def a(name, kind, py="cpython-312"):
        return {"name": name, "kind": kind, "material": {"python": py}}

    manifests = {"sm90-cu129": {"pythons": ["cpython-310", "cpython-312"],
                                "artifacts": {"1": a("k1", "cubin"), "2": a("e1", "ext"), "3": a("e1", "ext", "cpython-310"), "4": a("e2", "ext")}},
                 "sm90-cu130": {"pythons": ["cpython-310", "cpython-312"], "artifacts": {"1": a("e1", "ext"), "2": a("e1", "ext", "cpython-310")}}}
    got = sc.owed_replays(manifests)
    assert got[("sm90-cu129", "cpython-310")] == ["e2"]                         # e2 was built for cp312 only
    assert got[("sm90-cu130", "cpython-310")] == ["e2", "k1"]                   # the cubin once per build, under its first ABI
    assert got[("sm90-cu130", "cpython-312")] == ["e2"]
    assert ("sm90-cu129", "cpython-312") not in got
