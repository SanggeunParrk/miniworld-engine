"""``registry_module.csv`` is the contract: ONE expansion (``build.registry_calls``) feeds bundle coverage and ``dev bundle-verify``.

No GPU. The expansion on synthetic rows (what each column multiplies, what an option does, what an arch refuses) and on the real file
(no row is dropped silently: every row expands to at least one call or is listed as 'not an <arch> row' with its registry text)."""
from __future__ import annotations

import inspect

import pytest
import torch

from miniworld_engine.autotune.module_registry import ModuleRow, module_rows
from miniworld_engine.build import bundle_coverage, registry_calls


def row(**kw) -> ModuleRow:
    base = {"module": "projected_attention", "stream": "token_pair", "lengths": (128, 256), "dims": {"n_head": 4, "d_hidden": 128}, "options": (),
                "impls": ("miniworld",), "dtypes": ("bfloat16",), "computes": (), "modes": ("eval", "train"), "source": "synthetic"}
    base.update(kw)
    return ModuleRow(**base)


def test_the_product_is_length_impl_dtype_compute_mode():
    (e,) = registry_calls.expand("sm90", [row(impls=("miniworld", "portable"), dtypes=("bfloat16", "float32"), computes=("bfloat16", "float32"))])
    assert len(e.calls) == 2 * 2 * 2 * 2 * 2
    assert {c.compute for c in e.calls} == {"bfloat16", "float32"}
    assert all(c.module == "projected_attention" and c.row == 1 and c.dim_index == 0 for c in e.calls)


def test_an_option_is_swept_one_at_a_time_and_only_on_the_side_it_acts_on():
    options = (("ln_bwd_path", "persistent"), ("ln_bwd_path", "atomic"), ("gate_backend", "split"), ("infer_concat", "True"), ("p_drop", "0.25"))
    (e,) = registry_calls.expand("sm90", [row(lengths=(128,), options=options)])
    eval_options = [c.option for c in e.calls if c.mode == "eval"]
    train_options = [c.option for c in e.calls if c.mode == "train"]
    assert eval_options == [None, ("gate_backend", "split"), ("infer_concat", "True")]   # ln_bwd_path / p_drop act on the backward side
    assert train_options == [None, ("ln_bwd_path", "persistent"), ("ln_bwd_path", "atomic"), ("gate_backend", "split"), ("p_drop", "0.25")]


def test_augmentation_and_the_dim_index_follow_the_row():
    rows = [row(), row(dims={"n_head": 8, "d_hidden": 256}, eval_augmentation=5, train_augmentation=48), row(module="transition", dims={"d_hidden": 128})]
    a, b, c = registry_calls.expand("sm90", rows)
    assert (a.calls[0].dim_index, b.calls[0].dim_index, c.calls[0].dim_index) == (0, 1, 0)   # the builder's Case.dims index
    assert {(x.mode, x.augmentation) for x in b.calls} == {("eval", 5), ("train", 48)}


def test_a_combination_the_arch_refuses_is_listed_not_dropped():
    # the common rules: the trimul fused path is bf16-only, an fp32 run is torch
    (e,) = registry_calls.expand("sm90", [row(module="triangle_multiplication", dims={"d_pair": 128, "d_hidden": 128}, dtypes=("bfloat16", "float32"))])
    assert {c.dtype for c in e.calls} == {"bfloat16"}
    assert any(x.startswith("miniworld/float32:") for x in e.excluded)


def test_a_row_with_nothing_left_is_not_an_arch_row_with_the_registry_text():
    (e,) = registry_calls.expand("sm90", [row(module="triangle_multiplication", dims={"d_pair": 128, "d_hidden": 128}, dtypes=("float32",),
                                              impls=("miniworld",))])
    assert e.calls == []
    reason = e.not_arch_reason
    assert reason
    assert "dtypes=float32" in reason
    assert "impls=miniworld" in reason
    assert "float32" in reason


def test_a_call_round_trips_through_json():
    (e,) = registry_calls.expand("sm90", [row(lengths=(128,), options=(("p_drop", "0.25"),))])
    for c in e.calls:
        assert registry_calls.Call.from_json(c.to_json()) == c


def test_every_row_of_the_real_registry_expands_or_says_why_not():
    rows = module_rows()
    expanded = registry_calls.expand("sm90")
    assert len(expanded) == len(rows) >= 80           # no row dropped (88 module rows since the leaf probes went, 2026-11-03)
    assert [e.row for e in expanded] == list(range(1, len(rows) + 1))
    for e in expanded:
        assert e.calls or e.not_arch_reason, f"{e.id}: neither calls nor a reason"
        assert e.registry_text
    # a row of a module the trunk runs both ways expands to both directions
    assert {c.direction for e in expanded if e.module == "triangle_multiplication" for c in e.calls} == {"", "incoming"}
    # the contract is not empty: the declared product is thousands of calls
    assert sum(len(e.calls) for e in expanded) > 3000


def test_every_declared_module_has_a_builder_case():
    from miniworld_engine.autotune.builder import cases

    names = {c.name for c in cases()}
    assert {r.module for r in module_rows()} <= names


def test_the_kernel_level_rows_are_all_accounted_for():
    units = registry_calls.kernel_units("sm90")
    callable_ = [u for u in units if not u.reason]
    assert callable_
    assert all(u.backend == "cuda" and u.arch == "sm90" and u.driver for u in callable_)
    assert all(u.reason for u in units if u.backend != "cuda")   # Triton rows: a reason, never silent


def test_the_default_coverage_is_the_registry_expansion_and_the_older_cases_are_extras():
    default = bundle_coverage.cases("sm90")
    kinds = {c["kind"] for c in default}
    assert kinds == {"registry_module", "registry_kernel"}
    calls = sum(len(e.calls) for e in registry_calls.expand("sm90"))
    assert sum(c["kind"] == "registry_module" for c in default) == calls
    assert [c["id"] for c in default] == list(range(len(default)))
    extras = {c["kind"] for c in bundle_coverage.extra_cases()}
    assert extras.isdisjoint(kinds)
    assert "case" in extras
    assert "msa_module" in extras
    assert "trimul_wide_train" in extras


def test_every_default_kind_has_a_runner():
    source = inspect.getsource(bundle_coverage._run_descriptor)
    for kind in ("registry_module", "registry_kernel"):
        assert f'"{kind}"' in source


@pytest.mark.parametrize("arch", ["sm90", "sm100"])
def test_the_expansion_exists_for_every_arch_with_rules(arch: str):
    assert registry_calls.expand(arch)


def test_a_module_the_trunk_runs_both_ways_is_called_in_both_directions():
    (e,) = registry_calls.expand("sm90", [row(module="triangle_multiplication", lengths=(128,), modes=("eval",))])
    assert sorted(c.direction for c in e.calls) == ["", "incoming"]
    (e,) = registry_calls.expand("sm90", [row(module="triangle_attention_heads", lengths=(128,), modes=("eval",))])
    assert sorted(c.direction for c in e.calls) == ["", "ending"]


def test_the_default_descriptors_can_be_dealt_to_workers():
    from miniworld_engine.build import kernel_bundle

    default = bundle_coverage.cases("sm90")
    assert len(kernel_bundle._interleave(default)) == len(default)


def test_the_transition_runs_fp32_on_sm90_and_not_on_the_bf16_only_cards():
    # sm_90: the generic wgmma Transition runs fp32 activations on TF32; sm_80 / sm_86 / sm_100 keep the bf16-only rule
    r = row(module="transition", dims={"d_hidden": 128}, dtypes=("bfloat16", "float32"))
    (e,) = registry_calls.expand("sm90", [r])
    assert {c.dtype for c in e.calls} == {"bfloat16", "float32"}
    (e,) = registry_calls.expand("sm86", [r])
    assert "float32" not in {c.dtype for c in e.calls}


def test_master_is_a_precision_of_its_own():
    from miniworld_engine.autotune.module_registry import torch_dtypes

    assert torch_dtypes("master") == (torch.float32, torch.bfloat16)
    assert torch_dtypes("bfloat16") == (torch.bfloat16, torch.bfloat16)
    (e,) = registry_calls.expand("sm90", [row(module="triangle_multiplication", dims={"d_pair": 128, "d_hidden": 128},
                                              dtypes=("bfloat16", "master"))])
    assert {c.dtype for c in e.calls} == {"bfloat16", "master"}           # not refused by the fp32 rule: activations are bf16


def test_the_fp32_matmul_precision_switch_acts_on_fp32_calls_only():
    from miniworld_engine.autotune.builder import switch_applies

    assert switch_applies("matmul_precision", "train", "float32")
    assert not switch_applies("matmul_precision", "train", "bfloat16")
    (e,) = registry_calls.expand("sm90", [row(module="transition", dims={"d_hidden": 128}, dtypes=("bfloat16", "float32"),
                                              options=(("matmul_precision", "high"),))])
    pinned = [c for c in e.calls if c.option]
    assert pinned
    assert {c.dtype for c in pinned} == {"float32"}                         # a bf16 call under it would be a duplicate


def test_the_dit_family_declares_its_input_variants_as_rows():
    rows = {(r.module, tuple(sorted(r.dims))) for r in module_rows()}
    assert any(m == "bias_only_dit" and "per_sample_pair" in d for m, d in rows)
    assert any(m == "bias_only_dit" and "masked" in d for m, d in rows)
    assert any(m == "local_dit" and "esmfold2" in d for m, d in rows)
    assert any(m == "dit" and "use_qk_norm" in d for m, d in rows)


def test_an_extension_or_triton_out_of_memory_is_an_oom():
    assert registry_calls.out_of_memory(RuntimeError("Triton Error [CUDA]: out of memory"))
    assert registry_calls.out_of_memory(RuntimeError("CUDA error: out of memory\nCUDA kernel errors might be ..."))
    assert not registry_calls.out_of_memory(RuntimeError("CUDA error: invalid argument"))
    assert registry_calls.classify(RuntimeError("Triton Error [CUDA]: out of memory")) == "oom"
