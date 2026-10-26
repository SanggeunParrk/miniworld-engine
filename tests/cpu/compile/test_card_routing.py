"""Where `implementation="miniworld"` runs an op, card by card -- CPU only, with a fake GPU and a fake manifest directory.

The rule (modules/dispatch.py, autotune/records.py, docs/dispatch.md):
  * H100 / B200: native CUDA, an unserved call runs PyTorch, never Triton, never a warning;
  * every other card: a CUDA path that serves the call; else the Triton family IF the engine has a record for that exact card
    (every Triton kernel of the op has an `ok` manifest row); else one UnrecordedCardWarning per (op, card) and PyTorch;
  * `settings.force_portable_kernels` (the A/B switch) forces Triton everywhere.
"""
from __future__ import annotations

import dataclasses
import warnings
from functools import partial

import pytest
import torch
from fake_card import fake_card, write_manifest

from miniworld_engine import settings
from miniworld_engine.autotune import records
from miniworld_engine.modules import dispatch
from miniworld_engine.modules.dispatch import KernelBackend, UnrecordedCardWarning
from miniworld_engine.modules.exceptions import ImplementationType

LONG = "NVIDIA RTX A6000"          # a name the packaged manifests know (sm86)
OTHER = "NVIDIA A40"              # same capability, never recorded
OPS = ("transition", "triangle_multiplication", "triangle_attention", "adaptive_layernorm", "conditioned_transition",
       "augmented_attention", "outer_product_mean", "msa_pair_weighted_averaging", "rmsnorm")


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    previous = settings.current()
    settings.configure(force_portable_kernels=False, portable_on_hopper_blackwell=False, b200_portable_kernels=False, trimul_impl=None,
                       capture=False)
    dispatch._card_cache.clear()
    dispatch._unrecorded_warned.clear()
    dispatch._refused_warned.clear()
    yield
    settings.configure(**dataclasses.asdict(previous))
    dispatch._card_cache.clear()
    dispatch._unrecorded_warned.clear()
    dispatch._refused_warned.clear()


def _recorded(kind, *calls):
    """Run ``calls`` with every warning recorded: ``(results, warnings of class kind)``."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        results = [call() for call in calls]
    return results, [w for w in caught if issubclass(w.category, kind)]


@pytest.fixture
def manifests(tmp_path):
    with records.use_manifest_dir(tmp_path):
        yield tmp_path


# ---- (vii) the public enum -------------------------------------------------------------------------------------------
def test_the_public_enum_has_exactly_four_members():
    assert {m.value for m in ImplementationType} == {"pytorch", "miniworld", "cuequivariance", "anthropic"}
    assert not hasattr(ImplementationType, "TRITON")
    assert not hasattr(ImplementationType, "CUDA")


# ---- (v) the removed strings ----------------------------------------------------------------------------------------
@pytest.mark.parametrize("old", ["triton", "cuda", "TRITON", "Triton"])
def test_the_removed_strings_raise_and_list_the_four_valid_values(old):
    with pytest.raises(ValueError) as caught:
        ImplementationType(old)
    text = str(caught.value)
    for needle in ("pytorch", "miniworld", "cuequivariance", "anthropic", "routes per card automatically", "docs/dispatch.md"):
        assert needle in text, needle


def test_a_module_built_with_a_removed_string_raises():
    from miniworld_engine.modules import Transition

    with pytest.raises(ValueError, match="miniworld"):
        Transition(64, implementation="triton")


def test_an_unknown_string_is_still_an_error():
    with pytest.raises(ValueError, match="cutlass"):
        ImplementationType("cutlass")


@pytest.mark.parametrize(("old", "new"), [
    ("force_triton_kernels", "force_portable_kernels"),
    ("triton_on_hopper_blackwell", "portable_on_hopper_blackwell"),
    ("b200_engine_triton", "b200_portable_kernels"),
    ("engine_backend", "force_portable_kernels"),
])
def test_a_removed_setting_is_rejected_with_the_new_name(old, new):
    with pytest.raises(TypeError) as caught:
        settings.configure(**{old: True})
    assert old in str(caught.value) and repr(new) in str(caught.value)
    assert not hasattr(settings.current(), old)


def test_an_unknown_setting_is_rejected():
    with pytest.raises(TypeError, match="unknown setting"):
        settings.configure(no_such_setting=1)


@pytest.mark.parametrize("old", ["triton", "cuda"])
def test_trimul_impl_rejects_the_old_values(old):
    with pytest.raises(ValueError, match="portable"):
        settings.configure(trimul_impl=old)


@pytest.mark.parametrize(("route", "backend"), [("portable", KernelBackend.PORTABLE), ("native", KernelBackend.CUDA),
                                                ("pytorch", KernelBackend.PYTORCH)])
def test_trimul_impl_takes_the_neutral_vocabulary(route, backend):
    previous = settings.configure(trimul_impl=route)
    try:
        assert dispatch._trimul_known_best(None) is backend
    finally:
        settings.configure(**vars(previous))


# ---- (i) a record routes to Triton ------------------------------------------------------------------------------------
def test_a_recorded_card_routes_to_triton(monkeypatch, manifests):
    fake_card(monkeypatch, LONG, (8, 6))
    write_manifest(manifests, LONG, (8, 6), OPS)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for op in OPS:
            assert dispatch.resolve(op, "miniworld") is KernelBackend.PORTABLE, op
            assert dispatch.resolve(op, "miniworld", torch.device("cuda", 0)) is KernelBackend.PORTABLE, op


def test_one_missing_failed_or_wrong_dtype_kernel_unrecords_the_op(monkeypatch, manifests):
    fake_card(monkeypatch, LONG, (8, 6))
    victim = next(k for k in records.triton_kernels("transition") if records.kernel_mode(k) == "inference")
    write_manifest(manifests, LONG, (8, 6), ["transition"], skip=frozenset({victim}))
    assert records.missing("transition", LONG, (8, 6)) == (victim,)
    with pytest.warns(UnrecordedCardWarning, match=victim):
        assert dispatch.resolve("transition", "miniworld") is KernelBackend.PYTORCH
    write_manifest(manifests, LONG, (8, 6), ["transition"], status="failed")
    records._load.cache_clear()
    assert not records.recorded("transition", LONG, (8, 6))
    write_manifest(manifests, LONG, (8, 6), ["transition"], dtype="fp32")      # evidence at another precision is not evidence
    records._load.cache_clear()
    assert not records.recorded("transition", LONG, (8, 6), "bf16")


# ---- (ii) an unrecorded card warns once and runs PyTorch ------------------------------------------------------------
def test_an_unrecorded_card_warns_once_per_op_and_runs_pytorch(monkeypatch, manifests):
    fake_card(monkeypatch, "Brand New GPU", (8, 9))
    def twice():
        return dispatch.resolve("transition", "miniworld")

    results, caught = _recorded(UnrecordedCardWarning, twice, twice)
    assert results == [KernelBackend.PYTORCH, KernelBackend.PYTORCH]
    assert len(caught) == 1
    text = str(caught[0].message)
    for needle in ("transition", "Brand New GPU", "sm_89", "run_all", "force_portable_kernels", "Verify", "PyTorch"):
        assert needle in text, needle
    assert issubclass(UnrecordedCardWarning, UserWarning)
    with pytest.warns(UnrecordedCardWarning, match="triangle_attention"):   # another op on the same card warns on its own
        assert dispatch.resolve("triangle_attention", "miniworld") is KernelBackend.PYTORCH


def test_a_card_with_the_same_capability_and_another_name_is_not_recorded(monkeypatch, manifests):
    write_manifest(manifests, LONG, (8, 6), OPS)
    fake_card(monkeypatch, OTHER, (8, 6))
    with pytest.warns(UnrecordedCardWarning, match="A40"):
        assert dispatch.resolve("transition", "miniworld") is KernelBackend.PYTORCH
    fake_card(monkeypatch, LONG, (8, 6))
    assert dispatch.resolve("transition", "miniworld") is KernelBackend.PORTABLE


def test_the_same_name_at_another_capability_is_not_recorded(monkeypatch, manifests):
    write_manifest(manifests, LONG, (8, 6), OPS)
    fake_card(monkeypatch, LONG, (8, 9))
    with pytest.warns(UnrecordedCardWarning):
        assert dispatch.resolve("transition", "miniworld") is KernelBackend.PYTORCH


def test_an_unrecorded_card_other_requests_are_untouched(monkeypatch, manifests):
    fake_card(monkeypatch, "Brand New GPU", (8, 9))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert dispatch.resolve("transition", "pytorch") is KernelBackend.PYTORCH
        assert dispatch.resolve("triangle_multiplication", "cuequivariance") is KernelBackend.CUEQUIVARIANCE


def test_a_module_resolves_once_when_it_is_built(monkeypatch, manifests):
    from miniworld_engine.modules import Transition

    fake_card(monkeypatch, "Brand New GPU", (8, 9))
    with pytest.warns(UnrecordedCardWarning, match="transition"):
        module = Transition(64, implementation=ImplementationType.MINIWORLD)
    assert module._backend is KernelBackend.PYTORCH
    x = torch.randn(2, 8, 64)
    ref = Transition(64, implementation=ImplementationType.PYTORCH)
    ref.load_state_dict(module.state_dict())
    torch.testing.assert_close(module(x), ref(x))        # the PyTorch reference, bit for bit


# ---- mode-aware: a missing backward kernel blocks training only -------------------------------------------------------
def test_the_training_kernels_are_the_bwd_saveact_train_and_dw_ones():
    assert records.kernel_mode("trimul_output_f567_train_triton") == "training"
    assert records.kernel_mode("trimul_input_dual_bwd_triton") == "training"
    assert records.kernel_mode("cond_transition_expand_swiglu_saveact_triton") == "training"
    assert records.kernel_mode("swa_dit_swiglu_dw_triton") == "training"
    assert records.kernel_mode("transition_squeeze_residual_triton") == "inference"
    assert records.kernel_mode("adaln_fwd_gate_triton") == "inference"
    train = set(records.triton_kernels("triangle_multiplication", "bf16", "training"))
    infer = set(records.triton_kernels("triangle_multiplication", "bf16", "inference"))
    assert infer < train
    assert all(records.kernel_mode(k) == "training" for k in train - infer)


def test_an_inference_record_serves_inference_and_blocks_only_training(monkeypatch, manifests):
    from miniworld_engine.modules import TriangleMultiplication

    fake_card(monkeypatch, LONG, (8, 6))
    write_manifest(manifests, LONG, (8, 6), ["triangle_multiplication"], training=False)
    assert records.recorded("triangle_multiplication", LONG, (8, 6), mode="inference")
    assert not records.recorded("triangle_multiplication", LONG, (8, 6), mode="training")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert dispatch.resolve("triangle_multiplication", "miniworld") is KernelBackend.PORTABLE          # the build: inference
        module = TriangleMultiplication(64, implementation=ImplementationType.MINIWORLD)
        assert module._backend is KernelBackend.PORTABLE
        assert module._train_gate is not None
        x = torch.randn(1, 4, 4, 64)
        with torch.no_grad():
            assert dispatch.call_backend(module, module._backend, x) is KernelBackend.PORTABLE                # inference call
    with pytest.warns(UnrecordedCardWarning, match="backward"):
        assert dispatch.call_backend(module.train(), module._backend, x) is KernelBackend.PYTORCH           # training call
    with warnings.catch_warnings():
        warnings.simplefilter("error")                                                                       # warned once
        assert dispatch.call_backend(module.train(), module._backend, x) is KernelBackend.PYTORCH
    write_manifest(manifests, LONG, (8, 6), ["triangle_multiplication"], training=True)
    records._load.cache_clear()
    assert dispatch.train_gate("triangle_multiplication", "miniworld", KernelBackend.PORTABLE) is None


def test_a_card_missing_a_forward_kernel_blocks_inference_too(monkeypatch, manifests):
    fake_card(monkeypatch, LONG, (8, 6))
    victim = next(k for k in records.triton_kernels("transition") if records.kernel_mode(k) == "inference")
    write_manifest(manifests, LONG, (8, 6), ["transition"], skip=frozenset({victim}))
    with pytest.warns(UnrecordedCardWarning, match=victim):
        assert dispatch.resolve("transition", "miniworld") is KernelBackend.PYTORCH


def test_per_call_resolvers_ask_for_the_mode(monkeypatch, manifests):
    fake_card(monkeypatch, LONG, (8, 6))
    write_manifest(manifests, LONG, (8, 6), ["outer_product_mean"], training=False)
    assert dispatch.resolve("outer_product_mean", "miniworld", training=False) is KernelBackend.PORTABLE
    with pytest.warns(UnrecordedCardWarning):
        assert dispatch.resolve("outer_product_mean", "miniworld", training=True) is KernelBackend.PYTORCH


# ---- the ops.* entry points obey the same rule ------------------------------------------------------------------------
def test_ops_entry_points_are_blocked_on_an_unrecorded_card_only(monkeypatch, manifests):
    x = torch.randn(2, 8)
    dev = fake_card(monkeypatch, "Brand New GPU", (8, 9))
    with pytest.warns(UnrecordedCardWarning, match="transition"):
        assert dispatch.portable_blocked("transition", dev, x)
    fake_card(monkeypatch, LONG, (8, 6))
    write_manifest(manifests, LONG, (8, 6), ["transition"])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert not dispatch.portable_blocked("transition", dev, x)
    for name, capability in (("NVIDIA H100 80GB HBM3", (9, 0)), ("NVIDIA B200", (10, 0))):
        fake_card(monkeypatch, name, capability)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert not dispatch.portable_blocked("transition", dev, x)               # H100 / B200 keep their own rule
    fake_card(monkeypatch, "Brand New GPU", (8, 9))
    settings.configure(force_portable_kernels=True)
    assert not dispatch.portable_blocked("transition", dev, x)


def test_ops_swiglu_and_layer_norm_run_the_statements_on_an_unrecorded_card(monkeypatch, manifests):
    import torch.nn.functional as F

    from miniworld_engine import ops

    dev = fake_card(monkeypatch, "Brand New GPU", (8, 9))
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))   # the tensors are CPU; the card is faked
    monkeypatch.setattr(dispatch, "_card_of", lambda device=None: ("Brand New GPU", (8, 9)))
    x, wa, wb, ws = torch.randn(4, 16), torch.randn(32, 16), torch.randn(32, 16), torch.randn(16, 32)
    with pytest.warns(UnrecordedCardWarning):
        out = ops.swiglu_ffn(x, wa, wb, ws)
    torch.testing.assert_close(out, F.linear(F.silu(F.linear(x, wa)) * F.linear(x, wb), ws))
    w, b = torch.randn(16), torch.randn(16)
    with pytest.warns(UnrecordedCardWarning):
        out = ops.layer_norm(x, w, b, 1e-5)
    torch.testing.assert_close(out, F.layer_norm(x, (16,), w, b, 1e-5))
    assert dev.type == "cuda"


def test_layernorm_follows_the_rule_off_h100_and_b200(monkeypatch, manifests):
    fake_card(monkeypatch, LONG, (8, 6))
    write_manifest(manifests, LONG, (8, 6), ["layernorm"])
    assert dispatch.resolve("layernorm", "miniworld") is KernelBackend.CUDA
    fake_card(monkeypatch, "Brand New GPU", (8, 9))
    with pytest.warns(UnrecordedCardWarning, match="layernorm"):
        assert dispatch.resolve("layernorm", "miniworld") is KernelBackend.PYTORCH
    fake_card(monkeypatch, "NVIDIA H100 80GB HBM3", (9, 0))
    assert dispatch.resolve("layernorm", "miniworld") is KernelBackend.CUDA


# ---- (iii) a CUDA path that serves the call wins ----------------------------------------------------------------------
def test_a_native_cuda_path_needs_no_triton_record(monkeypatch, manifests):
    from miniworld_engine.modules import Transition

    fake_card(monkeypatch, "NVIDIA A100 80GB PCIe", (8, 0))
    write_manifest(manifests, "NVIDIA A100 80GB PCIe", (8, 0), ["layernorm"])      # the LayerNorm inside is recorded; Transition's kernels are not
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert Transition(128, n=4, implementation=ImplementationType.MINIWORLD)._backend is KernelBackend.PORTABLE  # fused sm_80 CUDA
        assert dispatch.resolve("transition", "miniworld", native_cuda=True) is KernelBackend.PORTABLE
    with pytest.warns(UnrecordedCardWarning):      # d = 64: no CUDA kernel, no record
        assert Transition(64, n=4, implementation=ImplementationType.MINIWORLD)._backend is KernelBackend.PYTORCH


def test_a_triton_record_does_not_displace_cuda_on_h100_and_b200(monkeypatch, manifests):
    for name, capability in (("NVIDIA H100 80GB HBM3", (9, 0)), ("NVIDIA B200", (10, 0))):
        fake_card(monkeypatch, name, capability)
        write_manifest(manifests, name, capability, OPS)
        assert dispatch.restricted_card()
        assert not dispatch.portable_leg_allowed(None, "transition")
        assert not dispatch.portable_leg_allowed(torch.device("cuda", 0), "triangle_multiplication")


# ---- (iv) H100 / B200 are unchanged -----------------------------------------------------------------------------------
@pytest.mark.parametrize(("name", "capability"), [("NVIDIA H100 80GB HBM3", (9, 0)), ("NVIDIA H200", (9, 0)), ("NVIDIA B200", (10, 0))])
def test_h100_and_b200_resolve_as_before_and_never_warn(monkeypatch, manifests, name, capability):
    fake_card(monkeypatch, name, capability)               # no manifest at all: still no warning
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for op in ("transition", "triangle_multiplication", "triangle_attention", "augmented_attention", "outer_product_mean",
                   "msa_pair_weighted_averaging"):
            assert dispatch.resolve(op, "miniworld") is KernelBackend.PORTABLE     # the family; the modules take their CUDA / PyTorch leg
        for op in ("adaptive_layernorm", "conditioned_transition"):
            assert dispatch.resolve(op, "miniworld") is (KernelBackend.CUDA if capability == (9, 0) else KernelBackend.PYTORCH)   # H100: sm_90a kernels
        assert dispatch.resolve("layernorm", "miniworld") is KernelBackend.CUDA
        assert dispatch.resolve("rmsnorm", "miniworld") is KernelBackend.CUDA
        assert dispatch.restricted_card()
        assert not dispatch.portable_leg_allowed(None)
    settings.configure(portable_on_hopper_blackwell=True)
    assert not dispatch.restricted_card()
    assert dispatch.portable_leg_allowed(None, "transition")


# ---- (vi) the A/B switch ----------------------------------------------------------------------------------------------
def test_the_ab_switch_forces_triton_on_any_card_without_a_warning(monkeypatch, manifests):
    fake_card(monkeypatch, "Brand New GPU", (8, 9))
    settings.configure(force_portable_kernels=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for op in OPS:
            assert dispatch.resolve(op, "miniworld") is KernelBackend.PORTABLE
        assert dispatch.portable_leg_allowed(None, "swa_dit")
    fake_card(monkeypatch, "NVIDIA H100 80GB HBM3", (9, 0))
    assert dispatch.resolve("adaptive_layernorm", "miniworld") is KernelBackend.PORTABLE 
    assert not dispatch.restricted_card()


def test_an_autotune_build_may_use_triton_on_a_new_card(monkeypatch, manifests):
    fake_card(monkeypatch, "Brand New GPU", (8, 9))
    settings.configure(capture=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert dispatch.resolve("transition", "miniworld") is KernelBackend.PORTABLE


def test_without_a_gpu_the_triton_family_stays(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for op in OPS:
            assert dispatch.resolve(op, "miniworld") is KernelBackend.PORTABLE
        assert dispatch.resolve("transition", "miniworld", torch.device("cpu")) is KernelBackend.PORTABLE


# ---- the records themselves -------------------------------------------------------------------------------------------
def test_the_packaged_manifests_are_the_recorded_cards():
    assert set(records.recorded_cards()) == {
        "NVIDIA A100 80GB PCIe (sm80)", "NVIDIA H100 80GB HBM3 (sm90)", "NVIDIA RTX A5000 (sm86)", "NVIDIA RTX A6000 (sm86)",
    }
    assert records.recorded("triangle_attention", "NVIDIA RTX A6000", (8, 6))
    assert not records.recorded("triangle_attention", "NVIDIA A40", (8, 6))
    assert not records.recorded("triangle_attention", "NVIDIA RTX A6000", (8, 9))


def test_every_op_in_the_table_names_registry_kernels():
    for op in records.OP_KERNELS:
        assert records.triton_kernels(op), op
    assert records.triton_kernels("no_such_op") == ()
    assert records.missing("no_such_op", "x", (8, 6))      # an op nothing vouches for is never recorded


def test_the_undeveloped_alternatives_are_not_held_against_an_op():
    from miniworld_engine.autotune import devices

    undeveloped = {r["kernel"] for r in devices.registry() if r["developed"] != "yes"}
    for op in records.OP_KERNELS:
        assert not undeveloped & set(records.triton_kernels(op))


def test_importing_the_records_needs_no_gpu_and_no_torch_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert records.card_key("NVIDIA RTX A6000", (8, 6)) == "NVIDIA RTX A6000 (sm86)"
    assert records.card_key("NVIDIA A/B", (8, 0)) == "NVIDIA A_B (sm80)"


# ---- a kernel REFUSING a call is a warning and the PyTorch reference, never an error ---------------------------------------
@pytest.fixture
def cuda_tensors(monkeypatch):
    """The tensors in these tests are CPU ones on a faked card: they must read as CUDA to reach the Triton leg's guards."""
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))


def test_protenix_template_trimul_with_a_wider_hidden_is_served_and_out_of_range_widths_warn(monkeypatch, manifests, cuda_tensors):
    """pair 64 / d_hidden 128 (Protenix's / FoldForge's template TriMul): the portable front serves it from the weights, no refusal; a
    width outside `contract.hidden_refusal`'s range is still a refusal (one warning, once) and the PyTorch reference."""
    from miniworld_engine.modules import (
        BidirectionalTriangleMultiplication,
        TriangleMultiplication,
    )
    from miniworld_engine.modules.dispatch import KernelRefusedWarning

    fake_card(monkeypatch, LONG, (8, 6))
    write_manifest(manifests, LONG, (8, 6), ["triangle_multiplication"])      # LayerNorm unrecorded: it takes the PyTorch statements
    x = torch.randn(1, 8, 8, 64).bfloat16()
    served = []
    for build in (TriangleMultiplication, BidirectionalTriangleMultiplication):
        monkeypatch.setattr(build, "_forward_triton", lambda self, pair, mask, ds=None: (served.append(self.d_hidden), pair)[1])
        extra = {"p_drop": 0.0} if build is BidirectionalTriangleMultiplication else {}
        dispatch._refused_warned.clear()
        mod = build(64, d_hidden=128, implementation=ImplementationType.MINIWORLD, **extra).bfloat16().eval()
        (y, _), caught = _recorded(KernelRefusedWarning, partial(mod, x), partial(mod, x))
        assert caught == [] and served[-1] == 128                                  # the portable path was reached, no refusal
        assert y is x or torch.equal(y, x)

        dispatch._refused_warned.clear()
        wide = build(64, d_hidden=1024, implementation=ImplementationType.MINIWORLD, **extra).bfloat16().eval()
        ref = build(64, d_hidden=1024, implementation=ImplementationType.PYTORCH, **extra).bfloat16().eval()
        ref.load_state_dict(wide.state_dict())
        n_served = len(served)
        (y, _), caught = _recorded(KernelRefusedWarning, partial(wide, x), partial(wide, x))     # the second call is silent
        assert len(caught) == 1 and len(served) == n_served
        assert "d_hidden (1024)" in str(caught[0].message)
        torch.testing.assert_close(y, ref(x))


def test_opm_refusal_warns_once_and_runs_pytorch(monkeypatch, manifests, cuda_tensors):
    from miniworld_engine.modules.dispatch import KernelRefusedWarning
    from miniworld_engine.modules.outer_product import OuterProductMean

    fake_card(monkeypatch, LONG, (8, 6))
    write_manifest(manifests, LONG, (8, 6), ["outer_product_mean"])
    mod = OuterProductMean(64, 128, 32, implementation=ImplementationType.MINIWORLD)
    ref = OuterProductMean(64, 128, 32, implementation=ImplementationType.PYTORCH)
    ref.load_state_dict(mod.state_dict())
    x = torch.randn(1, 8, 16, 64)                      # fp32: the Triton path is bf16
    (y, _), caught = _recorded(KernelRefusedWarning, partial(mod, x), partial(mod, x))
    assert len(caught) == 1
    assert "bf16" in str(caught[0].message)
    torch.testing.assert_close(y, ref(x))


def test_pwa_refusal_warns_once_and_runs_pytorch(monkeypatch, manifests, cuda_tensors):
    """A head the Triton path does not take (one head of 8 channels: 8 is not a multiple of 16) refuses with a warning, once. value_dim 8
    with 8 heads, AF3's, is served now (it used to be refused here)."""
    from miniworld_engine.modules.dispatch import KernelRefusedWarning
    from miniworld_engine.modules.msa_pair_weighted_averaging import (
        MSAPairWeightedAveraging,
    )

    fake_card(monkeypatch, LONG, (8, 6))
    write_manifest(manifests, LONG, (8, 6), ["msa_pair_weighted_averaging"])
    mod = MSAPairWeightedAveraging(64, 128, 1, 8, implementation=ImplementationType.MINIWORLD).bfloat16().eval()
    ref = MSAPairWeightedAveraging(64, 128, 1, 8, implementation=ImplementationType.PYTORCH).bfloat16().eval()
    ref.load_state_dict(mod.state_dict())
    msa, pair = torch.randn(1, 4, 16, 64).bfloat16(), torch.randn(1, 16, 16, 128).bfloat16()
    (y, _), caught = _recorded(KernelRefusedWarning, partial(mod, msa, pair), partial(mod, msa, pair))
    assert len(caught) == 1
    assert "multiple of 16" in str(caught[0].message)
    torch.testing.assert_close(y, ref(msa, pair))


def test_a_refusal_on_a_cpu_tensor_is_silent(monkeypatch, manifests):
    from miniworld_engine.modules.outer_product import OuterProductMean

    fake_card(monkeypatch, LONG, (8, 6))
    write_manifest(manifests, LONG, (8, 6), ["outer_product_mean", "layernorm"])
    mod = OuterProductMean(64, 128, 32, implementation=ImplementationType.MINIWORLD)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mod(torch.randn(1, 8, 16, 64))
