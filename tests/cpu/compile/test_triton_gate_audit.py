"""The record gate covers what the portable paths launch (docs/records/triton_gate_audit_20261031.md) -- CPU only, fake card, fake manifests.

* ``OP_KERNELS`` is checked against the derived launches (``kernels/registry/registry_kernel.units.json``: ``dev derive`` runs every
  ``registry_module.csv`` row and records each Triton launch, per dtype and eval / train): every kernel a unit launches is one its op
  counts, at that precision and in that mode;
* the ``ops.*`` facades that launched Triton directly (``rms_norm_modulation``, ``layer_norm_linear``, ``gated_residual``, the two TriMul
  updates, ``conditioned_transition``) run their PyTorch statements with one warning on a card the engine holds no record for;
* the kernels without a registry row (``UNREGISTERED_OPS``) are never recorded, and the precision the facades ask at is the call's.
"""
from __future__ import annotations

import json
import warnings

import pytest
import torch
import torch.nn.functional as F
from fake_card import fake_card, write_manifest
from paths import ROOT

from miniworld_engine import settings
from miniworld_engine.autotune import records
from miniworld_engine.modules import dispatch
from miniworld_engine.modules.dispatch import KernelRefusedWarning, UnrecordedCardWarning

NEW = "Brand New GPU"
LONG = "NVIDIA RTX A6000"

#: registry_module.csv module -> the op whose gate its portable path asks (pairformer_block: the union of its three trunk ops)
MODULE_OPS = {
    "adaptive_layernorm": ("adaptive_layernorm",),
    "augmented_attention": ("augmented_attention",),
    "attention_pair_bias": ("augmented_attention",),
    "conditioned_transition": ("conditioned_transition",),
    "gated_linear": ("gated_linear",),
    "layernorm_linear_atom_output": ("layer_norm_linear",),
    "layernorm_linear_native": ("layer_norm_linear",),
    "layernorm_native": ("layernorm",),
    "msa_pair_weighted_averaging": ("msa_pair_weighted_averaging",),
    "outer_product_mean": ("outer_product_mean",),
    "pairformer_block": ("transition", "triangle_multiplication", "triangle_attention"),
    "projected_attention": ("augmented_attention_pair_bias",),
    "rms_norm_modulation": ("rms_norm_modulation",),
    "swa_atom_attention": ("swa_atom_attention",),
    "swiglu_ffn": ("swiglu_ffn",),
    "transition": ("transition",),
    "triangle_attention_bidirectional": ("triangle_attention",),
    "triangle_attention_checkpoint": ("triangle_attention",),
    "triangle_attention_heads": ("triangle_attention",),
    "triangle_multiplication": ("triangle_multiplication",),
    "triangle_multiplication_bidirectional": ("triangle_multiplication",),
}


@pytest.fixture
def manifests(tmp_path):
    with records.use_manifest_dir(tmp_path):
        yield tmp_path


@pytest.fixture(autouse=True)
def _clean():
    previous = settings.current()
    settings.configure(force_portable_kernels=False, portable_on_hopper_blackwell=False, capture=False)
    for cache in (dispatch._card_cache, dispatch._unrecorded_warned, dispatch._refused_warned):
        cache.clear()
    yield
    settings.configure(**{k: getattr(previous, k) for k in ("force_portable_kernels", "portable_on_hopper_blackwell", "capture")})
    for cache in (dispatch._card_cache, dispatch._unrecorded_warned, dispatch._refused_warned):
        cache.clear()


def _units():
    path = ROOT / "src" / "miniworld_engine" / "kernels" / "registry" / "registry_kernel.units.json"
    return json.loads(path.read_text())["units"]


def _declared():
    return {r["kernel"]: {d.strip() for d in (r.get("dtypes") or "bf16").split("|")} for r in records._registry()
            if r["backend"] == "triton"}


def test_every_derived_launch_is_counted_by_its_op():
    declared, holes = _declared(), []
    for unit, launches in _units().items():
        head, mode = unit.split("]")[0], ("training" if " train" in unit else "inference")
        module, impl_dtype = head.split("[")
        dtype = "fp32" if impl_dtype.split("/")[1].split("->")[0] == "float32" else "bf16"
        ops = MODULE_OPS.get(module)
        assert ops is not None, f"registry_module.csv module {module!r} is not mapped to the op that gates it"
        counted = {k for op in ops for k in records.triton_kernels(op, dtype, mode)}
        for launch in launches:
            kernel = launch.split("|")[0]
            if kernel.startswith("<unautotuned:"):
                continue                      # a helper kernel with no registry row, launched inside an op's gated path (audit)
            if dtype not in declared.get(kernel, set()):
                continue                      # the row does not declare this precision (a fp32 accumulator kernel under a bf16 call)
            if kernel not in counted:
                holes.append(f"{module} {dtype} {mode}: {kernel}")
    assert not holes, "launched but not counted by the op's gate:\n  " + "\n  ".join(sorted(set(holes)))


def test_the_layernorm_forward_that_saves_its_statistics_counts_for_inference():
    assert records.kernel_mode("layernorm_fwd_saveact_triton") == "inference"
    assert records.kernel_mode("layernorm_fwd_saveact_strided_triton") == "training"
    for op in ("transition", "triangle_multiplication", "triangle_attention", "augmented_attention", "layernorm"):
        assert "layernorm_fwd_saveact_triton" in records.triton_kernels(op, "bf16", "inference"), op


def test_every_registered_family_is_counted_by_some_op():
    counted = {k for op in records.OP_KERNELS for d in ("bf16", "fp32") for m in ("inference", "training")
               for k in records.triton_kernels(op, d, m)}
    built = {r["kernel"] for r in records._registry() if r["backend"] == "triton" and r["developed"] == "yes"}
    assert not built - counted, sorted(built - counted)


def test_an_unregistered_op_is_never_recorded(manifests, monkeypatch):
    monkeypatch.setitem(records.UNREGISTERED_OPS, "some_op", ("kernels/x.py:_k",))    # the mechanism; the table is empty today
    for op, kernels in records.UNREGISTERED_OPS.items():
        assert op not in records.OP_KERNELS
        assert records.missing(op, LONG, (8, 6)) == tuple(f"<unregistered: {k}>" for k in kernels)
        assert not records.recorded(op, LONG, (8, 6), mode="inference")


def test_the_facades_ask_at_the_call_precision(monkeypatch, manifests):
    dev = fake_card(monkeypatch, LONG, (8, 6))
    write_manifest(manifests, LONG, (8, 6), ["rms_norm_modulation"])                       # bf16 rows only
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert not dispatch.portable_blocked("rms_norm_modulation", dev, torch.zeros(2, 8, dtype=torch.bfloat16))
    with pytest.warns(UnrecordedCardWarning, match="rmsnorm_adamod"):
        assert dispatch.portable_blocked("rms_norm_modulation", dev, torch.zeros(2, 8))      # fp32: no fp32 row
    assert dispatch.record_dtype(torch.zeros(1, dtype=torch.float16)) == "bf16"


@pytest.fixture
def unrecorded(monkeypatch, manifests):
    """A card the engine holds no record for, with CPU tensors that report ``is_cuda`` (the blocked legs are PyTorch statements)."""
    fake_card(monkeypatch, NEW, (8, 9))
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setattr(dispatch, "_card_of", lambda device=None: (NEW, (8, 9)))
    return NEW


def test_ops_rms_norm_modulation_runs_the_reference_on_an_unrecorded_card(unrecorded):
    from miniworld_engine import ops
    from miniworld_engine.kernels.rmsnorm.reference import rmsnorm_adamod_reference

    torch.manual_seed(0)
    q, c = torch.randn(3, 8, 16), torch.randn(3, 8, 12)
    w = [torch.randn(16, 12) * 0.1 for _ in range(3)]
    with pytest.warns(UnrecordedCardWarning, match="rms_norm_modulation"):
        y, g = ops.rms_norm_modulation(q, c, *w, eps=1e-6)
    ry, rg = rmsnorm_adamod_reference(q, c, *w, None, 1e-6)
    torch.testing.assert_close(y, ry)
    torch.testing.assert_close(g, rg)


def test_ops_layer_norm_linear_and_gated_residual_run_the_statements_on_an_unrecorded_card(unrecorded):
    from miniworld_engine import ops

    torch.manual_seed(0)
    x, lnw, pw = torch.randn(2, 5, 5, 16), torch.randn(16), torch.randn(4, 16)
    with pytest.warns(UnrecordedCardWarning, match="layer_norm_linear"):
        out = ops.layer_norm_linear(x, lnw, pw, 1e-5)
    torch.testing.assert_close(out, F.linear(F.layer_norm(x, (16,), lnw, None, 1e-5), pw))
    g, b = torch.randn_like(x), torch.randn_like(x)
    with pytest.warns(UnrecordedCardWarning, match="gated_residual"):
        out = ops.gated_residual(x, g, b)
    torch.testing.assert_close(out, x + g * b)


def test_ops_conditioned_transition_runs_the_statements_on_an_unrecorded_card(unrecorded):
    from miniworld_engine import ops

    torch.manual_seed(0)
    x, cond = torch.randn(2, 6, 16), torch.randn(2, 6, 8)
    wa, wb, ws, wsc, bsc = torch.randn(32, 16), torch.randn(32, 16), torch.randn(16, 32), torch.randn(16, 8), torch.randn(16)
    with pytest.warns(UnrecordedCardWarning, match="swiglu_ffn"):
        out = ops.conditioned_transition(x, cond, expand_a_weight=wa, expand_b_weight=wb, squeeze_weight=ws, to_scale_weight=wsc,
                                         to_scale_bias=bsc, n=2)
    ref = torch.sigmoid(F.linear(cond, wsc, bsc)) * F.linear(F.silu(F.linear(x, wa)) * F.linear(x, wb), ws)
    torch.testing.assert_close(out, ref)


@pytest.mark.parametrize("direction", ["outgoing", "incoming"])
@pytest.mark.parametrize("pair_mask", [False, True])
def test_ops_triangle_multiplicative_update_runs_the_reference_on_an_unrecorded_card(unrecorded, direction, pair_mask):
    from miniworld_engine import ops
    from miniworld_engine.modules.triangle_multiplication.reference import (
        triangle_multiplicative_update_pytorch,
    )

    torch.manual_seed(0)
    d, L = 16, 7
    x = torch.randn(1, L, L, d)
    res = torch.rand(1, L) > 0.2
    mask = (res[:, :, None] & res[:, None, :]) if pair_mask else res
    w = {"norm_in_weight": torch.randn(d), "norm_in_bias": torch.randn(d), "p_in_weight": torch.randn(2 * d, d) * 0.2,
             "g_in_weight": torch.randn(2 * d, d) * 0.2, "norm_out_weight": torch.randn(d), "norm_out_bias": torch.randn(d),
             "p_out_weight": torch.randn(d, d) * 0.2, "g_out_weight": torch.randn(d, d) * 0.2}
    with pytest.warns(KernelRefusedWarning, match="bf16 only"):
        out = ops.triangle_multiplicative_update(x, direction, mask, **w)
    m2d = res[:, :, None] & res[:, None, :]
    ref = x + triangle_multiplicative_update_pytorch(x, direction, m2d, **w)
    torch.testing.assert_close(out, ref)


def test_ops_bidirectional_triangle_multiplicative_update_matches_the_module_reference(monkeypatch, manifests):
    from miniworld_engine import ops
    from miniworld_engine.modules.exceptions import ImplementationType
    from miniworld_engine.modules.triangle_multiplication.bidirectional import (
        BidirectionalTriangleMultiplication,
    )

    torch.manual_seed(0)
    d, L = 16, 6
    module = BidirectionalTriangleMultiplication(d, implementation=ImplementationType.PYTORCH, p_drop=0.0).eval()
    for p in module.parameters():
        p.data.normal_(0, 0.2)
    x = torch.randn(1, L, L, d)
    mask = torch.rand(1, L) > 0.2
    with torch.no_grad():
        ref = module(x, mask)
        fake_card(monkeypatch, NEW, (8, 9))                      # the reference ran first, on a plain CPU host
        monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
        monkeypatch.setattr(dispatch, "_card_of", lambda device=None: (NEW, (8, 9)))
        with pytest.warns(KernelRefusedWarning, match="bf16 only"):
            out = ops.bidirectional_triangle_multiplicative_update(
                x, mask, norm_in_weight=module.ln_pair.weight, norm_in_bias=module.ln_pair.bias, to_left_weight=module.to_left.weight,
                to_left_gate_weight=module.to_left_gate.weight, to_right_weight=module.to_right.weight,
                to_right_gate_weight=module.to_right_gate.weight, norm_out_weight=module.ln_out.weight, norm_out_bias=module.ln_out.bias,
                to_out_weight=module.to_out.weight, to_gate_weight=module.to_gate.weight, eps=module.ln_pair.eps)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


def test_the_new_gates_leave_h100_and_b200_alone(monkeypatch, manifests):
    x = torch.zeros(2, 8, dtype=torch.bfloat16)
    for name, capability in (("NVIDIA H100 80GB HBM3", (9, 0)), ("NVIDIA B200", (10, 0))):
        dev = fake_card(monkeypatch, name, capability)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            for op in ("rms_norm_modulation", "layer_norm_linear", "gated_residual", "bias_only_pair_bias", "bias_only_core",
                       "swa_dit", "mpnn_message", "mpnn_edge_mlp", "triangle_multiplication", "swiglu_ffn"):
                assert not dispatch.portable_blocked(op, dev, x), (name, op)


def test_mpnn_auto_asks_the_record_and_an_explicit_backend_does_not(unrecorded):
    from miniworld_engine.kernels.mpnn_message import interface

    p = torch.zeros(2, 48, 128, dtype=torch.bfloat16)
    with pytest.warns(UnrecordedCardWarning, match="mpnn_message"):
        assert interface._auto_unrecorded(p, torch.zeros(128, 128), torch.zeros(128))


def test_a_bf16_trimul_facade_call_asks_the_record(unrecorded):
    """fp32 is a refusal (above: the kernels are bf16 only); a bf16 call on a card without a record is the unrecorded warning."""
    from miniworld_engine import ops

    d, L = 16, 5
    x = torch.randn(1, L, L, d).bfloat16()
    w = dict(norm_in_weight=torch.ones(d), norm_in_bias=torch.zeros(d), p_in_weight=torch.randn(2 * d, d).bfloat16(),
             g_in_weight=torch.randn(2 * d, d).bfloat16(), norm_out_weight=torch.ones(d), norm_out_bias=torch.zeros(d),
             p_out_weight=torch.randn(d, d).bfloat16(), g_out_weight=torch.randn(d, d).bfloat16())
    with pytest.warns(UnrecordedCardWarning, match="triangle_multiplication"):
        out = ops.triangle_multiplicative_update(x, "outgoing", None, **w)
    assert out.shape == x.shape
    assert out.dtype == torch.bfloat16


def test_every_kernel_with_a_row_and_a_former_unregistered_op_is_counted():
    """gated_residual and the bias-only pair bias have registry rows now (2026-10-31): their ops count them."""
    assert set(records.triton_kernels("gated_residual")) == {
        "gated_projection_fwd_residual_triton", "gated_projection_bwd_residual_triton"}
    assert set(records.triton_kernels("bias_only_pair_bias")) == {
        "bias_only_attention_layernorm_gemm_triton", "bias_only_attention_bwd_layernorm_gemm_triton"}
    assert set(records.triton_kernels("bias_only_pair_bias", "bf16", "inference")) == {"bias_only_attention_layernorm_gemm_triton"}
    assert "mpnn_message_fwd_gemm_gelu_reduce_triton" in records.triton_kernels("mpnn_message", "bf16", "inference")
    # the family-wide patterns of TriangleAttention / ops.gated_linear do not pick up the new rows
    for op in ("triangle_attention", "gated_linear"):
        assert not {"gated_projection_fwd_residual_triton", "bias_only_attention_layernorm_gemm_triton"} & set(records.triton_kernels(op))
    assert "augmented_attention_bwd_atomic_triton" in records.triton_kernels("augmented_attention_pair_bias")   # memory_efficient


def test_the_split_backward_reduce_is_asked_at_its_own_precision(monkeypatch, manifests):
    """augmented_attention_bwd_reduce_triton sums fp32 partials whatever the call's precision: a bf16 training call counts it, and asks
    its fp32 row."""
    assert "augmented_attention_bwd_reduce_triton" in records.triton_kernels("augmented_attention_pair_bias", "bf16", "training")
    assert "augmented_attention_bwd_reduce_triton" not in records.triton_kernels("augmented_attention_pair_bias", "bf16", "inference")
    assert records.row_dtype("augmented_attention_bwd_reduce_triton", "bf16") == "fp32"
    assert records.row_dtype("augmented_attention_fwd_triton", "bf16") == "bf16"
    fake_card(monkeypatch, LONG, (8, 6))
    write_manifest(manifests, LONG, (8, 6), ["augmented_attention_pair_bias"])            # writes that kernel's row at fp32
    assert records.recorded("augmented_attention_pair_bias", LONG, (8, 6), "bf16", "training")
    path = manifests / f"{records.card_key(LONG, (8, 6))}.csv"
    text = path.read_text().replace("augmented_attention_bwd_reduce_triton,triton,x,x.py,ok,,fp32",
                                    "augmented_attention_bwd_reduce_triton,triton,x,x.py,ok,,bf16")
    path.write_text(text)
    records._load.cache_clear()
    assert records.missing("augmented_attention_pair_bias", LONG, (8, 6), "bf16", "training") == ("augmented_attention_bwd_reduce_triton",)


def test_the_early_prunes_keep_the_launch_the_kernels_had_before_they_were_tuned():
    """Outside a build, the default config sets prune to the fixed launches of before: the pair-bias backward's warps by channel count,
    the MPNN no-grad fusion's policy by group count (and never a row tile that cannot hold its groups)."""
    import triton

    from miniworld_engine.kernels.bias_only_attention.triton.pair_bias import _bwd_prune
    from miniworld_engine.kernels.mpnn_message.triton.inference import _prune

    warps = [triton.Config({}, num_warps=4, num_stages=2), triton.Config({}, num_warps=8, num_stages=2)]
    assert [c.num_warps for c in _bwd_prune(warps, {}, C=128)] == [4]
    assert [c.num_warps for c in _bwd_prune(warps, {}, C=256)] == [8]
    policy = [triton.Config({"GROUPS_PER_CTA": 1, "BLOCK_M": 64, "BLOCK_K": 16, "COALESCED_WEIGHT_LOAD": 1}, num_warps=2),
              triton.Config({"GROUPS_PER_CTA": 2, "BLOCK_M": 128, "BLOCK_K": 32, "COALESCED_WEIGHT_LOAD": 0}, num_warps=4),
              triton.Config({"GROUPS_PER_CTA": 2, "BLOCK_M": 64, "BLOCK_K": 32, "COALESCED_WEIGHT_LOAD": 0}, num_warps=4)]
    assert [c.kwargs["GROUPS_PER_CTA"] for c in _prune(policy, {"groups": 100})] == [1]
    assert [(c.kwargs["GROUPS_PER_CTA"], c.kwargs["BLOCK_M"]) for c in _prune(policy, {"groups": 4096})] == [(2, 128)]
    settings.configure(run_autotune=True)
    try:
        assert len(_prune(policy, {"groups": 100})) == 2          # a build searches every valid config
        assert len(_bwd_prune(warps, {}, C=128)) == 2
    finally:
        settings.configure(run_autotune=False)


def test_the_mpnn_no_grad_fusion_counts_for_inference_only():
    """`mpnn_message_fwd_gemm_gelu_reduce_triton` runs only with autograd off: an inference call needs its row, a training call never
    launches it (so an A5000, which holds the four training-path rows and not this one, stays portable for training)."""
    infer = set(records.triton_kernels("mpnn_message", "bf16", "inference"))
    train = set(records.triton_kernels("mpnn_message", "bf16", "training"))
    assert "mpnn_message_fwd_gemm_gelu_reduce_triton" in infer
    assert "mpnn_message_fwd_gemm_gelu_reduce_triton" not in train
    assert records.recorded("mpnn_message", "NVIDIA RTX A5000", (8, 6), mode="training")
