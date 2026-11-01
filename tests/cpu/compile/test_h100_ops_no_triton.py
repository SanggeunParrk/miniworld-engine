"""H100 / B200 launch no Triton kernel through the functional ``ops.*`` facades and the module helpers around the CUDA paths
(docs/records/h100_ops_gate_20261101.md) -- CPU only, fake H100 card, every Triton entry point poisoned.

``modules.dispatch.portable_blocked`` is False on cc 9 / 10 (the record gate does not apply there), so a facade that only asked it kept its
Triton kernel on H100 / B200. Each facade now asks ``restricted_card`` first: the CUDA kernel where one serves the call, the PyTorch
statements otherwise. The poisoned names raise if anything calls them; ``settings.portable_on_hopper_blackwell`` (A/B work) brings them back.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from fake_card import fake_card

from miniworld_engine import settings
from miniworld_engine.modules import dispatch

H100 = "NVIDIA H100 80GB HBM3"
K = "miniworld_engine.kernels."

#: every Triton entry a facade used to reach (module, name): any call on a fake H100 is a failure
TRITON_ENTRIES = (
    (K + "transition.triton.fused", "triton_swiglu_ffn"),
    (K + "layernorm.triton.main", "triton_layernorm"),
    (K + "layernorm.cuda.masked_bwd", "layernorm_cuda_bwd_autograd"),
    (K + "rmsnorm.triton.main", "triton_rmsnorm_adamod"),
    (K + "rmsnorm.interface", "triton_rmsnorm"),
    (K + "layernorm_linear.triton.pair_bias", "triton_layer_norm_linear"),
    (K + "gated_projection.triton.residual", "gated_residual"),
    (K + "trimul_inproj.triton.unidirectional", "trimul_triton"),
    (K + "trimul_inproj.triton.bidirectional", "bidirectional_trimul_triton"),
    (K + "triangle_attention.interface", "triton_triangle_attention_pair_bias"),
    (K + "transition.triton.main", "triton_transition"),
    (K + "augmented_attention.interface", "triton_augmented_attention_pair_bias"),
    (K + "bias_only_attention.interface", "fused_gate_out"),
    (K + "bias_only_attention.interface", "sigmoid_gate_fused"),
    (K + "bias_only_attention.triton.pair_bias", "pair_bias_logits"),
    (K + "bias_only_attention.triton.core_t", "bias_only_core_t"),
)


def _boom(name):
    def run(*args, **kwargs):
        raise AssertionError(f"{name} (Triton) was called on H100 / B200")
    return run


def _install(monkeypatch):
    """A fake H100 whose CPU tensors report ``is_cuda``; every Triton entry point raises. Returns the settings to restore."""
    fake_card(monkeypatch, H100, (9, 0))
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setattr(dispatch, "_card_of", lambda device=None: (H100, (9, 0)))
    for module, name in TRITON_ENTRIES:
        monkeypatch.setattr(f"{module}.{name}", _boom(f"{module}.{name}"))
    previous = settings.configure(force_portable_kernels=False, portable_on_hopper_blackwell=False)
    for cache in (dispatch._card_cache, dispatch._unrecorded_warned, dispatch._refused_warned):
        cache.clear()
    return previous


@pytest.fixture
def h100(monkeypatch):
    previous = _install(monkeypatch)
    yield
    settings.configure(portable_on_hopper_blackwell=previous.portable_on_hopper_blackwell,
                       force_portable_kernels=previous.force_portable_kernels)


def test_the_gate_the_facades_ask_is_off_on_h100(h100):
    """The reason for this file: portable_blocked says False on cc 9 / 10, restricted_card says True."""
    x = torch.zeros(2, 8, dtype=torch.bfloat16)
    assert dispatch.restricted_card(x.device)
    assert not dispatch.portable_blocked("swiglu_ffn", x.device, x)


def test_swiglu_ffn_runs_the_statements(h100):
    from miniworld_engine import ops

    torch.manual_seed(0)
    x, wa, wb, ws = torch.randn(2, 5, 16), torch.randn(32, 16), torch.randn(32, 16), torch.randn(16, 32)
    torch.testing.assert_close(ops.swiglu_ffn(x, wa, wb, ws), F.linear(F.silu(F.linear(x, wa)) * F.linear(x, wb), ws))


@pytest.mark.parametrize("row_scale", [False, True])
def test_layer_norm_takes_the_general_cuda_layernorm_where_it_serves(h100, monkeypatch, row_scale):
    from miniworld_engine import ops

    calls = []

    def fake_cuda(x, weight, bias, eps=1e-5):
        calls.append(tuple(x.shape))
        return F.layer_norm(x, (x.shape[-1],), weight, bias, eps)

    monkeypatch.setattr(K + "layernorm.cuda.general.supported", lambda x: True)
    monkeypatch.setattr(K + "layernorm.cuda.general.layernorm_cuda", fake_cuda)
    torch.manual_seed(0)
    x, w, b = torch.randn(3, 6, 16), torch.randn(16), torch.randn(16)
    rs = torch.rand(3, 6) if row_scale else None
    ref = F.layer_norm(x, (16,), w, b, 1e-5)
    ref = ref if rs is None else ref * rs[..., None]
    torch.testing.assert_close(ops.layer_norm(x, w, b, 1e-5, rs), ref)
    assert calls == [(3, 6, 16)]


def test_layer_norm_runs_the_statements_where_the_cuda_kernel_does_not_serve(h100, monkeypatch):
    from miniworld_engine import ops

    monkeypatch.setattr(K + "layernorm.cuda.general.supported", lambda x: False)
    monkeypatch.setattr(K + "layernorm.cuda.general.layernorm_cuda", _boom("layernorm_cuda"))
    x, w = torch.randn(4, 16), torch.randn(16)
    torch.testing.assert_close(ops.layer_norm(x, w, None, 1e-5), F.layer_norm(x, (16,), w, None, 1e-5))


def test_rms_norm_modulation_runs_the_reference(h100):
    from miniworld_engine import ops
    from miniworld_engine.kernels.rmsnorm.reference import rmsnorm_adamod_reference

    torch.manual_seed(0)
    q, c = torch.randn(3, 8, 16), torch.randn(3, 8, 12)
    w = [torch.randn(16, 12) * 0.1 for _ in range(3)]
    y, g = ops.rms_norm_modulation(q, c, *w, eps=1e-6)
    ry, rg = rmsnorm_adamod_reference(q, c, *w, None, 1e-6)
    torch.testing.assert_close(y, ry)
    torch.testing.assert_close(g, rg)


def test_layer_norm_linear_and_gated_residual_run_the_statements(h100):
    from miniworld_engine import ops

    torch.manual_seed(0)
    x, lnw, pw = torch.randn(2, 5, 5, 16), torch.randn(16), torch.randn(4, 16)
    torch.testing.assert_close(ops.layer_norm_linear(x, lnw, pw, 1e-5), F.linear(F.layer_norm(x, (16,), lnw, None, 1e-5), pw))
    g, b = torch.randn_like(x), torch.randn_like(x)
    torch.testing.assert_close(ops.gated_residual(x, g, b), x + g * b)


def test_gated_linear_runs_the_statements(h100):
    from miniworld_engine import ops

    torch.manual_seed(0)
    gate, value = torch.randn(2, 5, 16).bfloat16(), torch.randn(2, 5, 16).bfloat16()
    w, bias = torch.randn(8, 16).bfloat16(), torch.randn(8).bfloat16()
    torch.testing.assert_close(ops.gated_linear(gate, value, w, bias), F.linear(torch.sigmoid(gate) * value, w, bias))


def test_conditioned_transition_runs_the_statements(h100):
    from miniworld_engine import ops

    torch.manual_seed(0)
    x, cond = torch.randn(2, 6, 16), torch.randn(2, 6, 8)
    wa, wb, ws, wsc, bsc = torch.randn(32, 16), torch.randn(32, 16), torch.randn(16, 32), torch.randn(16, 8), torch.randn(16)
    out = ops.conditioned_transition(x, cond, expand_a_weight=wa, expand_b_weight=wb, squeeze_weight=ws, to_scale_weight=wsc,
                                     to_scale_bias=bsc, n=2)
    ref = torch.sigmoid(F.linear(cond, wsc, bsc)) * F.linear(F.silu(F.linear(x, wa)) * F.linear(x, wb), ws)
    torch.testing.assert_close(out, ref)


def _attention_inputs(dtype, heads=2, length=5, dim=4):
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 1, heads, length, dim).to(dtype) for _ in range(3))
    return q, k, v, torch.randn(1, heads, length, length).to(dtype), torch.rand(2, 1, length) > 0.2


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_augmented_attention_runs_the_reference_where_no_cuda_core_serves(h100, monkeypatch, dtype):
    from miniworld_engine import ops
    from miniworld_engine.kernels.augmented_attention.reference import (
        augmented_attention_pair_bias_pytorch,
    )

    monkeypatch.setattr(K + "augmented_attention.cuda.available", lambda *a, **k: False)
    monkeypatch.setattr(K + "augmented_attention.cuda.augmented_attention_bf16_sm90", _boom("augmented_attention_bf16_sm90"))
    q, k, v, bias, mask = _attention_inputs(dtype)
    out = ops.augmented_attention_pair_bias(q, k, v, bias, mask)
    ref = augmented_attention_pair_bias_pytorch(q.transpose(2, 3), k.transpose(2, 3), v.transpose(2, 3),
                                                bias.permute(0, 2, 3, 1), mask).transpose(2, 3)
    assert out.dtype == dtype
    torch.testing.assert_close(out, ref)


def test_augmented_attention_takes_the_cuda_core_for_bf16_where_it_serves(h100, monkeypatch):
    from miniworld_engine import ops

    seen = []

    def core(q, k, v, bias, mask=None):
        seen.append((tuple(q.shape), tuple(bias.shape)))
        return q

    monkeypatch.setattr(K + "augmented_attention.cuda.available", lambda *a, **k: True)
    monkeypatch.setattr(K + "augmented_attention.cuda.augmented_attention_bf16_sm90", core)
    q, k, v, bias, mask = _attention_inputs(torch.bfloat16)
    out = ops.augmented_attention_pair_bias(q, k, v, bias, mask)
    assert seen == [((2, 1, 5, 2, 4), (1, 5, 5, 2))]            # token-major layout, as modules.AugmentedAttention hands it over
    torch.testing.assert_close(out, q)
    # an fp32 caller never takes the bf16 core
    seen.clear()
    ops.augmented_attention_pair_bias(*_attention_inputs(torch.float32))
    assert not seen


@pytest.mark.parametrize("qk_norm", [False, True])
def test_triangle_attention_facade_runs_the_pytorch_core(h100, qk_norm):
    from miniworld_engine import ops

    torch.manual_seed(0)
    d, heads, length = 16, 2, 6
    pair = torch.randn(1, length, length, d)
    weights = dict(
        n_head=heads, ln_pair_weight=torch.ones(d), ln_pair_bias=torch.zeros(d), to_value_weight=torch.randn(d, d) * 0.2,
        to_bias_weight=torch.randn(heads, d) * 0.2, to_gate_weight=torch.randn(d, d) * 0.2, to_out_weight=torch.randn(d, d) * 0.2,
        to_query_weight=torch.randn(d, d) * 0.2, to_key_weight=torch.randn(d, d) * 0.2)
    if qk_norm:
        weights.update(norm_query_weight=torch.ones(d // heads), norm_key_weight=torch.ones(d // heads))
    out = ops.triangle_attention(pair, torch.ones(1, length, dtype=torch.bool), **weights)
    assert out.shape == pair.shape
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("direction", ["outgoing", "incoming"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_trimul_facade_runs_the_statements_in_the_callers_dtype(h100, direction, dtype):
    """No Triton and no warning; the hand-CUDA TriMul stays bf16-only and module-bound, so the facade never runs a fp32 CUDA TriMul."""
    import warnings

    from miniworld_engine import ops
    from miniworld_engine.modules.triangle_multiplication.reference import (
        triangle_multiplicative_update_pytorch,
    )

    torch.manual_seed(0)
    d, length = 16, 7
    x = torch.randn(1, length, length, d).to(dtype)
    res = torch.rand(1, length) > 0.2
    w = {"norm_in_weight": torch.randn(d).to(dtype), "norm_in_bias": torch.randn(d).to(dtype), "p_in_weight": (torch.randn(2 * d, d) * 0.2).to(dtype),
         "g_in_weight": (torch.randn(2 * d, d) * 0.2).to(dtype), "norm_out_weight": torch.randn(d).to(dtype), "norm_out_bias": torch.randn(d).to(dtype),
         "p_out_weight": (torch.randn(d, d) * 0.2).to(dtype), "g_out_weight": (torch.randn(d, d) * 0.2).to(dtype)}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = ops.triangle_multiplicative_update(x, direction, res, **w)
    ref = x + triangle_multiplicative_update_pytorch(x, direction, res[:, :, None] & res[:, None, :], **w)
    assert out.dtype == dtype
    torch.testing.assert_close(out, ref, rtol=2e-2 if dtype is torch.bfloat16 else 1e-4, atol=2e-2 if dtype is torch.bfloat16 else 1e-4)


def test_bidirectional_trimul_facade_runs_the_statements(monkeypatch):
    from miniworld_engine import ops
    from miniworld_engine.modules.exceptions import ImplementationType
    from miniworld_engine.modules.triangle_multiplication.bidirectional import (
        BidirectionalTriangleMultiplication,
    )

    torch.manual_seed(0)
    d, length = 16, 6
    module = BidirectionalTriangleMultiplication(d, implementation=ImplementationType.PYTORCH, p_drop=0.0).eval()
    for p in module.parameters():
        p.data.normal_(0, 0.2)
    x, mask = torch.randn(1, length, length, d), torch.rand(1, length) > 0.2
    with torch.no_grad():
        ref = module(x, mask)                                   # the reference first, on a plain CPU host
        previous = _install(monkeypatch)
        out = ops.bidirectional_triangle_multiplicative_update(
            x, mask, norm_in_weight=module.ln_pair.weight, norm_in_bias=module.ln_pair.bias, to_left_weight=module.to_left.weight,
            to_left_gate_weight=module.to_left_gate.weight, to_right_weight=module.to_right.weight,
            to_right_gate_weight=module.to_right_gate.weight, norm_out_weight=module.ln_out.weight, norm_out_bias=module.ln_out.bias,
            to_out_weight=module.to_out.weight, to_gate_weight=module.to_gate.weight, eps=module.ln_pair.eps)
    settings.configure(portable_on_hopper_blackwell=previous.portable_on_hopper_blackwell,
                       force_portable_kernels=previous.force_portable_kernels)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


def test_the_ab_switch_brings_the_triton_entries_back(h100):
    """``portable_on_hopper_blackwell`` lifts the restriction: the facade reaches its Triton function (poisoned here, so it raises)."""
    from miniworld_engine import ops

    settings.configure(portable_on_hopper_blackwell=True)
    x = torch.randn(2, 5, 16)
    with pytest.raises(AssertionError, match="Triton"):
        ops.swiglu_ffn(x, torch.randn(32, 16), torch.randn(32, 16), torch.randn(16, 32))


def _bias_only_attention():
    from miniworld_engine.modules.bias_only_dit.module import BiasOnlyAttention
    from miniworld_engine.modules.exceptions import ImplementationType

    return BiasOnlyAttention(d_single=64, d_cond=32, d_pair=16, n_head=4, implementation=ImplementationType.MINIWORLD)


def test_bias_only_dit_training_helpers_do_not_pick_triton_on_h100(monkeypatch):
    """A training call (a gradient is recorded) whose shapes the CUDA block cannot take used to fall to the Triton core / pair-bias kernels."""
    attention = _bias_only_attention()                              # built on a plain CPU host, then the card appears
    previous = _install(monkeypatch)
    one_pair = torch.randn(2, 1, 8, 64, dtype=torch.bfloat16, requires_grad=True)         # [A, 1, L, d]: one pair for every sample
    per_sample = torch.randn(1, 2, 8, 64, dtype=torch.bfloat16, requires_grad=True)       # [1, B, L, d]: a pair of its own per sample
    pair = torch.randn(1, 8, 8, 16, dtype=torch.bfloat16, requires_grad=True)
    assert not attention._feature_major(one_pair)
    assert not attention._feature_major_psb(per_sample)
    assert not attention._fused_pair_bias(pair)
    # ... and the A/B switch is what brings them back (so the three checks above are the guard, not an unrelated refusal)
    settings.configure(portable_on_hopper_blackwell=True)
    assert attention._feature_major(one_pair)
    assert attention._feature_major_psb(per_sample)
    assert attention._fused_pair_bias(pair)
    settings.configure(portable_on_hopper_blackwell=previous.portable_on_hopper_blackwell,
                       force_portable_kernels=previous.force_portable_kernels)


def test_pair_bias_all_does_not_fold_on_triton_on_h100(monkeypatch):
    from miniworld_engine.modules.bias_only_dit import hoist
    from miniworld_engine.modules.exceptions import ImplementationType

    attention = _bias_only_attention()
    previous = _install(monkeypatch)
    monkeypatch.setattr(hoist._fused, "serves", lambda *a, **k: False)                    # no CUDA hoist serves this call
    monkeypatch.setattr(hoist, "_portable", _boom("hoist._portable"))
    pair = torch.randn(1, 8, 8, 16, dtype=torch.bfloat16)
    biases = hoist.pair_bias_all([attention], pair, implementation=ImplementationType.MINIWORLD)
    assert biases[0].shape == (1, 4, 8, 8)
    settings.configure(portable_on_hopper_blackwell=previous.portable_on_hopper_blackwell,
                       force_portable_kernels=previous.force_portable_kernels)


def test_the_cuda_path_prep_helpers_are_pytorch_ops(h100):
    """The three Triton helpers of the sm_90 AugmentedAttention cores (``_scale_bf16_kernel``, ``_prep_do_kernel``, ``_wf_kernel``) are PyTorch
    ops now: same arithmetic (fp32 multiply, one rounding to bf16; fp32 products and row sums)."""
    from miniworld_engine.kernels.augmented_attention import cuda as cu
    from miniworld_engine.kernels.augmented_attention.cuda import fused_pair as fp

    torch.manual_seed(0)
    a, length, nh, dh = 2, 8, 4, 16
    q = torch.randn(a, 1, length, nh, dh)
    qs, kb, vb = cu._prep_qkv(q, q.bfloat16(), q.bfloat16(), 0.25)
    assert qs.dtype is torch.bfloat16 and qs.shape == (a * length, nh * dh)
    torch.testing.assert_close(qs.float(), (q.reshape(a * length, nh * dh) * 0.25).bfloat16().float(), rtol=0, atol=0)
    bias = torch.randn(nh, length, length).bfloat16()
    torch.testing.assert_close(cu._bias_prep(bias).float(), (bias.float() * cu.LOG2E).bfloat16().float(), rtol=0, atol=0)
    do, o = torch.randn(a * length, nh * dh), torch.randn(a * length, nh * dh)
    dob, dd = cu._prep_do(do, o, a, length, nh, dh)
    assert dob.dtype is torch.bfloat16 and dd.shape == (a, nh, length)
    expected = (do.view(a, length, nh, dh) * o.view(a, length, nh, dh)).sum(-1).permute(0, 2, 1)
    torch.testing.assert_close(dd, expected)
    _, dd_only = cu._prep_do(do.bfloat16(), o.bfloat16(), a, length, nh, dh, write_dob=False)
    assert dd_only.shape == (a, nh, length)
    wb, lnw = torch.randn(nh, 128).bfloat16(), torch.randn(128).bfloat16()
    l2, nat = fp._wf(wb, lnw, True)
    w = wb.float() * lnw.float()[None, :]
    torch.testing.assert_close(l2.float(), (w * fp.LOG2E).bfloat16().float(), rtol=0, atol=0)
    torch.testing.assert_close(nat.float(), w.bfloat16().float(), rtol=0, atol=0)
    assert fp._wf(wb, lnw, False)[1] is None
    torch.testing.assert_close(fp._wf32(wb.float(), lnw.float()), wb.float() * lnw.float()[None, :])


@pytest.mark.parametrize("rows", [32, 64, 128])
def test_the_trimul_weight_pack_matches_the_triton_layout(h100, rows):
    """``pack_into_triton``'s body is PyTorch copies now; the layout is the one the Triton ``_pack`` wrote: 64-row groups of 32 gate rows
    then 32 value rows, the first R / 32 groups from the left leaves."""
    from miniworld_engine.kernels.trimul_inproj.cuda import _h100_pack as pack

    torch.manual_seed(0)
    d = 16
    wl, wlg, wr, wrg = (torch.randn(rows, d).bfloat16() for _ in range(4))
    dst = torch.empty(4 * rows, d, dtype=torch.bfloat16)
    pack._pack_blocks(dst, wl, wlg, wr, wrg)
    expected = torch.empty_like(dst)
    for row in range(4 * rows):                          # the Triton kernel's index arithmetic, one row at a time
        source = (row // 64) * 32 + row % 32
        right, gate, r = source >= rows, row % 64 < 32, source % rows
        leaf = (wrg if gate else wr) if right else (wlg if gate else wl)
        expected[row] = leaf[r]
    torch.testing.assert_close(dst, expected, rtol=0, atol=0)
