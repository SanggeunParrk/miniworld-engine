"""H100: the ``ops.*`` facades run the EXISTING hand-CUDA kernels where they serve the call (docs/records/h100_ops_gate_perf_20261101.md).

``test_ops_no_triton_gpu.py`` pins "no Triton launch through ``ops.*``"; this file pins that the routing that replaced Triton is the CUDA one
(``kernels/_h100_ops.py``) on shapes the kernels serve, that it equals the previous (Triton / statements) output within the existing
tolerances, that the new wrappers are bitwise repeatable (a determinism check), and that ``_op_triangle_attention`` at L = 1024 runs (the PyTorch
core's 16 GiB score tensors ran out of memory) in bf16 and fp32.

Every case also asserts the CUDA wrapper really answered (a wrapper that returns None sends the call to the statements: the test would pass
vacuously otherwise).
"""
from __future__ import annotations

import contextlib

import pytest
import torch
import torch.nn.functional as F
from test_ops_no_triton_gpu import (
    DTYPES,
    REPORT,
    TOL,
    ab,
    check,
    evaluate,
    randn,
    rel,
    triton_launches,
)

from miniworld_engine.kernels.augmented_attention.whole_op import augmented_attention_pair_bias as _op_augmented_attention_pair_bias

from miniworld_engine.kernels.trimul_inproj.whole_op import bidirectional_triangle_multiplicative_update as _op_bidirectional_triangle_multiplicative_update

from miniworld_engine.kernels.conditioned_transition.whole_op import conditioned_transition as _op_conditioned_transition


from miniworld_engine.kernels._ops_routed import gated_residual as _op_gated_residual

from miniworld_engine.kernels._ops_routed import layer_norm_linear as _op_layer_norm_linear

from miniworld_engine.kernels._ops_routed import rms_norm_modulation as _op_rms_norm_modulation

from miniworld_engine.kernels._ops_routed import swiglu_ffn as _op_swiglu_ffn

from miniworld_engine.kernels.triangle_attention.whole_op import triangle_attention as _op_triangle_attention

from miniworld_engine.kernels.trimul_inproj.whole_op import triangle_multiplicative_update as _op_triangle_multiplicative_update
from miniworld_engine.kernels import _h100_ops

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0), reason="H100 only"),
]

BF16, FP32 = torch.bfloat16, torch.float32


@contextlib.contextmanager
def served(monkeypatch, *names):
    """Count the non-None answers of the named ``_h100_ops`` wrappers (``trimul``, ``swiglu_ffn``, ...) and of the triangle core's CUDA
    function; yields the counter."""
    counts = {n: 0 for n in names}
    for name in names:
        if name == "_TriangleCore":
            original = _h100_ops._TriangleCore.apply

            def apply(*args, _o=original, _n=name):
                counts[_n] += 1
                return _o(*args)

            monkeypatch.setattr(_h100_ops._TriangleCore, "apply", staticmethod(apply))
            continue
        original = getattr(_h100_ops, name)

        def wrapper(*args, _o=original, _n=name, **kwargs):
            out = _o(*args, **kwargs)
            counts[_n] += out is not None
            return out

        monkeypatch.setattr(_h100_ops, name, wrapper)
    yield counts


def cases_check(name, dtype, call, tensors, grad, monkeypatch, wrapper, **kwargs):
    with served(monkeypatch, wrapper) as counts:
        check(name, dtype, call, tensors, grad, **kwargs)
    assert counts[wrapper] > 0, f"{name}: the CUDA wrapper {wrapper} did not serve the call (it ran the PyTorch statements)"


@pytest.mark.parametrize("grad", [False, True], ids=["infer", "train"])
@pytest.mark.parametrize("dtype", DTYPES, ids=["bf16", "fp32"])
class TestCudaWiring:
    def test_swiglu_ffn(self, dtype, grad, monkeypatch):
        # fp32 at the default matmul precision: the CUDA SwiGLU row kernel between two exact fp32 GEMMs (``_SwiGLUFFN``)
        for d, hidden, shape in ((128, 512, (1, 128, 128)), (768, 1536, (4, 256))):
            x = randn(*shape, d, dtype=dtype)
            ws = [randn(hidden, d, dtype=dtype, scale=0.1, seed=1), randn(hidden, d, dtype=dtype, scale=0.1, seed=2),
                  randn(d, hidden, dtype=dtype, scale=0.1, seed=3)]
            cases_check(f"_op_swiglu_ffn D{d}", dtype, _op_swiglu_ffn, [x, *ws], grad, monkeypatch, "swiglu_ffn")

    def test_conditioned_transition(self, dtype, grad, monkeypatch):
        if dtype is FP32:
            pytest.skip("fp32 runs the CUDA tail only under TF32 (allow_tf32)")
        for d, dc, length in ((768, 384, 256), (128, 128, 256)):
            x, cond = randn(4, length, d, dtype=dtype), randn(4, length, dc, dtype=dtype)
            wa, wb, ws = (randn(2 * d, d, dtype=dtype, scale=0.1, seed=1), randn(2 * d, d, dtype=dtype, scale=0.1, seed=2),
                          randn(d, 2 * d, dtype=dtype, scale=0.1, seed=3))
            wsc, bsc = randn(d, dc, dtype=dtype, scale=0.1), randn(d, dtype=dtype, scale=0.1)
            cases_check(f"_op_conditioned_transition D{d}", dtype,
                        lambda x, c, a, b, s, wc, bc: _op_conditioned_transition(
                            x, c, expand_a_weight=a, expand_b_weight=b, squeeze_weight=s, to_scale_weight=wc, to_scale_bias=bc, n=2),
                        [x, cond, wa, wb, ws, wsc, bsc], grad, monkeypatch, "conditioned_transition")

    def test_gated_residual(self, dtype, grad):
        x, g, b = (randn(1, 128, 128, 128, dtype=dtype, seed=s) for s in (1, 2, 3))
        check("_op_gated_residual", dtype, _op_gated_residual, [x, g, b], grad)

    def test_layer_norm_linear(self, dtype, grad, monkeypatch):
        x, lnw, pw = randn(1, 128, 128, 128, dtype=dtype), randn(128, scale=0.1) + 1, randn(16, 128, scale=0.1)
        cases_check("_op_layer_norm_linear", dtype, lambda x, l, p: _op_layer_norm_linear(x, l, p, 1e-5), [x, lnw, pw], grad, monkeypatch,
                    "layer_norm_linear")
        # the fp32 statements are the yardstick
        with ab(False):
            got, gg = evaluate(lambda x, l, p: _op_layer_norm_linear(x, l, p, 1e-5), [x, lnw, pw], grad)
            truth, gt = evaluate(lambda x, l, p: F.linear(F.layer_norm(x, (128,), l, None, 1e-5), p), [t.float() for t in (x, lnw, pw)], grad)
        assert rel(got[0], truth[0]) <= TOL[dtype]
        for a, b in zip(gg, gt, strict=True):
            assert rel(a, b) <= 3 * TOL[dtype]

    def test_rms_norm_modulation(self, dtype, grad):
        for d, dc in ((768, 384), (128, 128)):
            q, c = randn(4, 256, d, dtype=dtype), randn(4, 256, dc, dtype=dtype)
            ws = [randn(d, dc, dtype=dtype, scale=0.1, seed=s) for s in (1, 2, 3)]
            check(f"_op_rms_norm_modulation D{d}", dtype, lambda q, c, a, b, g: _op_rms_norm_modulation(q, c, a, b, g, None, 1e-5),
                  [q, c, *ws], grad)

    @pytest.mark.parametrize(("heads", "length", "dim"), [(8, 128, 64), (16, 200, 48), (16, 256, 48), (4, 256, 32)])
    def test_augmented_attention_pair_bias(self, dtype, grad, heads, length, dim):
        q, k, v = (randn(2, 1, heads, length, dim, dtype=dtype, seed=s) for s in (1, 2, 3))
        bias = randn(1, heads, length, length, dtype=dtype, seed=4)
        mask = torch.rand(2, 1, length, device="cuda") > 0.1
        # the sm_90 core (bf16, 16 x 48 / 4 x 32, L % 128 == 0) or, for every other call, SDPA with the pair bias (cuDNN without a gradient)
        check(f"_op_augmented_attention_pair_bias H{heads} L{length} D{dim}", dtype,
              lambda q, k, v, b: _op_augmented_attention_pair_bias(q, k, v, b, mask), [q, k, v, bias], grad)

    @pytest.mark.parametrize(("length", "qk_norm"), [(128, False), (128, True), (200, True)])
    def test_triangle_attention(self, dtype, grad, length, qk_norm, monkeypatch):
        d, heads = 128, 4
        pair = randn(1, length, length, d, dtype=dtype)
        names = ["ln_w", "ln_b", "wv", "wb", "wg", "wo", "wq", "wk"]
        shapes = [(d,), (d,), (d, d), (heads, d), (d, d), (d, d), (d, d), (d, d)]
        ws = [randn(*s, dtype=dtype, scale=0.05 if len(s) == 2 else 0.1, seed=i) + (1 if n == "ln_w" else 0)
              for i, (n, s) in enumerate(zip(names, shapes, strict=True))]
        norm = [0.25 * (randn(d // heads, dtype=dtype, scale=0.1, seed=11) + 1), 0.25 * (randn(d // heads, dtype=dtype, scale=0.1, seed=12) + 1)] if qk_norm else []
        mask = torch.rand(1, length, device="cuda") > 0.1

        def call(pair, ln_w, ln_b, wv, wb, wg, wo, wq, wk, *nq):
            return _op_triangle_attention(
                pair, mask, n_head=heads, ln_pair_weight=ln_w, ln_pair_bias=ln_b, to_value_weight=wv, to_bias_weight=wb, to_gate_weight=wg,
                to_out_weight=wo, to_query_weight=wq, to_key_weight=wk,
                norm_query_weight=nq[0] if nq else None, norm_key_weight=nq[1] if nq else None)

        inputs = [pair, *ws, *norm]
        # the fp32 statements are the yardstick (the previous Triton core is wrong for QK-normed q / k with a strided v)
        with served(monkeypatch, "_TriangleCore") as counts, triton_launches() as launches:
            with ab(False):
                got, got_grad = evaluate(call, inputs, grad)
                again, again_grad = evaluate(call, inputs, grad)
                truth, truth_grad = evaluate(call, [t.float() for t in inputs], grad)
        assert not launches, dict(launches)
        assert (counts["_TriangleCore"] > 0) == (dtype is BF16), "the CUDA core must serve bf16 (and only bf16)"
        for a, b in zip(got + got_grad, again + again_grad, strict=True):
            assert torch.equal(a, b), "not bitwise repeatable"
        err = rel(got[0], truth[0])
        gerr = max([rel(a, b) for a, b in zip(got_grad, truth_grad, strict=True)] or [0.0])
        REPORT.append(f"_op_triangle_attention L{length} qk_norm={qk_norm} {str(dtype).split('.')[-1]:5s} {'train' if grad else 'infer':5s} "
                      f"CUDA-core={counts['_TriangleCore'] > 0} out-vs-fp32={err:.2e} grad-vs-fp32={gerr:.2e}")
        assert err <= 2 * TOL[dtype]
        assert gerr <= (6e-2 if dtype is BF16 else 6e-3)

    @pytest.mark.parametrize("direction", ["outgoing", "incoming"])
    @pytest.mark.parametrize("length", [128, 256])
    def test_trimul(self, dtype, grad, direction, length, monkeypatch):
        if dtype is FP32:
            pytest.skip("TriMul is bf16 only: fp32 runs the statements in fp32 (never a bf16 path)")
        d = 128
        x = randn(1, length, length, d, dtype=dtype)
        mask = torch.rand(1, length, device="cuda") > 0.1
        tensors = [x, randn(d, scale=0.1) + 1, randn(d, scale=0.1), randn(2 * d, d, dtype=dtype, scale=0.05, seed=1),
                   randn(2 * d, d, dtype=dtype, scale=0.05, seed=2), randn(d, scale=0.1) + 1, randn(d, scale=0.1),
                   randn(d, d, dtype=dtype, scale=0.05, seed=3), randn(d, d, dtype=dtype, scale=0.05, seed=4)]

        def call(x, nw, nb, pin, gin, now, nob, pout, gout):
            return _op_triangle_multiplicative_update(x, direction, mask, nw, nb, pin, gin, now, nob, pout, gout)

        cases_check(f"_op_triangle_multiplicative_update {direction} L{length}", dtype, call, tensors, grad, monkeypatch, "trimul")

    @pytest.mark.parametrize("length", [128, 256])
    def test_trimul_bidirectional(self, dtype, grad, length, monkeypatch):
        if dtype is FP32:
            pytest.skip("TriMul is bf16 only")
        d = 128
        x = randn(1, length, length, d, dtype=dtype)
        mask = torch.rand(1, length, device="cuda") > 0.1
        ws = [randn(2 * d, d, dtype=dtype, scale=0.05, seed=s) for s in (1, 2, 3, 4)]
        tensors = [x, randn(d, scale=0.1) + 1, randn(d, scale=0.1), *ws, randn(2 * d, scale=0.1) + 1, randn(2 * d, scale=0.1),
                   randn(d, 2 * d, dtype=dtype, scale=0.05, seed=5), randn(d, d, dtype=dtype, scale=0.05, seed=6)]

        def call(x, nw, nb, wl, wlg, wr, wrg, now, nob, wo, wg):
            return _op_bidirectional_triangle_multiplicative_update(
                x, mask, norm_in_weight=nw, norm_in_bias=nb, to_left_weight=wl, to_left_gate_weight=wlg, to_right_weight=wr,
                to_right_gate_weight=wrg, norm_out_weight=now, norm_out_bias=nob, to_out_weight=wo, to_gate_weight=wg)

        cases_check(f"_op_bidirectional_triangle_multiplicative_update L{length}", dtype, call, tensors, grad, monkeypatch, "trimul")


@pytest.mark.parametrize("grad", [False, True], ids=["infer", "train"])
@pytest.mark.parametrize("dtype", DTYPES, ids=["bf16", "fp32"])
@pytest.mark.parametrize("qk_norm", [False, True])
def test_triangle_attention_at_l1024_does_not_run_out_of_memory(dtype, grad, qk_norm, monkeypatch):
    """The PyTorch core builds B H L L L fp32 scores (16 GiB at L = 1024, 4 heads): it ran out of memory. bf16 runs the CUDA core, fp32 the PyTorch core
    chunked over the rows (checkpointed in training): both fit, and both repeat bit for bit."""
    d, heads, length = 128, 4, 1024
    pair = randn(1, length, length, d, dtype=dtype)
    names = ["ln_w", "ln_b", "wv", "wb", "wg", "wo", "wq", "wk"]
    shapes = [(d,), (d,), (d, d), (heads, d), (d, d), (d, d), (d, d), (d, d)]
    ws = [randn(*s, dtype=dtype, scale=0.05 if len(s) == 2 else 0.1, seed=i) + (1 if n == "ln_w" else 0) for i, (n, s) in enumerate(zip(names, shapes, strict=True))]
    norm = [0.25 * (randn(d // heads, dtype=dtype, scale=0.1, seed=11) + 1), 0.25 * (randn(d // heads, dtype=dtype, scale=0.1, seed=12) + 1)] if qk_norm else []
    mask = torch.rand(1, length, device="cuda") > 0.1

    def call(pair, ln_w, ln_b, wv, wb, wg, wo, wq, wk, *nq):
        return _op_triangle_attention(
            pair, mask, n_head=heads, ln_pair_weight=ln_w, ln_pair_bias=ln_b, to_value_weight=wv, to_bias_weight=wb, to_gate_weight=wg,
            to_out_weight=wo, to_query_weight=wq, to_key_weight=wk, norm_query_weight=nq[0] if nq else None, norm_key_weight=nq[1] if nq else None)

    inputs = [pair, *ws, *norm]
    with ab(False), triton_launches() as launches, served(monkeypatch, "_TriangleCore") as counts:
        first, first_grad = evaluate(call, inputs, grad)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / 2**30
        second, second_grad = evaluate(call, inputs, grad)
    assert not launches, dict(launches)
    assert (counts["_TriangleCore"] > 0) == (dtype is BF16)
    for a, b in zip(first + first_grad, second + second_grad, strict=True):
        assert torch.isfinite(a).all() and torch.equal(a, b), "not finite / not bitwise repeatable"
    REPORT.append(f"_op_triangle_attention L1024 qk_norm={qk_norm} {str(dtype).split('.')[-1]:5s} {'train' if grad else 'infer':5s} ran, bitwise repeatable, "
                  f"CUDA-core={counts['_TriangleCore'] > 0} peak-allocated={peak:.1f} GiB")


def test_the_chunked_pytorch_triangle_core_equals_the_one_shot_form(monkeypatch):
    """The row chunks (a budget forced small) give the one-shot result bit for bit in fp32 inference, and the same gradients through the checkpoints."""
    from miniworld_engine.kernels.triangle_attention.reference import triangle_attention_pair_bias_pytorch as core

    b, h, length, dh = 1, 4, 96, 32
    q, k, v = (randn(b, h, length, length, dh, seed=s).requires_grad_() for s in (1, 2, 3))
    bias = randn(b, h, length, length, seed=4).requires_grad_()
    one = core(q, k, v, bias)
    one.square().sum().backward()
    ref_grads = [t.grad.clone() for t in (q, k, v, bias)]
    for t in (q, k, v, bias):
        t.grad = None
    monkeypatch.setattr(_h100_ops, "_SCORE_BUDGET", b * h * length * length * 4 * 16)   # 16 rows per chunk
    chunked = _h100_ops.triangle_core_pytorch(q, k, v, bias)
    chunked.square().sum().backward()
    assert rel(chunked, one) < 1e-6
    for t, ref in zip((q, k, v, bias), ref_grads, strict=True):
        assert rel(t.grad, ref) < 1e-5


def test_the_triangle_core_gradients_match_fp32_autograd():
    """The backward of the CUDA core (delta, key side, query side, dbias) against the fp32 PyTorch core's autograd, including a masked key column."""
    from miniworld_engine.kernels.triangle_attention.reference import triangle_attention_pair_bias_pytorch as core

    b, h, length, dh = 1, 4, 128, 32
    q, k, v = (randn(b, h, length, length, dh, dtype=BF16, seed=s).requires_grad_() for s in (1, 2, 3))
    bias = (randn(b, h, length, length, dtype=BF16, seed=4)).requires_grad_()
    cot = randn(b, h, length, length, dh, dtype=BF16, seed=5)
    out = _h100_ops.triangle_core(q, k, v, bias)
    out.backward(cot)
    leaves32 = [t.detach().float().requires_grad_() for t in (q, k, v, bias)]
    core(*leaves32).backward(cot.float())
    assert rel(out, core(*leaves32)) < 2e-2
    for name, t, t32 in zip("qkv", (q, k, v), leaves32, strict=False):
        assert rel(t.grad, t32.grad) < 6e-2, name
    assert rel(bias.grad, leaves32[3].grad) < 6e-2, "dbias"


def teardown_module(module):
    print("\n==== H100 ops CUDA wiring: new routing vs previous path ====")
    print("\n".join(REPORT))
