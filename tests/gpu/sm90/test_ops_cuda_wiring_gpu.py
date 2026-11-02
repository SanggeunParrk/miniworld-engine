"""H100: the ``ops.*`` facades run the EXISTING hand-CUDA kernels where they serve the call (docs/records/h100_ops_gate_perf_20261101.md).

``test_ops_no_triton_gpu.py`` pins "no Triton launch through ``ops.*``"; this file pins that the routing that replaced Triton is the CUDA one
(``kernels/*/cuda/`` wrappers) on shapes the kernels serve, that it equals the previous (Triton / statements) output within the existing
tolerances, that the new wrappers are bitwise repeatable (a determinism check).

Every case also asserts the CUDA wrapper really answered (a wrapper that returns None sends the call to the statements: the test would pass
vacuously otherwise).
"""
from __future__ import annotations

import contextlib
import importlib

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
)

from miniworld_engine.kernels.augmented_attention.whole_op import augmented_attention_pair_bias as _op_augmented_attention_pair_bias


from miniworld_engine.kernels.gated_projection.interface import gated_residual as _op_gated_residual

from miniworld_engine.kernels.layernorm_linear.interface import layer_norm_linear as _op_layer_norm_linear

from miniworld_engine.kernels.rmsnorm.interface import rms_norm_modulation as _op_rms_norm_modulation

from miniworld_engine.kernels.transition.interface import swiglu_ffn as _op_swiglu_ffn

from miniworld_engine.kernels.trimul_inproj.interface import triangle_multiplicative_update as _op_triangle_multiplicative_update

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0), reason="H100 only"),
]

BF16, FP32 = torch.bfloat16, torch.float32


#: wrapper name -> (module, attribute) of the H100 CUDA wrapper the routed entry point offers the call to first
WRAPPERS = {
    "swiglu_ffn": ("miniworld_engine.kernels.conditioned_transition.cuda.swiglu_ffn", "swiglu_ffn"),
    "layer_norm_linear": ("miniworld_engine.kernels.layernorm_linear.cuda.pair_bias", "layer_norm_linear"),
    "trimul": ("miniworld_engine.kernels.trimul_inproj.whole_op", "_trimul_h100"),
}


@contextlib.contextmanager
def served(monkeypatch, *names):
    """Count the non-None answers of the named CUDA wrappers (``trimul``, ``swiglu_ffn``, ...); yields the counter."""
    counts = {n: 0 for n in names}
    for name in names:
        module = importlib.import_module(WRAPPERS[name][0])
        attribute = WRAPPERS[name][1]
        original = getattr(module, attribute)

        def wrapper(*args, _o=original, _n=name, **kwargs):
            out = _o(*args, **kwargs)
            counts[_n] += out is not None
            return out

        monkeypatch.setattr(module, attribute, wrapper)
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


def teardown_module(module):
    print("\n==== H100 ops CUDA wiring: new routing vs previous path ====")
    print("\n".join(REPORT))
