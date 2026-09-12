"""GPU build exclusions also constrain nested GEMM runtime choices."""
from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
import torch

from miniworld_engine.build import matrix
from miniworld_engine.kernels.trimul_inproj.cute import dispatch


@pytest.fixture(autouse=True)
def reset_dispatch():
    dispatch.reset()
    yield
    dispatch.reset()


@pytest.mark.parametrize("sm", [(8, 0), (8, 6), (9, 0), (10, 0)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_runtime_uses_same_csv_verdict_as_builder(monkeypatch, sm, dtype):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: sm)
    assert dispatch._cute_allowed(torch.device("cuda:1"), dtype, "triangle_multiplication") == (
        matrix.allows(matrix.sm_tag(sm), "triangle_multiplication", "cute",
                      str(dtype).removeprefix("torch.")))


@pytest.mark.parametrize("op", ["mm", "addmm", "bmm"])
@pytest.mark.parametrize("enabled", [True, False])
def test_sm80_primitives_never_import_or_calibrate_quack(monkeypatch, op, enabled):
    actual_policy = dispatch._cute_allowed
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (8, 0))
    monkeypatch.setattr(dispatch, "_cute_allowed", lambda device, dtype, case:
                        actual_policy(torch.device("cuda:0"), dtype, case))
    monkeypatch.setattr(dispatch, "_ENABLED", enabled)
    def forbidden(*args, **kwargs):
        pytest.fail("unsupported Quack candidate was reached")
    monkeypatch.setitem(sys.modules, "miniworld_engine.kernels._quack_compat",
                        SimpleNamespace(gemm=forbidden, gemm_act=forbidden))
    monkeypatch.setattr(dispatch, "_calibrate", forbidden)
    a, b = torch.randn(4, 8), torch.randn(8, 5)
    if op == "mm":
        out, expected = dispatch.mm("dw", a, b), a @ b
    elif op == "addmm":
        c = torch.randn(4, 5)
        out, expected = dispatch.addmm("dw", c, a, b), torch.addmm(c, a, b)
    else:
        a, b = a[None], b[None]
        out, expected = dispatch.bmm("dw", a, b), torch.bmm(a, b)
    torch.testing.assert_close(out, expected)


def test_warm_cute_winner_is_filtered_before_cache_lookup(monkeypatch):
    allowed = True
    monkeypatch.setattr(dispatch, "_cute_allowed", lambda *args: allowed)
    monkeypatch.setattr(dispatch, "_ENABLED", True)
    calls = []
    def calibrate(name, key, candidates):
        calls.append(key)
        dispatch._CACHE.setdefault(name, {})[key] = 0
        return 0
    monkeypatch.setattr(dispatch, "_calibrate", calibrate)
    operands = (torch.empty(2, 2),)
    candidates = [("cute", lambda: "cute"), ("cublas", lambda: "cublas")]
    assert dispatch.pick("same", (2, 2), candidates, operands=operands) == "cute"
    allowed = False
    assert dispatch.pick("same", (2, 2), candidates, operands=operands) == "cublas"
    assert len(calls) == 1


def test_cache_separates_dtype_stride_and_candidate_order(monkeypatch):
    monkeypatch.setattr(dispatch, "_cute_allowed", lambda *args: True)
    monkeypatch.setattr(dispatch, "_ENABLED", True)
    seen = []
    def calibrate(name, key, candidates):
        seen.append(key)
        dispatch._CACHE.setdefault(name, {})[key] = 0
        return 0
    monkeypatch.setattr(dispatch, "_calibrate", calibrate)
    a = torch.empty(2, 2)
    candidates = [("cute", lambda: 1), ("cublas", lambda: 2)]
    for operand in (a, a, a.t(), a.to(torch.bfloat16)):
        dispatch.pick("same", (2, 2), candidates, operands=(operand,))
    dispatch.pick("same", (2, 2), candidates[::-1], operands=(a,))
    assert len(seen) == 4


@pytest.mark.gpu
@pytest.mark.parametrize("op", ["mm", "addmm", "bmm"])
def test_ampere_eager_and_fullgraph_backward_without_quack(op, monkeypatch):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 8:
        pytest.skip("requires allocated Ampere GPU")
    def forbidden(*args, **kwargs):
        pytest.fail("Quack called on Ampere")
    monkeypatch.setitem(sys.modules, "miniworld_engine.kernels._quack_compat",
                        SimpleNamespace(gemm=forbidden, gemm_act=forbidden))
    a = torch.randn(16, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    b = torch.randn(32, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    c = torch.randn(16, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    if op == "mm":
        fn = lambda a, b, c: dispatch.mm("test", a, b)
        ref = lambda a, b, c: a @ b
    elif op == "addmm":
        fn = lambda a, b, c: dispatch.addmm("test", c, a, b)
        ref = lambda a, b, c: torch.addmm(c, a, b)
    else:
        fn = lambda a, b, c: dispatch.bmm("test", a[None], b[None])
        ref = lambda a, b, c: torch.bmm(a[None], b[None])
    expected = ref(a, b, c)
    expected_grads = torch.autograd.grad(expected.sum(), (a, b, c), allow_unused=True)
    for implementation in (fn, torch.compile(fn, fullgraph=True)):
        output = implementation(a, b, c)
        torch.testing.assert_close(output, expected)
        grads = torch.autograd.grad(output.sum(), (a, b, c), allow_unused=True)
        for actual, wanted in zip(grads, expected_grads, strict=True):
            if wanted is None:
                assert actual is None
            else:
                torch.testing.assert_close(actual, wanted)
