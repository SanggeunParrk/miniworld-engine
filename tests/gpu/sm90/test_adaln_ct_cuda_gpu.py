"""H100: the hand-CUDA AdaptiveLayerNorm and ConditionedTransition (``kernels/adaln/cuda``, ``kernels/conditioned_transition/cuda``) behind the modules.

What is pinned: the route (CUDA on H100, the portable family under ``force_portable_kernels``), that the CUDA kernels are the ones that launch -- eager, under
``torch.compile`` and captured in a CUDA graph -- parity with an fp64 reference within 1.5x the error of the same-dtype PyTorch statements (inference, and
every training gradient), and that a repeated call is bitwise the same.
"""
from __future__ import annotations

import pytest
import torch

from miniworld_engine import settings
from miniworld_engine.modules.adaptive_layernorm.module import AdaptiveLayerNorm
from miniworld_engine.modules.conditioned_transition.module import ConditionedTransition
from miniworld_engine.modules.dispatch import KernelBackend
from miniworld_engine.modules.exceptions import ImplementationType as IT

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0), reason="H100 only"),
]

A, L, D, DC = 2, 128, 768, 384


@pytest.fixture(autouse=True)
def _settings():
    previous = settings.configure(force_portable_kernels=False, portable_on_hopper_blackwell=False, adaln_h100_cuda=True)
    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    yield
    torch.backends.cuda.matmul.allow_tf32 = prev_tf32
    settings.configure(**{f: getattr(previous, f) for f in ("force_portable_kernels", "portable_on_hopper_blackwell", "adaln_h100_cuda")})


def _kernels(fn):
    from torch.profiler import ProfilerActivity, profile

    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        fn()
        torch.cuda.synchronize()
    return {e.key for e in p.key_averages() if e.device_time_total > 0}


def _has(names, *needles):
    return all(any(n in k for k in names) for n in needles)


def _build(op, impl, dtype=torch.float32):
    torch.manual_seed(0)
    m = (AdaptiveLayerNorm(D, DC, implementation=impl) if op == "adaln" else ConditionedTransition(D, DC, implementation=impl, dtype=dtype)).cuda()
    for p in m.parameters():
        p.data.normal_(0, 0.05) if p.ndim > 1 else p.data.normal_(0, 0.5)
    return m


def _inputs(dtype, grad=False):
    torch.manual_seed(1)
    x = torch.randn(A, 1, L, D, device="cuda", dtype=dtype, requires_grad=grad)
    c = torch.randn(A, 1, L, DC, device="cuda", dtype=dtype, requires_grad=grad)
    return x, c


def _call(m, x, c, bf16):
    if bf16:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return m(x, c)
    return m(x, c)


def _ref64(op, m):
    ref = _build(op, IT.PYTORCH, torch.float64).double() if op == "ct" else AdaptiveLayerNorm(D, DC, implementation=IT.PYTORCH, dtype=torch.float64).cuda()
    ref.load_state_dict({k: v.double() for k, v in m.state_dict().items()})
    return ref


def _err(a, ref):
    return ((a.double() - ref).norm() / ref.norm()).item()


@pytest.mark.parametrize("op", ["adaln", "ct"])
def test_the_route_is_cuda_on_h100_and_portable_under_the_ab_switch(op):
    assert _build(op, IT.MINIWORLD)._backend is KernelBackend.CUDA
    settings.configure(force_portable_kernels=True)
    assert _build(op, IT.MINIWORLD)._backend is KernelBackend.PORTABLE


@pytest.mark.parametrize("op", ["adaln", "ct"])
@pytest.mark.parametrize("bf16", [True, False], ids=["bf16", "fp32"])
@pytest.mark.parametrize("mode", ["eager", "compiled", "graph"])
def test_inference_launches_the_cuda_kernels_and_matches_the_reference(op, bf16, mode):
    dtype = torch.bfloat16 if bf16 else torch.float32
    m = _build(op, IT.MINIWORLD)
    ref, pt = _ref64(op, m), _build(op, IT.PYTORCH)
    pt.load_state_dict(m.state_dict())
    x, c = _inputs(dtype)
    fn = torch.compile(m) if mode == "compiled" else m
    with torch.no_grad():
        run = lambda: _call(fn, x, c, bf16)  # noqa: E731
        y = run()
        if mode == "graph":
            run()
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                out = run()
            g.replay()
            torch.cuda.synchronize()
            y = out
            replay = lambda: g.replay()  # noqa: E731
        else:
            replay = run
        names = _kernels(replay)
        yr = ref(x.double(), c.double())
        yp = _call(pt, x, c, bf16)
        assert _err(y, yr) <= 1.5 * _err(yp, yr) + 1e-6, (_err(y, yr), _err(yp, yr))
        again = run() if mode != "graph" else (g.replay(), out.clone())[1]
        assert torch.equal(again, y)
    assert _has(names, "adaln_gemm_k", "ln_rows_k"), names
    if op == "ct":
        assert _has(names, "expand_swiglu_k", "squeeze_gate_k"), names
    assert not any("_kernel" in k and k.startswith("_") for k in names), names   # no Triton kernel


@pytest.mark.parametrize("op", ["adaln", "ct"])
@pytest.mark.parametrize("bf16", [True, False], ids=["bf16", "fp32"])
@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "compiled"])
def test_training_gradients_match_the_reference(op, bf16, compiled):
    dtype = torch.bfloat16 if bf16 else torch.float32
    m = _build(op, IT.MINIWORLD)
    ref, pt = _ref64(op, m), _build(op, IT.PYTORCH)
    pt.load_state_dict(m.state_dict())
    x, c = _inputs(dtype, grad=True)
    dy = torch.randn(A, 1, L, D, device="cuda", dtype=dtype)

    def run(mod, xx, cc, ac):
        for t in (xx, cc, *mod.parameters()):
            t.grad = None
        y = _call(mod, xx, cc, ac)
        y.backward(dy.to(y.dtype))
        return [y.detach()] + [t.grad.detach().clone() for t in (xx, cc, *mod.parameters())]

    fn = torch.compile(m) if compiled else m
    mine = run(fn, x, c, bf16)
    x64, c64 = x.detach().double().requires_grad_(), c.detach().double().requires_grad_()
    for t in (x64, c64, *ref.parameters()):
        t.grad = None
    y64 = ref(x64, c64)
    y64.backward(dy.double())
    r64 = [y64.detach()] + [t.grad for t in (x64, c64, *ref.parameters())]
    theirs = run(pt, x, c, bf16)
    for i, (a, b, p) in enumerate(zip(mine, r64, theirs)):
        assert _err(a, b) <= 1.5 * _err(p, b) + 1e-6, (i, _err(a, b), _err(p, b))
    names = _kernels(lambda: run(fn, x, c, bf16))
    assert _has(names, "ln_rows_k") or _has(names, "ln_cond_fwd_k"), names
    if op == "ct":
        assert _has(names, "sigmul_bwd_k"), names
    assert not any("_kernel" in k and k.startswith("_") for k in names), names


@pytest.mark.parametrize("op", ["adaln", "ct"])
def test_the_ab_switch_runs_the_portable_kernels(op, monkeypatch):
    import collections

    import triton.runtime.jit as jit

    launches = collections.Counter()
    original = jit.JITFunction.run

    def run(self, *args, **kwargs):
        launches[getattr(self.fn, "__name__", "?")] += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(jit.JITFunction, "run", run)
    settings.configure(force_portable_kernels=True)
    m = _build(op, IT.MINIWORLD)
    x, c = _inputs(torch.float32)
    with torch.no_grad():
        names = _kernels(lambda: m(x, c))
    assert launches, "no Triton kernel launched"
    assert not any(k.startswith(("adaln_gemm_k", "expand_swiglu_k", "squeeze_gate_k")) or "ln_rows_k" in k for k in names), names
