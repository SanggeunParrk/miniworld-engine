"""H100: the width-specific paths of the hand-CUDA AdaptiveLayerNorm / ConditionedTransition (``kernels/adaln/cuda``, ``kernels/conditioned_transition/cuda``).

* the atom width (d = c = 128): the one-launch forward ``adaln_atom_k`` (inference and the training forward that saves LN(cond), the pre-activation and the
  statistics) and the backward ``adaln_dgrad_atom_k`` (dgrad + the LayerNorm backward of cond), bf16 and fp32;
* the fp32 token width with a 768-wide cond (the ESMFold2 shape): the feature-major backward row pass ``adaln_bwd_t_k`` feeding the TF32 weight-gradient GEMM, and the
  one-launch weight packing ``pack_f32_k`` (AdaLN and the ConditionedTransition tail);
* the small-M squeeze tile of the ConditionedTransition inference tail.

Each is pinned as the other width tests are: the kernels that launch (eager and captured in a CUDA graph), parity with an fp64 reference within 1.5x the error of the
same-dtype PyTorch statements for the output and every gradient, and a bitwise-identical repeat.
"""
from __future__ import annotations

import pytest
import torch

from miniworld_engine import settings
from miniworld_engine.modules.adaptive_layernorm.module import AdaptiveLayerNorm
from miniworld_engine.modules.conditioned_transition.module import ConditionedTransition
from miniworld_engine.modules.exceptions import ImplementationType as IT

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0), reason="H100 only"),
]


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


def _build(op, d, dc, impl, dtype=torch.float32):
    torch.manual_seed(0)
    m = (AdaptiveLayerNorm(d, dc, implementation=impl) if op == "adaln" else ConditionedTransition(d, dc, implementation=impl, dtype=dtype)).cuda()
    for p in m.parameters():
        p.data.normal_(0, 0.05) if p.ndim > 1 else p.data.normal_(0, 0.5)
    return m


def _ref64(op, m, d, dc):
    ref = _build(op, d, dc, IT.PYTORCH, torch.float64).double() if op == "ct" else AdaptiveLayerNorm(d, dc, implementation=IT.PYTORCH, dtype=torch.float64).cuda()
    ref.load_state_dict({k: v.double() for k, v in m.state_dict().items()})
    return ref


def _err(a, ref):
    return ((a.double() - ref).norm() / ref.norm()).item()


def _call(m, x, c, bf16):
    if bf16:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return m(x, c)
    return m(x, c)


# (d, dc, rows): 3 x 700 rows is not a multiple of the 64-row tile of the atom kernels; 2 x 4 x 128 rows exercise the multi-tile loops
ATOM = (128, 128, (3, 700))
ESM = (768, 768, (2, 160))


@pytest.mark.parametrize("op", ["adaln", "ct"])
@pytest.mark.parametrize("bf16", [True, False], ids=["bf16", "fp32"])
@pytest.mark.parametrize("graph", [False, True], ids=["eager", "graph"])
def test_atom_width_inference(op, bf16, graph):
    d, dc, (a, l) = ATOM
    dtype = torch.bfloat16 if bf16 else torch.float32
    m = _build(op, d, dc, IT.MINIWORLD)
    ref, pt = _ref64(op, m, d, dc), _build(op, d, dc, IT.PYTORCH)
    pt.load_state_dict(m.state_dict())
    torch.manual_seed(1)
    x, c = torch.randn(a, 1, l, d, device="cuda", dtype=dtype), torch.randn(a, 1, l, dc, device="cuda", dtype=dtype)
    with torch.no_grad():
        run = lambda: _call(m, x, c, bf16)  # noqa: E731
        y = run()
        if graph:
            run()
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                out = run()
            g.replay()
            torch.cuda.synchronize()
            y, replay = out, (lambda: g.replay())
        else:
            replay = run
        names = _kernels(replay)
        yr, yp = ref(x.double(), c.double()), _call(pt, x, c, bf16)
        assert _err(y, yr) <= 1.5 * _err(yp, yr) + 1e-6, (_err(y, yr), _err(yp, yr))
        again = run() if not graph else (g.replay(), out.clone())[1]
        assert torch.equal(again, y)
    assert _has(names, "adaln_atom_k"), names
    assert not _has(names, "ln_rows_k"), names   # one launch, not two
    if op == "ct":
        assert _has(names, "expand_swiglu_k", "squeeze_gate_k"), names


@pytest.mark.parametrize("cfg", [ATOM, ESM], ids=["atom", "token_c768"])
@pytest.mark.parametrize("op", ["adaln", "ct"])
@pytest.mark.parametrize("bf16", [True, False], ids=["bf16", "fp32"])
def test_training_gradients_and_kernels(cfg, op, bf16):
    d, dc, (a, l) = cfg
    dtype = torch.bfloat16 if bf16 else torch.float32
    m = _build(op, d, dc, IT.MINIWORLD)
    ref, pt = _ref64(op, m, d, dc), _build(op, d, dc, IT.PYTORCH)
    pt.load_state_dict(m.state_dict())
    torch.manual_seed(1)
    x = torch.randn(a, 1, l, d, device="cuda", dtype=dtype, requires_grad=True)
    c = torch.randn(a, 1, l, dc, device="cuda", dtype=dtype, requires_grad=True)
    dy = torch.randn(a, 1, l, d, device="cuda", dtype=dtype)

    def run(mod, xx, cc, ac):
        for t in (xx, cc, *mod.parameters()):
            t.grad = None
        y = _call(mod, xx, cc, ac)
        y.backward(dy.to(y.dtype))
        return [y.detach()] + [t.grad.detach().clone() for t in (xx, cc, *mod.parameters())]

    mine = run(m, x, c, bf16)
    x64, c64 = x.detach().double().requires_grad_(), c.detach().double().requires_grad_()
    for t in (x64, c64, *ref.parameters()):
        t.grad = None
    y64 = ref(x64, c64)
    y64.backward(dy.double())
    r64 = [y64.detach()] + [t.grad for t in (x64, c64, *ref.parameters())]
    theirs = run(pt, x, c, bf16)
    for i, (u, v, w) in enumerate(zip(mine, r64, theirs)):
        assert _err(u, v) <= 1.5 * _err(w, v) + 1e-6, (i, _err(u, v), _err(w, v))
    again = run(m, x, c, bf16)
    assert all(torch.equal(u, v) for u, v in zip(mine, again)), "the step is not bitwise repeatable"
    names = _kernels(lambda: run(m, x, c, bf16))
    if cfg is ATOM:
        assert _has(names, "adaln_atom_k", "adaln_dgrad_atom_k"), names
    elif not bf16:
        assert _has(names, "adaln_bwd_t_k", "pack_f32_k"), names   # the feature-major row pass; the one-launch weight packing
    if op == "ct" and not bf16:
        assert _has(names, "pack_f32_k"), names
    assert not any("_kernel" in k and k.startswith("_") for k in names), names   # no Triton kernel


@pytest.mark.parametrize("bf16", [True, False], ids=["bf16", "fp32"])
def test_ct_inference_small_m_squeeze_tile(bf16):
    """M = 5 x 64 rows: 3 row blocks x 6 column blocks of 128 would leave most SMs idle, the squeeze takes 64-column tiles; same result as the 128 tile."""
    from miniworld_engine.kernels.conditioned_transition.cuda import ct_inference as ci

    dtype = torch.bfloat16 if bf16 else torch.float32
    torch.manual_seed(2)
    d, nd, dc, m = 768, 1536, 384, 320
    x, cond, res = (torch.randn(m, k, device="cuda", dtype=dtype) for k in (d, dc, d))
    wa, wb, ws, wsc, bsc = (torch.randn(*s, device="cuda") * 0.05 for s in ((nd, d), (nd, d), (d, nd), (d, dc), (d,)))
    y64 = ci.ct_inference(x, cond, wa, wb, ws, wsc, bsc, res, bn=64)
    y128 = ci.ct_inference(x, cond, wa, wb, ws, wsc, bsc, res, bn=128)
    auto = ci.ct_inference(x, cond, wa, wb, ws, wsc, bsc, res)
    assert torch.equal(auto, y64)   # the small-M rule picked 64
    assert (y64.double() - y128.double()).abs().max() <= 5e-2 * y128.double().abs().max()
