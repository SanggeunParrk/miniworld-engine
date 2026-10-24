"""The sm_90a Transition beyond n = 4 / aligned rows / bf16: hidden = 2 D on every width, ragged row counts, fp32 on TF32 tensor cores.

Two code paths are checked here against a float64 reference (forward output and every gradient) and for bitwise determinism:

* ``fused_generic_sm90a`` -- the generic kernels (any D multiple of 64 up to 1024, hidden a multiple of 128, any row count, bf16 or
  TF32), and
* the tuned fused kernels (``fused_sm90a`` D = 128, ``fused_wide_sm90a`` D = 64 / 256 / 384 / 512) at n = 2.

Plus the module dispatch: which path ``modules.Transition`` takes for which shape.
"""

import copy

import pytest
import torch
import torch.nn.functional as F
from settings_helpers import override_setting

from miniworld_engine.modules.exceptions import ImplementationType

pytestmark = pytest.mark.gpu

CUDA = torch.cuda.is_available()
HOPPER = CUDA and torch.cuda.get_device_capability() == (9, 0)
needs_hopper = pytest.mark.skipif(
    not HOPPER, reason="the sm_90a Transition kernels need an H100"
)


def _reference(x, gamma, beta, wa, wb, ws, eps=1e-5):
    xn = F.layer_norm(x, (x.shape[-1],), gamma, beta, eps)
    return x + (F.silu(xn @ wa.t()) * (xn @ wb.t())) @ ws.t()


def _operands(m, d, n, dtype, seed=0):
    torch.manual_seed(seed)
    hid = n * d
    dev = "cuda"
    x = torch.randn(m, d, device=dev, dtype=dtype)
    gamma = (1 + 0.2 * torch.randn(d, device=dev)).float()
    beta = (0.2 * torch.randn(d, device=dev)).float()
    wa = (torch.randn(hid, d, device=dev) * d**-0.5).to(dtype)
    wb = (torch.randn(hid, d, device=dev) * d**-0.5).to(dtype)
    ws = (torch.randn(d, hid, device=dev) * hid**-0.5).to(dtype)
    dy = torch.randn(m, d, device=dev, dtype=dtype)
    return x, gamma, beta, wa, wb, ws, dy


def _rel(got, want):
    return float(
        (got.double() - want.double()).norm() / want.double().norm().clamp_min(1e-30)
    )


def _check_against_float64(fn, m, d, n, dtype, tol, deterministic_ln_grads=True):
    x, gamma, beta, wa, wb, ws, dy = _operands(m, d, n, dtype)
    ref_in = [
        t.detach().double().requires_grad_() for t in (x, gamma, beta, wa, wb, ws)
    ]
    y_ref = _reference(*ref_in)
    y_ref.backward(dy.double())

    def step():
        ins = [
            t.detach().clone().requires_grad_() for t in (x, gamma, beta, wa, wb, ws)
        ]
        y = fn(*ins, 1e-5)
        y.backward(dy)
        return y, ins

    y, ins = step()
    names = ("out", "dx", "dgamma", "dbeta", "dWa", "dWb", "dWs")
    got = [y, *(t.grad for t in ins)]
    want = [y_ref, *(t.grad for t in ref_in)]
    for name, g, w in zip(names, got, want, strict=True):
        assert _rel(g, w) < tol, f"M={m} D={d} n={n} {dtype}: {name} {_rel(g, w):.3e}"
    # replay is bit-identical, and the no-grad forward is the training forward
    y2, ins2 = step()
    assert torch.equal(y, y2)
    for i, (a, b) in enumerate(zip(ins, ins2, strict=True)):
        if (
            i in (1, 2) and not deterministic_ln_grads
        ):  # dgamma / dbeta of the engine LayerNorm backward used at D >= 384: atomics
            assert _rel(a.grad, b.grad) < 1e-5
        else:
            assert torch.equal(a.grad, b.grad)
    with torch.no_grad():
        assert torch.equal(fn(x, gamma, beta, wa, wb, ws, 1e-5), y)


# -------------------------------------------------------------------------------------------------------------------- the gate
@needs_hopper
def test_generic_gate_accepts_and_rejects():
    from miniworld_engine.kernels.transition.cuda import fused_generic_sm90a as gen

    def operands(m, d, hid, dtype=torch.bfloat16):
        return (
            torch.randn(m, d, device="cuda", dtype=dtype),
            torch.randn(hid, d, device="cuda", dtype=dtype),
            torch.randn(d, hid, device="cuda", dtype=dtype),
        )

    assert gen.supported(*operands(100, 768, 1536))  # ragged rows, D768, n = 2
    assert gen.supported(*operands(1, 64, 128))
    assert gen.supported(
        torch.randn(2, 7, 9, 192, device="cuda", dtype=torch.bfloat16),
        *operands(1, 192, 384)[1:],
    )  # 4-D activation, D192
    assert not gen.supported(*operands(128, 96, 384))  # width not a multiple of 64
    assert not gen.supported(*operands(128, 128, 192))  # hidden not a multiple of 128
    assert not gen.supported(*operands(128, 1152, 2304))  # width above 1024
    assert not gen.supported(
        torch.randn(128, 128, device="cuda", dtype=torch.float16),
        *operands(1, 128, 256, torch.float16)[1:],
    )  # fp16
    x, wa, ws = operands(128, 128, 256)
    assert gen.supported(x, wa.float(), ws)  # fp32 masters under bf16 activations are cast inside
    assert not gen.supported(x, wa.half(), ws)  # a weight dtype the kernels do not read
    assert not gen.supported(x.cpu(), wa.cpu(), ws.cpu())
    assert not gen.supported(
        torch.empty(0, 128, device="cuda", dtype=torch.bfloat16), wa, ws
    )
    xf, waf, wsf = operands(128, 128, 256, torch.float32)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    assert not gen.supported(xf, waf, wsf)  # fp32 only on TF32 tensor cores
    torch.set_float32_matmul_precision("high")
    try:
        assert gen.supported(xf, waf, wsf)
    finally:
        torch.set_float32_matmul_precision("highest")


@needs_hopper
def test_tuned_gates_take_n2():
    from miniworld_engine.kernels.transition.cuda import fused_sm90a, fused_wide_sm90a

    def operands(d, hid):
        return (
            torch.randn(128, d, device="cuda", dtype=torch.bfloat16),
            torch.randn(hid, d, device="cuda", dtype=torch.bfloat16),
            torch.randn(d, hid, device="cuda", dtype=torch.bfloat16),
        )

    assert fused_sm90a.supported(*operands(128, 256))
    assert not fused_sm90a.supported(*operands(128, 384))
    for d in (64, 256, 384, 512):
        assert fused_wide_sm90a.supported(*operands(d, 4 * d))
        assert fused_wide_sm90a.supported(*operands(d, 2 * d))  # n = 2 is built for D64 ... D512
        assert not fused_wide_sm90a.supported(*operands(d, 3 * d))
    assert fused_wide_sm90a.supported(*operands(768, 1536))  # D768 n = 2 has a tuned build in the committed wide family


# -------------------------------------------------------------------------------------------------------------- generic kernels
BF16_SHAPES = [
    (1, 64, 2),
    (100, 128, 2),
    (129, 64, 4),
    (1000, 256, 2),
    (130, 192, 2),
    (257, 320, 2),
    (384, 384, 2),
    (513, 512, 2),
    (200, 768, 2),
    (300, 768, 4),
    (65, 1024, 2),
    (777, 384, 4),
    (128, 128, 4),
]


@needs_hopper
@pytest.mark.parametrize(("m", "d", "n"), BF16_SHAPES)
def test_generic_bf16_matches_float64(m, d, n):
    from miniworld_engine.kernels.transition.cuda import fused_generic_sm90a as gen

    _check_against_float64(gen.transition_generic_sm90a, m, d, n, torch.bfloat16, 8e-3)


FP32_SHAPES = [
    (1, 64, 2),
    (100, 128, 2),
    (1000, 256, 2),
    (130, 192, 2),
    (384, 384, 2),
    (200, 768, 2),
    (300, 512, 4),
]


@needs_hopper
@pytest.mark.parametrize(("m", "d", "n"), FP32_SHAPES)
def test_generic_fp32_tf32_matches_float64(m, d, n):
    from miniworld_engine.kernels.transition.cuda import fused_generic_sm90a as gen

    torch.set_float32_matmul_precision("high")
    try:
        _check_against_float64(
            gen.transition_generic_sm90a, m, d, n, torch.float32, 4e-3
        )
    finally:
        torch.set_float32_matmul_precision("highest")


@needs_hopper
def test_generic_graph_replay_is_bit_identical():
    """CUDA-graph capture of forward + backward, replayed over the same buffers."""
    from miniworld_engine.kernels.transition.cuda import fused_generic_sm90a as gen

    x, gamma, beta, wa, wb, ws, dy = _operands(1000, 384, 2, torch.bfloat16)
    ins = [t.clone().requires_grad_() for t in (x, gamma, beta, wa, wb, ws)]

    def step():
        y = gen.transition_generic_sm90a(*ins, 1e-5)
        return (y, *torch.autograd.grad(y, ins, dy))

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            step()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = step()
    graph.replay()
    first = [t.clone() for t in out]
    for _ in range(5):
        graph.replay()
    for a, b in zip(first, out, strict=True):
        assert torch.equal(a, b)
    eager = step()
    for a, b in zip(first, eager, strict=True):
        assert torch.equal(a, b)


# ------------------------------------------------------------------------------------------------ tuned kernels at n = 2
@needs_hopper
@pytest.mark.parametrize(
    ("d", "m"),
    [(128, 256), (128, 1024), (64, 256), (64, 1024), (256, 256), (256, 1024)],
)
def test_tuned_n2_matches_float64(d, m):
    from miniworld_engine.kernels.transition.cuda import fused_sm90a, fused_wide_sm90a

    x, _, _, wa, _, ws, _ = _operands(m, d, 2, torch.bfloat16)
    if d == 128:
        assert fused_sm90a.available(x, wa, ws)
        fn = fused_sm90a.transition_fused_sm90a
    else:
        assert fused_wide_sm90a.available(x, wa, ws)
        fn = fused_wide_sm90a.transition_wide_sm90a
    _check_against_float64(
        fn, m, d, 2, torch.bfloat16, 8e-3, deterministic_ln_grads=True
    )


# ------------------------------------------------------------------------------------------------------------ module dispatch
def _module(d, n, dtype, seed=72):
    from miniworld_engine.modules import Transition

    torch.manual_seed(seed)
    module = (
        Transition(d, n=n, implementation=ImplementationType.MINIWORLD).cuda().bfloat16()
    )
    with torch.no_grad():
        for param in module.parameters():
            if param.ndim == 2:
                param.normal_(std=param.shape[-1] ** -0.5)
            elif param is module.ln_in.weight:
                param.copy_(1 + 0.2 * torch.randn_like(param))
            else:
                param.normal_(std=0.2)
    return module.to(dtype) if dtype is torch.float32 else module


def _run(module, x, dy, *, fused):
    from miniworld_engine import settings

    mod = copy.deepcopy(module)
    settings.configure(
        force_portable_kernels=not fused, transition_fused_sm90a=fused
    )
    xx = x.clone().requires_grad_()
    y = mod(xx)
    y.backward(dy)
    return {
        "out": y.detach().float(),
        "dx": xx.grad.detach().float(),
        **{name: p.grad.detach().float() for name, p in mod.named_parameters()},
    }


def _spy(monkeypatch, module_obj, name):
    calls = []
    original = getattr(module_obj, name)
    monkeypatch.setattr(
        module_obj, name, lambda *a, **k: (calls.append(1), original(*a, **k))[1]
    )
    return calls


@needs_hopper
@pytest.mark.parametrize(
    ("d", "n", "lead", "which"),
    [
        (128, 2, (1, 16, 16), "fused_sm90a.transition_fused_sm90a"),
        (64, 2, (1, 16, 16), "fused_wide_sm90a.transition_wide_sm90a"),
        (256, 2, (1, 16, 16), "fused_wide_sm90a.transition_wide_sm90a"),
        (384, 2, (1, 16, 16), "fused_wide_sm90a.transition_wide_sm90a"),
        (512, 4, (1, 16, 16), "fused_wide_sm90a.transition_wide_sm90a"),
        (768, 2, (1, 16, 16), "fused_wide_sm90a.transition_wide_sm90a"),
        (
            128,
            4,
            (1, 15, 15),
            "fused_generic_sm90a.transition_generic_sm90a",
        ),  # 225 rows: ragged
        (
            64,
            2,
            (3, 11, 13),
            "fused_generic_sm90a.transition_generic_sm90a",
        ),  # 429 rows: ragged
        (
            384,
            3,
            (1, 16, 16),
            "fused_generic_sm90a.transition_generic_sm90a",
        ),  # hidden 3 D = 1152: not a tuned multiple
    ],
)
def test_module_takes_the_path_and_is_no_less_accurate_than_triton(
    d, n, lead, which, monkeypatch
):
    from miniworld_engine import settings
    from miniworld_engine.kernels.transition.cuda import (
        fused_generic_sm90a,
        fused_sm90a,
        fused_wide_sm90a,
    )

    monkeypatch.setattr(settings, "_ACTIVE", settings.current())
    override_setting(
        monkeypatch, "transition_generic_small", "0"
    )  # this test targets the fused / wide build itself, not the small-row preference
    mod_name, fn_name = which.split(".")
    target = {
        "fused_sm90a": fused_sm90a,
        "fused_wide_sm90a": fused_wide_sm90a,
        "fused_generic_sm90a": fused_generic_sm90a,
    }[mod_name]
    calls = _spy(monkeypatch, target, fn_name)
    module = _module(d, n, torch.bfloat16)
    x = torch.randn(*lead, d, device="cuda", dtype=torch.bfloat16)
    dy = torch.randn_like(x)
    fused = _run(module, x, dy, fused=True)
    assert calls, f"{which} was never entered"
    triton = _run(module, x, dy, fused=False)
    reference = _run(_module(d, n, torch.float32), x.float(), dy.float(), fused=False)
    for name in reference:
        got, base = (
            _rel(fused[name], reference[name]),
            _rel(triton[name], reference[name]),
        )
        assert got <= max(base * 1.6, 1e-5), (
            f"D{d} n{n} {name}: fused {got:.3e} vs triton {base:.3e}"
        )


@needs_hopper
def test_module_fp32_runs_the_generic_kernels_on_tf32(monkeypatch):
    from miniworld_engine import settings
    from miniworld_engine.kernels.transition.cuda import fused_generic_sm90a

    monkeypatch.setattr(settings, "_ACTIVE", settings.current())
    calls = _spy(monkeypatch, fused_generic_sm90a, "transition_generic_sm90a")
    module = _module(256, 2, torch.float32)
    x = torch.randn(1, 16, 20, 256, device="cuda")
    dy = torch.randn_like(x)
    torch.set_float32_matmul_precision("highest")
    off = _run(module, x, dy, fused=True)
    assert not calls, "fp32 without TF32 must keep the exact-fp32 path"
    torch.set_float32_matmul_precision("high")
    try:
        on = _run(module, x, dy, fused=True)
    finally:
        torch.set_float32_matmul_precision("highest")
    assert calls, "fp32 with TF32 never entered the generic kernels"
    for name in off:
        assert _rel(on[name], off[name]) < 5e-3, name


@needs_hopper
@pytest.mark.parametrize(
    ("d", "rows", "training", "expect_generic"),
    [
        (128, 1024, False, True),
        (128, 1024, True, True),
        (128, 16384, False, False),
        (256, 12288, False, False),
        (256, 12288, True, True),
        (384, 12288, False, True),
        (512, 49152, True, False),
        (96, 1024, False, False),
    ],
)
def test_small_rows_prefer_the_generic_kernels(
    d, rows, training, expect_generic, monkeypatch
):
    """Below ~132 row tiles the fused / wide builds leave SMs idle; the generic kernels (tiles over the columns too) are faster
    (measured on H100 against those builds, anthropic and the compiled reference)."""
    from miniworld_engine.kernels.transition.cuda import fused_generic_sm90a

    override_setting(monkeypatch, "transition_generic_small", None)
    x = torch.empty(rows, d, device="cuda", dtype=torch.bfloat16)
    assert fused_generic_sm90a.prefers_generic(x, training) is expect_generic
    override_setting(monkeypatch, "transition_generic_small", "0")
    assert fused_generic_sm90a.prefers_generic(x, training) is False
    assert fused_generic_sm90a.prefers_generic(x.float(), training) is False
