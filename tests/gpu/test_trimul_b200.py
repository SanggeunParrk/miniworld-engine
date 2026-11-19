# pyright: reportOptionalMemberAccess=false, reportOptionalSubscript=false, reportOptionalIterable=false, reportArgumentType=false, reportGeneralTypeIssues=false, reportPrivateImportUsage=false
"""B200 (sm_100a) TriMul through the classes' ``cuda`` backend (``kernels/integrations/trimul_b200``): the route serves every requirement
width (D64-D512, both directions, bidirectional, hidden 2 D at D64 / D128, RoseTTAFold3's length factor), inference and training; the
outputs and gradients stay near the fp32 statements; runs are bitwise repeatable (the b1s / b1g / b7g ring flags spin across CTAs:
repeated backwards must neither hang nor change a bit); a batch equals its one-sample calls; an L that is not a multiple of 16 is padded
on the host; a CUDA-graph replay gives the eager bits."""

import copy
from typing import Any

import pytest
import torch

from miniworld_engine.measure import config as C
from miniworld_engine.modules import BidirectionalTriangleMultiplication, TriangleMultiplication, TriangleMultiplicationRF3

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0), reason="needs a B200 (sm_100a)"),
]

AUTOCAST = {"device_type": "cuda", "dtype": torch.bfloat16}


def _module(kind: str, d: int, *, hidden: int | None = None, seed: int = 0):
    if kind == "bidir":
        m = BidirectionalTriangleMultiplication(d, d_hidden=hidden)
    elif kind == "rf3":
        m = TriangleMultiplicationRF3(d, outgoing=True)
    else:
        m = TriangleMultiplication(d, d_hidden=hidden, outgoing=kind == "out")
    m = m.cuda().float()
    C.randomize(m, torch.Generator(device="cuda").manual_seed(seed))
    m.set_master_weights()
    return m


def _inputs(m, length: int, samples: int = 1, seed: int = 1, masked: bool = True):
    x = torch.randn(
        samples, length, length, m.ln_pair.weight.shape[0], device="cuda", generator=torch.Generator(device="cuda").manual_seed(seed)
    )
    mask = torch.ones(samples, length, dtype=torch.bool, device="cuda")
    if masked:
        for b in range(samples):
            mask[b, length - (length // 8) * (b + 1) % length :] = False
    return x.bfloat16(), mask


def _run(m, x, mask, train: bool, dy=None) -> tuple[torch.Tensor, Any, Any]:
    """(y, dx, parameter grads) under bf16 autocast; train: forward + backward of a fixed upstream gradient."""
    if not train:
        with torch.no_grad(), torch.autocast(**AUTOCAST):
            return m(x, mask=mask), None, None
    xx = x.clone().requires_grad_(True)
    for p in m.parameters():
        p.grad = None
    with torch.autocast(**AUTOCAST):
        y = m(xx, mask=mask)
    if dy is None:
        dy = torch.randn(y.shape, device="cuda", generator=torch.Generator(device="cuda").manual_seed(7))
    dy = dy.to(y.dtype)
    y.backward(dy)
    return y.detach(), xx.grad, [p.grad.clone() for p in m.parameters()]


def _rel(a, b) -> float:
    return ((a.double() - b.double()).abs().max() / b.double().abs().max().clamp_min(1e-30)).item()


CASES = [
    ("out", 64, None),
    ("in", 64, None),
    ("out", 64, 128),
    ("in", 64, 128),
    ("rf3", 64, None),
    ("out", 128, None),
    ("in", 128, None),
    ("rf3", 128, None),
    ("out", 256, None),
    ("in", 384, None),
    ("out", 512, None),
    ("bidir", 64, None),
    ("bidir", 128, None),
    ("bidir", 256, None),
    ("bidir", 384, None),
    ("bidir", 512, None),
]


@pytest.mark.parametrize("train", [False, True], ids=["eval", "train"])
@pytest.mark.parametrize(("kind", "d", "hidden"), CASES)
def test_b200_route_and_accuracy(kind, d, hidden, train) -> None:
    base = _module(kind, d, hidden=hidden).train(train)
    base.p_drop = 0.0
    x, mask = _inputs(base, 256)
    kern = copy.deepcopy(base).set_backend("cuda")
    ref = copy.deepcopy(base)  # fp32 statements: the masters, the input cast up
    with torch.set_grad_enabled(train), torch.autocast(**AUTOCAST):
        assert kern.unsupported(x, mask=mask) is None
    y, dx, gw = _run(kern, x, mask, train)
    with torch.autocast("cuda", enabled=False):
        ry, rdx, rgw = _run(ref, x.float(), mask, train)
    assert _rel(y.float() - x.float(), ry - x.float()) < 2e-2
    if train:
        assert _rel(dx, rdx) < 2e-2
        for g, r in zip(gw, rgw, strict=True):
            assert _rel(g, r) < 3e-2


@pytest.mark.parametrize(
    ("kind", "d", "hidden"),
    [("bidir", 64, None), ("out", 64, None), ("out", 64, 128), ("bidir", 128, None), ("in", 128, None), ("out", 256, None)],
)
def test_b200_train_repeatable(kind, d, hidden) -> None:
    """Twenty backwards of the ring-flag kernels (b1s / b1g / b7g at D64 / D128): no hang, the same bits every time."""
    m = _module(kind, d, hidden=hidden).set_backend("cuda").train()
    m.p_drop = 0.0
    x, mask = _inputs(m, 384)
    y0, dx0, g0 = _run(m, x, mask, True)
    for _ in range(19):
        y, dx, g = _run(m, x, mask, True)
        assert torch.equal(y, y0) and torch.equal(dx, dx0)
        assert all(torch.equal(a, b) for a, b in zip(g, g0, strict=True))


@pytest.mark.parametrize("train", [False, True], ids=["eval", "train"])
@pytest.mark.parametrize(("kind", "d"), [("bidir", 64), ("out", 128), ("in", 256)])
def test_b200_batch_equals_samples(kind, d, train) -> None:
    m = _module(kind, d).set_backend("cuda").train(train)
    m.p_drop = 0.0
    x, mask = _inputs(m, 128, samples=3)
    dy = torch.randn(x.shape, device="cuda", generator=torch.Generator(device="cuda").manual_seed(7))
    y, dx, _ = _run(m, x, mask, train, dy)
    for b in range(3):
        yb, dxb, _ = _run(m, x[b : b + 1], mask[b : b + 1], train, dy[b : b + 1])
        assert torch.equal(y[b : b + 1], yb)
        if train:
            assert torch.equal(dx[b : b + 1], dxb)


@pytest.mark.parametrize(("kind", "d"), [("out", 64), ("bidir", 128), ("in", 384)])
def test_b200_ragged_length_inference(kind, d) -> None:
    """L 200 (not a multiple of 16): padded on the host with masked pad tokens."""
    m = _module(kind, d).eval()
    x, mask = _inputs(m, 200)
    kern = copy.deepcopy(m).set_backend("cuda")
    with torch.no_grad(), torch.autocast(**AUTOCAST):
        assert kern.unsupported(x, mask=mask) is None
    y, _, _ = _run(kern, x, mask, False)
    with torch.autocast("cuda", enabled=False):
        ry, _, _ = _run(m, x.float(), mask, False)
    assert _rel(y.float() - x.float(), ry - x.float()) < 2e-2


def test_b200_rf3_masked_batch() -> None:
    """RoseTTAFold3's per-sample length factor with a mask over two samples (one launch per sample)."""
    m = _module("rf3", 128).eval()
    x, mask = _inputs(m, 256, samples=2)
    kern = copy.deepcopy(m).set_backend("cuda")
    y, _, _ = _run(kern, x, mask, False)
    with torch.autocast("cuda", enabled=False):
        ry, _, _ = _run(m, x.float(), mask, False)
    assert _rel(y.float() - x.float(), ry - x.float()) < 2e-2


@pytest.mark.parametrize(("kind", "d"), [("bidir", 128), ("out", 64), ("out", 512)])
def test_b200_graph_replay_matches_eager(kind, d) -> None:
    m = _module(kind, d).set_backend("cuda").train()
    m.p_drop = 0.0
    x, mask = _inputs(m, 256)
    xs = x.clone().requires_grad_(True)
    dy = torch.randn(x.shape, device="cuda", generator=torch.Generator(device="cuda").manual_seed(3)).bfloat16()

    def step():
        xs.grad = None
        for p in m.parameters():
            p.grad = None
        with torch.autocast(**AUTOCAST):
            y = m(xs, mask=mask)
        y.backward(dy)
        return y

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            y_eager = step().detach().clone()
    torch.cuda.current_stream().wait_stream(s)
    assert xs.grad is not None
    dx_eager = xs.grad.clone()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        y_static = step()
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(y_static, y_eager)
    assert xs.grad is not None and torch.equal(xs.grad, dx_eager)


@pytest.mark.parametrize("p", [0.25, 0.15])
def test_b200_dropout_scale_is_the_reference_draw(p) -> None:
    """``b200_infer.dropout_scale`` (the draw + one compare / cast / scale kernel): the reference's bits for the same generator state,
    draw after draw."""
    from miniworld_engine.kernels.trimul_inproj.cuda.b200_infer import dropout_scale
    from miniworld_engine.modules.functional import dropout_scale as reference

    pair = torch.empty(2, 128, 128, 64, device="cuda", dtype=torch.bfloat16)
    torch.manual_seed(11)
    want = [reference(pair, p, 1) for _ in range(3)]  # consecutive draws: the generator offset advances alike
    torch.manual_seed(11)
    got = [dropout_scale(pair, p) for _ in range(3)]
    for g, w in zip(got, want, strict=True):
        assert g.shape == w.shape and g.dtype == w.dtype
        assert torch.equal(g, w)


@pytest.mark.parametrize("train", [False, True], ids=["eval", "train"])
@pytest.mark.parametrize("masked", [False, True], ids=["full", "masked"])
@pytest.mark.parametrize("d", [64, 128])
def test_b200_rf3_factor_in_forms(d, masked, train) -> None:
    """RoseTTAFold3 on one sample, D64 / D128: the length factor formed in the forms launch (``forms_rf3_op``, the gradient of W_l scaled in
    ``grad_reduce``) gives the bits of the ``_LeftScaled`` path (the factor as torch ops on the weight)."""
    from miniworld_engine.modules import triangle as T
    from miniworld_engine.modules.functional import inverse_length

    m = _module("rf3", d).set_backend("cuda").train(train)
    assert isinstance(m, TriangleMultiplicationRF3)
    m.p_drop = 0.0
    x, mask = _inputs(m, 256, masked=masked)
    dy = torch.randn(x.shape, device="cuda", generator=torch.Generator(device="cuda").manual_seed(7)).bfloat16()
    outs = []
    for fused in (True, False):
        for q in m.parameters():
            q.grad = None
        xx = x.clone().requires_grad_(train)
        with torch.autocast(**AUTOCAST), torch.set_grad_enabled(train):
            if fused:
                y = m._forward_cuda(xx, mask)
            else:
                f = inverse_length(xx, mask).reshape(-1)
                y = T._trimul_cuda_forward(T._LeftScaled(m, f[:1]), xx, mask, None, bidirectional=False)
        if train:
            y.backward(dy)
            outs.append([y.detach(), xx.grad, *[q.grad.clone() for q in m.parameters() if q.grad is not None]])
        else:
            outs.append([y])
    for a, b in zip(*outs, strict=True):
        if train:
            assert torch.equal(a, b)
        else:
            # inference (S90 part 3): the factor is applied on the left projection's accumulator inside the front kernels (power-of-two factors give
            # the same bits; a counted n not a power of two rounds the weight product differently): inside the bf16 noise of the output
            d = (a.float() - b.float()).abs()
            assert d.max().item() <= 8e-2 and d.pow(2).mean().sqrt().item() <= 4e-3


@pytest.mark.parametrize(("kind", "d"), [("out", 64), ("in", 64), ("rf3", 64), ("out", 128)])
def test_b200_multi_tile_pipelines(kind, d) -> None:
    """L384 (1152 tiles: about eight per CTA) in training, hidden == pair width: k3g's double-buffered accumulators and xn staging
    tiles (D64; the residual x is no longer reloaded), LN_out on warps 8-11 only, b7m's double-buffered x / dy tiles -- the
    fp32 statements within tolerance, and the same bits on every one of five runs."""
    base = _module(kind, d).train()
    base.p_drop = 0.0
    x, mask = _inputs(base, 384)
    kern = copy.deepcopy(base).set_backend("cuda")
    ref = copy.deepcopy(base)
    y, dx, gw = _run(kern, x, mask, True)
    with torch.autocast("cuda", enabled=False):
        ry, rdx, rgw = _run(ref, x.float(), mask, True)
    assert _rel(y.float() - x.float(), ry - x.float()) < 2e-2
    assert _rel(dx, rdx) < 2e-2
    for g, r in zip(gw, rgw, strict=True):
        assert _rel(g, r) < 3e-2
    for _ in range(4):
        y1, dx1, g1 = _run(kern, x, mask, True)
        assert torch.equal(y1, y) and torch.equal(dx1, dx)
        assert all(torch.equal(a, b) for a, b in zip(g1, gw, strict=True))


@pytest.mark.parametrize("train", [False, True], ids=["eval", "train"])
@pytest.mark.parametrize(
    ("kind", "d", "length"), [("out", 64, 128), ("in", 64, 128), ("bidir", 64, 128), ("out", 128, 128), ("rf3", 64, 256)]
)
def test_b200_pdl_chain_graph_matches_eager(kind, d, length, train) -> None:
    """The programmatic-dependent launches (k3g after the contraction, one tile per CTA at L128 with its two epilogue halves;
    forms / k1w / b1s / b7m / grad_reduce in training): ten replays of a captured step give the eager step's bits, and the module
    stays inside the fp32 statements' tolerance."""
    base = _module(kind, d).train(train)
    base.p_drop = 0.0
    x, mask = _inputs(base, length)
    m = copy.deepcopy(base).set_backend("cuda")
    xs = x.clone().requires_grad_(train)
    dy = torch.randn(x.shape, device="cuda", generator=torch.Generator(device="cuda").manual_seed(3)).bfloat16()

    def step():
        if train:
            xs.grad = None
            for p in m.parameters():
                p.grad = None
            with torch.autocast(**AUTOCAST):
                y = m(xs, mask=mask)
            y.backward(dy)
            return y
        with torch.no_grad(), torch.autocast(**AUTOCAST):
            return m(xs, mask=mask)

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            y_eager = step().detach().clone()
    torch.cuda.current_stream().wait_stream(s)
    dx_eager = xs.grad.clone() if train else None
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        y_static = step()
    for _ in range(10):
        g.replay()
        torch.cuda.synchronize()
        assert torch.equal(y_static, y_eager)
        if train:
            assert xs.grad is not None and torch.equal(xs.grad, dx_eager)
    ry, _, _ = _run(copy.deepcopy(base), x.float(), mask, False)
    assert _rel(y_eager.float() - x.float(), ry - x.float()) < 2e-2


@pytest.mark.parametrize(("d", "r"), [(256, 256), (256, 512), (384, 384), (512, 512)])
def test_b200_wide_forms_and_grad_pack(d, r) -> None:
    """D256-512 training: ``wide_forms`` (one launch) is bitwise the six bf16 casts and the packed [W1; Wg] operand it replaced (fp32
    masters, one column-major); ``wide_grad_pack`` is bitwise the unpack / dWp expression / copies it replaced (2026-11-17)."""
    from miniworld_engine.kernels.trimul_inproj.cuda import b200_train as T

    g = torch.Generator(device="cuda").manual_seed(3)
    rnd = lambda *s: torch.randn(*s, device="cuda", generator=g)
    ps = [rnd(r, d) for _ in range(4)] + [rnd(d, d), rnd(d, r)]
    ps[1] = ps[1].t().contiguous().t()
    gi, bi, go, bo = rnd(d), rnd(d), rnd(r), rnd(r)
    lv = T._kernel_leaves([rnd(1, 4, 4, d).bfloat16(), *ps, gi, bi, go, bo])
    ref = [w.to(torch.bfloat16) for w in ps]
    assert all(torch.equal(a, b) for a, b in zip(lv[1:7], ref, strict=True))
    assert torch.equal(lv[11], torch.cat((T._pack_w1(*ref[:4]), ref[4])))
    dw, mm, acc = rnd(4 * r + d, d), rnd(d, r), rnd(4 * d + 2 * r)
    out = torch.empty(T._grad_total(lv), device="cuda")
    T._ext().wide_grad_pack(dw, mm, acc, go, bo, out)
    p = 2 * r
    r0, r1 = acc[:d], acc[d : 2 * d]
    expect = [
        *T._unpack_w1(dw[: 2 * p], p),
        dw[2 * p :],
        go[None, :] * (mm - r1[:, None]) + r0[:, None] * bo[None, :],
        acc[2 * d : 3 * d],
        acc[3 * d : 4 * d],
        acc[4 * d : 4 * d + r],
        acc[4 * d + r :],
    ]
    assert all(torch.equal(a, b) for a, b in zip(T._split_grads(out, lv), expect, strict=True))


def test_b200_dropout_scale_graph_replays() -> None:
    """Captured in a CUDA graph, ``b200_infer.dropout_scale`` draws, replay after replay, what the reference's captured draw gives from
    the same seed (2026-11-17)."""
    from miniworld_engine.kernels.trimul_inproj.cuda.b200_infer import dropout_scale
    from miniworld_engine.modules.functional import dropout_scale as reference

    pair = torch.empty(1, 256, 256, 64, device="cuda", dtype=torch.bfloat16)

    def replays(fn):
        torch.manual_seed(5)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            fn()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            out = fn()
        res = []
        for _ in range(3):
            g.replay()
            res.append(out.clone())
        torch.cuda.synchronize()
        return res

    got, want = replays(lambda: dropout_scale(pair, 0.25)), replays(lambda: reference(pair, 0.25, 1))
    assert not torch.equal(got[0], got[1])
    assert all(torch.equal(a, b) for a, b in zip(got, want, strict=True))


@pytest.mark.parametrize(("kind", "length", "samples"), [("out", 128, 1), ("in", 384, 1), ("out", 256, 2), ("rf3", 512, 1)])
def test_b200_unified_b1_matches_ring_b1s(kind, length, samples) -> None:
    """D64 one direction: the unified B1 (``b1u``: front epilogue and LN_out backward in one CTA, no ring / flags / cooperative launch)
    gives the ring b1s path's y, dx and parameter gradients up to the bf16 rounding of the (S1 | S2) column partials' order, and
    is bitwise repeatable (five backwards; B = 2: the sample-major 4-D maps)."""
    from miniworld_engine.kernels.trimul_inproj.cuda import b200_train as BT

    m = _module(kind, 64).set_backend("cuda").train()
    x, mask = _inputs(m, length, samples=samples)
    flag = BT.B1U
    try:
        BT.B1U = False
        torch.manual_seed(3)
        y0, dx0, g0 = _run(m, x, mask, True)
        BT.B1U = True
        torch.manual_seed(3)
        y1, dx1, g1 = _run(m, x, mask, True)
        for _ in range(4):
            torch.manual_seed(3)
            y, dx, g = _run(m, x, mask, True)
            assert torch.equal(y, y1) and torch.equal(dx, dx1)
            assert all(torch.equal(a, b) for a, b in zip(g, g1, strict=True))
    finally:
        BT.B1U = flag
    assert torch.equal(y0, y1)
    assert _rel(dx1, dx0) < 1e-2
    for a, b in zip(g1, g0, strict=True):
        assert _rel(a, b) < 1e-2


@pytest.mark.parametrize(
    ("kind", "hidden", "length", "samples"), [("out", 128, 128, 1), ("in", 128, 384, 1), ("bidir", None, 256, 2), ("bidir", None, 128, 1)]
)
def test_b200_unified_b1_h128_matches_ring_b1s(kind, hidden, length, samples) -> None:
    """D64 at contraction width 128 (hidden 128 one direction, bidirectional): the unified B1 at H = 128 (``b1v``: A^T on 128 TMEM
    lanes, dnr in a two-slot ring of 64-channel halves) gives the ring b1s path's y, dx and parameter gradients up to the bf16 rounding
    of the per-tile sums' order, and is bitwise repeatable (five backwards; B = 2: two samples per launch)."""
    from miniworld_engine.kernels.trimul_inproj.cuda import b200_train as BT

    m = _module(kind, 64, hidden=hidden).set_backend("cuda").train()
    x, mask = _inputs(m, length, samples=samples)
    flag = BT.B1V
    try:
        BT.B1V = False
        torch.manual_seed(3)
        y0, dx0, g0 = _run(m, x, mask, True)
        BT.B1V = True
        torch.manual_seed(3)
        y1, dx1, g1 = _run(m, x, mask, True)
        for _ in range(4):
            torch.manual_seed(3)
            y, dx, g = _run(m, x, mask, True)
            assert torch.equal(y, y1) and torch.equal(dx, dx1)
            assert all(torch.equal(a, b) for a, b in zip(g, g1, strict=True))
    finally:
        BT.B1V = flag
    assert torch.equal(y0, y1)
    assert _rel(dx1, dx0) < 1e-2
    for a, b in zip(g1, g0, strict=True):
        assert _rel(a, b) < 1e-2


@pytest.mark.parametrize(("kind", "length", "samples"), [("out", 128, 1), ("in", 128, 1), ("rf3", 128, 1), ("out", 128, 2), ("in", 256, 1)])
def test_b200_unified_b1_d128_matches_ring_b1g(kind, length, samples) -> None:
    """D128 one direction: the unified B1 at C = 128 (``b1w``: front epilogue and LN_out backward in one CTA, dnr in G's TMEM columns, no
    ring / flags / cooperative launch) gives the ring b1g path's y, dx and parameter gradients up to the order of the per-CTA sums, and is
    bitwise repeatable (five backwards). B = 2 at L128 and L256 run several tiles per CTA (the single-buffered stages in sequence)."""
    from miniworld_engine.kernels.trimul_inproj.cuda import b200_train as BT

    m = _module(kind, 128).set_backend("cuda").train()
    x, mask = _inputs(m, length, samples=samples)
    flag, top = BT.B1W, BT.B1W_MAX_L
    try:
        BT.B1W_MAX_L = 1 << 20
        BT.B1W = False
        torch.manual_seed(3)
        y0, dx0, g0 = _run(m, x, mask, True)
        BT.B1W = True
        torch.manual_seed(3)
        y1, dx1, g1 = _run(m, x, mask, True)
        for _ in range(4):
            torch.manual_seed(3)
            y, dx, g = _run(m, x, mask, True)
            assert torch.equal(y, y1) and torch.equal(dx, dx1)
            assert all(torch.equal(a, b) for a, b in zip(g, g1, strict=True))
    finally:
        BT.B1W, BT.B1W_MAX_L = flag, top
    assert torch.equal(y0, y1)
    assert _rel(dx1, dx0) < 1e-2
    for a, b in zip(g1, g0, strict=True):
        assert _rel(a, b) < 1e-2


@pytest.mark.parametrize("direction", [0, 1, 2])
@pytest.mark.parametrize(("sm", "length", "samples"), [(True, 128, 1), (True, 256, 2), (False, 128, 2), (False, 256, 1)])
def test_b200_contraction_backward_one_launch(direction, sm, length, samples) -> None:
    """``cgb_backward`` (every product of the contraction backward in one tcgen05 launch) against the cuBLAS products it replaces
    (``_contract_bwd_sm`` / ``_contract_bwd``): equal up to the order of the fp32 sums over L before the one bf16 rounding (rel <= 1e-2
    on random planes), and bitwise repeatable."""
    from miniworld_engine.kernels.trimul_inproj.cuda import b200_train as BT

    g = torch.Generator(device="cuda").manual_seed(11)
    k = 64 if sm else 128  # channels per contraction operand
    nops = 4 if direction == 0 else 2
    if sm:
        planes = torch.randn(nops, samples, 64, length, length, device="cuda", generator=g).bfloat16()
        dt = torch.randn(nops // 2, samples, 64, length, length, device="cuda", generator=g).bfloat16()
        ref = BT._contract_bwd_sm(planes, dt, direction)
    else:
        planes = torch.randn(nops * k, samples, length, length, device="cuda", generator=g).bfloat16()
        dt = torch.randn(nops // 2 * k, samples, length, length, device="cuda", generator=g).bfloat16()
        ref = BT._contract_bwd(planes, dt, k, direction)
    E = BT._ext()
    out = torch.empty_like(planes)
    E.cgb_backward(planes, dt, out, length, direction)
    again = torch.empty_like(planes)
    E.cgb_backward(planes, dt, again, length, direction)
    torch.cuda.synchronize()
    assert torch.equal(out, again)
    assert ((out.float() - ref.float()).abs().max() / ref.float().abs().max()).item() < 1e-2


@pytest.mark.parametrize("length", [128, 256])
@pytest.mark.parametrize(
    ("kind", "d", "hidden"), [("out", 64, None), ("in", 64, None), ("out", 64, 128), ("bidir", 64, None), ("out", 128, None)]
)
def test_b200_m64_tiles(kind, d, hidden, length, monkeypatch) -> None:
    """The M = 64 tcgen05 kernels (k1h front, k3h output at D64; forced in every cell here, one tile per CTA at L128, several at
    L256): inside the fp32 statements' tolerance, within bf16 rounding of the k1w / k3g path, the same bits on repeated calls, and ten
    replays of a captured eval step give the eager bits."""
    from miniworld_engine.kernels.trimul_inproj.cuda import b200_infer as BI

    base = _module(kind, d, hidden=hidden).eval()
    x, mask = _inputs(base, length)
    m = copy.deepcopy(base).set_backend("cuda")
    y128, _, _ = _run(m, x, mask, False)
    cell = (length, kind == "bidir", d, hidden or d)
    monkeypatch.setattr(BI, "M64_FRONT", frozenset({cell}))
    monkeypatch.setattr(BI, "M64_OUT", frozenset({cell}) if d == 64 else frozenset())
    y, _, _ = _run(m, x, mask, False)
    with torch.autocast("cuda", enabled=False):
        ry, _, _ = _run(copy.deepcopy(base), x.float(), mask, False)
    assert _rel(y.float() - x.float(), ry - x.float()) < 2e-2
    assert _rel(y.float() - x.float(), y128.float() - x.float()) < 1e-2
    for _ in range(3):
        assert torch.equal(_run(m, x, mask, False)[0], y)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s), torch.no_grad(), torch.autocast(**AUTOCAST):
        for _ in range(2):
            m(x, mask=mask)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g), torch.no_grad(), torch.autocast(**AUTOCAST):
        y_static = m(x, mask=mask)
    for _ in range(10):
        g.replay()
        torch.cuda.synchronize()
        assert torch.equal(y_static, y)


@pytest.mark.parametrize(("samples", "length", "d"), [(1, 128, 64), (1, 256, 64), (1, 768, 128), (2, 128, 64), (8, 640, 128)])
def test_b200_philox_forms_draw_is_torch_rand(samples, length, d) -> None:
    """``forms_philox_op`` (the draw made inside the forms launch, 2026-11-17 part 6): the row-dropout scale is bitwise the reference's
    torch.rand draw turned into the scale, and the generator stands where torch.rand leaves it (the next draw is equal), draw after
    draw; (8, 640, 128) covers more elements than one pass of torch's grid (several Philox calls per thread)."""
    from miniworld_engine.kernels.trimul_inproj.cuda import b200_train as BT
    from miniworld_engine.kernels.trimul_inproj.cuda.b200_infer import drop_scale_op

    m = _module("out", d)
    params = [
        m.to_left.weight,
        m.to_left_gate.weight,
        m.to_right.weight,
        m.to_right_gate.weight,
        m.to_gate.weight,
        m.to_out.weight,
        m.ln_out.weight.detach().float().contiguous(),
    ]
    rows = samples * length
    torch.manual_seed(3)
    want = [drop_scale_op(torch.rand((samples, 1, length, d), device="cuda", dtype=torch.bfloat16), 0.25).view(rows, d) for _ in range(3)]
    after_want = torch.rand(97, device="cuda")
    torch.manual_seed(3)
    got = [BT.forms_philox_op(params, rows, d, 0.25) for _ in range(3)]
    after_got = torch.rand(97, device="cuda")
    for (buf, ds), w in zip(got, want, strict=True):
        assert torch.equal(ds, w)
        assert torch.equal(buf, BT.forms_op(params))
    assert torch.equal(after_got, after_want)


@pytest.mark.parametrize("writer", [2, 1, 0], ids=["writer_pdl", "writer", "aten_fill"])
def test_b200_philox_forms_graph_replays(writer) -> None:
    """Captured in a CUDA graph, the in-forms draw gives, replay after replay, the reference's captured draw from the same seed, with the
    graph RNG prologue served by this extension's writer (one programmatic launch) or by ATen's two fills."""
    from miniworld_engine.kernels.trimul_inproj.cuda import b200_train as BT
    from miniworld_engine.kernels.trimul_inproj.cuda.b200_infer import drop_scale_op

    m = _module("out", 64)
    params = [
        m.to_left.weight,
        m.to_left_gate.weight,
        m.to_right.weight,
        m.to_right_gate.weight,
        m.to_gate.weight,
        m.to_out.weight,
        m.ln_out.weight.detach().float().contiguous(),
    ]
    BT.GRAPH_RNG_WRITER = writer

    def replays(fn):
        torch.manual_seed(5)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            fn()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            out = fn()
        res = []
        for _ in range(4):
            g.replay()
            res.append(out.clone())
        torch.cuda.synchronize()
        return res

    try:
        got = replays(lambda: BT.forms_philox_op(params, 256, 64, 0.25)[1])
        want = replays(lambda: drop_scale_op(torch.rand((1, 1, 256, 64), device="cuda", dtype=torch.bfloat16), 0.25).view(256, 64))
    finally:
        BT.GRAPH_RNG_WRITER = 2
    assert not torch.equal(got[0], got[1])
    assert all(torch.equal(a, b) for a, b in zip(got, want, strict=True))


@pytest.mark.parametrize(("kind", "d", "hidden"), [("out", 64, None), ("in", 64, 128), ("rf3", 64, None), ("bidir", 128, None)])
def test_b200_philox_in_forms_step_bits(kind, d, hidden) -> None:
    """A dropout training step with the draw in the forms launch (``PHILOX_IN_FORMS``) gives the torch.rand draw's y, dx and parameter
    gradients bitwise, two steps in a row (the generator advances alike)."""
    from miniworld_engine.kernels.trimul_inproj.cuda import b200_train as BT

    m = _module(kind, d, hidden=hidden).set_backend("cuda").train()
    m.p_drop = 0.25
    x, mask = _inputs(m, 128)
    res = {}
    try:
        for flag in (False, True):
            BT.PHILOX_IN_FORMS = flag
            torch.manual_seed(9)
            res[flag] = [_run(m, x, mask, True) for _ in range(2)]
    finally:
        BT.PHILOX_IN_FORMS = True
    for (y0, dx0, g0), (y1, dx1, g1) in zip(res[False], res[True], strict=True):
        assert torch.equal(y0, y1) and torch.equal(dx0, dx1)
        assert all(torch.equal(a, b) for a, b in zip(g0, g1, strict=True))
    assert not torch.equal(res[True][0][0], res[True][1][0])


@pytest.mark.parametrize(("kind", "hidden", "masked"), [("out", None, True), ("in", 128, False), ("rf3", None, True)])
def test_b200_b7m_extra_ctas_reduce_bits(kind, hidden, masked) -> None:
    """D64 L128 (b7m tiles < SMs): b7m's extra CTAs summing the B1 kernel's parameter-gradient partials (``B7M_XR``, 2026-11-17 part 6)
    give grad_reduce's bits for every gradient, and repeated backwards stay bitwise equal."""
    from miniworld_engine.kernels.trimul_inproj.cuda import b200_train as BT

    m = _module(kind, 64, hidden=hidden).set_backend("cuda").train()
    m.p_drop = 0.0
    x, mask = _inputs(m, 128, masked=masked)
    res = {}
    try:
        for flag in (False, True):
            BT.B7M_XR = flag
            res[flag] = [_run(m, x, mask, True) for _ in range(3)]
    finally:
        BT.B7M_XR = True
    for y, dx, g in res[False] + res[True]:
        assert torch.equal(y, res[False][0][0]) and torch.equal(dx, res[False][0][1])
        assert all(torch.equal(a, b) for a, b in zip(g, res[False][0][2], strict=True))


@pytest.mark.parametrize("writer", [2, 0], ids=["writer_pdl", "aten_fill"])
def test_b200_graph_rng_writer_shared_with_triattn(writer) -> None:
    """ONE graph RNG writer (``kernels/_graph_rng_sm100``, 2026-11-17 part 7) serves the RNG prologue of graphs holding TriangleAttention's
    draw (``b200_triattn.drop_scale``, pack.cu), TriMul's (the forms launch) or both, all captured in one process: replay after replay
    the draws equal those of ATen's fills (writer 0), and with the writer on no replay launches ATen's fill kernel (before, the family
    that registered its fill_ override last served only its own scalars)."""
    from torch.profiler import ProfilerActivity, profile

    from miniworld_engine.kernels import _graph_rng_sm100
    from miniworld_engine.kernels.triangle_attention.cuda import b200_triattn
    from miniworld_engine.kernels.trimul_inproj.cuda import b200_train as BT

    m = _module("out", 64)
    params = [
        m.to_left.weight,
        m.to_left_gate.weight,
        m.to_right.weight,
        m.to_right_gate.weight,
        m.to_gate.weight,
        m.to_out.weight,
        m.ln_out.weight.detach().float().contiguous(),
    ]
    pair = torch.zeros(1, 256, 256, 64, device="cuda", dtype=torch.bfloat16)

    def ta():
        return b200_triattn.drop_scale(pair, 0.25)

    def tm():
        return BT.forms_philox_op(params, 256, 64, 0.25)[1]

    def both():
        return torch.cat([ta().reshape(-1), tm().reshape(-1)])

    def graphs(writer):
        BT.GRAPH_RNG_WRITER = writer
        _graph_rng_sm100.set_mode(writer)
        out = {}
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        for name, fn in (("triattn", ta), ("trimul", tm), ("both", both)):
            torch.manual_seed(5)
            with torch.cuda.stream(s):
                fn()
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=s):
                o = fn()
            out[name] = (g, o)
        return out

    try:
        ref = graphs(0)
        want = {k: [] for k in ref}
        for _ in range(3):
            for k, (g, o) in ref.items():
                g.replay()
                want[k].append(o.clone())
        torch.cuda.synchronize()
        del ref
        got = graphs(writer)
        assert _graph_rng_sm100.ext().pointer_count() >= 2  # the default generator's seed / offset scalars, registered by both families
        res = {k: [] for k in got}
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(3):
                for k, (g, o) in got.items():
                    g.replay()
                    res[k].append(o.clone())
            torch.cuda.synchronize()
    finally:
        BT.GRAPH_RNG_WRITER = 2
        _graph_rng_sm100.set_mode(2)
    for k in want:
        assert not torch.equal(res[k][0], res[k][1])
        assert all(torch.equal(a, b) for a, b in zip(res[k], want[k], strict=True)), k
    names = [e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    fills = sum("FillFunctor<long>" in n for n in names)
    writes = sum("rng_scalar_write" in n for n in names)
    if writer:
        assert fills == 0 and writes == 9  # one launch per replay: 3 graphs x 3 replays
    else:
        assert fills == 18 and writes == 0


@pytest.mark.parametrize(("kind", "hidden", "length"), [("out", None, 128), ("in", None, 128), ("out", 128, 128), ("in", 128, 128)])
def test_b200_k1wt_master_front_bits(kind, hidden, length) -> None:
    """The training front from the fp32 master weights right after the forms launch (``k1wt``, 2026-11-18 part 8): a dropout training
    step's y, dx and parameter gradients are bitwise those of the k1w path, two steps in a row, eager and replayed in a CUDA graph."""
    from miniworld_engine.kernels.trimul_inproj.cuda import b200_train as BT

    m = _module(kind, 64, hidden=hidden).set_backend("cuda").train()
    m.p_drop = 0.25
    x, mask = _inputs(m, length)
    calls = []
    real = BT.forward_small_master

    def counted(*a):
        calls.append(1)
        return real(*a)

    BT.forward_small_master = counted
    tf_saved = BT.TRAIN_TF
    BT.TRAIN_TF = False  # the k1w path (the one-launch forward is not bitwise the chain)
    res = {}
    try:
        for flag in (True, False):  # k1wt first: the forms are cached after the first launch (eager)
            BT.K1WT = flag
            torch.manual_seed(9)
            res[flag] = [_run(m, x, mask, True) for _ in range(2)]
    finally:
        BT.K1WT = False
        BT.forward_small_master = real
    assert calls  # the cell took k1wt
    for (y0, dx0, g0), (y1, dx1, g1) in zip(res[False], res[True], strict=True):
        assert torch.equal(y0, y1) and torch.equal(dx0, dx1)
        assert all(torch.equal(a, b) for a, b in zip(g0, g1, strict=True))

    def graph_out(flag):
        BT.K1WT = flag
        xx = x.clone().requires_grad_(True)
        dy = torch.randn(x.shape, device="cuda", generator=torch.Generator(device="cuda").manual_seed(7)).to(x.dtype)

        def step():
            for p in m.parameters():
                p.grad = None
            xx.grad = None
            with torch.autocast(**AUTOCAST):
                y = m(xx, mask=mask)
            y.backward(dy)
            return y

        torch.manual_seed(4)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            step()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            y = step()
        out = []
        for _ in range(3):
            g.replay()
            out.append([y.clone(), xx.grad.clone(), *(p.grad.clone() for p in m.parameters())])
        torch.cuda.synchronize()
        return out

    try:
        a, b = graph_out(False), graph_out(True)
    finally:
        BT.K1WT = False
        BT.TRAIN_TF = tf_saved
    assert all(torch.equal(u, v) for ra, rb in zip(a, b, strict=True) for u, v in zip(ra, rb, strict=True))


@pytest.mark.parametrize("kind", ["out", "in", "rf3"])
@pytest.mark.parametrize("masked", [False, True])
def test_b200_l128_one_launch(kind, masked) -> None:
    """D64 L = 128 inference through ``tf_sm100.cu`` (one cluster launch) against the three-kernel chain (TRIMUL_B200_L128 off) and an
    fp64 reference of the same module: the one-launch error is the chain's (not bitwise: the contraction sums in another order), a
    CUDA-graph replay gives the eager bits, and repeated runs (the plane-row counters are monotonic across launches) do not change a bit."""
    from miniworld_engine.kernels.trimul_inproj.cuda import b200_infer as BI

    m = _module(kind, 64)
    m.set_backend("cuda")
    m.eval()
    x, mask = _inputs(m, 128, masked=masked)
    ref_m = copy.deepcopy(m).double()
    ref_m.set_backend("pytorch")
    with torch.no_grad():
        ref = ref_m(x.double(), mask=mask)
    assert BI._tf_ext().tf_supported()
    saved = BI.FUSED
    try:
        BI.FUSED = False
        chain = _run(m, x, mask, False)[0].clone()
        BI.FUSED = True
        one = _run(m, x, mask, False)[0].clone()
        for _ in range(5):
            assert torch.equal(_run(m, x, mask, False)[0], one)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            _run(m, x, mask, False)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            og = _run(m, x, mask, False)[0]
        for _ in range(10):
            g.replay()
        torch.cuda.synchronize()
        assert torch.equal(og, one)
    finally:
        BI.FUSED = saved
    e_chain = (chain.double() - ref).abs()
    e_one = (one.double() - ref).abs()
    assert torch.isfinite(one).all()
    assert e_one.max().item() <= 1.25 * e_chain.max().item() + 1e-3
    assert e_one.pow(2).mean().sqrt().item() <= 1.05 * e_chain.pow(2).mean().sqrt().item()


@pytest.mark.parametrize("kind", ["out", "in", "rf3"])
@pytest.mark.parametrize("masked", [False, True])
def test_b200_l128_train_one_launch(kind, masked) -> None:
    """D64 L = 128 training forward through ``tf_train_sm100.cu`` (one cluster launch, 2026-11-18 training part 11) against the
    k1w -> cuBLAS -> k3g chain (``TRAIN_TF`` off) and an fp64 reference: y, dx and every parameter gradient (the unchanged backward on
    the one-launch saved set) carry the chain's error (not bitwise: the contraction sums in another order); repeated steps and a
    CUDA-graph replay give the same bits."""
    from miniworld_engine.kernels.trimul_inproj.cuda import b200_train as BT

    m = _module(kind, 64).set_backend("cuda").train()
    m.p_drop = 0.0
    x, mask = _inputs(m, 128, masked=masked)
    dy = torch.randn(x.shape, device="cuda", generator=torch.Generator(device="cuda").manual_seed(7))
    ref_m = copy.deepcopy(m).double().set_backend("pytorch").train()
    xr = x.double().requires_grad_(True)
    yr = ref_m(xr, mask=mask)
    yr.backward(dy.double())
    ref = [yr.detach(), xr.grad, *(p.grad for p in ref_m.parameters())]
    calls = []
    real = BT._tft_ok

    def counted(*a):
        ok = real(*a)
        calls.append(ok)
        return ok

    BT._tft_ok = counted
    saved = BT.TRAIN_TF
    try:
        BT.TRAIN_TF = False
        chain = _run(m, x, mask, True, dy)
        BT.TRAIN_TF = True
        one = _run(m, x, mask, True, dy)
        for _ in range(3):
            again = _run(m, x, mask, True, dy)
            assert torch.equal(again[0], one[0]) and torch.equal(again[1], one[1])
            assert all(torch.equal(a, b) for a, b in zip(again[2], one[2], strict=True))
        xx = x.clone().requires_grad_(True)

        def step():
            for p in m.parameters():
                p.grad = None
            xx.grad = None
            with torch.autocast(**AUTOCAST):
                y = m(xx, mask=mask)
            y.backward(dy.to(y.dtype))
            return y

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            step()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            yg = step()
        for _ in range(3):
            g.replay()
        torch.cuda.synchronize()
        assert torch.equal(yg, one[0]) and torch.equal(xx.grad, one[1])
        assert all(torch.equal(p.grad, b) for p, b in zip(m.parameters(), one[2], strict=True))
    finally:
        BT.TRAIN_TF = saved
        BT._tft_ok = real
    assert any(calls)  # the cell took the one launch
    for k, (c, o, r) in enumerate(zip([chain[0], chain[1], *chain[2]], [one[0], one[1], *one[2]], ref, strict=True)):
        ec, eo = (c.double() - r).abs(), (o.double() - r).abs()
        assert torch.isfinite(o).all(), k
        assert eo.max().item() <= 1.5 * ec.max().item() + 1e-3 * r.abs().max().item(), k
        assert eo.pow(2).mean().sqrt().item() <= 1.1 * ec.pow(2).mean().sqrt().item() + 1e-4 * r.abs().max().item(), k


@pytest.mark.parametrize(
    ("kind", "hidden", "chain"), [("out", None, True), ("in", None, True), ("out", None, False), ("out", 128, False), ("in", 128, False)]
)
@pytest.mark.parametrize("masked", [False, True])
def test_b200_l128_train_master_after_forms(kind, hidden, chain, masked) -> None:
    """(2026-11-18 training part 12) the L128 training front reading the fp32 master weights itself -- hidden 64: the one-launch forward
    (``TFT_MASTER``) as the first kernel with the forms launch beside it (``TFT_CHAIN``) or right after a small wait-first forms launch,
    hidden 128: k1wt after such a forms launch (``K1WT_AF``) -- with a live row dropout: y, dx and every parameter gradient bitwise those
    of the forms-fed route, eager (the forms launch's steps and a cached-forms step) and in a CUDA-graph replay under the same
    generator seed."""
    from miniworld_engine.kernels.trimul_inproj.cuda import b200_train as BT

    saved_chain = BT.TFT_CHAIN
    BT.TFT_CHAIN = chain
    try:
        _master_ab(BT, kind, hidden, masked)
    finally:
        BT.TFT_CHAIN = saved_chain


def _master_ab(BT, kind, hidden, masked) -> None:
    m = _module(kind, 64, hidden=hidden).set_backend("cuda").train()
    m.p_drop = 0.25
    x, mask = _inputs(m, 128, masked=masked)
    dy = torch.randn(x.shape, device="cuda", generator=torch.Generator(device="cuda").manual_seed(7))
    calls = []
    flag, plan = ("TFT_MASTER", "tftm_plan") if hidden is None else ("K1WT_AF", "k1wt_plan")
    real = getattr(BT, plan)

    def counted(*a):
        ok = real(*a)
        calls.append(bool(ok))
        return ok

    def eager():
        outs = []
        for k in range(3):
            if k < 2:  # a new parameter version: the step makes its forms launch (the third reads the cached forms)
                with torch.no_grad():
                    for p in m.parameters():
                        p.mul_(1.0)
            torch.manual_seed(11)
            outs.append(_run(m, x, mask, True, dy))
        return outs

    def graphed():
        xx = x.clone().requires_grad_(True)

        def step():
            for p in m.parameters():
                p.grad = None
            xx.grad = None
            with torch.autocast(**AUTOCAST):
                y = m(xx, mask=mask)
            y.backward(dy.to(y.dtype))
            return y

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            step()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            yg = step()
        res = []
        for _ in range(2):
            torch.manual_seed(13)
            g.replay()
            torch.cuda.synchronize()
            res.append((yg.clone(), xx.grad.clone(), [p.grad.clone() for p in m.parameters()]))
        return res

    setattr(BT, plan, counted)
    saved = getattr(BT, flag)
    try:
        setattr(BT, flag, False)
        base_e, base_g = eager(), graphed()
        setattr(BT, flag, True)
        calls.clear()
        new_e, new_g = eager(), graphed()
    finally:
        setattr(BT, flag, saved)
        setattr(BT, plan, real)
    assert any(calls)  # the cell took the master path
    for a, b in zip([*base_e, *base_g], [*new_e, *new_g], strict=True):
        assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
        assert all(torch.equal(p, q) for p, q in zip(a[2], b[2], strict=True))
    assert torch.equal(new_g[0][0], new_g[1][0]) and torch.equal(new_e[0][0], new_e[2][0])


@pytest.mark.parametrize(("kind", "d", "length", "samples"), [("out", 256, 128, 1), ("in", 384, 128, 2), ("bidir", 256, 128, 1), ("out", 384, 256, 1), ("bidir", 384, 256, 2)])
@pytest.mark.parametrize("masked", [False, True])
def test_b200_folded_front(kind, d, length, samples, masked) -> None:
    """The LN_in-folded k1w front (D256 / D384 inference, ``b200_infer.fold_cell``; raw x into TMEM, one correction MMA per chunk):
    the folded call is at least as close to the fp32 statements as the normalise-then-MMA front, several tiles a CTA included; it is
    deterministic and a graph replay gives the eager bits."""
    from miniworld_engine.kernels.trimul_inproj.cuda import b200_infer as BI

    m = _module(kind, d).eval().set_backend("cuda")
    m.p_drop = 0.0
    ref = copy.deepcopy(m).set_backend("pytorch")
    x, mask = _inputs(m, length, samples=samples, masked=masked)
    assert BI.fold_cell(length, d, 0 if kind == "bidir" else 1)
    with torch.autocast("cuda", enabled=False), torch.no_grad():
        ry = ref(x.float(), mask=mask)
    saved = BI.K1W_FOLD
    try:
        BI.K1W_FOLD = False
        y0, _, _ = _run(m, x, mask, False)
        BI.K1W_FOLD = True
        y1, _, _ = _run(m, x, mask, False)
        assert torch.equal(y1, _run(m, x, mask, False)[0])
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            _run(m, x, mask, False)
        torch.cuda.synchronize()
        with torch.cuda.graph(g):
            yg, _, _ = _run(m, x, mask, False)
        g.replay()
        torch.cuda.synchronize()
        assert torch.equal(yg, y1)
    finally:
        BI.K1W_FOLD = saved
    assert not torch.equal(y0, y1)  # the folded front ran
    e0, e1 = _rel(y0.float() - x.float(), ry - x.float()), _rel(y1.float() - x.float(), ry - x.float())
    assert e1 < 2e-2 and e1 <= 1.25 * e0


@pytest.mark.parametrize(("kind", "d", "length"), [("out", 256, 128), ("in", 384, 128), ("bidir", 256, 128), ("out", 512, 256)])
def test_b200_wide_philox_in_forms_step_bits(kind, d, length) -> None:
    """S90: a D256-512 dropout training step with the draw and its scale made by the wide forms launch (``wide_forms_philox_op``) gives the
    torch.rand draw's y, dx and parameter gradients bitwise (two steps in a row: the generator advances alike), and the same bits again from
    the same seed."""
    from miniworld_engine.kernels.trimul_inproj.cuda import b200_train as BT

    m = _module(kind, d).set_backend("cuda").train()
    m.p_drop = 0.25
    x, mask = _inputs(m, length)
    res = {}
    try:
        for flag in (False, True, True):
            BT.PHILOX_IN_FORMS = flag
            torch.manual_seed(9)
            res[len(res)] = [_run(m, x, mask, True) for _ in range(2)]
    finally:
        BT.PHILOX_IN_FORMS = True
    for other in (1, 2):
        for (y0, dx0, g0), (y1, dx1, g1) in zip(res[0], res[other], strict=True):
            assert torch.equal(y0, y1) and torch.equal(dx0, dx1)
            # the LayerNorm / to_out gradients are column sums of the per-block partials: equal between the two draws' runs (same kernels)
            assert all(torch.equal(a, b) for a, b in zip(g0, g1, strict=True))
    assert not torch.equal(res[1][0][0], res[1][1][0])


@pytest.mark.parametrize("writer", [2, 0], ids=["writer_pdl", "aten_fill"])
def test_b200_wide_philox_forms_graph_replays(writer) -> None:
    """S90: captured in a CUDA graph, the wide forms launch's draw gives, replay after replay, the reference's captured draw from the same seed."""
    from miniworld_engine.kernels.trimul_inproj.cuda import b200_train as BT
    from miniworld_engine.kernels.trimul_inproj.cuda.b200_infer import drop_scale_op

    m = _module("out", 256)
    params = [m.to_left.weight, m.to_left_gate.weight, m.to_right.weight, m.to_right_gate.weight, m.to_gate.weight, m.to_out.weight]
    BT.GRAPH_RNG_WRITER = writer

    def replays(fn):
        torch.manual_seed(5)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            fn()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            out = fn()
        res = []
        for _ in range(4):
            g.replay()
            res.append(out.clone())
        torch.cuda.synchronize()
        return res

    try:
        got = replays(lambda: BT.wide_forms_philox_op(params, 256, 256, 0.25)[1])
        want = replays(lambda: drop_scale_op(torch.rand((1, 1, 256, 256), device="cuda", dtype=torch.bfloat16), 0.25).view(256, 256))
    finally:
        BT.GRAPH_RNG_WRITER = 2
    assert not torch.equal(got[0], got[1])
    assert all(torch.equal(a, b) for a, b in zip(got, want, strict=True))


@pytest.mark.parametrize(("kind", "d", "length"), [("out", 64, 128), ("in", 128, 256), ("out", 256, 128), ("bidir", 64, 128), ("rf3", 128, 128)])
def test_b200_train_graph_captures_equal_bits_with_poisoned_memory(kind, d, length) -> None:
    """S90: two fresh CUDA-graph captures of one training step (y, dx, every parameter gradient) are bitwise equal, also when a capture's pool is
    poisoned (0xFF blocks allocated, filled and freed inside the capture before the step): no kernel reads uninitialised memory."""
    m = _module(kind, d).set_backend("cuda").train()
    m.p_drop = 0.25
    x, mask = _inputs(m, length)
    xx = x.clone().requires_grad_(True)
    dy = torch.randn(x.shape, device="cuda", generator=torch.Generator(device="cuda").manual_seed(7)).to(x.dtype)

    def step():
        for p in m.parameters():
            p.grad = None
        xx.grad = None
        with torch.autocast(**AUTOCAST):
            y = m(xx, mask=mask)
        y.backward(dy)
        return y.detach()

    def capture(poisoned):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                step()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            if poisoned:
                blocks = [torch.empty(mb << 20, dtype=torch.uint8, device="cuda") for mb in (1, 2, 4, 8, 16, 32, 64, 128, 256) for _ in range(3)]
                for b in blocks:
                    b.fill_(0xFF)
                del blocks
            y = step()
        torch.cuda.synchronize()
        torch.manual_seed(1)
        g.replay()
        torch.cuda.synchronize()
        return [y.clone(), xx.grad.clone(), *[p.grad.clone() for p in m.parameters()]]

    a, b, c = capture(False), capture(False), capture(True)
    for other in (b, c):
        assert all(torch.equal(u, v) for u, v in zip(a, other, strict=True))
