"""B200 TriangleAttention at lengths that are not a multiple of the 128-token tile (FoldForge's crops, e.g. L 731), without padded copies.

The kernels that touch the caller's pair-shaped tensors (front, tail, gate / head backward, weight gradients; fused and wide paths)
read and write the unpadded [B, L, L, C] tensors through tensor maps bounded by L: rows and columns beyond L load as zeros and their
stores are dropped, and the front masks the keys beyond L; the intermediates keep the padded length.  Checked here, at L 333, 517, 700
and 731, for the fused d_pair-128 path (both nodes) and the wide path's cores (heads of 16 / 32 zero-padded to 32, 64, 96 zero-padded
to 128):

- the same computation as the explicitly padded call (the pair zero-padded to whole tiles, the pad keys masked, the result cut): y the
  same bits; in training (fp32 master parameters) dx within 2e-4, and on the wide path dW_q|k|v|g the same bits, dW_o within 5e-5 (dWo =
  dy_s^T u runs over the caller's B L^2 rows instead of B Lp^2 in cuBLAS: 1e-5 measured), dW_b / dgamma / dbeta within 1e-4 (they carry
  the query side's fp32 dbias reduce-add, unordered: the padded call against itself differs by up to 3.8e-5 at L 731); on the fused path
  (fp32 atomics for the weight gradients) every parameter gradient within 1e-4;
- against the module's PyTorch statements in fp64: within 1.25x the bf16 statements' own error (+1e-3 output, +2e-3 gradients), inference
  and training with dropout 0.25 (a fixed draw);
- CUDA-graph replay of a ragged call (inference and a training step, fp32 master parameters) equals the eager call: y the same bits,
  dx within 2e-4 and the parameter gradients within 1e-4 (the unordered fp32 reductions above).
"""

import gc

import pytest
import torch
import torch.nn.functional as F

from miniworld_engine import settings
from miniworld_engine.integrations import triattn_b200 as tri
from miniworld_engine.modules.exceptions import ImplementationType
from miniworld_engine.modules.triangle_attention.module import TriangleAttention

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]

# (d_pair, d_hidden, heads, starting, batch)
CONFIGS = {
    "C128-start": (128, 128, 4, True, 1),        # the fused path (AF3 / Boltz-2 / Protenix trunks at 4 x 32)
    "C128-end": (128, 128, 4, False, 1),
    "C64-H4x16-end": (64, 64, 4, False, 2),      # wide, 16-channel heads zero-padded to 32 (AF3 template stack), two samples
    "C256-H8x32-start": (256, 256, 8, True, 1),  # wide, 32-channel heads (Protenix-v2 trunk)
    "C256-H4x64-end": (256, 256, 4, False, 1),   # wide, the head-dim-64 core (Chai-1 trunk)
    "C384-H4x96-start": (384, 384, 4, True, 1),  # wide, the head-dim-128 core (96 zero-padded)
}
LENGTHS = (333, 517, 700, 731)


@pytest.fixture(autouse=True)
def policy():
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("B200 (sm_100) required")
    old = settings.configure(force_portable_kernels=False)
    try:
        yield
    finally:
        settings.configure(**vars(old))
        gc.collect()
        torch.cuda.empty_cache()


def relative(a, b):
    return float((a.detach().double() - b.detach().double()).norm() / b.detach().double().norm().clamp_min(1e-30))


def make(cfg, impl=ImplementationType.MINIWORLD, p_drop=0.0, dtype=torch.bfloat16):
    c, hd, h, starting, _ = cfg
    torch.manual_seed(0)
    m = TriangleAttention(c, h, d_hidden=hd, starting=starting, p_drop=p_drop, implementation=impl)
    with torch.no_grad():
        for name, p in m.named_parameters():
            if p.ndim == 2:
                p.normal_(std=p.shape[-1] ** -0.5)
            elif "weight" in name:
                p.copy_(1 + 0.1 * torch.randn_like(p))
            else:
                p.normal_(std=0.05)
    return m.cuda().to(dtype)


def inputs(B, L, C, seed=1):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(B, L, L, C, device="cuda", dtype=torch.bfloat16, generator=g)
    keep = torch.rand(B, L, device="cuda", generator=g) > 0.1          # ~10 % of the keys masked, the last 11 always
    keep[:, -11:] = False
    keep[:, 0] = True
    dy = 0.1 * torch.randn(B, L, L, C, device="cuda", dtype=torch.bfloat16, generator=g)
    return x, keep, dy


def drop_scale(module, B, L, C, p=0.25, seed=7):
    shape = [B, L, L, C]
    shape[module._drop_dim] = 1
    g = torch.Generator(device="cuda").manual_seed(seed)
    return ((torch.rand(shape, device="cuda", generator=g) > p).to(torch.bfloat16) / (1.0 - p))


def fix_drop(module, scale):
    module._make_drop_scale = lambda pair, p: scale.to(pair.dtype)


def run(m, x, mask, dy, train):
    """-> (y, dx, parameter gradients) of one call (dx and the gradients None in inference)."""
    if not train:
        with torch.no_grad():
            return m.eval()(x, mask), None, None
    m.zero_grad(set_to_none=True)
    xi = x.detach().clone().requires_grad_()
    y = m.train()(xi, mask)
    y.backward(dy.to(y.dtype))
    return y.detach(), xi.grad, [p.grad.detach().clone() for p in m.parameters()]


def padded_call(m, x, mask, dy, train, scale=None):
    """The explicitly padded call (the computation of the path before: zero rows / columns to whole tiles, the pad keys masked, the
    dropout scale zero-padded), cut back to L."""
    B, L = x.shape[:2]
    Lp = -(-L // 128) * 128
    xp = F.pad(x, (0, 0, 0, Lp - L, 0, Lp - L))
    keep = torch.zeros(B, Lp, dtype=torch.bool, device=x.device)
    keep[:, :L] = mask
    dyp = F.pad(dy, (0, 0, 0, Lp - L, 0, Lp - L)) if dy is not None else None
    if scale is not None:
        sp = F.pad(scale, (0, 0, 0, Lp - L)) if m._drop_dim == 1 else F.pad(scale, (0, 0, 0, 0, 0, Lp - L))
        fix_drop(m, sp)
    y, dx, grads = run(m, xp, keep, dyp, train)
    return y[:, :L, :L], None if dx is None else dx[:, :L, :L], grads


@pytest.mark.parametrize("train", [False, True], ids=["inference", "training"])
@pytest.mark.parametrize("length", LENGTHS)
@pytest.mark.parametrize("name", list(CONFIGS))
def test_ragged_matches_padded_call(name, length, train):
    cfg = CONFIGS[name]
    B, C = cfg[4], cfg[0]
    # training with fp32 master parameters: the gradients come back in fp32 (a bf16 gradient's rounding would hide the comparison)
    m = make(cfg, p_drop=0.25 if train else 0.0, dtype=torch.float32 if train else torch.bfloat16)
    x, mask, dy = inputs(B, length, C)
    assert tri.kind_of(m.train(train), x, mask) is not None
    scale = drop_scale(m, B, length, C) if train else None
    if scale is not None:
        fix_drop(m, scale)
    y, dx, grads = run(m, x, mask, dy, train)
    yp, dxp, gp = padded_call(m, x, mask, dy, train, scale)
    assert torch.isfinite(y).all()
    assert torch.equal(y, yp), relative(y.float() - x.float(), yp.float() - x.float())
    if train:
        assert relative(dx, dxp) <= 2e-4, relative(dx, dxp)
        wide = tri.kind_of(m, x, mask) == tri.GENERIC
        for (n, _), a, b in zip(m.named_parameters(), grads, gp, strict=True):
            assert a.dtype == torch.float32
            if wide and n.split(".")[0] in ("to_query", "to_key", "to_value", "to_gate"):
                assert torch.equal(a, b), (n, relative(a, b))
            else:
                bound = 5e-5 if wide and n.startswith("to_out") else 1e-4
                assert relative(a, b) <= bound, (n, relative(a, b))


# B L^2 = 258 064 / 332 929: cuBLAS 12.9's split-K GEMM dropped the last K % 64 rows of the wide path's dWo = dy_s^T u (and of d bo)
# over the caller's rows (2026-10-29: 0.4-2.7 % off the padded call); the wide configs only (the fused path's weight gradients are
# fp32 atomics in its kernels)
TAIL_LENGTHS = (508, 577)


@pytest.mark.parametrize("length", TAIL_LENGTHS)
@pytest.mark.parametrize("name", [n for n, c in CONFIGS.items() if c[0] != 128])
def test_ragged_long_reduction_keeps_every_row(name, length):
    test_ragged_matches_padded_call(name, length, True)


def statements(cfg, ours, dtype, p_drop):
    ref = make(cfg, ImplementationType.PYTORCH, p_drop, dtype=dtype)
    ref.load_state_dict({k: v.to(dtype) for k, v in ours.state_dict().items()})
    return ref


@pytest.mark.parametrize("train", [False, True], ids=["inference", "training"])
@pytest.mark.parametrize("length", LENGTHS)
@pytest.mark.parametrize("name", list(CONFIGS))
def test_ragged_against_fp64(name, length, train):
    cfg = CONFIGS[name]
    B, C = cfg[4], cfg[0]
    p = 0.25 if train else 0.0
    m = make(cfg, p_drop=p)
    x, mask, dy = inputs(B, length, C, seed=3)
    scale = drop_scale(m, B, length, C, p) if train else None
    if scale is not None:
        fix_drop(m, scale)
    y, dx, grads = run(m, x, mask, dy, train)
    del m
    out = {}
    for tag, dt in (("ref16", torch.bfloat16), ("ref64", torch.float64)):
        r = statements(cfg, make(cfg, p_drop=p), dt, p)
        if scale is not None:
            fix_drop(r, scale)
        out[tag] = run(r, x.to(dt), mask, dy, train)
        del r
        gc.collect()
        torch.cuda.empty_cache()
    xd = x.double()
    e = relative(y.double() - xd, out["ref64"][0] - xd)
    e16 = relative(out["ref16"][0].double() - xd, out["ref64"][0] - xd)
    assert e <= 1.25 * e16 + 1e-3, ("y", e, e16)
    if train:
        e, e16 = relative(dx, out["ref64"][1]), relative(out["ref16"][1], out["ref64"][1])
        assert e <= 1.25 * e16 + 2e-3, ("dx", e, e16)
        names = [n for n, _ in make(cfg).named_parameters()]
        for n, a, b16, b64 in zip(names, grads, out["ref16"][2], out["ref64"][2], strict=True):
            e, e16 = relative(a, b64), relative(b16, b64)
            assert e <= 1.25 * e16 + 2e-3, (n, e, e16)


@pytest.mark.parametrize("train", [False, True], ids=["inference", "training"])
@pytest.mark.parametrize("name", ["C128-end", "C256-H4x64-end"])
def test_ragged_cuda_graph(name, train):
    cfg = CONFIGS[name]
    B, C = cfg[4], cfg[0]
    m = make(cfg, p_drop=0.25 if train else 0.0, dtype=torch.float32 if train else torch.bfloat16)
    x, mask, dy = inputs(B, 333, C, seed=5)
    if train:
        fix_drop(m, drop_scale(m, B, 333, C))
    xi = x.detach().clone().requires_grad_(train)
    params = list(m.parameters())
    m.train(train)

    def step():
        if not train:
            with torch.no_grad():
                return (m(xi, mask),)
        y = m(xi, mask)
        return (y, *torch.autograd.grad(y, [xi, *params], dy))

    eager = [t.detach().clone() for t in step()]
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        step()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        outs = step()
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(outs[0], eager[0])
    for i, (a, b) in enumerate(zip(outs[1:], eager[1:], strict=True)):
        assert relative(a, b) <= (2e-4 if i == 0 else 1e-4), (i, relative(a, b))
