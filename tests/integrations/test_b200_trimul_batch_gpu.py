"""B200 TriMul D64, native batch: B samples in one call (tokens b-major, one launch per stage) against B calls of one sample.

The kernels are the same per token; what the batch changes is the sample index of the token mask, the per-sample row-dropout scale (k3g / b1s reload
it when a tile's sample changes) and cuBLAS's batch dim (d B planes). Each sample draws its own dropout scale and mask, so an index mix-up between
samples shows as a large error."""

import pytest
import torch

from miniworld_engine.kernels.trimul_inproj.cuda import b200_infer, b200_train

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]

D = 64
EPS = 1e-5


@pytest.fixture(autouse=True)
def b200():
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("B200 (sm_100) required")


def relative(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12))


def leaves_for(direction, x, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    bf = torch.bfloat16
    p = (4 if direction == 0 else 2) * D
    h = p // 2                                                   # channels of t
    r = lambda *s, sc: (torch.randn(*s, device="cuda", generator=g) * sc)
    wl, wlg, wr, wrg = (r(p // 2, D, sc=D ** -0.5).to(bf) for _ in range(4))
    wg = r(D, D, sc=D ** -0.5).to(bf)
    wp = r(D, h, sc=h ** -0.5).to(bf)
    gi, bi = 1 + 0.1 * r(D, sc=1.0), 0.1 * r(D, sc=1.0)
    go_, bo = 1 + 0.1 * r(h, sc=1.0), 0.1 * r(h, sc=1.0)
    # wp is [out = D, in = h]; front matrices [out = p/2, in = D]
    return [x, wl, wlg, wr, wrg, wg, wp, gi.float(), bi.float(), go_.float(), bo.float()]


CASES = [(L, B, direction) for L in (128, 256, 384) for B in (2, 4) for direction in (0, 1, 2)]
IDS = [f"L{L}-B{B}-dir{d}" for L, B, d in CASES]


def per_sample(fn, x, mask, ds):
    return torch.cat([fn(x[i:i + 1], None if mask is None else mask[i], None if ds is None else ds[i]) for i in range(x.shape[0])], 0)


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_inference_batch_matches_samples(case):
    L, B, direction = case
    torch.manual_seed(1)
    x = torch.randn(B, L, L, D, device="cuda", dtype=torch.bfloat16)
    leaves = leaves_for(direction, x)
    mask = torch.rand(B, L, device="cuda") > 0.15            # a different mask per sample
    ds = (torch.rand(B, L, D, device="cuda") > 0.25).to(torch.bfloat16) / 0.75

    def run(xb, mk, d_):
        lv = [xb, *leaves[1:]]
        return b200_infer.inference(lv, None if mk is None else mk.reshape(-1), None if d_ is None else d_.reshape(-1, D).contiguous(), direction)

    got = b200_infer.inference([x, *leaves[1:]], mask.reshape(-1), ds.reshape(-1, D).contiguous(), direction)
    want = per_sample(run, x, mask, ds)
    assert torch.isfinite(got.float()).all()
    assert relative(got, want) < 2e-3, relative(got, want)
    # no dropout, no mask
    got0 = b200_infer.inference([x, *leaves[1:]], None, None, direction)
    want0 = per_sample(lambda xb, m_, d_: b200_infer.inference([xb, *leaves[1:]], None, None, direction), x, None, None)
    assert relative(got0, want0) < 2e-3


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_training_batch_matches_samples(case):
    L, B, direction = case
    torch.manual_seed(2)
    xb = torch.randn(B, L, L, D, device="cuda", dtype=torch.bfloat16)
    base = leaves_for(direction, xb)
    mask = torch.rand(B, L, device="cuda") > 0.15
    ds = (torch.rand(B, L, D, device="cuda") > 0.25).to(torch.bfloat16) / 0.75
    dy = torch.randn(B, L, L, D, device="cuda", dtype=torch.bfloat16)

    def grads(x, mk, d_, dyy):
        lv = [x.detach().clone().requires_grad_(), *(t.detach().clone().requires_grad_(t.is_floating_point()) for t in base[1:])]
        y = b200_train.trimul_train(lv, None if mk is None else mk.reshape(-1), None if d_ is None else d_.reshape(-1, D).contiguous(), direction)
        y.backward(dyy)
        return y.detach(), [t.grad.clone() for t in lv]

    y_b, g_b = grads(xb, mask, ds, dy)
    ys, gs = [], []
    for i in range(B):
        yi, gi = grads(xb[i:i + 1], mask[i], ds[i], dy[i:i + 1])
        ys.append(yi); gs.append(gi)
    y_s = torch.cat(ys, 0)
    assert relative(y_b, y_s) < 2e-3, ("y", relative(y_b, y_s))
    dx_s = torch.cat([g[0] for g in gs], 0)
    assert relative(g_b[0], dx_s) < 5e-3, ("dx", relative(g_b[0], dx_s))
    for k in range(1, len(base)):                             # weight gradients: the batch sums over samples
        want = sum(g[k].float() for g in gs)
        assert relative(g_b[k], want) < 1e-2, (k, relative(g_b[k], want))
