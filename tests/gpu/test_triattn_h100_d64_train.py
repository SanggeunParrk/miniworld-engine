"""H100 TriangleAttention training core of 64-channel heads (RF3 4 x 64, Chai-1 / confidence d_pair 256 x 4, AF3 d_pair 512 x 8): the wgmma
forward (``triattn_d64t_sm90.cu``: o, base-2 lse, u) against an fp64 reference and the mma.sync forward, and the query-side backward taking two
pair rows a CTA (``BWD_Q_ROWS`` 2: their dS tiles added in shared memory, half the dbias planes) against one row a CTA and an fp64 reference:
dq / dk / dv bitwise the same, dbias within the bf16 rounding of the planes, every run bitwise equal to its repeat (deterministic)."""

import pytest
import torch

from miniworld_engine.kernels.triangle_attention.cuda import h100_dh as dh

pytestmark = [pytest.mark.gpu, pytest.mark.h100]
D = 64


def _inputs(L, H, seed=0):
    torch.manual_seed(seed)
    HD = H * D
    proj = (torch.randn(L * L, 4 * HD, device="cuda") * 1.5).bfloat16()
    bias = (torch.randn(1, H, L, L, device="cuda") * 2).bfloat16()
    bias[..., L - L // 8 :] = float("-inf")  # masked trailing keys
    return proj, bias


def _ref(proj, bias, do, L, H):
    """fp64 forward (o, lse2, u) and backward (dq, dk, dv, dbias) of the core, natural-unit bias."""
    HD = H * D
    sc = D**-0.5
    rows = proj.double().view(L, L, 4, H, D)  # [n, q, 4, h, d]
    q, k, v, g = rows[:, :, 0], rows[:, :, 1], rows[:, :, 2], rows[:, :, 3]
    s = torch.einsum("nqhd,nkhd->nhqk", q, k) * sc + bias[0].double()[None]
    lse = torch.logsumexp(s, -1)
    p = torch.exp(s - lse[..., None])
    o = torch.einsum("nhqk,nkhd->nqhd", p, v)
    do_ = do.double().view(L, L, H, D)
    dp = torch.einsum("nqhd,nkhd->nhqk", do_, v)
    ds = p * (dp - (do_ * o).sum(-1).permute(0, 2, 1)[..., None])
    dq = torch.einsum("nhqk,nkhd->nqhd", ds, k) * sc
    dk = torch.einsum("nhqk,nqhd->nkhd", ds, q) * sc
    dv = torch.einsum("nhqk,nqhd->nkhd", p, do_)
    return o.reshape(L * L, HD), lse * 1.4426950408889634, (o * torch.sigmoid(g)).reshape(L * L, HD), dq.reshape(L * L, HD), dk.reshape(
        L * L, HD
    ), dv.reshape(L * L, HD), ds.sum(0)


def _rel(a, b):
    return ((a.double() - b.double()).pow(2).mean().sqrt() / b.double().pow(2).mean().sqrt()).item()


@pytest.mark.parametrize("L,H", [(128, 2), (256, 4), (384, 2)])
def test_forward_wgmma(L, H):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("needs an H100 (sm_90)")
    proj, bias = _inputs(L, H)
    scale = D**-0.5
    o, lse, u = dh.ext_d64t().triattn_d64_forward_train(proj, bias, H, scale)
    o2, lse2, u2 = dh.ext_d64t().triattn_d64_forward_train(proj, bias, H, scale)
    assert torch.equal(o, o2) and torch.equal(lse, lse2) and torch.equal(u, u2), "not deterministic"
    do = torch.zeros(L * L, H * D, device="cuda", dtype=torch.bfloat16)
    ro, rlse, ru, *_ = _ref(proj, bias, do, L, H)
    assert _rel(o, ro) < 6e-3 and _rel(u, ru) < 6e-3
    assert (lse.permute(0, 1, 3, 2).reshape(L * L, H).double() - rlse.permute(0, 2, 1).reshape(L * L, H)).abs().max() < 2e-2
    # the mma.sync core: the same quantities
    om, lm, um = dh.ext().triattn_dh_forward_train(proj, bias, H, D, scale)
    assert _rel(o, om) < 8e-3 and _rel(u, um) < 8e-3
    assert (lse - lm).abs().max() < 2e-2


@pytest.mark.parametrize("L,H", [(256, 2), (384, 2)])
def test_backward_two_rows(L, H):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("needs an H100 (sm_90)")
    proj, bias = _inputs(L, H)
    scale = D**-0.5
    HD = H * D
    o, lse, u = dh.ext_d64t().triattn_d64_forward_train(proj, bias, H, scale)
    do = torch.randn(L * L, HD, device="cuda").bfloat16()
    delta = (do.float().view(L * L, H, D) * o.float().view(L * L, H, D)).sum(-1).view(1, L, L, H).permute(0, 1, 3, 2).contiguous()
    out = {}
    for nr in (1, 2):
        dh.BWD_Q_ROWS = nr
        runs = []
        for _ in range(2):
            dall = torch.zeros(L * L, 4 * HD, device="cuda", dtype=torch.bfloat16)
            db = dh.backward(proj, bias, do, lse, delta, dall, H, D, scale)
            runs.append((dall[:, :3 * HD].clone(), db))
        assert torch.equal(runs[0][0], runs[1][0]) and torch.equal(runs[0][1], runs[1][1]), f"nr {nr}: not deterministic"
        out[nr] = runs[0]
    dh.BWD_Q_ROWS = 2
    assert torch.equal(out[1][0], out[2][0]), "dq / dk / dv must not depend on the rows a CTA takes"
    assert _rel(out[2][1], out[1][1]) < 1e-2
    _, _, _, rdq, rdk, rdv, rdb = _ref(proj, bias, do, L, H)
    for nr in (1, 2):
        assert _rel(out[nr][0][:, :HD], rdq) < 2e-2 and _rel(out[nr][0][:, HD : 2 * HD], rdk) < 2e-2 and _rel(out[nr][0][:, 2 * HD :], rdv) < 2e-2
    assert _rel(out[2][1], rdb) <= _rel(out[1][1], rdb) * 1.05 + 1e-3  # the rounding of a sum is no worse than a sum of roundings
