"""H100 TriangleAttention training backward, query side: the row-split kernel (two warpgroups sum their dbias partials in shared
memory, ``TB_Q8``, used from S = 512) against the one-warpgroup-per-row-block kernel (``TB_Q8=0``) and an fp64 reference:
dq bitwise the same, dbias within fp32 summation order, both runs bitwise equal (deterministic), any slab count."""

import pytest
import torch

from miniworld_engine.kernels.triangle_attention.cuda import h100_sm90 as hs

pytestmark = [pytest.mark.gpu, pytest.mark.h100]


def _ref64(q, k, v, sb, do):
    sc = 32**-0.5
    qf, kf, vf, dof = (t.double() for t in (q, k, v, do))
    p = torch.softmax((torch.einsum("bnqhd,bnkhd->bnhqk", qf, kf) + sb.double()[:, None]) * sc, -1)
    dp = torch.einsum("bnqhd,bnkhd->bnhqk", dof, vf)
    ds = p * (dp - (p * dp).sum(-1, keepdim=True))
    return torch.einsum("bnhqk,bnkhd->bnqhd", ds, kf) * sc, ds.sum(1)


@pytest.mark.parametrize("n_rows,heads,slabs", [(512, 4, 0), (512, 4, 3), (640, 2, 0)])
def test_rowsplit_matches(n_rows, heads, slabs):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("needs an H100 (sm_90)")
    torch.manual_seed(0)
    L, H = n_rows, heads
    shp = (1, 8, L, H, 32)  # 8 pair rows: one task group of the row-split kernel; S = L keys
    q, k, v, do = (torch.randn(shp, device="cuda", dtype=torch.bfloat16) for _ in range(4))
    sb = torch.randn(1, H, L, L, device="cuda", dtype=torch.bfloat16)
    o, lse, _, _ = hs._core().triattn_fwd(q, k, v, sb, 32**-0.5, True)
    new, old = hs._bwd(), hs._bwd(("TB_Q8=0",))
    delta = old.triattn_delta(do, o.view(shp))

    def run(e):
        d = torch.zeros(1, H, L, L, device="cuda")
        return e.triattn_bwd_q(q, k, v, sb, do, lse, delta, 32**-0.5, d, None, slabs).float(), d

    (dq_n, db_n), (dq_n2, db_n2), (dq_o, db_o) = run(new), run(new), run(old)
    assert torch.equal(dq_n, dq_n2) and torch.equal(db_n, db_n2), "not deterministic"
    assert torch.equal(dq_n, dq_o), "dq must not change"
    assert (db_n - db_o).pow(2).mean().sqrt() <= 1e-5 * db_o.pow(2).mean().sqrt()
    rq, rd = _ref64(q, k, v, sb, do)
    assert (db_n.double() - rd).pow(2).mean().sqrt() <= 2e-3 * rd.pow(2).mean().sqrt()
    assert (dq_n.double() - rq).pow(2).mean().sqrt() <= 1e-2 * rq.pow(2).mean().sqrt()
