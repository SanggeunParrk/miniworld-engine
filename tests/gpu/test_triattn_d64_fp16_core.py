"""H100 TriangleAttention inference core of 64-channel heads on fp16 operands (``h100_sources/triattn_d64_sm90.cu``, ``h100_dh.ext_d64``):

- the core against an fp64 reference of the same fp16 operands: masked keys, a fully masked sample half, a query row without an attended key
  (u = 0), a gate bias; error well inside the bf16 output's rounding;
- the key tiles past each sample's last attended key skipped (``key_tiles(mask, L, KEY_TILE)``): bitwise the same; deterministic;
- the module route (``h100_dh.forward``: fp16 LayerNorm / bias glue, fp16 GEMMs, this core) for AF3 / RF3 / Chai-1 / Chai-1 confidence, both
  nodes, against the fp32 PyTorch module.
"""

import copy

import pytest
import torch

from miniworld_engine.kernels.triangle_attention.cuda import h100_dh
from miniworld_engine.kernels.triangle_attention.cuda import h100_sm90 as hs
from miniworld_engine.measure.config import randomize
from miniworld_engine.modules.triangle import (
    TriangleAttentionAF3,
    TriangleAttentionChai1,
    TriangleAttentionChai1Confidence,
    TriangleAttentionRF3,
)

pytestmark = [pytest.mark.gpu, pytest.mark.h100]


def _needs_h100() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("needs an H100 (sm_90)")


@pytest.mark.parametrize(("L", "H"), [(128, 4), (384, 8), (512, 4)])
def test_core_matches_fp64(L: int, H: int) -> None:
    _needs_h100()
    D, B = 64, 2
    scale = D**-0.5
    g = torch.Generator(device="cuda").manual_seed(11)
    proj = torch.randn(B * L * L, 4 * H * D, device="cuda", generator=g).half()
    mask = torch.ones(B, L, dtype=torch.bool, device="cuda")
    mask[1, L // 2 :] = False
    mask[1, 3] = False
    bias = 2 * torch.randn(B, H, L, L, device="cuda", generator=g)
    bias[0, :, 7, :] = float("-inf")  # query 7 of sample 0: no attended key
    bias = bias.masked_fill(~mask[:, None, None, :], float("-inf"))
    sbias = (bias / scale).half().contiguous()
    gb = torch.randn(H * D, device="cuda", generator=g)
    ext = h100_dh.ext_d64()
    kt = hs.key_tiles(mask, L, ext.KEY_TILE)
    u = ext.triattn_d64_forward(proj, sbias, gb, H, scale, kt)
    assert torch.equal(u, ext.triattn_d64_forward(proj, sbias, gb, H, scale, kt))  # deterministic
    assert torch.equal(u, ext.triattn_d64_forward(proj, sbias, gb, H, scale))  # the skipped tiles change nothing
    assert torch.isfinite(u).all()
    # fp64 reference on a few pair rows of each sample
    P = proj.view(B, L, L, 4, H, D).double()
    for b in range(B):
        for n in (0, L // 2 + 1, L - 1):
            q, k, v, gt = (P[b, n, :, i].permute(1, 0, 2) for i in range(4))  # [H, L, D]
            s = torch.einsum("hqd,hkd->hqk", q, k) * scale + sbias[b].double() * scale
            p = torch.softmax(s, -1).nan_to_num(0.0)  # a row without an attended key: u = 0
            o = torch.einsum("hqk,hkd->hqd", p, v)
            ref = torch.sigmoid(gt + gb.view(H, 1, D).double()) * o
            got = u.view(B, L, L, H, D)[b, n].permute(1, 0, 2).double()
            err = (got - ref).abs()
            assert err.max() < 1.5e-2 and err.pow(2).mean().sqrt() < 2e-3, (b, n, err.max().item())
    assert u.view(B, L, L, H, D)[0, :, 7].abs().max() == 0  # the row without an attended key


def _module(cls, d: int, h: int, kw: dict, starting: bool):
    torch.manual_seed(0)
    m = cls(d, h, starting=starting, master_weights=True, backend="cuda", **kw).cuda().eval()
    with torch.no_grad():
        randomize(m, torch.Generator(device="cuda").manual_seed(1))
    return m


@pytest.mark.parametrize(
    "cls,d,h,kw",
    [
        (TriangleAttentionAF3, 512, 8, {}),
        (TriangleAttentionRF3, 64, 4, {"d_hidden": 256}),
        (TriangleAttentionChai1, 256, 4, {}),
        (TriangleAttentionChai1Confidence, 256, 4, {}),
    ],
)
@pytest.mark.parametrize("starting", [True, False])
def test_module_route(cls, d: int, h: int, kw: dict, starting: bool) -> None:
    _needs_h100()
    L = 256
    m = _module(cls, d, h, kw, starting)
    g = torch.Generator(device="cuda").manual_seed(3)
    x = torch.randn(1, L, L, d, device="cuda", generator=g).bfloat16()
    mask = torch.ones(1, L, dtype=torch.bool, device="cuda")
    mask[:, L - 40 :] = False
    mask[:, 5] = False
    ref_m = copy.deepcopy(m).float().set_backend("pytorch")
    with torch.no_grad():
        assert m._cuda_route(x, mask)[0] == "dh"
        assert h100_dh.wgmma_d64(L, 64)
        out = m(x, mask)
        assert torch.equal(out, m(x, mask))  # deterministic
        with torch.autocast("cuda", torch.bfloat16):  # as the master-weight models call it: the fp16 front is outside the autocast
            assert torch.equal(out, m(x, mask))
        ref = ref_m(x.float(), mask)
    upd, ref_upd = out.float() - x.float(), ref - x.float()
    assert torch.isfinite(out).all()
    err = (upd - ref_upd).norm() / ref_upd.norm()
    assert err < 1.5e-2, err
