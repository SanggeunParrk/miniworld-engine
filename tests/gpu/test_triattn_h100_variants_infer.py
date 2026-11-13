"""H100 TriangleAttention inference of the RoseTTAFold3 / Chai-1 variants.

- Chai-1's output layouts (the ending node's untransposed update, the confidence head's second out-projection) with heads of 32
  channels take the width-generic kernels (``triattn_h100_sm90``, fp16 core): the result matches the fp32 reference as the
  standard layouts do.
- Heads of 64 channels (the ``h100_dh`` core): the 64-key tiles past each sample's last attended key are not computed
  (``h100_sm90.key_tiles(mask, L, 64)``): bitwise the same output as with every tile.
- Both deterministic (two calls, the same bits).
"""

import copy

import pytest
import torch

from miniworld_engine.kernels.triangle_attention.cuda import h100_sm90 as hs
from miniworld_engine.measure.config import randomize
from miniworld_engine.modules.triangle import TriangleAttentionChai1, TriangleAttentionChai1Confidence, TriangleAttentionRF3

pytestmark = [pytest.mark.gpu, pytest.mark.h100]


def _needs_h100() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("needs an H100 (sm_90)")


def _module(cls, d: int, h: int, kw: dict, starting: bool):
    torch.manual_seed(0)
    m = cls(d, h, starting=starting, master_weights=True, backend="cuda", **kw).cuda().eval()
    with torch.no_grad():
        randomize(m, torch.Generator(device="cuda").manual_seed(1))
    return m


def _masks(L: int, g: torch.Generator) -> list[torch.Tensor]:
    pad = torch.ones(2, L, dtype=torch.bool, device="cuda")
    pad[0, L - 100 :] = False
    pad[1, L - 8 :] = False
    holes = torch.rand(2, L, device="cuda", generator=g) > 0.2
    holes[:, 200:] = False
    return [pad, holes]


@pytest.mark.parametrize("cls", [TriangleAttentionChai1, TriangleAttentionChai1Confidence])
@pytest.mark.parametrize("starting", [True, False])
@pytest.mark.parametrize("L", [256, 384])
def test_chai_layouts_on_the_sm90_path(cls, starting: bool, L: int) -> None:
    _needs_h100()
    m = _module(cls, 64, 4, {"d_hidden": 128}, starting)
    g = torch.Generator(device="cuda").manual_seed(3)
    x = torch.randn(2, L, L, 64, device="cuda", generator=g).bfloat16()
    ref_m = copy.deepcopy(m).float().set_backend("pytorch")
    for mask in _masks(L, g):
        with torch.no_grad():
            assert m._cuda_route(x, mask)[0] == "sm90"
            out = m(x, mask).float()
            again = m(x, mask).float()
            ref = ref_m(x.float(), mask)
        assert torch.equal(out, again)  # deterministic
        upd, ref_upd = out - x.float(), ref - x.float()
        err = (upd - ref_upd).norm() / ref_upd.norm()
        assert torch.isfinite(out).all()
        assert err < 1.5e-2, err


@pytest.mark.parametrize(
    "cls,d,kw",
    [(TriangleAttentionRF3, 64, {"d_hidden": 256}), (TriangleAttentionChai1, 256, {}), (TriangleAttentionChai1Confidence, 64, {"d_hidden": 256})],
)
@pytest.mark.parametrize("starting", [True, False])
def test_dh_skip_is_bitwise(cls, d: int, kw: dict, starting: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    _needs_h100()
    L = 384
    m = _module(cls, d, 4, kw, starting)
    g = torch.Generator(device="cuda").manual_seed(5)
    x = torch.randn(2, L, L, d, device="cuda", generator=g).bfloat16()
    real = hs.key_tiles
    for mask in _masks(L, g):
        with torch.no_grad():
            assert m._cuda_route(x, mask)[0] == "dh"
            skip = m(x, mask)
            assert torch.equal(skip, m(x, mask))  # deterministic
            monkeypatch.setattr(hs, "key_tiles", lambda mk, n, tile=hs.TOK: None)
            full = m(x, mask)
            monkeypatch.setattr(hs, "key_tiles", real)
        assert torch.isfinite(skip).all()
        assert not torch.equal(skip, x)
        assert torch.equal(skip, full)


@pytest.mark.parametrize("L", [128, 384, 512])
def test_d64_wgmma_core_matches_the_mma_sync_core(L: int) -> None:
    """The wgmma core of 64-channel heads (``h100_dh.ext_d64``) against the mma.sync core on the same operands: two samples (one with a
    hole and half its keys padded), a gate bias; both cores round P to bf16, so they agree to a few bf16 ulps of u."""
    _needs_h100()
    from miniworld_engine.kernels.triangle_attention.cuda import h100_dh

    H, D, B = 4, 64, 2
    g = torch.Generator(device="cuda").manual_seed(7)
    proj = torch.randn(B * L * L, 4 * H * D, device="cuda", generator=g).bfloat16()
    mask = torch.ones(B, L, dtype=torch.bool, device="cuda")
    mask[1, L // 2 :] = False
    mask[1, 3] = False
    bias = (2 * torch.randn(B, H, L, L, device="cuda", generator=g)).bfloat16().masked_fill(~mask[:, None, None, :], float("-inf"))
    gb = torch.randn(H * D, device="cuda", generator=g)
    kt = hs.key_tiles(mask, L, 64)
    assert h100_dh.wgmma_d64(L, D)
    ref = h100_dh.ext().triattn_dh_forward(proj, bias, gb, H, D, D**-0.5, h100_dh.query_tile(L, D), kt).float()
    u = h100_dh.ext_d64().triattn_d64_forward(proj, bias, gb, H, D**-0.5, kt)
    assert torch.equal(u, h100_dh.ext_d64().triattn_d64_forward(proj, bias, gb, H, D**-0.5, kt))  # deterministic
    assert torch.equal(u, h100_dh.ext_d64().triattn_d64_forward(proj, bias, gb, H, D**-0.5))  # the skipped tiles change nothing
    d = (u.float() - ref).abs()
    assert torch.isfinite(u).all()
    assert d.max() < 0.05 and d.mean() < 5e-4, (d.max(), d.mean())
