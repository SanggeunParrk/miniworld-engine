"""H100 TriangleAttention inference, fp16 core: the 128-key tiles past each batch element's last attended key (``h100_sm90.key_tiles``)
are not streamed.  Their keys are -inf in every row, so their exponentials are exact zeros: the module output is bitwise the same as
with every tile computed -- d_pair 128 / 256 (fused fronts), the width-generic path (d_pair 64, d_hidden 128), starting / ending node,
AF3's flip, two batch elements with different paddings, a mask whose last key is attended (no tile skipped) and a mask with holes."""

import pytest
import torch

from miniworld_engine.kernels.triangle_attention.cuda import h100_sm90 as hs
from miniworld_engine.modules.triangle import TriangleAttention

pytestmark = [pytest.mark.gpu, pytest.mark.h100]


def _needs_h100() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("needs an H100 (sm_90)")


def _module(d: int, h: int, kw: dict, starting: bool) -> TriangleAttention:
    torch.manual_seed(0)
    m = TriangleAttention(d, h, starting=starting, master_weights=True, backend="cuda", **kw).cuda().eval()
    with torch.no_grad():
        for p in m.parameters():
            if p.dim() == 2:
                p.normal_(0, p.shape[1] ** -0.5)
    return m


def test_key_tiles_values() -> None:
    _needs_h100()
    L = 512
    m = torch.ones(4, L, dtype=torch.bool, device="cuda")
    m[0, 448:] = False  # last attended 447 -> 4 tiles
    m[1, 384:] = False  # last attended 383 -> 3 tiles
    m[2] = False  # none attended -> 1 tile (the core flags the rows)
    m[3, 1:] = False  # key 0 only -> 1 tile
    kt = hs.key_tiles(m, L)
    assert kt is not None
    assert kt.tolist() == [4, 3, 1, 1]
    assert hs.key_tiles(None, L) is None


CASES = [
    (128, 4, {}, 384),
    (128, 4, {"model": "af3"}, 384),
    (64, 4, {"d_hidden": 128}, 384),  # the width-generic path (fp16 GEMMs from L 256 on)
    (256, 8, {}, 384),
]


@pytest.mark.parametrize("d,h,kw,L", CASES)
@pytest.mark.parametrize("starting", [True, False])
def test_skip_is_bitwise(d: int, h: int, kw: dict, L: int, starting: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    _needs_h100()
    m = _module(d, h, kw, starting)
    g = torch.Generator(device="cuda").manual_seed(3)
    x = torch.randn(2, L, L, d, device="cuda", generator=g).bfloat16()
    masks = []
    pad = torch.ones(2, L, dtype=torch.bool, device="cuda")
    pad[0, L - 200 :] = False  # 1 tile dead (keys 256.. of sample 0)
    pad[1, L - 40 :] = False  # none dead
    masks.append(pad)
    holes = torch.rand(2, L, device="cuda", generator=g) > 0.2
    holes[:, 250:] = False  # a ragged mask ending inside tile 1: tile 2 dead in both samples
    masks.append(holes)
    real = hs.key_tiles
    kt = hs.key_tiles(pad, L)
    assert kt is not None
    assert kt.tolist() == [2, 3]
    with torch.no_grad():
        for mask in masks:
            skip = m(x, mask)
            monkeypatch.setattr(hs, "key_tiles", lambda mk, n: None)
            full = m(x, mask)
            monkeypatch.setattr(hs, "key_tiles", real)
            assert torch.isfinite(skip).all()
            assert not torch.equal(skip, x)  # the update is live
            assert torch.equal(skip, full)
