"""A100: ``AtomDiTESMFold2Pair`` inference hoists its conditioning table and windowed pair bias per fold (``local_dit_esm._infer_a100_hoisted``).

The hoisted builds are ``_block``'s own statements, so every call equals the call with the hoist off (``MINIWORLD_ESM_A100_HOIST=0``) BITWISE; a new
conditioning or pair tensor, an in-place weight change and another key mask must not be served the old tables."""

import pytest
import torch

from foldjet_engine.modules import AtomDiTESMFold2Pair

pytestmark = [pytest.mark.gpu]


def _a100() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (8, 0)


def _block(dtype):
    torch.manual_seed(0)
    blk = AtomDiTESMFold2Pair(backend="cuda").cuda()
    with torch.no_grad():
        for p in blk.parameters():
            p.copy_(torch.randn_like(p) / p.shape[-1] ** 0.5 if p.ndim > 1 else 1 + torch.randn_like(p) * 0.1)
    return blk.to(dtype).eval()


def _inputs(a, n, real, dtype, seed=1):
    g = torch.Generator(device="cuda").manual_seed(seed)
    s = torch.randn(a, 1, n, 128, device="cuda", generator=g).to(dtype)
    c = torch.randn(1, 1, n, 128, device="cuda", generator=g).to(dtype).expand(a, 1, n, 128)
    z = torch.randn(1, (n + 31) // 32, 32, 192, 16, device="cuda", generator=g).to(dtype)
    mask = torch.zeros(1, n, dtype=torch.bool, device="cuda")
    mask[:, :real] = True
    angle = 3 * torch.randn(1, n, 16, device="cuda", generator=g)
    return s, c, z, mask, (angle.cos(), angle.sin())


def _run(blk, s, c, z, mask, rope, hoist, monkeypatch):
    monkeypatch.setenv("MINIWORLD_ESM_A100_HOIST", hoist)
    with torch.no_grad():
        return blk(s, c, z, mask=mask, rope=rope).clone()


@pytest.mark.skipif(not _a100(), reason="A100 (sm_80)")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize(("a", "n", "real"), [(5, 1024, 1000), (1, 384, 150), (3, 640, 193)])
def test_hoist_is_bitwise_and_reads_new_inputs(dtype, a, n, real, monkeypatch):
    blk = _block(dtype)
    s, c, z, mask, rope = _inputs(a, n, real, dtype)
    off = _run(blk, s, c, z, mask, rope, "0", monkeypatch)
    on = _run(blk, s, c, z, mask, rope, "1", monkeypatch)
    again = _run(blk, s, c, z, mask, rope, "1", monkeypatch)  # served from the hoisted tables
    assert torch.isfinite(on).all()
    assert torch.equal(off, on) and torch.equal(on, again)

    # an in-place change of the pair / the conditioning / a weight, and another mask: each read, each still bitwise the unhoisted call
    z.mul_(0.5)
    assert torch.equal(_run(blk, s, c, z, mask, rope, "1", monkeypatch), _run(blk, s, c, z, mask, rope, "0", monkeypatch))
    c2 = (c[:1] * 1.5).expand_as(c)
    assert torch.equal(_run(blk, s, c2, z, mask, rope, "1", monkeypatch), _run(blk, s, c2, z, mask, rope, "0", monkeypatch))
    with torch.no_grad():
        blk.attention.to_bias.weight.mul_(1.25)
    assert torch.equal(_run(blk, s, c2, z, mask, rope, "1", monkeypatch), _run(blk, s, c2, z, mask, rope, "0", monkeypatch))
    mask2 = mask.clone()
    mask2[:, real // 2 :] = False
    got = _run(blk, s, c2, z, mask2, rope, "1", monkeypatch)
    assert torch.equal(got, _run(blk, s, c2, z, mask2, rope, "0", monkeypatch))
    assert not torch.equal(got, _run(blk, s, c2, z, mask, rope, "1", monkeypatch))
