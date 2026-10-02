"""flash_window_fa4 vs the legacy flash_window_seqused: same output, same gradients; and no NaN from the unwritten padding rows."""
import pytest
import torch

from miniworld_engine.modules.swa_atom_attention import module as M

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or M._flash_backend(torch.device("cuda")) != "fa4", reason="FA4 on CUDA")


def _mk(n, s, seed):
    g = torch.Generator().manual_seed(seed)
    q, k, v = (torch.randn(n, s, 4, 32, generator=g).cuda().requires_grad_() for _ in range(3))
    lens = ([s, s - 17, 5, s - 1] * n)[:n]
    seqused = torch.tensor(lens, dtype=torch.int32, device="cuda")
    valid = torch.arange(s, device="cuda").view(1, s) < seqused.view(n, 1)
    cu = torch.arange(0, (n + 1) * s, s, dtype=torch.int32, device="cuda")
    return q, k, v, seqused, valid, cu


def _rel(a, e):
    return float((a.double() - e.double()).norm() / e.double().norm().clamp_min(1e-30))


@pytest.mark.parametrize("hw", [64, 10**6, -1])
@pytest.mark.parametrize(("n", "s"), [(1, 256), (3, 384), (2, 1024)])
def test_matches_the_legacy_op(n, s, hw):
    q, k, v, seqused, valid, cu = _mk(n, s, 0)
    dy = torch.randn(n, s, 4, 32, device="cuda")
    scale = 32 ** -0.5
    # junk in the allocator so unwritten rows would show
    junk = [torch.full((1 << 24,), float("nan"), device="cuda") for _ in range(4)]
    del junk
    o_new = M.flash_window_fa4(q, k, v, seqused, valid, scale, hw)
    g_new = torch.autograd.grad(o_new, (q, k, v), dy)
    o_old = M.flash_window_seqused(q, k, v, cu, seqused, s, valid, n, s, scale, hw)
    g_old = torch.autograd.grad(o_old, (q, k, v), dy)
    assert torch.isfinite(o_new).all()
    assert all(torch.isfinite(g).all() for g in g_new)
    assert _rel(o_new, o_old) < 5e-3, _rel(o_new, o_old)
    for nm, a, b in zip("qkv", g_new, g_old, strict=True):
        assert _rel(a, b) < 1e-2, (nm, _rel(a, b))
    pad = ~valid
    assert (o_new[pad] == 0).all()
    print(f"n{n} s{s} hw{hw}: out {_rel(o_new, o_old):.1e} dq {_rel(g_new[0], g_old[0]):.1e} dk {_rel(g_new[1], g_old[1]):.1e} dv {_rel(g_new[2], g_old[2]):.1e}")
