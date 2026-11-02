"""The trunk_seed-grouped call shapes -- token DiT ``single`` [A, G, L, 768] with ``pair`` [G, L, L, 128], atom block ``single`` [A, G, N, 128]
with ``pair`` [G, nwin, 32, 128, 16] -- on EVERY route of the ladder: the hand-CUDA kernels, the portable (Triton) path
(``force_portable_kernels``) and the PyTorch module. Each must run, put every group's rows on that group's pair, and hold the repo's
yardstick against an fp64 PyTorch run: error no worse than 1.5x the error of the bf16 PyTorch module."""
import pytest
import torch
from portable_helpers import randomize_half_std as _randomize
from settings_helpers import override_setting

from miniworld_engine.modules.dit import DiTBlock
from miniworld_engine.modules.exceptions import ImplementationType as IT
from miniworld_engine.modules.local_dit import LocalDiTBlock

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0), reason="Hopper (sm_90) required"
    ),
]
BF = torch.bfloat16
ROUTES = ["cuda", "portable", "pytorch"]


def _rel(a, b):
    a, b = a.detach().double(), b.detach().double()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def _token_modules():
    torch.manual_seed(11)
    eng = _randomize(DiTBlock(implementation=IT.MINIWORLD).cuda()).eval()
    pt = DiTBlock(implementation=IT.PYTORCH).cuda().eval()
    pt.load_state_dict(eng.state_dict())
    ref = DiTBlock(implementation=IT.PYTORCH).cuda().double().eval()
    ref.load_state_dict({k: v.double() for k, v in eng.state_dict().items()})
    return eng, pt, ref


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize(("a", "groups", "length", "masked"), [(5, 5, 256, False), (3, 4, 512, True)])
def test_token_dit_groups(monkeypatch, route, a, groups, length, masked):
    eng, pt, ref = _token_modules()
    if route == "portable":
        override_setting(monkeypatch, "force_portable_kernels", True)
    mod = pt if route == "pytorch" else eng
    single = torch.randn(a, groups, length, 768, device="cuda").to(BF)
    cond = torch.randn(1, groups, length, 384, device="cuda").to(BF).expand(a, -1, -1, -1)
    pair = torch.randn(groups, length, length, 128, device="cuda").to(BF)
    mask = (torch.rand(1, length, device="cuda") > 0.15).expand(groups, length) if masked else None
    with torch.no_grad(), torch.autocast("cuda", dtype=BF):
        got = mod(single, cond, pair, mask)
        base = torch.cat([pt(single[:, g : g + 1], cond[:, g : g + 1], pair[g : g + 1], None if mask is None else mask[g : g + 1])
                          for g in range(groups)], dim=1)
    with torch.no_grad():
        want = [ref(single[:, g : g + 1].double(), cond[:, g : g + 1].double(), pair[g : g + 1].double(),
                    None if mask is None else mask[g : g + 1]) for g in range(groups)]
    assert got.shape == single.shape and torch.isfinite(got.float()).all()
    for g in range(groups):
        e_got, e_base = _rel(got[:, g : g + 1], want[g]), _rel(base[:, g : g + 1], want[g])
        assert e_got <= 1.5 * e_base, (route, g, e_got, e_base)
        for other in range(groups):
            if other != g:
                assert _rel(got[:, g : g + 1], want[other]) > 20 * e_got, (route, g, other)


def _atom_modules(**options):
    def make(impl):
        torch.manual_seed(0)
        blk = LocalDiTBlock(128, 128, 16, 4, n=2, cross_attention=True, implementation=impl, **options)
        with torch.no_grad():
            for p in blk.parameters():
                p.copy_(torch.randn_like(p) / p.shape[-1] ** 0.5 if p.ndim > 1 else torch.randn_like(p) * 0.1)
        return blk

    eng = make(IT.MINIWORLD).cuda().float()
    pt = make(IT.PYTORCH).cuda().float()
    pt.load_state_dict(eng.state_dict())
    ref = make(IT.PYTORCH).cuda().double()
    ref.load_state_dict(eng.state_dict())
    return eng, pt, ref


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize(("a", "groups", "n", "real"), [(5, 5, 512, 512), (3, 4, 512, 301)])
def test_atom_block_groups(monkeypatch, route, a, groups, n, real):
    eng, pt, ref = _atom_modules(kv_norm="parallel", key_window="slide")
    if route == "portable":
        override_setting(monkeypatch, "force_portable_kernels", True)
    mod = pt if route == "pytorch" else eng
    gen = torch.Generator(device="cuda").manual_seed(3)
    s = torch.randn(a, groups, n, 128, device="cuda", generator=gen).to(BF)
    c = torch.randn(1, groups, n, 128, device="cuda", generator=gen).to(BF).expand(a, -1, -1, -1)
    z = torch.randn(groups, (n + 31) // 32, 32, 128, 16, device="cuda", generator=gen).to(BF)
    mask = torch.zeros(1, n, dtype=torch.bool, device="cuda")
    mask[:, :real] = True
    with torch.no_grad(), torch.autocast("cuda", dtype=BF):
        got = mod(s, c, z, mask.expand(groups, n))     # the modules' key mask is [B, N]: one input's mask expanded over the G pairs
        base = torch.cat([pt(s[:, g : g + 1], c[:, g : g + 1], z[g : g + 1], mask) for g in range(groups)], dim=1)
    with torch.no_grad():
        want = [ref(s[:, g : g + 1].double(), c[:, g : g + 1].double(), z[g : g + 1].double(), mask)[:, :, :real] for g in range(groups)]
    assert got.shape == s.shape and torch.isfinite(got[:, :, :real].float()).all()
    for g in range(groups):
        e_got, e_base = _rel(got[:, g : g + 1, :real], want[g]), _rel(base[:, g : g + 1, :real], want[g])
        assert e_got <= 1.5 * e_base, (route, g, e_got, e_base)
        for other in range(groups):
            if other != g:
                assert _rel(got[:, g : g + 1, :real], want[other]) > 20 * e_got, (route, g, other)
