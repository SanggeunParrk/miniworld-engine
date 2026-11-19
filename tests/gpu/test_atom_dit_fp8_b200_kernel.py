"""B200 atom DiT e4m3 row kernels (pre_fwd / post_fwd ``-DF8=1``) against an fp64 reference on random rows, next to the bf16 kernels (stress: scale 1e-3 .. 40, offset)."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0), reason="B200 (sm_100a) only")


def _pre_ref(a, mod, W, bq):
    af = a.double()
    xn = (af - af.mean(1, keepdim=True)) / (af.var(1, unbiased=False, keepdim=True) + 1e-5).sqrt()
    x1 = xn * mod[:, :128].double() + mod[:, 128:256].double()
    Wd = W.double()
    return [x1 @ Wd[:128].T + bq.double(), x1 @ Wd[128:256].T, x1 @ Wd[256:384].T, torch.sigmoid(x1 @ Wd[384:].T)]


@pytest.mark.parametrize("M", [32, 1024, 5120])
@pytest.mark.parametrize("scale,offset", [(1.0, 0.0), (1e-3, 0.0), (40.0, 0.0), (1.0, 25.0)])
def test_pre_f8(M, scale, offset):
    from miniworld_engine.kernels.augmented_attention.cuda import b200_atom as R
    from miniworld_engine.kernels.augmented_attention.cuda.b200_atom import f8

    torch.manual_seed(0)
    dev = "cuda"
    a = ((torch.randn(M, 128, device=dev) * 3 + 1) * scale + offset).bfloat16()
    mod = torch.zeros(M, 768, device=dev, dtype=torch.bfloat16)
    mod[:, :128] = torch.rand(M, 128, device=dev).bfloat16()
    mod[:, 128:256] = (torch.randn(M, 128, device=dev) * 0.2).bfloat16()
    W = (torch.randn(512, 128, device=dev) / 128**0.5 * 8).bfloat16()
    bq = torch.randn(128, device=dev) * 0.1
    ref = _pre_ref(a, mod, W, bq)
    base = R.pre_fwd(a, mod, W, bq)[:4]
    got = f8.pre_fwd(a, mod, W, bq)
    for r, b, f in zip(ref, base, got):
        sc = r.pow(2).mean().sqrt()
        eb = float((b.double() - r).pow(2).mean().sqrt() / sc)
        ef = float((f.double() - r).pow(2).mean().sqrt() / sc)
        assert torch.isfinite(f).all() and ef < 0.06 and ef < 25 * max(eb, 1e-3), (eb, ef)


@pytest.mark.parametrize("M", [32, 1024, 5120])
@pytest.mark.parametrize("scale", [1e-3, 1.0, 40.0])
def test_post_f8(M, scale):
    from miniworld_engine.kernels.augmented_attention.cuda import b200_atom as R
    from miniworld_engine.kernels.augmented_attention.cuda.b200_atom import f8

    torch.manual_seed(0)
    dev = "cuda"
    a = (torch.randn(M, 128, device=dev) * 2 * scale).bfloat16()
    o = (torch.randn(M, 128, device=dev) * scale).bfloat16()
    g = torch.rand(M, 128, device=dev).bfloat16()
    mod = torch.zeros(M, 768, device=dev, dtype=torch.bfloat16)
    mod[:, 256:384] = torch.rand(M, 128, device=dev).bfloat16()
    mod[:, 384:512] = torch.rand(M, 128, device=dev).bfloat16()
    mod[:, 512:640] = (torch.randn(M, 128, device=dev) * 0.2).bfloat16()
    mod[:, 640:768] = torch.rand(M, 128, device=dev).bfloat16()
    Wo = (torch.randn(128, 128, device=dev) / 128**0.5).bfloat16()
    Wu = (torch.randn(512, 128, device=dev) / 128**0.5).bfloat16()
    Wd = (torch.randn(128, 256, device=dev) / 256**0.5).bfloat16()
    d = lambda t: t.double()  # noqa: E731
    u = (d(g) * d(o)) @ d(Wo).T
    a2 = d(a) + d(mod[:, 256:384]) * u
    x2 = (a2 - a2.mean(1, keepdim=True)) / (a2.var(1, unbiased=False, keepdim=True) + 1e-5).sqrt() * d(mod[:, 384:512]) + d(mod[:, 512:640])
    A, Bm = x2 @ d(Wu[:256]).T, x2 @ d(Wu[256:]).T
    ref = a2 + d(mod[:, 640:768]) * ((A * torch.sigmoid(A) * Bm) @ d(Wd).T)
    sc = (ref - d(a)).pow(2).mean().sqrt()
    base = R.post_fwd(a, o, g, mod, Wo, Wu, Wd)[0]
    got = f8.post_fwd(a, o, g, mod, Wo, Wu, Wd)
    eb = float((d(base) - ref).pow(2).mean().sqrt() / sc)
    ef = float((d(got) - ref).pow(2).mean().sqrt() / sc)
    assert torch.isfinite(got).all() and ef < 0.08 and ef < 12 * max(eb, 5e-3), (eb, ef)


@pytest.mark.parametrize("M", [32, 1024, 5120])
@pytest.mark.parametrize("scale", [1e-3, 1.0, 40.0])
def test_post_f8_up_gate(M, scale):
    """Boltz-2's up-gated block (``wup``): h = silu(a) b u."""
    from miniworld_engine.kernels.augmented_attention.cuda import b200_atom as R
    from miniworld_engine.kernels.augmented_attention.cuda.b200_atom import f8

    torch.manual_seed(1)
    dev = "cuda"
    a = (torch.randn(M, 128, device=dev) * 2 * scale).bfloat16()
    o = (torch.randn(M, 128, device=dev) * scale).bfloat16()
    g = torch.rand(M, 128, device=dev).bfloat16()
    mod = torch.zeros(M, 768, device=dev, dtype=torch.bfloat16)
    mod[:, 256:384] = torch.rand(M, 128, device=dev).bfloat16()
    mod[:, 384:512] = torch.rand(M, 128, device=dev).bfloat16()
    mod[:, 512:640] = (torch.randn(M, 128, device=dev) * 0.2).bfloat16()
    mod[:, 640:768] = torch.rand(M, 128, device=dev).bfloat16()
    Wo = (torch.randn(128, 128, device=dev) / 128**0.5).bfloat16()
    Wu = (torch.randn(512, 128, device=dev) / 128**0.5).bfloat16()
    Wd = (torch.randn(128, 256, device=dev) / 256**0.5).bfloat16()
    Wup = (torch.randn(256, 128, device=dev) / 128**0.5).bfloat16()
    d = lambda t: t.double()  # noqa: E731
    u = (d(g) * d(o)) @ d(Wo).T
    a2 = d(a) + d(mod[:, 256:384]) * u
    x2 = (a2 - a2.mean(1, keepdim=True)) / (a2.var(1, unbiased=False, keepdim=True) + 1e-5).sqrt() * d(mod[:, 384:512]) + d(mod[:, 512:640])
    A, Bm, U = x2 @ d(Wu[:256]).T, x2 @ d(Wu[256:]).T, x2 @ d(Wup).T
    ref = a2 + d(mod[:, 640:768]) * ((A * torch.sigmoid(A) * Bm * U) @ d(Wd).T)
    sc = (ref - d(a)).pow(2).mean().sqrt()
    base = R.post_fwd(a, o, g, mod, Wo, Wu, Wd, wup=Wup)[0]
    got = f8.post_fwd(a, o, g, mod, Wo, Wu, Wd, wup=Wup)
    eb = float((d(base) - ref).pow(2).mean().sqrt() / sc)
    ef = float((d(got) - ref).pow(2).mean().sqrt() / sc)
    assert torch.isfinite(got).all() and ef < 0.12 and ef < 15 * max(eb, 5e-3), (eb, ef)
