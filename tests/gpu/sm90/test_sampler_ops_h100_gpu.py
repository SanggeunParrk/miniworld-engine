"""The sampler glue kernels (``kernels.sampler_ops``) against their PyTorch reference: random rigid motion + re-noising and Kabsch
alignment + Euler update, fp32."""

import pytest
import torch

from miniworld_engine.kernels.sampler_ops import interface, reference

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
        reason="Hopper (sm_90) required",
    ),
]

CHURN = {
    "af3": dict(window=0, gamma=0.8, gamma_min=1.0, sigma_min=0.0, sigma_max=float("inf"), noise_scale=1.003, variance_floor=0.0, clamp=False),
    "edm": dict(window=1, gamma=0.4, gamma_min=1.0, sigma_min=0.05, sigma_max=50.0, noise_scale=1.003, variance_floor=1e-6, clamp=True),
}


def _scene(b, n, real, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    coords = 10 * torch.randn(b, n, 3, device="cuda", generator=g) + 3
    mask = (torch.arange(n, device="cuda") < real).float()
    return g, coords, mask


@pytest.mark.parametrize("b,n,real", [(1, 128 * 24, 3000), (5, 128 * 24, 3072), (5, 1024 * 24, 20000), (3, 1000, 1000), (2, 7, 5)])
@pytest.mark.parametrize("churn", list(CHURN))
@pytest.mark.parametrize("sigma", [(160.0, 120.0), (2.0, 1.5), (0.03, 0.02)])
def test_augment_noise(b, n, real, churn, sigma):
    g, coords, mask = _scene(b, n, real)
    rv = torch.randn(b, 2, 3, device="cuda", generator=g)
    tr = torch.randn(b, 3, device="cuda", generator=g)
    noise = torch.randn_like(coords)
    s, sn = (torch.tensor(v, device="cuda") for v in sigma)
    got, sc = interface.augment_noise(coords, mask, rv, tr, noise, s, sn, **CHURN[churn])
    want, want_sc = reference.augment_noise(coords, mask, rv, tr, noise, s, sn, **CHURN[churn])
    assert torch.allclose(sc[:2], want_sc, rtol=1e-6, atol=0)
    scale = want.abs().max()
    assert (got - want).abs().max() / scale < 1e-5


@pytest.mark.parametrize("b,n,real", [(1, 128 * 24, 3000), (5, 128 * 24, 3072), (5, 1024 * 24, 20000), (3, 1000, 1000), (2, 7, 5)])
@pytest.mark.parametrize("noise", [0.0, 0.5, 20.0])
def test_align_euler(b, n, real, noise):
    g, coords, mask = _scene(b, n, real, seed=1)
    angle = torch.tensor(0.7, device="cuda")
    rot = torch.tensor([[angle.cos(), -angle.sin(), 0], [angle.sin(), angle.cos(), 0], [0, 0, 1]], device="cuda")
    noisy = coords @ rot.T + 5 + noise * torch.randn_like(coords)
    denoised = coords + 0.1 * torch.randn_like(coords)
    sc = torch.tensor([4.0, 0.3, 0.0, 0.0], device="cuda")
    sn = torch.tensor(2.5, device="cuda")
    got = interface.align_euler(noisy, denoised, mask, sc, sn, 1.5)
    want = reference.align_euler(noisy, denoised, mask, sc, sn, 1.5)
    assert (got - want).abs().max() / want.abs().max() < 1e-4


@pytest.mark.parametrize("samples,tokens,atoms,channels", [(5, 128, 24, 768), (1, 256, 24, 768), (5, 128, 24, 128)])
@pytest.mark.parametrize("mask_dtype", [torch.float32, torch.bool])
def test_pool_atoms(samples, tokens, atoms, channels, mask_dtype):
    g = torch.Generator(device="cuda").manual_seed(2)
    rows = tokens * atoms + 64
    x = torch.randn(samples, rows, channels, device="cuda", dtype=torch.bfloat16, generator=g)
    idx = torch.randint(0, rows, (tokens, atoms), device="cuda", generator=g)
    atom_mask = torch.rand(tokens, atoms, device="cuda", generator=g) > 0.3
    gather_mask = atom_mask | (torch.rand(tokens, atoms, device="cuda", generator=g) > 0.9)
    atom_mask[5] = False   # a token without atoms: a zero row (the denominator is clamped)
    got = interface.pool_atoms(x, idx, gather_mask.to(mask_dtype), atom_mask.to(mask_dtype))
    want = reference.pool_atoms(x, idx, gather_mask, atom_mask)
    assert got.dtype == torch.bfloat16 and got.shape == want.shape
    assert (got.float() - want.float()).abs().max() <= 2 ** -7 * want.float().abs().max()
