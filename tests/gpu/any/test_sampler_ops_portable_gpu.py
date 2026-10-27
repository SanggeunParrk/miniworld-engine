"""The portable kernels of the sampler glue (``kernels/sampler_ops/triton``) on any card: output against the PyTorch reference (``sampler_ops.reference``),
ragged and tiny structures, B up to 48, both churn rules, zero mask tail, bitwise determinism (reruns, and across the A/B switch), every candidate launch
configuration, CUDA-graph capture. The A/B switch puts every call on the portable kernels (H100 included); the comparison with the CUDA kernels is in
``tests/gpu/sm90/test_sampler_ops_portable_vs_cuda_gpu.py``."""
from __future__ import annotations

import pytest
import torch
from settings_helpers import forced_triton as portable_switch   # the internal A/B switch: the one place the tests name it

from miniworld_engine.kernels.sampler_ops import interface, reference

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]
CHURN = {
    "af3": dict(window=0, gamma=0.8, gamma_min=1.0, sigma_min=0.0, sigma_max=float("inf"), noise_scale=1.003, variance_floor=0.0, clamp=False),
    "edm": dict(window=1, gamma=0.4, gamma_min=1.0, sigma_min=0.05, sigma_max=50.0, noise_scale=1.003, variance_floor=1e-6, clamp=True),
}
SHAPES = [(1, 128 * 24, 3000), (5, 128 * 24, 3072), (5, 1024 * 24, 20000), (3, 1000, 1000), (2, 7, 5), (1, 1, 1), (48, 4096, 4000), (2, 1025, 700)]


def _scene(b, n, real, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    coords = 10 * torch.randn(b, n, 3, device="cuda", generator=g) + 3
    mask = (torch.arange(n, device="cuda") < real).float()
    return g, coords, mask


def _rel(a, e):
    return float((a.double() - e.double()).abs().max() / e.double().abs().max().clamp_min(1e-30))


def _draws(g, coords):
    b = coords.shape[0]
    return torch.randn(b, 2, 3, device="cuda", generator=g), torch.randn(b, 3, device="cuda", generator=g), torch.randn_like(coords)


@pytest.mark.parametrize(("b", "n", "real"), SHAPES)
@pytest.mark.parametrize("churn", list(CHURN))
@pytest.mark.parametrize("sigma", [(160.0, 120.0), (2.0, 1.5), (0.03, 0.02)])
def test_augment_noise_matches_the_reference(b, n, real, churn, sigma):
    g, coords, mask = _scene(b, n, real)
    rv, tr, noise = _draws(g, coords)
    s, sn = (torch.tensor(v, device="cuda") for v in sigma)
    with portable_switch():
        got, sc = interface.augment_noise(coords, mask, rv, tr, noise, s, sn, **CHURN[churn])
    want, want_sc = reference.augment_noise(coords, mask, rv, tr, noise, s, sn, **CHURN[churn])
    assert torch.isfinite(got).all()
    assert torch.allclose(sc[:2], want_sc, rtol=1e-6, atol=0)
    assert _rel(got, want) < 1e-5
    if real < n:
        assert (got[:, real:] - noise[:, real:] * want_sc[1]).abs().max() < 1e-4 * max(1.0, float(want_sc[1]))      # the padding is exactly the noise


@pytest.mark.parametrize(("b", "n", "real"), SHAPES)
@pytest.mark.parametrize("noise_level", [0.0, 0.5, 20.0])
def test_align_euler_matches_the_reference(b, n, real, noise_level):
    g, coords, mask = _scene(b, n, real, seed=1)
    angle = torch.tensor(0.7, device="cuda")
    rot = torch.tensor([[angle.cos(), -angle.sin(), 0], [angle.sin(), angle.cos(), 0], [0, 0, 1]], device="cuda")
    noisy = coords @ rot.T + 5 + noise_level * torch.randn_like(coords)
    denoised = coords + 0.1 * torch.randn_like(coords)
    sc = torch.tensor([4.0, 0.3, 0.0, 0.0], device="cuda")
    sn = torch.tensor(2.5, device="cuda")
    with portable_switch():
        got = interface.align_euler(noisy, denoised, mask, sc, sn, 1.5)
    want = reference.align_euler(noisy, denoised, mask, sc, sn, 1.5)
    assert _rel(got, want) < 1e-4


def test_align_euler_recovers_a_rigid_motion():
    """Noise-free: the aligned structure is `denoised` itself, so the update is the identity step and the output sits on `denoised`."""
    g, coords, mask = _scene(3, 4096, 4096, seed=3)
    rot = torch.linalg.qr(torch.randn(3, 3, 3, device="cuda", generator=g))[0]
    rot = rot * torch.linalg.det(rot).sign()[:, None, None]
    noisy = torch.einsum("bni,bij->bnj", coords, rot) + torch.tensor([3.0, -4.0, 5.0], device="cuda")
    sc, sn = torch.tensor([2.0, 0.0, 0.0, 0.0], device="cuda"), torch.tensor(1.0, device="cuda")
    with portable_switch():
        out = interface.align_euler(noisy, coords, mask, sc, sn, 1.0)
    # aligned == coords; out = aligned + step (sigma_next - sigma_hat) (aligned - coords) / sigma_hat = coords
    assert (out - coords).abs().max() < 2e-4


@pytest.mark.parametrize(("samples", "tokens", "atoms", "channels"), [(5, 128, 24, 768), (1, 256, 24, 768), (5, 128, 24, 128), (3, 37, 24, 100), (2, 9, 5, 6)])
@pytest.mark.parametrize("mask_dtype", [torch.float32, torch.bool])
def test_pool_atoms_matches_the_reference(samples, tokens, atoms, channels, mask_dtype):
    g = torch.Generator(device="cuda").manual_seed(2)
    rows = tokens * atoms + 64
    x = torch.randn(samples, rows, channels, device="cuda", dtype=torch.bfloat16, generator=g)
    idx = torch.randint(0, rows, (tokens, atoms), device="cuda", generator=g)
    atom_mask = torch.rand(tokens, atoms, device="cuda", generator=g) > 0.3
    gather_mask = atom_mask | (torch.rand(tokens, atoms, device="cuda", generator=g) > 0.9)
    atom_mask[min(5, tokens - 1)] = False      # a token without atoms: a zero row (the denominator is clamped)
    with portable_switch():
        got = interface.pool_atoms(x, idx, gather_mask.to(mask_dtype), atom_mask.to(mask_dtype))
    want = reference.pool_atoms(x, idx, gather_mask, atom_mask)
    assert got.dtype == torch.bfloat16 and got.shape == want.shape
    assert (got.float() - want.float()).abs().max() <= 2 ** -7 * want.float().abs().max()
    assert (got[:, min(5, tokens - 1)] == 0).all()


def test_the_results_are_bitwise_reproducible_and_the_switch_does_not_change_them():
    g, coords, mask = _scene(5, 6000, 5000, seed=4)
    rv, tr, noise = _draws(g, coords)
    s, sn = torch.tensor(2.0, device="cuda"), torch.tensor(1.5, device="cuda")
    noisy_in = coords + torch.randn_like(coords)
    sc = torch.tensor([4.0, 0.3, 0.0, 0.0], device="cuda")
    with portable_switch():
        first = interface.augment_noise(coords, mask, rv, tr, noise, s, sn, **CHURN["af3"])[0], interface.align_euler(noisy_in, coords, mask, sc, sn, 1.5)
        for _ in range(3):
            again = interface.augment_noise(coords, mask, rv, tr, noise, s, sn, **CHURN["af3"])[0], interface.align_euler(noisy_in, coords, mask, sc, sn, 1.5)
            assert torch.equal(first[0], again[0]) and torch.equal(first[1], again[1])


def _tuners():
    from miniworld_engine.kernels.sampler_ops.triton import main

    return {"stats": main._stats, "augment": main._augment, "cov": main._cov, "align": main._align, "pool": main._pool}


@pytest.mark.parametrize("which", ["stats", "augment", "cov", "align", "pool"])
def test_every_candidate_config_is_correct(which, monkeypatch):
    """Each config of the default set alone gives the right numbers, and the same bits twice."""
    tuner = _tuners()[which]
    configs = list(tuner.configs)
    assert len(configs) >= 3
    g, coords, mask = _scene(3, 3000, 2500, seed=5)
    rv, tr, noise = _draws(g, coords)
    s, sn = torch.tensor(2.0, device="cuda"), torch.tensor(1.5, device="cuda")
    noisy_in = coords + torch.randn_like(coords)
    sc = torch.tensor([4.0, 0.3, 0.0, 0.0], device="cuda")
    x = torch.randn(2, 400, 128, device="cuda", dtype=torch.bfloat16, generator=g)
    idx = torch.randint(0, 400, (16, 24), device="cuda", generator=g)
    am = torch.rand(16, 24, device="cuda", generator=g) > 0.3
    want_aug = reference.augment_noise(coords, mask, rv, tr, noise, s, sn, **CHURN["af3"])[0]
    want_al = reference.align_euler(noisy_in, coords, mask, sc, sn, 1.5)
    want_pool = reference.pool_atoms(x, idx, am, am)
    for config in configs:
        monkeypatch.setattr(tuner, "configs", [config])
        monkeypatch.setattr(tuner, "cache", {})
        with portable_switch():
            runs = [(interface.augment_noise(coords, mask, rv, tr, noise, s, sn, **CHURN["af3"])[0], interface.align_euler(noisy_in, coords, mask, sc, sn, 1.5),
                     interface.pool_atoms(x, idx, am, am)) for _ in range(2)]
        assert _rel(runs[0][0], want_aug) < 1e-5 and _rel(runs[0][1], want_al) < 1e-4, config
        assert (runs[0][2].float() - want_pool.float()).abs().max() <= 2 ** -7 * want_pool.float().abs().max(), config
        assert all(torch.equal(a, b) for a, b in zip(runs[0], runs[1], strict=True)), config


def test_cuda_graph_capture():
    g, coords, mask = _scene(5, 4096, 4000, seed=6)
    rv, tr, noise = _draws(g, coords)
    s, sn = torch.tensor(2.0, device="cuda"), torch.tensor(1.5, device="cuda")
    x = torch.randn(2, 400, 128, device="cuda", dtype=torch.bfloat16, generator=g)
    idx = torch.randint(0, 400, (16, 24), device="cuda", generator=g)
    am = torch.rand(16, 24, device="cuda", generator=g) > 0.3

    def step():
        noisy, sc = interface.augment_noise(coords, mask, rv, tr, noise, s, sn, **CHURN["af3"])
        return noisy, interface.align_euler(noisy, coords, mask, sc, sn, 1.5), interface.pool_atoms(x, idx, am, am)

    with portable_switch():
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):                       # eager runs and warm-up (the autotuner runs here) on the stream the capture will use
            eager = step()
            for _ in range(2):
                step()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = step()
        graph.replay()
        torch.cuda.synchronize()
    for a, e in zip(out, eager, strict=True):
        assert torch.equal(a, e)
