"""The portable kernels of the sampler glue against the H100 CUDA kernels on the same inputs: same frame, same churn scalars, same algorithm (Horn's quaternion
method by 12 squarings), the pooling bitwise equal; and the routing on H100: CUDA by default, the portable kernels only under the A/B switch."""
import pytest
import torch
from settings_helpers import forced_triton as portable_switch

from miniworld_engine.kernels.sampler_ops import interface

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0), reason="Hopper (sm_90) required"),
]
CHURN = dict(window=0, gamma=0.8, gamma_min=1.0, sigma_min=0.0, sigma_max=float("inf"), noise_scale=1.003, variance_floor=0.0, clamp=False)


def _rel(a, e):
    return float((a.double() - e.double()).abs().max() / e.double().abs().max())


@pytest.mark.parametrize(("b", "n", "real"), [(1, 3072, 3000), (5, 3072, 3072), (5, 24576, 20000), (3, 1000, 1000), (2, 7, 5), (48, 8192, 8000)])
def test_the_coordinate_ops_agree_with_cuda(b, n, real, monkeypatch):
    g = torch.Generator(device="cuda").manual_seed(0)
    coords = 10 * torch.randn(b, n, 3, device="cuda", generator=g) + 3
    mask = (torch.arange(n, device="cuda") < real).float()
    rv, tr = torch.randn(b, 2, 3, device="cuda", generator=g), torch.randn(b, 3, device="cuda", generator=g)
    noise = torch.randn_like(coords)
    s, sn = torch.tensor(2.0, device="cuda"), torch.tensor(1.5, device="cuda")
    cu, cu_sc = interface.augment_noise(coords, mask, rv, tr, noise, s, sn, **CHURN)
    noisy = coords @ torch.tensor([[0.8, -0.6, 0.0], [0.6, 0.8, 0.0], [0.0, 0.0, 1.0]], device="cuda") + 2 + 0.5 * torch.randn_like(coords)
    cu_al = interface.align_euler(noisy, coords, mask, cu_sc, sn, 1.5)
    with portable_switch():
        po, po_sc = interface.augment_noise(coords, mask, rv, tr, noise, s, sn, **CHURN)
        po_al = interface.align_euler(noisy, coords, mask, cu_sc, sn, 1.5)
    assert torch.equal(po_sc[:2], cu_sc[:2])                    # the churn scalars: the same IEEE arithmetic
    assert _rel(po, cu) < 1e-6 and _rel(po_al, cu_al) < 1e-5


@pytest.mark.parametrize(("samples", "tokens", "channels"), [(5, 128, 768), (1, 256, 128), (3, 37, 100)])
def test_the_pooling_is_bitwise_the_cuda_kernels(samples, tokens, channels):
    if channels % 4:
        pytest.skip("the CUDA kernel needs C % 4 == 0")
    g = torch.Generator(device="cuda").manual_seed(2)
    rows = tokens * 24 + 64
    x = torch.randn(samples, rows, channels, device="cuda", dtype=torch.bfloat16, generator=g)
    idx = torch.randint(0, rows, (tokens, 24), device="cuda", generator=g)
    am = torch.rand(tokens, 24, device="cuda", generator=g) > 0.3
    gm = am | (torch.rand(tokens, 24, device="cuda", generator=g) > 0.9)
    cu = interface.pool_atoms(x, idx, gm.float(), am.float())
    with portable_switch():
        po = interface.pool_atoms(x, idx, gm.float(), am.float())
    assert torch.equal(po, cu)


def test_h100_runs_cuda_by_default_and_the_portable_kernels_only_under_the_switch(monkeypatch):
    from miniworld_engine.kernels.sampler_ops import triton as portable

    calls = []
    for name in ("portable_augment_noise", "portable_align_euler", "portable_pool_atoms"):
        real = getattr(portable.main, name)
        monkeypatch.setattr(portable.main, name, lambda *a, _r=real, _n=name, **k: (calls.append(_n), _r(*a, **k))[1])
        monkeypatch.setattr(portable, name, getattr(portable.main, name))
    coords = torch.randn(2, 256, 3, device="cuda")
    mask = torch.ones(256, device="cuda")
    s, sn = torch.tensor(2.0, device="cuda"), torch.tensor(1.5, device="cuda")
    rv, tr, noise = torch.randn(2, 2, 3, device="cuda"), torch.randn(2, 3, device="cuda"), torch.randn_like(coords)
    noisy, sc = interface.augment_noise(coords, mask, rv, tr, noise, s, sn, **CHURN)
    interface.align_euler(noisy, coords, mask, sc, sn, 1.5)
    assert interface.serves(coords, mask) and not calls
    with portable_switch():
        noisy, sc = interface.augment_noise(coords, mask, rv, tr, noise, s, sn, **CHURN)
        interface.align_euler(noisy, coords, mask, sc, sn, 1.5)
    assert calls == ["portable_augment_noise", "portable_align_euler"]
