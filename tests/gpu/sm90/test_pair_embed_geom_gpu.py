"""The fused backbone pair embedding (kernels/pair_embed_geom) agrees with the dense PyTorch builder: values (a bin flipped by a last-bit
difference in a distance or an angle is allowed for a tiny fraction of the pairs), masks (padding, missing CA / CB), several lengths."""
import pytest
import torch

from miniworld_engine.kernels.pair_embed_geom import pair_embed_geom, pair_embed_geom_reference, refusal

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0), reason="Hopper (sm_90) required"
    ),
]


def _backbone(B, n, g):
    """A chain-like random backbone in nm: residues ~0.38 nm apart along a random walk, N / C / CB offsets around each CA."""
    steps = torch.randn(B, n, 3, device="cuda", generator=g)
    ca = (steps / steps.norm(dim=-1, keepdim=True) * 0.38).cumsum(1)
    off = torch.randn(B, n, 3, 3, device="cuda", generator=g) * 0.12
    return torch.stack([ca + off[:, :, 0], ca, ca + off[:, :, 1], ca + off[:, :, 2]], 2)


@pytest.mark.parametrize(("B", "n"), [(2, 64), (3, 100), (2, 200), (1, 257)])
def test_matches_the_dense_builder(B, n):
    g = torch.Generator(device="cuda").manual_seed(B * n)
    w = torch.randn(256, 317, device="cuda", generator=g) * 0.3
    bb = _backbone(B, n, g)
    lengths = torch.randint(n // 2, n + 1, (B,), device="cuda", generator=g)
    mask = torch.arange(n, device="cuda")[None] < lengths[:, None]
    ca = mask & (torch.rand(B, n, device="cuda", generator=g) > 0.03)
    cb = ca & (torch.rand(B, n, device="cuda", generator=g) > 0.1)
    bb[..., 3, :] = bb[..., 3, :] * cb[..., None]    # a residue without CB (glycine) has its CB at the origin; N, CA, C stay
    idx = torch.randint(1, 4, (B, n), device="cuda", generator=g).cumsum(1)
    chains = torch.where(mask, torch.randint(0, 3, (B, n), device="cuda", generator=g), torch.full((B, n), -1, device="cuda"))
    assert refusal(w, bb, ca, cb, mask, idx, chains) is None
    out = pair_embed_geom(w, bb, ca, cb, mask, idx, chains).float()
    ref = pair_embed_geom_reference(w, bb, ca, cb, mask, idx, chains)
    bad = (out - ref).abs() > 0.02 + 0.02 * ref.abs()
    frac = float(bad.any(-1).float().mean())
    if frac >= 2e-3:  # where do they differ? (by the masks of the two residues)
        pairbad = bad.any(-1)
        rows = []
        for name, m in (("ca_i", ca), ("cb_i", cb)):
            for v in (0, 1):
                sel = (m[:, :, None] == v) & mask[:, :, None] & mask[:, None, :]
                rows.append(f"{name}={v}: {float(pairbad[sel.expand_as(pairbad)].float().mean()):.3f}")
        for name, m in (("ca_j", ca), ("cb_j", cb)):
            for v in (0, 1):
                sel = (m[:, None, :] == v) & mask[:, :, None] & mask[:, None, :]
                rows.append(f"{name}={v}: {float(pairbad[sel.expand_as(pairbad)].float().mean()):.3f}")
        diff = (out - ref).abs().amax(-1)
        rows.append(f"max abs diff {float(diff.max()):.3f}, ref scale {float(ref.abs().mean()):.3f}")
    assert frac < 2e-3, f"{frac:.2e} of the pairs differ; " + "; ".join(rows)
    pm = (mask[:, :, None] & mask[:, None, :])[..., None]
    assert not out[~pm.expand_as(out)].any(), "padded pairs must be exactly zero"
