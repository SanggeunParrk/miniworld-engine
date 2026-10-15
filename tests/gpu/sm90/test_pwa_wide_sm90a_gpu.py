"""The widened H100 MSAPairWeightedAveraging path (integrations.pwa_wide): d_msa 64 / 128, d_pair 128 / 256 / 384, 8 heads x 8 / 16
value channels, ragged lengths and MSA depths, inference and training. Reference: the module's own PyTorch statements in fp32.

Needs an H100 (sm_90a); skipped elsewhere.
"""

import pytest
import torch

from miniworld_engine.integrations import pwa_train as pt
from miniworld_engine.modules.exceptions import ImplementationType
from miniworld_engine.modules.msa_pair_weighted_averaging import (
    MSAPairWeightedAveraging,
)

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU"),
    pytest.mark.skipif(
        torch.cuda.is_available() and torch.cuda.get_device_capability() != (9, 0),
        reason="the wide kernels are sm_90a",
    ),
]

DEV, DT = "cuda", torch.bfloat16
SHAPES = [
    (128, 256, 16),
    (128, 256, 8),
    (128, 384, 8),
    (64, 128, 8),
]  # (d_msa, d_pair, d_hidden): ESMFold2, Protenix-v2, OpenDDE, d_hidden 8


def cos(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return (a @ b / (a.norm() * b.norm())).item()


def build(d_msa, d_pair, d_hidden, impl, dtype, p_drop=0.0):
    torch.manual_seed(0)
    m = MSAPairWeightedAveraging(
        d_msa, d_pair, 8, d_hidden, implementation=impl, p_drop=p_drop
    ).to(DEV)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if n.startswith("ln_"):
                p.normal_(1.0 if n.endswith("weight") else 0.0, 0.2)
            else:
                p.normal_(0, 0.1)
    return m.to(dtype)


def run(module, msa, pair, mask, gz):
    msa = msa.clone().requires_grad_(True)
    pair = pair.clone().requires_grad_(True)
    module.zero_grad(set_to_none=True)
    out = module(msa, pair, mask)
    out.backward(gz)
    return (
        out.detach(),
        msa.grad.clone(),
        pair.grad.clone(),
        {n: p.grad.clone() for n, p in module.named_parameters()},
    )


@pytest.mark.parametrize(("d_msa", "d_pair", "d_hidden"), SHAPES)
@pytest.mark.parametrize(
    ("length", "depth", "masked"),
    [(128, 256, True), (100, 301, True), (128, 256, False)],
)  # ragged length / odd depth
def test_matches_fp32_reference_and_is_deterministic(
    d_msa, d_pair, d_hidden, length, depth, masked
):
    torch.manual_seed(1)
    msa = torch.randn(1, depth, length, d_msa, device=DEV).to(DT)
    pair = torch.randn(1, length, length, d_pair, device=DEV).to(DT)
    gz = (torch.randn(1, depth, length, d_msa, device=DEV) * 0.1).to(DT)
    mask = (torch.rand(1, length, device=DEV) < 0.85) if masked else None
    ref = build(d_msa, d_pair, d_hidden, ImplementationType.PYTORCH, torch.float32)
    r_out, r_dm, r_dz, r_g = run(ref, msa.float(), pair.float(), mask, gz.float())
    fused = build(d_msa, d_pair, d_hidden, ImplementationType.MINIWORLD, DT)
    pt.STATS["served"] = 0
    f_out, f_dm, f_dz, f_g = run(fused, msa, pair, mask, gz)
    assert pt.STATS["served"] == 1, pt.STATS["refused"]
    assert cos(f_out, r_out) > 0.9999
    assert cos(f_dm, r_dm) > 0.9999
    assert cos(f_dz, r_dz) > 0.9998
    for n, g in f_g.items():
        assert cos(g, r_g[n]) > 0.9998, n
    f_out2, f_dm2, f_dz2, f_g2 = run(
        fused, msa, pair, mask, gz
    )  # bitwise deterministic
    assert torch.equal(f_out, f_out2)
    assert torch.equal(f_dm, f_dm2)
    assert torch.equal(f_dz, f_dz2)
    assert all(torch.equal(f_g[n], f_g2[n]) for n in f_g)
    with torch.no_grad():  # inference shares the training forward
        assert torch.equal(fused(msa, pair, mask), f_out)


@pytest.mark.parametrize(
    ("d_msa", "d_pair", "d_hidden"), [(128, 256, 16), (128, 384, 8)]
)
def test_row_dropout_is_fused_and_keeps_the_residual(d_msa, d_pair, d_hidden):
    length, depth, p = 128, 256, 0.25
    torch.manual_seed(1)
    msa = torch.randn(1, depth, length, d_msa, device=DEV).to(DT)
    pair = torch.randn(1, length, length, d_pair, device=DEV).to(DT)
    module = build(
        d_msa, d_pair, d_hidden, ImplementationType.MINIWORLD, DT, p_drop=p
    ).train()
    with torch.no_grad():
        torch.manual_seed(5)
        out = module(msa, pair, None)
        torch.manual_seed(5)
        keep = (
            torch.rand(length, d_msa, device=DEV, dtype=DT) > p
        )  # the keep-mask the kernel path draws
        module.eval()
        full = module(msa, pair, None)
    assert 0.6 < keep.float().mean().item() < 0.9
    dropped = ~keep[None, None].expand_as(out)
    assert torch.equal(
        out[dropped], msa[dropped]
    )  # a dropped (token, channel) leaves the residual untouched
    kept = keep[None, None].expand_as(out)
    upd_full, upd_drop = (full - msa).float()[kept], (out - msa).float()[kept]
    assert cos(upd_drop, upd_full) > 0.999
    assert abs((upd_drop.norm() / upd_full.norm()).item() - 1 / (1 - p)) < 0.02


def test_refusals():
    msa = torch.zeros(1, 64, 100, 128, device=DEV, dtype=DT)
    pair = torch.zeros(1, 100, 100, 256, device=DEV, dtype=DT)
    assert pt.refusal(msa, pair, 128, 256, 8, 16) is None
    assert pt.refusal(msa, pair, 128, 256, 8, 8) is None
    assert (
        pt.refusal(msa, pair, 128, 256, 8, 32) is not None
    )  # 8 x 32 on d_msa 128 is not a registry shape
    assert pt.refusal(msa, pair, 128, 256, 4, 16) is not None
    big = torch.zeros(1, 2, 1040, 128, device=DEV, dtype=DT)
    assert pt.refusal(big, pair, 128, 256, 8, 16) is not None  # N <= 1024
