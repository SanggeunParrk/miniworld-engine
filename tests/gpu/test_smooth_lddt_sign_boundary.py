"""SmoothLddtLoss gradient at the kink of |d_pred - d_gt|: the sign must be the exact one (fp64 reference), not the rounded fp32 one.

Cause of the old fp32-train grad:x_pred RED at N 5120 / 7168 (error ~1e-3 against fp64): a pair with |d_pred - d_gt| below the fp32 error of the
distances (~1e-5 A) took the wrong sign. The input here puts many pairs there on purpose: a block of atoms of x_pred is a rigid motion of the
ground truth (d_pred == d_gt up to the rounding of the coordinates), the rest is noisy."""

import pytest
import torch

from miniworld_engine.modules.smooth_lddt import SmoothLddtLoss

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

N, BLOCK = 1024, 384


def _inputs(device):
    g = torch.Generator(device=device).manual_seed(0)
    gt = torch.randn((1, N, 3), device=device, generator=g) * 10.0
    noisy = gt + torch.randn((1, 1, N, 3), device=device, generator=g)[0]
    # rigid motion of the first BLOCK atoms: rotation about z by 0.7 rad and a shift (fp32 rounding leaves |df| ~ 1e-6 A)
    c, s = 0.7648421872844885, 0.644217687237691
    rot = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], device=device)
    x = noisy.clone()
    x[:, :BLOCK] = gt[:, :BLOCK] @ rot.T + torch.tensor([3.0, -2.0, 1.0], device=device)
    is_nuc = torch.rand((1, N), device=device, generator=g) < 0.125
    mask = torch.rand((1, N), device=device, generator=g) < 0.95
    return x[None].contiguous(), gt, is_nuc, mask


def _grad(backend, x, gt, is_nuc, mask, dtype):
    xx = x.to(dtype).clone().requires_grad_(True)
    out = SmoothLddtLoss(backend=backend)(xx, gt.to(dtype), is_nuc, mask)
    out.backward()
    return xx.grad.double()


@pytest.mark.parametrize("backend", ["cuda", "triton"])
def test_grad_sign_at_the_kink_matches_fp64(backend):
    x, gt, is_nuc, mask = _inputs("cuda")
    mod = SmoothLddtLoss(backend=backend)
    if mod.unsupported(x, gt, is_nuc, mask) is not None:
        pytest.skip(f"{backend} unsupported here")
    gold = _grad("pytorch", x, gt, is_nuc, mask, torch.float64)
    err = ((_grad(backend, x, gt, is_nuc, mask, torch.float32) - gold).abs().max() / gold.abs().max()).item()
    assert err < 1e-5, f"{backend}: grad:x_pred error against fp64 {err:.3e} (wrong sign at the kink?)"
