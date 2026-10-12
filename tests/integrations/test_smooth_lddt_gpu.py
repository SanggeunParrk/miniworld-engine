"""Smooth lDDT on B200 (``kernels/smooth_lddt``): value and coordinate gradient against the dense fp32 reference, ragged lengths,
nucleotide radii, masks, B > 1, bf16 input, determinism and CUDA-graph capture."""
from __future__ import annotations

import pytest
import torch

from miniworld_engine.kernels.smooth_lddt import refusal, smooth_lddt, smooth_lddt_reference

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0), reason="B200 (sm_100)"),
]
DEV = "cuda"


def _case(a, b, n, seed=0, frac_nuc=0.2, frac_valid=0.9, noise=1.5):
    g = torch.Generator().manual_seed(seed)
    gt = (torch.cumsum(torch.randn(b, n, 3, generator=g), dim=1) * 1.5).to(DEV)      # a chain-like structure
    nuc = (torch.rand(b, n, generator=g) < frac_nuc).to(DEV)
    mask = (torch.rand(b, n, generator=g) < frac_valid).to(DEV)
    x = (gt.unsqueeze(0) + torch.randn(a, b, n, 3, generator=g).to(DEV) * noise).contiguous()
    return x, gt, nuc, mask


def _rel(a, e):
    return float((a.double() - e.double()).norm() / e.double().norm().clamp_min(1e-30))


@pytest.mark.parametrize("a,b,n", [(3, 2, 200), (4, 1, 500), (2, 2, 1024), (48, 1, 1000)])
def test_value_and_gradient_match_the_dense_reference(a, b, n):
    x, gt, nuc, mask = _case(a, b, n)
    assert refusal(x, gt, nuc, mask) is None
    xk, xr = x.clone().requires_grad_(), x.clone().requires_grad_()
    lk, lr = smooth_lddt(xk, gt, nuc, mask), smooth_lddt_reference(xr, gt, nuc, mask)
    lk.backward(); lr.backward()
    assert abs(float(lk) - float(lr)) < 1e-6 * max(1.0, abs(float(lr)))
    assert _rel(xk.grad, xr.grad) < 5e-5


def test_nothing_resolved_and_nothing_in_radius_give_zero_gradient():
    x, gt, nuc, mask = _case(2, 1, 256)
    mask[:] = False
    xk = x.clone().requires_grad_()
    loss = smooth_lddt(xk, gt, nuc, mask)
    loss.backward()
    assert float(loss) == pytest.approx(1.0) and float(xk.grad.abs().max()) == 0.0


def test_bf16_input_gets_a_bf16_gradient_of_the_fp32_value():
    x, gt, nuc, mask = _case(3, 1, 384)
    xb = x.to(torch.bfloat16).requires_grad_()
    loss = smooth_lddt(xb, gt, nuc, mask)
    loss.backward()
    ref = smooth_lddt_reference(xb.detach().float(), gt, nuc, mask)
    assert xb.grad.dtype == torch.bfloat16 and abs(float(loss) - float(ref)) < 1e-6


def test_the_backward_is_deterministic():
    x, gt, nuc, mask = _case(8, 2, 640)
    grads = []
    for _ in range(3):
        xk = x.clone().requires_grad_()
        smooth_lddt(xk, gt, nuc, mask).backward()
        grads.append(xk.grad)
    assert all(torch.equal(grads[0], g) for g in grads[1:])


def test_a_cuda_graph_replays_value_and_gradient():
    x, gt, nuc, mask = _case(6, 1, 512)
    xk = x.clone().requires_grad_()
    for _ in range(2):                                   # warm-up: builds the cubin
        xk.grad = None
        smooth_lddt(xk, gt, nuc, mask).backward()
    xk.grad = None
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        loss = smooth_lddt(xk, gt, nuc, mask)
        loss.backward()
    graph.replay()
    torch.cuda.synchronize()
    xr = x.clone().requires_grad_()
    lr = smooth_lddt_reference(xr, gt, nuc, mask)
    lr.backward()
    assert abs(float(loss) - float(lr)) < 1e-6 and _rel(xk.grad, xr.grad) < 5e-5
