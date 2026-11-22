"""The RTX 5090 bias-only TriangleAttention path: the P v contraction and the out-projection with its residual on CUTLASS tiles
(``bidir_triattn_h100._pv`` / ``_out_proj``, from L 256) against the cuBLAS route they replace (``SM120_GEMM = False``), both measured against the fp32 PyTorch
reference (the accuracy standard: no worse than the route it replaces, output and every gradient), for the single-direction module (both directions) and the
bidirectional one, masked and not."""

from __future__ import annotations

import pytest
import torch

pytestmark = [pytest.mark.gpu]
pytestmark.append(
    pytest.mark.skipif(
        not (torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0)), reason="RTX 5090 (sm_120a)"
    )
)


def _module(kind, starting, backend, seed=0):
    from miniworld_engine.modules.triangle import BidirectionalTriangleAttention, TriangleAttention

    torch.manual_seed(seed)
    if kind == "bi":
        m = BidirectionalTriangleAttention(128, 4, backend=backend, master_weights=True)
    else:
        m = TriangleAttention(128, 4, starting=starting, use_self_attention=False, master_weights=True, backend=backend)
    for p in m.parameters():
        if p.dim() > 1:
            torch.nn.init.normal_(p, std=0.03)
    return m.cuda()


def _run(m, x, mask, train):
    xx = x.clone().requires_grad_(train)
    for p in m.parameters():
        p.grad = None
    with torch.enable_grad() if train else torch.no_grad():
        y = m(xx, mask) if mask is not None else m(xx)
        grads = []
        if train:
            y.backward(torch.linspace(-1, 1, y.numel(), device=y.device).view_as(y).to(y.dtype))
            grads = [xx.grad] + [p.grad for p in m.parameters() if p.grad is not None]
    return [t.detach().float() for t in (y, *grads)]


def _rms(a, b):
    return (a - b).pow(2).mean().sqrt().item()


@pytest.mark.parametrize("kind,starting", [("single", True), ("single", False), ("bi", True)])
@pytest.mark.parametrize("length", [256, 384])
@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("train", [False, True])
def test_cutlass_contractions_as_accurate_as_cublas(kind, starting, length, masked, train):
    from miniworld_engine.kernels.integrations import bidir_triattn_h100 as k

    torch.manual_seed(1)
    x = torch.randn(1, length, length, 128, device="cuda", dtype=torch.bfloat16)
    mask = None
    if masked:
        mask = torch.ones(1, length, dtype=torch.bool, device="cuda")
        mask[:, length - 13 :] = False
    m = _module(kind, starting, "cuda")
    saved = k.SM120_GEMM
    try:
        k.SM120_GEMM = True
        new = _run(m, x, mask, train)
        k.SM120_GEMM = False
        old = _run(m, x, mask, train)
    finally:
        k.SM120_GEMM = saved
    # the reference: the fp32 PyTorch module on the same (bf16-valued) input and weights
    ref_m = _module(kind, starting, "pytorch")
    ref_m.load_state_dict(m.state_dict())
    ref = _run(ref_m, x.float(), mask, train)
    assert len(new) == len(old) == len(ref)
    for i, (a, b, r) in enumerate(zip(new, old, ref, strict=True)):
        scale = r.pow(2).mean().sqrt().item() + 1e-12
        e_new, e_old = _rms(a, r), _rms(b, r)
        assert e_new <= 1.5 * e_old + 2e-3 * scale, f"tensor {i}: rms error vs fp32 {e_new / scale:.3e} (cutlass) against {e_old / scale:.3e} (cublas)"


def test_route_is_taken_from_l256_only():
    from miniworld_engine.kernels.integrations import bidir_triattn_h100 as k

    assert k._sm120_gemm()
    c = torch.randn(4, 32 * 128, 128, device="cuda", dtype=torch.bfloat16)
    p = torch.randn(4, 128, 128, device="cuda", dtype=torch.bfloat16)
    out = torch.empty_like(c)
    k._pv(c, p, out)  # L 128: cuBLAS (the bmm)
    assert torch.equal(out, torch.bmm(c, p.transpose(1, 2)))
