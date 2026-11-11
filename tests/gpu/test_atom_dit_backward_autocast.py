"""R20 (FoldForge): the atom-DiT block backward called inside an autocast region (FoldForge's training step runs ``.backward()`` under
its bf16 autocast; autograd carries the caller's autocast state into the backward) must give the gradients of a backward outside it.

FoldForge's Protenix v1 pattern: 8 noisy samples [8, 1, 1024, 128], the conditioning one row set expanded over the samples (stride 0), one
pair [1, 32, 32, 128, 16], the slid key window, 610 real atoms first in the key mask, fp32 master parameters under bf16 autocast. Before the
fix the H100 weight-gradient GEMMs ran bf16 under the caller's autocast and ``finalize90`` read them as fp32 rows (out of bounds:
CUDA_ERROR_ILLEGAL_ADDRESS in FoldForge's step; garbage parameter gradients where it did not fault)."""

import typing

import pytest
import torch

from miniworld_engine.measure import config as C
from miniworld_engine.modules.atom_dit import AtomDiTProtenix

pytestmark = [pytest.mark.gpu]


def _backends() -> list[str]:
    if not torch.cuda.is_available():
        return []
    return [b for b in ("cuda", "triton") if b in AtomDiTProtenix.backends]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device")
@pytest.mark.parametrize("backend", _backends())
@pytest.mark.parametrize("key_window", ["slide", "pad"])
@pytest.mark.parametrize("shared", [True, False])
def test_backward_under_autocast_equals_backward_outside(backend: str, key_window: str, shared: bool) -> None:
    torch.manual_seed(0)
    m = AtomDiTProtenix(d_single=128, d_cond=128, d_pair=16, n_head=4, n=2, key_window=key_window).cuda().float()
    C.randomize(m, torch.Generator(device="cuda").manual_seed(0))
    m = typing.cast("AtomDiTProtenix", C.cast(m, "master").train(True))
    m.set_backend(backend)
    g = torch.Generator(device="cuda").manual_seed(1)
    n, samples, real = 1024, 8, 610
    single = torch.randn(samples, 1, n, 128, device="cuda", generator=g).bfloat16()
    cond1 = torch.randn(1, 1, n, 128, device="cuda", generator=g).bfloat16()
    cond_full = torch.randn(samples, 1, n, 128, device="cuda", generator=g).bfloat16()
    pair = torch.randn(1, n // 32, 32, 128, 16, device="cuda", generator=g).bfloat16()
    mask = (torch.arange(n, device="cuda") < real)[None]
    dy = torch.randn(samples, 1, n, 128, device="cuda", generator=g)
    why = m.unsupported(single, cond1.expand(samples, 1, n, 128) if shared else cond_full, pair, mask)
    if why is not None:
        pytest.skip(f"{backend}: {why}")

    results = []
    for autocast_backward in (False, True):
        m.zero_grad(set_to_none=True)
        x = single.clone().requires_grad_(True)
        z = pair.clone().requires_grad_(True)
        c = (cond1 if shared else cond_full).clone().requires_grad_(True)
        call_c = c.expand(samples, 1, n, 128) if shared else c
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = m(x, call_c, z, mask)
            if autocast_backward:  # FoldForge: the loss and its backward inside the autocast region
                (y.float() * dy).sum().backward()
        if not autocast_backward:
            (y.float() * dy).sum().backward()
        torch.cuda.synchronize()
        grads = {"single": x.grad, "pair": z.grad, "cond": c.grad}
        grads.update({k: p.grad for k, p in m.named_parameters()})
        results.append(grads)
    outside, inside = results
    # H100: the same kernels in the same order, bitwise. A100 (the sm_80 block): its backward is not bitwise run to run (two backwards
    # outside autocast ~5e-5 apart on the fp32 weight gradients, the same inside), so there the two agree to that level
    exact = torch.cuda.get_device_capability() != (8, 0) or backend != "cuda"
    rtol, atol = (0.0, 0.0) if exact else (1e-3, 2e-4)
    for k, a in outside.items():
        b = inside[k]
        assert (a is None) == (b is None), k
        if a is not None:
            assert torch.isfinite(b).all(), k
            torch.testing.assert_close(b, a, rtol=rtol, atol=atol, msg=lambda s, k=k: f"{k}: {s}")
