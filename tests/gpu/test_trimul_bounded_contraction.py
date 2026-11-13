"""A100: the bounded TriMul contraction (``trimul_inproj/cuda/_bounded.py``, ``sm80/contract_bounded.cu``).

- the kernel on zero-tailed planes (as K1 writes them for a padded sample): with the live extents it writes the full product bit for bit
  (its own full-K launch and the cuBLAS contraction), including a dead tile region, for every config and swizzle;
- the module (TriangleMultiplication, cuda backend, inference) with the bounded route forced on its shape: bitwise the module with
  cuBLAS, at every position, on padded masks, a mask with holes and no mask;
- a CUDA graph captured with one mask and replayed after another mask is copied into the same tensor (FoldForge R26 refreshes the mask per
  fold): equal to eager on the new mask (the live extents are recomputed on the device by every replay, never baked at capture).
"""

from __future__ import annotations

import pytest
import torch

from miniworld_engine.kernels.trimul_inproj.cuda import _bounded

pytestmark = [pytest.mark.gpu]


def _needs_a100() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 0):
        pytest.skip("needs an A100 (sm_80)")


def _zero_tails(t: torch.Tensor, n: int) -> torch.Tensor:
    t = t.clone()
    t[:, n:] = 0
    t[:, :, n:] = 0
    return t


@pytest.mark.parametrize("mode", [0, 1])
@pytest.mark.parametrize("live", [512 - 64 - 7, 300, 512])
def test_kernel_bitwise(mode: int, live: int) -> None:
    _needs_a100()
    g = torch.Generator(device="cuda").manual_seed(0)
    P, n = 8, 512
    a = _zero_tails(torch.randn(P, n, n, device="cuda", generator=g).bfloat16(), live)
    b = _zero_tails(torch.randn(P, n, n, device="cuda", generator=g).bfloat16(), live)
    A, B = (a, b.transpose(1, 2)) if mode == 0 else (a.transpose(1, 2), b)
    ref = torch.bmm(A, B)
    bounds = torch.tensor([[live, live, live]], dtype=torch.int32, device="cuda")
    for cfg in range(12):
        for lg in (0, 2):
            full = torch.full_like(a, 7.0)
            _bounded.contract(a, b, full, None, (cfg, lg), mode)
            got = torch.full_like(a, 7.0)  # every element written (the dead tiles as zeros)
            _bounded.contract(a, b, got, bounds, (cfg, lg), mode)
            assert torch.equal(got, full), (cfg, lg)
            assert torch.equal(got, ref), (cfg, lg)


def test_live_bounds() -> None:
    _needs_a100()
    m = torch.ones(3, 64, dtype=torch.bool, device="cuda")
    m[0, 40:] = False
    m[1, :] = False
    m[2, 10:20] = False  # holes: the last live token counts
    assert _bounded.live_bounds(m.reshape(-1), 3).tolist() == [[40, 40, 40], [0, 0, 0], [64, 64, 64]]
    assert _bounded.live_bounds(m[0].view(torch.uint8), 1).tolist() == [[40, 40, 40]]


def _module(d: int, outgoing: bool):
    from miniworld_engine.modules.triangle import TriangleMultiplication

    torch.manual_seed(0)
    m = TriangleMultiplication(d, outgoing=outgoing, master_weights=True, backend="cuda").cuda().eval()
    with torch.no_grad():
        for p in m.parameters():
            if p.dim() == 2:
                p.normal_(0, p.shape[1] ** -0.5)
            else:
                p.normal_(1.0 if p is m.ln_pair.weight or p is m.ln_out.weight else 0.0, 0.1)
    return m


def _force(monkeypatch: pytest.MonkeyPatch, planes: int, n: int) -> None:
    """The bounded route on this shape (both modes), whatever ``CELLS`` holds."""
    monkeypatch.setattr(_bounded, "CELLS", {**_bounded.CELLS, (planes, n, 0): (2, 0), (planes, n, 1): (2, 0)})


@pytest.mark.parametrize("d,L", [(128, 384), (256, 512), (64, 1024)])
@pytest.mark.parametrize("outgoing", [True, False])
def test_module_bitwise(d: int, L: int, outgoing: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    _needs_a100()
    m = _module(d, outgoing)
    g = torch.Generator(device="cuda").manual_seed(1)
    x = torch.randn(1, L, L, d, device="cuda", generator=g).bfloat16()
    pad8 = torch.ones(1, L, dtype=torch.bool, device="cuda")
    pad8[:, L - L // 8 :] = False
    pad4 = torch.ones(1, L, dtype=torch.bool, device="cuda")
    pad4[:, L - L // 4 - 5 :] = False
    holes = torch.rand(1, L, device="cuda", generator=g) > 0.2
    holes[:, L - 100 :] = False
    real = _bounded.contract
    with torch.no_grad():
        for mask in (pad8, pad4, holes, None):
            monkeypatch.setattr(_bounded, "CELLS", {})
            ref = m(x, mask)
            _force(monkeypatch, d, L)
            calls: list[int] = []
            monkeypatch.setattr(_bounded, "contract", lambda *a, calls=calls, **k: (calls.append(1), real(*a, **k))[1])
            got = m(x, mask)
            monkeypatch.setattr(_bounded, "contract", real)
            assert calls, "the bounded contraction did not run"
            assert torch.equal(got, ref)


def test_graph_replay_new_mask(monkeypatch: pytest.MonkeyPatch) -> None:
    _needs_a100()
    d, L = 128, 384
    _force(monkeypatch, d, L)
    m = _module(d, True)
    g = torch.Generator(device="cuda").manual_seed(2)
    x = torch.randn(1, L, L, d, device="cuda", generator=g).bfloat16()
    mask = torch.ones(1, L, dtype=torch.bool, device="cuda")
    mask[:, 300:] = False  # fold A
    with torch.no_grad():
        for _ in range(2):
            m(x, mask)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = m(x, mask)
        for live in (200, 384, 333):  # folds B, C, D: the same tensors, new contents
            mask.fill_(True)
            mask[:, live:] = False
            x.copy_(torch.randn(1, L, L, d, device="cuda", generator=g).bfloat16())
            graph.replay()
            eager = m(x, mask)
            assert torch.equal(out, eager), live
