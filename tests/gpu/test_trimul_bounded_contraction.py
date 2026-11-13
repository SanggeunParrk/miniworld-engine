"""The bounded TriMul contraction (``trimul_inproj/cuda/_bounded.py``; H100 ``h100_sources/bounded``, A100 ``sm80/contract_bounded.cu``).

- the kernels on zero-tailed planes (as K1 writes them for a padded sample): with the live extents they write the full product bit for bit
  (their own full-K launch and the cuBLAS contraction), including a dead tile region, two samples with different extents and every config;
- the module (TriangleMultiplication, cuda backend, inference) with the bounded route forced on its cell: bitwise the module with cuBLAS,
  at every position, on padded masks, a mask with holes and no mask;
- a CUDA graph captured with one mask and replayed after another mask is copied into the same tensor (FoldForge R26 refreshes the mask per
  fold): equal to eager on the new mask (the live extents are recomputed on the device by every replay, never baked at capture).
"""

from __future__ import annotations

import pytest
import torch

from miniworld_engine.kernels.trimul_inproj.cuda import _bounded

pytestmark = [pytest.mark.gpu]


def _card() -> str:
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    cap = torch.cuda.get_device_capability()
    if cap == (9, 0):
        return "h100"
    if cap == (8, 0):
        return "a100"
    pytest.skip("needs an H100 (sm_90) or an A100 (sm_80)")
    raise AssertionError


def _configs(card: str) -> list[tuple]:
    if card == "h100":
        return [(c, s, r) for c in range(_bounded.H100_CONFIGS) for s, r in ((1, 1), (2, 2))]
    return [(c, g) for c in range(9) for g in (0, 2)]


def _zero_tails(t: torch.Tensor, live: list[int]) -> torch.Tensor:
    """[P, n, n] planes, plane l of sample l % len(live): rows / columns past that sample's live count zeroed."""
    t = t.clone()
    for l in range(t.shape[0]):
        n = live[l % len(live)]
        t[l, n:] = 0
        t[l, :, n:] = 0
    return t


@pytest.mark.parametrize("mode", [0, 1])
def test_kernel_bitwise(mode: int) -> None:
    card = _card()
    from miniworld_engine.kernels.trimul_inproj.cuda._h100_bmm import contract as cublas

    g = torch.Generator(device="cuda").manual_seed(0)
    P, n = 16, 512
    live = [512 - 64 - 7, 300]  # two samples: a ragged tail (inside a tile) and a large dead region
    a = _zero_tails(torch.randn(P, n, n, device="cuda", generator=g).bfloat16(), live)
    b = _zero_tails(torch.randn(P, n, n, device="cuda", generator=g).bfloat16(), live)
    A, B = (a, b.transpose(1, 2)) if mode == 0 else (a.transpose(1, 2), b)
    ref = torch.empty_like(a)
    cublas(A, B, ref) if card == "h100" else torch.bmm(A, B, out=ref)
    bounds = torch.tensor([[x, x, x] for x in live], dtype=torch.int32, device="cuda")
    for cfg in _configs(card):
        full = torch.full_like(a, 7.0)
        _bounded.contract(A, B, full, None, 2, cfg)
        got = torch.full_like(a, 7.0)  # every element written (the dead tiles as zeros)
        _bounded.contract(A, B, got, bounds, 2, cfg)
        assert torch.equal(got, full), cfg
        assert torch.equal(got, ref), cfg


def test_live_bounds() -> None:
    _card()
    m = torch.ones(3, 64, dtype=torch.bool, device="cuda")
    m[0, 40:] = False
    m[1, :] = False
    m[2, 10:20] = False  # holes: the last live token counts
    lb = _bounded.live_bounds(m.reshape(-1), 3)
    assert lb is not None
    assert lb.tolist() == [[40, 40, 40], [0, 0, 0], [64, 64, 64]]
    assert _bounded.live_bounds(None, 3) is None


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


def _force_cells(monkeypatch: pytest.MonkeyPatch, card: str, planes: int, n: int) -> None:
    """The bounded route on this shape (both modes), whatever ``CELLS`` holds."""
    cfg = (0, 1, 1) if card == "h100" else (2, 0)
    cells = dict(_bounded.CELLS)
    for mode in (0, 1):
        for p in (planes, planes // 2):
            cells[(card, p, n, mode)] = cfg
    monkeypatch.setattr(_bounded, "CELLS", cells)


SHAPES = [(128, 384), (256, 512), (64, 1024)]


@pytest.mark.parametrize("d,L", SHAPES)
@pytest.mark.parametrize("outgoing", [True, False])
def test_module_bitwise(d: int, L: int, outgoing: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    card = _card()
    m = _module(d, outgoing)
    g = torch.Generator(device="cuda").manual_seed(1)
    x = torch.randn(1, L, L, d, device="cuda", generator=g).bfloat16()
    pad8 = torch.ones(1, L, dtype=torch.bool, device="cuda")
    pad8[:, L - L // 8 :] = False
    pad4 = torch.ones(1, L, dtype=torch.bool, device="cuda")
    pad4[:, L - L // 4 - 5 :] = False
    holes = torch.rand(1, L, device="cuda", generator=g) > 0.2
    holes[:, L - 100 :] = False
    with torch.no_grad():
        for mask in (pad8, pad4, holes, None):
            monkeypatch.setattr(_bounded, "CELLS", {})
            ref = m(x, mask)
            _force_cells(monkeypatch, card, d, L)
            calls = []
            real = _bounded.contract
            monkeypatch.setattr(_bounded, "contract", lambda *a, calls=calls, real=real, **k: (calls.append(1), real(*a, **k))[1])
            got = m(x, mask)
            monkeypatch.setattr(_bounded, "contract", real)
            assert calls, "the bounded contraction did not run"
            assert torch.equal(got, ref)


def test_graph_replay_new_mask(monkeypatch: pytest.MonkeyPatch) -> None:
    card = _card()
    d, L = 128, 384
    _force_cells(monkeypatch, card, d, L)
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
        for live in (200, 384, 333):  # folds B, C, D: the same tensor, new contents
            mask.fill_(True)
            mask[:, live:] = False
            x.copy_(torch.randn(1, L, L, d, device="cuda", generator=g).bfloat16())
            graph.replay()
            eager = m(x, mask)
            assert torch.equal(out, eager), live
