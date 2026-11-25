"""H100 bidirectional wide TriMul training (``h100_wide_training``): every frozen cuBLASLt entry of ``lt_selection.json`` is well formed and the
dwg / dwp weight-gradient GEMMs of the served cells find their frozen algorithm in this runtime (no "not offered by this cuBLASLt" warning), with
every gradient bit-equal to the first heuristic's (the 3.0.0 behaviour on this cuBLASLt: only bit-equal candidates are frozen)."""

import json
import warnings

import pytest
import torch

from foldjet_engine.kernels.trimul_inproj.cuda import h100_wide_training as W
from foldjet_engine.modules.triangle import BidirectionalTriangleMultiplication

pytestmark = [pytest.mark.gpu, pytest.mark.h100]


def test_table_entries_are_well_formed() -> None:
    table = json.loads((W.R / "lt_selection.json").read_text())
    for cell, entries in table.items():
        if not cell[0].isdigit():
            continue
        for name, entry in entries.items():
            candidates = [entry] if isinstance(entry[0], int) else entry
            assert candidates, (cell, name)
            for index, words in candidates:
                assert isinstance(index, int) and len(words) == 8 and all(isinstance(w, int) for w in words), (cell, name)


def _run(m, x, mask, dy):
    for p in (x, *m.parameters()):
        p.grad = None
    torch.manual_seed(3)
    m(x, mask).backward(dy)
    return [x.grad.clone(), *[p.grad.clone() for p in m.parameters()]]


@pytest.mark.parametrize("cell", [(256, 384), (256, 768), (384, 768), (512, 384)])
def test_frozen_algorithms_are_offered_and_accurate(cell: tuple[int, int], monkeypatch: pytest.MonkeyPatch) -> None:
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("needs an H100 (sm_90)")
    D, L = cell
    torch.manual_seed(0)
    m = BidirectionalTriangleMultiplication(D, d_hidden=D, master_weights=True, backend="cuda").cuda().train()
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(0, D**-0.5) if p.dim() > 1 else p.normal_(1.0, 0.1)
    g = torch.Generator(device="cuda").manual_seed(1)
    x = torch.randn(1, L, L, D, device="cuda", generator=g).bfloat16().requires_grad_()
    mask = torch.ones(1, L, dtype=torch.bool, device="cuda")
    dy = torch.randn(1, L, L, D, device="cuda", generator=g).bfloat16()
    W._WARNED.clear()
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        served = _run(m, x, mask, dy)
    again = _run(m, x, mask, dy)
    assert all(torch.equal(a, b) for a, b in zip(served, again, strict=True)), "two runs must be bitwise equal"
    monkeypatch.setattr(W, "LT_SELECTION", "heuristic")
    first = _run(m, x, mask, dy)
    assert all(torch.equal(a, b) for a, b in zip(served, first, strict=True)), "frozen algorithms are bit-equal to the first heuristic"
