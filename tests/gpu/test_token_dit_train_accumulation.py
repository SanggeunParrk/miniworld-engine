"""Gradient accumulation of the H100 token DiT training block (``kernels/integrations/token_dit_train._native_caller_hooks``).

The parameters' ``AccumulateGrad`` nodes carry no Python hook (their ``grad += new`` is ATen's ``add_`` in the autograd engine; a pre /
post hook pair and a ``TorchDispatchMode`` per accumulation cost ~50 ms of host time per training micro step of a 24-block stack): the
accumulated gradient is the fp32 sum of the single backward passes' gradients, bitwise and deterministic. A sample-expanded input (stride 0
over the samples) keeps the hook on its ``ExpandBackward0`` node (the fixed-order sum over the samples).
"""

from __future__ import annotations

import typing

import pytest
import torch

from miniworld_engine.measure import config as C
from miniworld_engine.modules.token_dit import TokenDiTProtenix

pytestmark = [pytest.mark.gpu, pytest.mark.h100]
FLAG = "miniworld_h100_native_caller"


def _h100() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)


def _hooked(node) -> bool:
    return bool(typing.cast(dict, node.metadata).get(FLAG))


def _blocks(n: int) -> list:
    out = []
    for i in range(n):
        module = TokenDiTProtenix(d_single=768, d_cond=384, d_pair=128, n_head=16, n=2).cuda().float()
        C.randomize(module, torch.Generator(device="cuda").manual_seed(i))
        m: typing.Any = C.cast(module, "master")
        out.append(m.train().set_backend("cuda"))
    return out


def _loss(blocks, inp):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        h = inp["single"]
        rest = {k: v for k, v in inp.items() if k != "single"}
        for b in blocks:
            h = b(h, **rest)
        return h.float().square().mean()


@pytest.mark.skipif(not _h100(), reason="H100 (sm_90) token DiT")
@pytest.mark.parametrize("samples", [1, 5])
def test_accumulated_gradients_are_the_sum_of_the_passes(samples) -> None:
    blocks = _blocks(3)
    params = [p for b in blocks for p in b.parameters() if p.requires_grad]
    g = torch.Generator(device="cuda").manual_seed(1)
    inps = [
        blocks[0].make_inputs(length=128, samples=samples, depth=None, dtype=torch.bfloat16, device="cuda", generator=g) for _ in range(3)
    ]

    def single(inp):
        for p in params:
            p.grad = None
        _loss(blocks, inp).backward()
        return [typing.cast(torch.Tensor, p.grad).clone() for p in params]

    def accumulated():
        for p in params:
            p.grad = None
        for inp in inps:
            loss = _loss(blocks, inp)
            for p in params:  # no Python hook between the engine and a parameter's accumulation
                node = torch.autograd.graph.get_gradient_edge(p).node
                assert not _hooked(node), node.name()
            loss.backward()
        return [typing.cast(torch.Tensor, p.grad).clone() for p in params]

    parts = [single(inp) for inp in inps]
    acc, again = accumulated(), accumulated()
    for i, got in enumerate(acc):
        assert got.dtype == torch.float32 and bool(torch.isfinite(got).all())
        want = (parts[0][i] + parts[1][i]) + parts[2][i]  # AccumulateGrad's order
        assert torch.equal(got, want), (i, float((got - want).abs().max()))
        assert torch.equal(got, again[i]), i


@pytest.mark.skipif(not _h100(), reason="H100 (sm_90) token DiT")
def test_sample_expanded_conditioning_keeps_its_sum_hook() -> None:
    blocks = _blocks(2)
    g = torch.Generator(device="cuda").manual_seed(1)
    inp = blocks[0].make_inputs(length=128, samples=5, depth=None, dtype=torch.bfloat16, device="cuda", generator=g)
    base = inp["cond"][:1].detach().clone().requires_grad_(True)  # one conditioning for the samples

    def grad():
        base.grad = None
        cond = base.expand(5, *base.shape[1:])
        loss = _loss(blocks, {**inp, "cond": cond})
        node = torch.autograd.graph.get_gradient_edge(cond).node
        assert node.name() == "ExpandBackward0" and _hooked(node)
        loss.backward()
        return typing.cast(torch.Tensor, base.grad).clone()

    a, b = grad(), grad()
    assert bool(torch.isfinite(a.float()).all()) and float(a.float().abs().max()) > 0
    assert torch.equal(a, b)
