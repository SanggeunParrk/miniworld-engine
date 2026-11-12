"""The lifetime of the token DiT's hoisted pair bias (``kernels/conditioned_transition/cuda``, ``kernels._capture``), FoldForge 5I28 OpenDDE n = 16.

FoldForge captures one sampler graph per fold under ``static_weights()`` + ``static_inputs()``, never refreshes it, and drops the handle; after the
fold it drops its pair and calls ``release_dead_inputs``. Before the fix the refresh records of that capture (one per block, held globally) kept the
pair and every block's bias alive for good: 3.75 GiB of pair conditioning and 24 x 0.47 GiB of biases per fold at 3968 structural tokens, through
the next fold's trunk. Checked here on the real H100 path:

- the records go with the dropped handle, so once the caller drops the pair the biases are freed (allocated memory returns to where it was);
- a replay of that graph while the pair is held is still bitwise the eager output (nothing it reads was freed);
- a pair written in place (a new version) replaces its old biases in the cache instead of adding a second set (``lookup_inputs``'s ``lineage``),
  and the eager output on it equals a fresh pair's, bitwise.
"""

from __future__ import annotations

import gc

import pytest
import torch
from torch.multiprocessing.reductions import StorageWeakRef

pytestmark = [pytest.mark.gpu]

S, L, DC, DP = 5, 256, 384, 128


def _h100() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)


def _blocks(n: int = 4):
    from miniworld_engine.modules import TokenDiT

    torch.manual_seed(0)
    out = []
    for _ in range(n):
        b = TokenDiT("protenix", 768, DC, DP, 16, 2)
        for p in b.parameters():
            torch.nn.init.normal_(p, std=0.05) if p.dim() > 1 else torch.nn.init.normal_(p, mean=0.3, std=0.2)
        out.append(b.float().cuda().eval().requires_grad_(False).set_backend("cuda"))
    return out


def _bias_entries(pair: torch.Tensor) -> int:
    from miniworld_engine.kernels.conditioned_transition import cuda as C

    cdata = pair.untyped_storage()._cdata
    return sum(1 for e in C._PAIR_BIAS.values() if not e[1].expired() and e[1].cdata == cdata)


@pytest.mark.skipif(not _h100(), reason="H100 (sm_90) path")
def test_dropped_capture_handle_frees_the_fold_with_its_pair():
    from miniworld_engine.capture import static_inputs, static_weights
    from miniworld_engine.kernels import _capture

    blocks = _blocks()
    g = torch.Generator(device="cuda").manual_seed(1)
    x = torch.randn(S, 1, L, 768, device="cuda", generator=g).bfloat16()
    cond = torch.randn(1, 1, L, DC, device="cuda", generator=g).bfloat16().expand(S, 1, L, DC)
    mask = torch.ones(1, L, device="cuda", dtype=torch.bool)
    mask[0, -7:] = False
    held = {"pair": torch.randn(1, L, L, DP, device="cuda", generator=g).bfloat16()}

    def stack():
        h = x
        for b in blocks:
            h = b(h, cond, held["pair"], mask)
        return h

    with torch.no_grad():  # the weight packs and workspaces, made once on another pair: not part of what is measured
        warm = {"pair": torch.randn(1, L, L, DP, device="cuda", generator=g).bfloat16()}
        held, warm = warm, held
        stack()
        with static_weights(), static_inputs():
            stack()
        held, warm = warm, held
        del warm
    gc.collect()
    _capture.release_dead_inputs()
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    gone = StorageWeakRef(held["pair"].untyped_storage())
    with torch.no_grad():
        with static_weights(), static_inputs():  # the handle is not kept: FoldForge's sampler graph
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(2):
                    stack()
            torch.cuda.current_stream().wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                gout = stack()
        assert _bias_entries(held["pair"]) == len(blocks)
        graph.replay()
        eager = stack()
        torch.cuda.synchronize()
        assert torch.equal(gout, eager)
        biases = len(blocks) * 16 * L * L * 2  # [heads, L, L] bf16 per block: what the cache holds for this pair
        del graph, gout, eager
        gc.collect()
        torch.cuda.synchronize()
        assert torch.cuda.memory_allocated() - base >= biases  # the pair's biases are cached while the pair lives
        before = torch.cuda.memory_allocated()
        del held["pair"]
        gc.collect()
        _capture.release_dead_inputs()
        torch.cuda.synchronize()
        assert gone.expired()  # nothing of the engine kept the pair
        freed = before - torch.cuda.memory_allocated()
        assert freed >= biases + L * L * DP * 2, freed / 2**20  # nor its biases


@pytest.mark.skipif(not _h100(), reason="H100 (sm_90) path")
def test_pair_written_in_place_replaces_its_biases():
    blocks = _blocks(2)
    g = torch.Generator(device="cuda").manual_seed(2)
    x = torch.randn(S, 1, L, 768, device="cuda", generator=g).bfloat16()
    cond = torch.randn(1, 1, L, DC, device="cuda", generator=g).bfloat16().expand(S, 1, L, DC)
    mask = torch.ones(1, L, device="cuda", dtype=torch.bool)
    pair = torch.randn(1, L, L, DP, device="cuda", generator=g).bfloat16()
    new = torch.randn(1, L, L, DP, device="cuda", generator=g).bfloat16()

    def stack(p):
        h = x
        for b in blocks:
            h = b(h, cond, p, mask)
        return h

    with torch.no_grad():
        stack(pair)
        assert _bias_entries(pair) == len(blocks)
        pair.copy_(new)  # the next fold written into the same buffer: a new version
        out = stack(pair)
        assert _bias_entries(pair) == len(blocks)  # replaced, not a second set
        assert torch.equal(out, stack(new.clone()))
