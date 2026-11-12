"""The H100 token DiT's per-block call cache (``modules/token_dit._call_cache``) on served calls: a cached route still refuses an unsupported
call (with its reason), a replaced weight is read by the next call, and the cached path's output is the uncached one, bit for bit."""

from __future__ import annotations

import typing

import pytest
import torch

from miniworld_engine.measure import config as C
from miniworld_engine.modules.primitives import BackendUnsupported
from miniworld_engine.modules.token_dit import _CALL_CACHE, TokenDiTProtenix

pytestmark = [pytest.mark.gpu, pytest.mark.h100]


def _h100() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)


def _master(seed: int = 0) -> typing.Any:
    module = TokenDiTProtenix(d_single=768, d_cond=384, d_pair=128, n_head=16, n=2).cuda().float()
    C.randomize(module, torch.Generator(device="cuda").manual_seed(seed))
    out: typing.Any = C.cast(module, "master")
    return out.eval().set_backend("cuda")


def _inputs(m, length: int):
    g = torch.Generator(device="cuda").manual_seed(1)
    return m.make_inputs(length=length, samples=5, depth=None, dtype=torch.bfloat16, device="cuda", generator=g)


def _ctx():
    return torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16)


@pytest.mark.skipif(not _h100(), reason="H100 (sm_90) token DiT")
def test_cached_route_still_refuses_unsupported_call() -> None:
    m = _master()
    ok, bad = _inputs(m, 128), _inputs(m, 136)  # L 136: not a multiple of 128 (the H100 inference core's tiles)
    a, b = _ctx()
    with a, b:
        for _ in range(2):  # the second call is served from the cache
            assert m.unsupported(**ok) is None
            m(**ok)
        for _ in range(2):
            why = m.unsupported(**bad)
            assert isinstance(why, str) and "inference" in why
            with pytest.raises(BackendUnsupported, match="inference"):
                m(**bad)
        assert m.unsupported(**ok) is None


@pytest.mark.skipif(not _h100(), reason="H100 (sm_90) token DiT")
def test_cached_output_bitwise_and_weight_replacement_seen() -> None:
    m = _master()
    inp = _inputs(m, 128)
    a, b = _ctx()
    with a, b:
        first = m(**inp)
        assert _CALL_CACHE in m.__dict__
        again = m(**inp)  # route and leaves from the cache
        torch.testing.assert_close(again, first, rtol=0, atol=0)
        # a new Parameter for the output projection: the cache is stale, the next call reads the new weight
        w = m.attention.to_out.weight
        m.attention.to_out.weight = torch.nn.Parameter(w.detach() * 0.5, requires_grad=False)
        replaced = m(**inp)
        fresh = _master()
        fresh.attention.to_out.weight = torch.nn.Parameter(w.detach() * 0.5, requires_grad=False)
        ref = fresh(**inp)
    assert not torch.equal(replaced, first)
    torch.testing.assert_close(replaced, ref, rtol=0, atol=0)
