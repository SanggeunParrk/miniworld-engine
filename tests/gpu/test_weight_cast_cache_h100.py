"""FoldForge (2026-11-13): with fp32 master weights under bf16 activations, the H100 TriMul inference (d_pair 256 path) and the generic
Transition inference cast their weights on every call (2712 copy launches per ESMFold2 n = 1 trunk; ~1200 per OpenDDE fold). The casts
are now made once per parameter version (``kernels._capture.param_forms``): a repeat call launches no cast, the output bits are those of
an uncached call, and an in-place weight update is seen by the next call.
"""

import copy

import pytest
import torch
from torch.autograd import DeviceType  # pyright: ignore[reportPrivateImportUsage]

from miniworld_engine.kernels import _capture
from miniworld_engine.measure import config as C
from miniworld_engine.modules import BidirectionalTriangleMultiplication, Transition, TriangleMultiplication

pytestmark = [pytest.mark.gpu]

H100 = torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)


def _module(kind: str):
    if kind.startswith("trimul"):
        m = TriangleMultiplication(int(kind[6:]), outgoing=True)
    elif kind == "bidir":
        m = BidirectionalTriangleMultiplication(64)
    else:
        m = Transition(int(kind[10:]), n=4)
    m = m.cuda().float().eval()
    C.randomize(m, torch.Generator(device="cuda").manual_seed(0))
    m.set_master_weights()
    return m.set_backend("cuda")


def _inputs(m, length: int):
    gen = torch.Generator(device="cuda").manual_seed(1)
    return m.make_inputs(length=length, samples=1, depth=None, dtype=torch.bfloat16, device="cuda", generator=gen)


def _casts(fn) -> list[str]:
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    names = [e.name for e in (prof.events() or []) if e.device_type == DeviceType.CUDA]
    return [n for n in names if "copy" in n.lower() or "cast" in n.lower()]


@pytest.mark.skipif(not H100, reason="H100")
@pytest.mark.parametrize("length", [128, 384])
@pytest.mark.parametrize("kind", ["trimul64", "trimul128", "trimul256", "bidir", "transition128", "transition384"])
def test_master_weight_casts_once_per_version(kind: str, length: int) -> None:
    m = _module(kind)
    inputs = _inputs(m, length)
    with torch.no_grad():
        why = m.unsupported(**inputs)
    if why is not None:
        pytest.skip(f"not served: {why}")
    with torch.no_grad():
        first = m(**inputs).clone()
        casts = _casts(lambda: m(**inputs))
        assert not casts, f"a repeat call cast the weights again: {casts}"
        second = m(**inputs)
        assert torch.equal(first, second)
        _capture.clear_weight_caches()
        assert torch.equal(m(**inputs), first), "cached casts changed the output bits"
        # an in-place update bumps the parameters' version: the next call casts the new values
        for p in m.parameters():
            p.mul_(0.5)
        updated = m(**inputs).clone()  # no cache clear: the version bump alone must miss the old entry
        assert not torch.equal(updated, first)
        fresh = copy.deepcopy(m)
        _capture.clear_weight_caches()
        assert torch.equal(updated, fresh(**inputs))
