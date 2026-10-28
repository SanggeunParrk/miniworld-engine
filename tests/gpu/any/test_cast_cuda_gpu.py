"""The one CUDA weight cast (``kernels/_cast_cuda``) is bit-identical to ``tensor.to(dtype)``: dense and strided (column-major) sources, the
four-element fp32 -> bf16 path and its scalar tail, both directions, and the aliasing / ``force`` contract its callers rely on."""

import pytest
import torch

from miniworld_engine.kernels import _cast_cuda

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]

bf, f32 = torch.bfloat16, torch.float32


def _sources():
    g = torch.Generator(device="cuda").manual_seed(0)
    dense = 3 * torch.randn(512, 384, device="cuda", generator=g)          # the vector path
    odd = 3 * torch.randn(1001, device="cuda", generator=g)                 # n % 4 != 0: the scalar loop
    column_major = (3 * torch.randn(256, 128, device="cuda", generator=g)).t()  # a strided 2-D source
    offset = (3 * torch.randn(4097, device="cuda", generator=g))[1:]        # misaligned for float4
    tiny = torch.tensor([1e-40, -0.0, float("inf"), float("nan"), 65504.0, 3.3895e38], device="cuda")
    return [dense, odd, column_major, offset, tiny]


def _same_bits(a, b):
    assert a.dtype == b.dtype
    assert a.shape == b.shape
    view = torch.int16 if a.dtype == bf else torch.int32
    assert torch.equal(a.contiguous().view(view), b.contiguous().view(view))


@pytest.mark.parametrize("target", [bf, f32])
@pytest.mark.parametrize("source", [f32, bf])
def test_cast_matches_to(source, target):
    srcs = [s.to(source) for s in _sources()]
    outs = _cast_cuda.cast(srcs, [target] * len(srcs), force=True)
    for s, o in zip(srcs, outs, strict=True):
        assert o.is_contiguous()
        _same_bits(o, s.to(target))


def test_cast_returns_a_matching_contiguous_input_as_it_is():
    a = torch.randn(64, 64, device="cuda")
    b = torch.randn(64, 64, device="cuda", dtype=bf)
    t = torch.randn(64, 32, device="cuda", dtype=bf).t()
    oa, ob, ot = _cast_cuda.cast([a, b, t], [bf, bf, bf])
    assert ob is b
    assert ot is not t
    assert oa is not a
    _same_bits(ot, t.contiguous())
    fa, fb = _cast_cuda.cast([a, b], [f32, bf], force=True)
    assert fa is not a
    assert fb is not b
    _same_bits(fa, a)


def test_cast_into_packs_row_blocks():
    srcs = _sources()[:3]
    rows = [s.reshape(-1).numel() for s in srcs]
    buf = torch.empty(sum(rows), device="cuda", dtype=bf)
    dsts, r = [], 0
    for n, s in zip(rows, srcs, strict=True):
        dsts.append(buf[r:r + n].view(s.shape) if s.is_contiguous() else buf[r:r + n].view(s.shape[0], s.shape[1]))
        r += n
    _cast_cuda.cast_into(srcs, dsts)
    for s, d in zip(srcs, dsts, strict=True):
        _same_bits(d, s.to(bf))
