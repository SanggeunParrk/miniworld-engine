"""``kernels._gemm.mm``: long bf16 / fp16 reductions keep every row (cuBLAS 12.9's split-K tail drop on B200, see the module).

At reduction lengths where the plain cuBLAS call dropped the last K % 64 rows on B200 (2026-10-29: 2 049, 9 409, 16 641, 16 643,
258 064 = 508^2, 603 667; 0.4-2.7 % relative error) the helper is within 1e-4 of fp64 of the same operands with an fp32 result
(aligned calls: up to 4.5e-5 at K 700 000), and within the bf16 rounding of the fp64 result otherwise; a reduction it does not split
(short, a multiple of 64, fp32) is the plain call, bit for bit; the split is capturable in a CUDA graph.
"""

import pytest
import torch

from miniworld_engine.kernels import _gemm

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]

BAD_K = (2049, 9409, 16641, 16643, 258064, 603667)


def _operands(k, m, n, dtype, layout, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    y = torch.randn(k, n, device="cuda", dtype=dtype, generator=g)
    if layout == "TN":  # a weight gradient dY^T X: both operands row-major over the reduction
        a = torch.randn(k, m, device="cuda", dtype=dtype, generator=g).t()
    else:
        a = torch.randn(m, k, device="cuda", dtype=dtype, generator=g)
    return a, y


def _rel(x, ref):
    return float((x.double() - ref).norm() / ref.norm())


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("layout", ["TN", "NN"])
@pytest.mark.parametrize(("m", "n"), [(128, 128), (256, 256), (8, 384)])
@pytest.mark.parametrize("k", BAD_K)
def test_fp32_result_keeps_every_row(k, m, n, layout, dtype):
    a, b = _operands(k, m, n, dtype, layout)
    ref = a.double() @ b.double()
    y = _gemm.mm(a, b, out_dtype=torch.float32)
    assert y.dtype == torch.float32 and y.shape == (m, n)
    assert _rel(y, ref) < 1e-4


@pytest.mark.parametrize("k", (16641, 258064))
def test_low_precision_result_and_out(k):
    a, b = _operands(k, 256, 256, torch.bfloat16, "TN", seed=1)
    ref = a.double() @ b.double()
    floor = _rel(ref.to(torch.bfloat16), ref)  # the bf16 rounding of the exact result
    y = _gemm.mm(a, b)
    assert y.dtype == torch.bfloat16
    assert _rel(y, ref) < 1.05 * floor
    out = torch.empty(256, 256, device="cuda", dtype=torch.float32)
    r = _gemm.mm(a, b, out=out)
    assert r.data_ptr() == out.data_ptr()
    assert _rel(out, ref) < 1e-4


@pytest.mark.parametrize(("k", "dtype"), [(16640, torch.bfloat16), (1000, torch.bfloat16), (16641, torch.float32)])
def test_unsplit_is_the_plain_call(k, dtype):
    a, b = _operands(k, 128, 128, dtype, "TN", seed=2)
    assert not _gemm.ragged(k, dtype)
    if dtype == torch.float32:
        assert torch.equal(_gemm.mm(a, b), torch.mm(a, b))
    else:
        assert torch.equal(_gemm.mm(a, b, out_dtype=torch.float32), torch.mm(a, b, out_dtype=torch.float32))


def test_cuda_graph_capture():
    a, b = _operands(16641, 128, 128, torch.bfloat16, "TN", seed=3)
    ref = a.double() @ b.double()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        _gemm.mm(a, b, out_dtype=torch.float32)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        y = _gemm.mm(a, b, out_dtype=torch.float32)
    g.replay()
    torch.cuda.synchronize()
    assert _rel(y, ref) < 1e-4
