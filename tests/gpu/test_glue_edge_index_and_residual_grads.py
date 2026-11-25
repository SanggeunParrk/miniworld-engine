"""3.0.1 glue removal: (1) the slid edge windows' index tables (``edge.cu`` ``edge_index``) equal the torch int64 arithmetic they replaced,
every bit, for every mask / conformer / ``slide_qblock`` case; (2) ``ParallelResidual`` on ``cuda``: a leaf branch's gradient is a fresh tensor
written in the one backward pass (AccumulateGrad takes it: no full-tensor copy), a non-leaf branch or a leaf that already has a ``.grad`` keeps
the shared ``g``; the values equal the reference's."""

from __future__ import annotations

import pytest
import torch

from foldjet_engine.modules import ParallelResidual, parallel_residual

pytestmark = pytest.mark.gpu

DS = 128


def _reference(n, kmask, n0, qblock, uid):
    """The torch arithmetic ``local_dit_h100._edge_index`` / ``_edge_valid`` ran before 3.0.1 (windows, key positions, validity [5, 32, 128])."""
    dev = "cuda"
    nwin = n // 32
    n_real = kmask.sum() if kmask is not None else torch.full((), n0, device=dev, dtype=torch.int64)
    n_rule = (n_real + 31) // 32 * 32 if qblock else n_real
    last = (n_real - 1) // 32
    idx = torch.stack([last * 0, last * 0 + 1, last - 2, last - 1, last]).clamp(0, nwin - 1)
    starts = torch.minimum((32 * idx - 48).clamp(min=0), (n_rule - DS).clamp(min=0))
    keys = starts[:, None] + torch.arange(DS, device=dev)
    valid = (keys < n_real)[:, None, :]
    if uid is not None:
        rows = 32 * idx[:, None] + torch.arange(32, device=dev)
        same = uid[rows][:, :, None] == uid[keys][:, None, :]
        if kmask is not None:
            same = same | ~kmask[rows][:, :, None]
        valid = valid & same
    return idx, keys, valid.expand(5, 32, DS).to(torch.uint8).contiguous()


@pytest.mark.h100
@pytest.mark.parametrize("n", [128, 256, 1024, 2048])
@pytest.mark.parametrize("n0", [1, 31, 100, 127, 128, 129, 200, 997, 1024, 2000])
@pytest.mark.parametrize("qblock", [False, True])
@pytest.mark.parametrize("variant", ["none", "kmask", "uid_kmask", "uid_only"])
@pytest.mark.parametrize("uid_dtype", [torch.int32, torch.int64])
def test_edge_index_equals_the_torch_arithmetic(n, n0, qblock, variant, uid_dtype):
    from foldjet_engine.kernels.dit_variants import cuda

    if n0 > n:
        pytest.skip("n0 <= n")
    if variant == "none" and uid_dtype != torch.int32:
        pytest.skip("one dtype without a conformer mask")
    g = torch.Generator(device="cuda").manual_seed(n * 7 + n0)
    kmask = (torch.arange(n, device="cuda") < n0) if variant in ("kmask", "uid_kmask") else None
    uid = None
    if variant in ("uid_kmask", "uid_only"):
        uid = torch.randint(0, 3, (n,), device="cuda", generator=g).to(uid_dtype)
        uid[n0:] = -1
    like = torch.empty(1, device="cuda")
    idx, keys, valid = cuda.edge_ext().edge_index(kmask, uid, n, n0, qblock, like)
    ridx, rkeys, rvalid = _reference(n, kmask, n0, qblock, uid)
    assert torch.equal(idx, ridx) and torch.equal(keys, rkeys) and torch.equal(valid, rvalid)
    assert idx.dtype == torch.int64 and keys.dtype == torch.int64 and valid.dtype == torch.uint8


def _copy_kernels(fn) -> int:
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as p:
        fn()
        torch.cuda.synchronize()
    return sum(1 for e in p.events() or [] if e.name == "aten::copy_")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("branches", [1, 2, 3, 4])
@pytest.mark.parametrize("shape", [(1, 129, 130, 64), (3, 5, 7)])
def test_leaf_branch_gradients_are_fresh_and_equal(dtype, branches, shape):
    g = torch.Generator(device="cuda").manual_seed(3)
    x = torch.randn(shape, device="cuda", generator=g).to(dtype)
    ys = [(x.float() + 0.1 * torch.randn(x.shape, device="cuda", generator=g)).to(dtype) for _ in range(branches)]
    up = torch.randn(x.shape, device="cuda", generator=g).to(dtype)
    m = ParallelResidual(d=shape[-1], branches=branches, backend="cuda")

    def run(backend):
        xs = x.detach().requires_grad_(True)
        yg = [y.detach().requires_grad_(True) for y in ys]
        out = m(xs, *yg) if backend == "cuda" else parallel_residual(xs, *yg, backend="pytorch")
        out.backward(up)
        return xs, yg

    xs, yg = run("cuda")
    rx, ry = run("pytorch")
    assert torch.equal(xs.grad, rx.grad)  # pyright: ignore[reportArgumentType]
    for a, b in zip(yg, ry, strict=True):
        assert torch.equal(a.grad, b.grad)  # pyright: ignore[reportArgumentType]
        assert a.grad.data_ptr() != up.data_ptr()  # pyright: ignore[reportOptionalMemberAccess]: a fresh tensor, not the shared upstream
    # no AccumulateGrad copy of g: the backward holds no aten::copy_ of the leaves' gradients
    xs = x.detach().requires_grad_(True)
    yg = [y.detach().requires_grad_(True) for y in ys]
    out = m(xs, *yg)
    assert _copy_kernels(lambda: out.backward(up)) == 0


def test_non_leaf_and_accumulated_branches_keep_the_shared_gradient():
    g = torch.Generator(device="cuda").manual_seed(5)
    x = torch.randn(2, 33, 64, device="cuda", generator=g).bfloat16().requires_grad_(True)
    y0 = torch.randn(2, 33, 64, device="cuda", generator=g).bfloat16().requires_grad_(True)
    y1 = torch.randn(2, 33, 64, device="cuda", generator=g).bfloat16().requires_grad_(True)
    up = torch.randn(2, 33, 64, device="cuda", generator=g).bfloat16()
    y1.grad = torch.ones_like(y1)  # a leaf that already has a .grad: accumulates
    m = ParallelResidual(d=64, branches=3, backend="cuda")
    y2 = (y0 * 1).bfloat16()  # a non-leaf branch
    m(x, y0, y1, y2).backward(up)
    ref_y1 = torch.ones_like(y1) + up
    assert torch.equal(y1.grad, ref_y1)  # pyright: ignore[reportArgumentType]
    assert torch.equal(y0.grad, up + up)  # pyright: ignore[reportArgumentType]  # its own branch and the one through y2
