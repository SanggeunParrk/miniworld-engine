"""TMA B4: feature splits, tails, idle CTAs and asynchronous graph replay."""

import pytest
import torch

CONFIGS = [
    (1, 64, 1, 1),
    (2, 128, 2, 2),
    (4, 256, 4, 3),
    (8, 64, 32, 5),
    (16, 1024, 16, 1),
    (32, 256, 8, 2),
    (64, 128, 4, 3),
]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("layout", ["row", "col"])
@pytest.mark.parametrize("tile", CONFIGS)
def test_tma_backward(dtype, layout, tile):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("Hopper required")
    from miniworld_engine.kernels.layernorm.cute.tma_backward import (
        prepare,
        config_rejection,
    )

    bm, bk, nw, ns = tile
    c = dict(BLOCK_M1=bm, BLOCK_K=bk, num_warps=nw, num_stages=ns)
    reason = config_rejection(
        c,
        n=137,
        itemsize=torch.empty((), dtype=dtype).element_size(),
        m_major=layout == "col",
        smem_limit=torch.cuda.get_device_properties(0).shared_memory_per_block_optin,
    )
    if reason:
        pytest.skip(reason)
    torch.manual_seed(519)
    m, n = 263, 137
    strides = (144, 1) if layout == "row" else (1, 272)
    x = torch.empty_strided((m, n), strides, dtype=dtype, device="cuda").normal_()
    dy = torch.empty_strided((m, n), strides, dtype=dtype, device="cuda").normal_()
    w = torch.randn(n, device="cuda")
    mean = x.float().mean(1)
    rs = (x.float().var(1, unbiased=False) + 1e-5).rsqrt()
    dx = torch.empty_strided((m, n), strides, dtype=dtype, device="cuda")
    # More CTAs than row tiles checks zero partials for inactive programs.
    dw = torch.empty((272, n), device="cuda")
    db = torch.empty_like(dw)
    fn = prepare(x, dy, w, mean, rs, dx, dw, db, c)
    fn()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    dx.fill_(float("nan"))
    dw.fill_(float("nan"))
    db.fill_(float("nan"))
    graph.replay()
    xh = (x.float() - mean[:, None]) * rs[:, None]
    wd = dy.float() * w
    refs = [
        (
            (wd - ((wd * xh).mean(1)[:, None] * xh + wd.mean(1)[:, None])) * rs[:, None]
        ).to(dtype),
        (dy.float() * xh).sum(0),
        dy.float().sum(0),
    ]
    for actual, ref in zip((dx, dw.sum(0), db.sum(0)), refs):
        assert torch.isfinite(actual).all()
        rel = (actual.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-8)
        assert rel < (0.004 if dtype == torch.bfloat16 else 3e-6)


def test_tma_contract_rejects_unaligned_without_copy():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("Hopper required")
    from miniworld_engine.kernels.layernorm.cute.tma_backward import input_rejection

    x = torch.randn(263, 137, device="cuda", dtype=torch.bfloat16)
    w = torch.ones(137, device="cuda")
    stats = torch.ones(263, device="cuda")
    assert "aligned" in input_rejection(x, x, w, stats, stats, x.stride())
