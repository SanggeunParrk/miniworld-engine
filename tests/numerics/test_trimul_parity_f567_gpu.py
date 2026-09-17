"""SM90 F567 parity: independent reductions, grouped tails and stage recycling."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("shape", [(513, 256, 128, 128, 17), (135, 80, 152, 96, 13)])
@pytest.mark.parametrize("layout", [0, 1])
@pytest.mark.parametrize(
    "tiles",
    [
        (64, 32, 16, 4, 4, 2),
        (64, 64, 32, 2, 4, 3),
        (64, 128, 64, 1, 4, 4),
        (128, 32, 32, 8, 8, 2),
        (128, 64, 64, 2, 4, 3),
        (64, 64, 64, 1, 8, 2),
    ],
)
def test_f567_sm90_matches_triton(shape, layout, tiles):
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 required")
    from miniworld_engine.kernels.trimul_inproj.cute.parity_f567 import output_f567_impl
    from miniworld_engine.kernels.trimul_inproj.triton.output_fused import (
        _output_f567_kernel,
    )

    m, kp, kg, n, L = shape
    torch.manual_seed(123)

    def rand(*dims):
        return torch.randn(*dims, device="cuda", dtype=torch.bfloat16) * 0.2

    norm, x, wp, wg = rand(m, kp), rand(m, kg), rand(n, kp), rand(n, kg).t()
    if layout:
        wp = wp.t().contiguous().t()
        wg = wg.contiguous()
    residual = rand(m, n)
    ds = (torch.rand(L, n, device="cuda") > 0.2).bfloat16() * 1.25
    c = dict(
        zip(
            ("BLOCK_M1", "BLOCK_N", "BLOCK_K", "GROUP_M", "num_warps", "num_stages"),
            tiles,
            strict=True,
        )
    )
    got = output_f567_impl(norm, x, wp, wg, residual, ds, L, c)
    y, p, g = (torch.empty_like(residual) for _ in range(3))
    grid = (
        (m + c["BLOCK_M1"] - 1)
        // c["BLOCK_M1"]
        * ((n + c["BLOCK_N"] - 1) // c["BLOCK_N"]),
    )
    _output_f567_kernel.fn[grid](
        norm,
        x,
        wp,
        wg,
        p,
        g,
        y,
        residual,
        ds,
        m,
        L,
        kp,
        kg,
        n,
        *wp.stride(),
        *wg.stride(),
        shape_key=0,
        **c,
    )
    torch.cuda.synchronize()
    for actual, expected in zip(got, (y, p, g), strict=True):
        relative = (
            actual.float() - expected.float()
        ).norm() / expected.float().norm().clamp_min(1.0e-20)
        assert relative.item() < 1.0e-4


def test_f567_native_selector_replays_prepared_launch(monkeypatch):
    """Native tuning can replay candidates and return a winner with live outputs."""
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 required")
    from miniworld_engine.autotune import trimul_sm90_config
    from miniworld_engine.kernels.trimul_inproj.cute.parity_f567 import output_f567_impl

    torch.manual_seed(456)
    norm = torch.randn(129, 256, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(129, 128, device="cuda", dtype=torch.bfloat16)
    wp = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    wg = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    residual = torch.randn(129, 128, device="cuda", dtype=torch.bfloat16)
    dropscale = torch.ones(17, 128, device="cuda", dtype=torch.bfloat16)
    config = {
        "BLOCK_M1": 64,
        "BLOCK_N": 64,
        "BLOCK_K": 32,
        "GROUP_M": 2,
        "num_warps": 4,
        "num_stages": 2,
    }
    replayed = []

    def select(op, tensors, *, extra, feasibility, run):
        assert op == "trimul_output_f567_train_sm90_cute"
        for stages in (2, 3):
            candidate = {**config, "num_stages": stages}
            assert feasibility(candidate) is None
            run(candidate)
            replayed.append(candidate)
        return config

    monkeypatch.setattr(trimul_sm90_config, "resolve", select)
    actual = output_f567_impl(norm, x, wp, wg, residual, dropscale, 17)
    expected = output_f567_impl(norm, x, wp, wg, residual, dropscale, 17, config)
    torch.cuda.synchronize()
    assert len(replayed) == 2
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("reduction", [(16, 8), (256, 128)])
def test_f567_full_range_sigmoid_matches_triton(reduction):
    """Preserve tiny BF16 gate values and exponential overflow behavior."""
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 required")
    from miniworld_engine.kernels.trimul_inproj.cute.parity_f567 import output_f567_impl
    from miniworld_engine.kernels.trimul_inproj.triton.output_fused import (
        _output_f567_kernel,
    )

    kp, kg = reduction
    m, n, length = 129, 128, 128
    kw = {"device": "cuda", "dtype": torch.bfloat16}
    norm = torch.ones(m, kp, **kw)
    x = torch.zeros(m, kg, **kw)
    x[:, 0] = 1
    wp = torch.ones(n, kp, **kw)
    wg = torch.zeros(kg, n, **kw)
    wg[0] = torch.linspace(-100, 100, n, device="cuda").bfloat16()
    residual = torch.zeros(m, n, **kw)
    ds = torch.ones(length, n, **kw)
    config = {
        "BLOCK_M1": 64,
        "BLOCK_N": 64,
        "BLOCK_K": 64,
        "GROUP_M": 4,
        "num_warps": 4,
        "num_stages": 2,
    }
    actual = output_f567_impl(norm, x, wp, wg, residual, ds, length, config)
    y, proj, gate = (torch.empty_like(residual) for _ in range(3))
    _output_f567_kernel.fn[(6,)](
        norm,
        x,
        wp,
        wg,
        proj,
        gate,
        y,
        residual,
        ds,
        m,
        length,
        kp,
        kg,
        n,
        *wp.stride(),
        *wg.stride(),
        shape_key=0,
        **config,
    )
    torch.cuda.synchronize()
    for a, b in zip(actual, (y, proj, gate), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert torch.any((gate > 0) & (gate < torch.finfo(torch.bfloat16).tiny))
