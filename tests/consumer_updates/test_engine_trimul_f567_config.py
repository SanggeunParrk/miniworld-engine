"""CSV, shape-key and tiled F567 correctness contracts for the staged engine."""

import itertools

import pytest
import torch
import triton

from miniworld_engine.autotune import configs
from miniworld_engine.autotune.shape_key import token_key
from miniworld_engine.kernels.trimul_inproj.triton.output_fused import (
    _output_f567_kernel,
    output_f567_train,
    prune_output_configs,
)

OP = "trimul_output_f567_train_triton"


def test_csv_owns_every_tiling_axis():
    grid = configs.configs_for(OP)
    assert len(grid) == 3072
    axes = {"BLOCK_M1", "BLOCK_N", "BLOCK_K", "GROUP_M"}
    assert all(set(c.kwargs) == axes for c in grid)
    assert {c.kwargs["BLOCK_K"] for c in grid} == {16, 32, 64, 128}
    assert {c.num_warps for c in grid} == {1, 2, 4, 8}
    assert {c.num_stages for c in grid} == {2, 3, 4}
    assert len(
        {(tuple(sorted(c.kwargs.items())), c.num_warps, c.num_stages) for c in grid}
    ) == len(grid)
    assert set(_output_f567_kernel.keys) == {
        "shape_key",
        "M",
        "L",
        "wp0",
        "wp1",
        "wg0",
        "wg1",
    }


def test_independent_shape_axes_do_not_alias():
    keys = {
        token_key(l, KP=kp, KG=kg, N=n)
        for l, kp, kg, n in itertools.product(
            [128, 384, 768], [128, 256], [64, 128], [64, 128]
        )
    }
    assert len(keys) == 24
    cfg = triton.Config(
        dict(BLOCK_M1=128, BLOCK_N=256, BLOCK_K=128, GROUP_M=8),
        num_warps=8,
        num_stages=4,
    )
    # An intentionally oversized one-config mask probe is still executable.
    assert prune_output_configs([cfg], dict(M=7, N=13, KP=19, KG=17)) == [cfg]


def probes():
    """Every axis value, unequal K extents, multi-N tiles and incomplete M groups."""
    choices = []
    for i, (m, n) in enumerate(
        itertools.product([16, 32, 64, 128], [32, 64, 128, 256])
    ):
        choices.append(
            dict(
                BLOCK_M1=m,
                BLOCK_N=n,
                BLOCK_K=[16, 32, 64, 128][i % 4],
                GROUP_M=[1, 2, 4, 8][i % 4],
                num_warps=[4, 8][i % 2],
                num_stages=[2, 3, 4][i % 3],
            )
        )
    for w in (1, 2):
        choices.append(
            dict(
                BLOCK_M1=32,
                BLOCK_N=64,
                BLOCK_K=32,
                GROUP_M=4,
                num_warps=w,
                num_stages=2,
            )
        )
    choices.append(
        dict(
            BLOCK_M1=128, BLOCK_N=128, BLOCK_K=64, GROUP_M=8, num_warps=4, num_stages=3
        )
    )
    return choices


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("cfg", probes())
@pytest.mark.parametrize("transpose_weights", [False, True])
def test_masks_groups_and_independent_reductions(cfg, transpose_weights):
    torch.manual_seed(329)
    torch.backends.cuda.matmul.allow_tf32 = False
    m, n = 5 * cfg["BLOCK_M1"] + 7, 2 * cfg["BLOCK_N"] + 3
    kp, kg, length = 193, 67, 17
    kw = dict(device="cuda", dtype=torch.bfloat16)
    norm, x = torch.randn(m, kp, **kw), torch.randn(m, kg, **kw)
    wp, wg = torch.randn(n, kp, **kw) / kp**0.5, torch.randn(kg, n, **kw) / kg**0.5
    if transpose_weights:
        wp, wg = wp.t().contiguous().t(), wg.t().contiguous().t()
    res = torch.randn(m, n, **kw)
    ds = (torch.rand(length, n, device="cuda") > 0.25).to(torch.bfloat16) / 0.75
    y, proj, gate = (torch.full((m, n), float("nan"), **kw) for _ in range(3))
    grid = (triton.cdiv(m, cfg["BLOCK_M1"]) * triton.cdiv(n, cfg["BLOCK_N"]),)
    _output_f567_kernel.fn[grid](
        norm,
        x,
        wp,
        wg,
        proj,
        gate,
        y,
        res,
        ds,
        m,
        length,
        kp,
        kg,
        n,
        *wp.stride(),
        *wg.stride(),
        shape_key=token_key(length, KP=kp, KG=kg, N=n),
        **cfg,
    )
    p = (norm.float() @ wp.float().t()).to(torch.bfloat16).float()
    g = torch.sigmoid((x.float() @ wg.float()).to(torch.bfloat16).float())
    expected = res.float() + p * g * ds.float()[torch.arange(m, device="cuda") % length]
    for a, b in [(proj, p), (gate, g), (y, expected)]:
        assert torch.isfinite(a).all()
        rel = (a.float() - b).norm() / b.norm().clamp_min(1e-8)
        assert rel < 0.004, (cfg, transpose_weights, rel.item())


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_public_shape_validation_before_launch():
    kw = dict(device="cuda", dtype=torch.bfloat16)
    args = [
        torch.empty(7, 19, **kw),
        torch.empty(7, 17, **kw),
        torch.empty(13, 19, **kw),
        torch.empty(17, 13, **kw),
        torch.empty(7, 13, **kw),
        torch.empty(3, 13, **kw),
        3,
    ]
    wrong = list(args)
    wrong[5] = torch.empty(4, 13, **kw)
    with pytest.raises(ValueError, match="shapes disagree"):
        output_f567_train(*wrong)
    wrong = list(args)
    wrong[0] = wrong[0].float()
    with pytest.raises(TypeError, match="BF16"):
        output_f567_train(*wrong)
