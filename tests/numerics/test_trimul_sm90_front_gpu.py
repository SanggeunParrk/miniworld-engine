"""TMA/WGMMA F2 keeps Triton's mask rounding and saved backward contract."""
import pytest
import torch


@pytest.fixture
def sm90_front():
    # Device discovery stays inside test execution, never module import.
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("requires an SM90 GPU")
    from miniworld_engine.kernels.trimul_inproj.cute.parity_front import launch_front
    from miniworld_engine.kernels.trimul_inproj.triton.bidirectional import (
        _bidir_front_kernel,
    )
    return launch_front, _bidir_front_kernel.fn


def _assert_rel(actual, expected):
    error = (actual.float() - expected.float()).norm()
    scale = expected.float().norm().clamp_min(1e-12)
    assert (error / scale).item() <= 1e-4


@pytest.mark.parametrize("warps", [4, 8])
@pytest.mark.parametrize(("m", "k", "h2", "mask_kind", "save", "bm", "bk", "bh", "stages"), [
    (144, 96, 40, "fractional", True, 64, 32, 32, 2),
    (256, 128, 64, "none", True, 64, 16, 16, 1),
    (256, 128, 64, "binary", True, 64, 64, 64, 4),
    (256, 128, 64, "fractional", False, 64, 32, 32, 3),
    (144, 96, 40, "fractional", True, 128, 32, 32, 2),
    (256, 128, 64, "binary", True, 128, 64, 64, 1),
    # Non-power-of-two N groups:3 and5, including short-M inference tails.
    (24, 96, 40, "fractional", False, 128, 64, 32, 2),
    (144, 96, 72, "binary", True, 64, 32, 32, 2),
    (24, 96, 72, "none", False, 128, 64, 32, 3),
    # Non-divisible K-trip/stage ratios across multiple channel chunks.
    (144, 224, 40, "fractional", True, 64, 32, 32, 3),
    (144, 144, 40, "binary", True, 64, 16, 16, 10),
    (144, 256, 40, "none", True, 128, 16, 16, 10),
    # The fast geometry with saved values, plus an inference-only storage fit.
    (144, 128, 72, "fractional", True, 128, 64, 32, 2),
    (144, 128, 256, "fractional", False, 128, 64, 64, 6),
])
def test_front_outputs_and_saved_contract(sm90_front, m, k, h2, mask_kind, save,
                                           bm, bk, bh, stages, warps):
    launch, triton_front = sm90_front
    torch.manual_seed(419)
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * .2
    w = torch.randn(k, 4 * h2, device="cuda", dtype=torch.bfloat16) * .2
    mask = None
    if mask_kind != "none":
        mask = torch.rand(m, device="cuda")
        if mask_kind == "binary":
            mask = mask > .25
        mask = mask.to(torch.bfloat16)
    output = a.new_empty((2 * h2, m))
    preact = a.new_empty((4 * h2, m)) if save else None
    expected = torch.empty_like(output)
    saved = a.new_empty((4 * h2, m))
    config = {"BLOCK_M1": bm, "BLOCK_K_D": bk, "BLOCK_K_H2": bh,
                  "num_warps": warps, "num_stages": stages}
    triton_front[((m + bm - 1) // bm,)](
        a, w, expected[:h2], expected[h2:], saved, mask, m, m,
        K=k, H2=h2, shape_key=0, SAVE_PREACT=save, **config,
    )
    launch(a, w, output, preact, mask, config)
    torch.cuda.synchronize()
    _assert_rel(output, expected)
    if save:
        _assert_rel(preact, saved)
    # FP32 GEMM reference independently checks raw preactivations and the
    # crucial BF16-before-fractional-mask boundary, not only backend agreement.
    raw = a.float() @ w.float()
    value = raw[:, 0::2].sigmoid() * raw[:, 1::2]
    if mask is not None:
        value = value.to(torch.bfloat16).float() * mask.float()[:, None]
        assert output[:, mask == 0].count_nonzero().item() == 0
    _assert_rel(output, value.to(torch.bfloat16).T)
    if save:
        _assert_rel(preact, raw.to(torch.bfloat16).T)


def test_front_group_is_derived_from_physical_projection_tiles(sm90_front):
    from miniworld_engine.kernels.trimul_inproj.cute.parity_front import (
        _front_n_tile_group,
    )

    assert _front_n_tile_group(1024, 64) == 8
    assert _front_n_tile_group(1024, 16) == 32
    assert _front_n_tile_group(160, 32) == 3
    assert _front_n_tile_group(288, 32) == 5


@pytest.mark.parametrize("warps", [4, 8])
@pytest.mark.parametrize("save", [False, True])
def test_grouped_front_fullgraph_and_cuda_graph(sm90_front, save, warps):
    from miniworld_engine.kernels.trimul_inproj.cute.parity_front import front_sm90

    torch.manual_seed(987)
    a = torch.randn(144, 96, device="cuda", dtype=torch.bfloat16) * .2
    w = torch.randn(96, 160, device="cuda", dtype=torch.bfloat16) * .2
    mask = torch.rand(144, device="cuda", dtype=torch.bfloat16)

    def run(a, w, mask):
        return front_sm90(a, w, mask, save, 128, 32, 32, warps, 2)

    expected, saved = run(a, w, mask)
    compiled = torch.compile(run, fullgraph=True, dynamic=False)
    actual, preact = compiled(a, w, mask)
    torch.cuda.synchronize()
    _assert_rel(actual, expected)
    if save:
        _assert_rel(preact, saved)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual, preact = compiled(a, w, mask)
    graph.replay()
    torch.cuda.synchronize()
    _assert_rel(actual, expected)
    if save:
        _assert_rel(preact, saved)


@pytest.mark.parametrize("warps", [4, 8])
@pytest.mark.parametrize("projection", [1.0, 256.0])
@pytest.mark.parametrize("mask_kind", ["none", "fractional"])
def test_front_extreme_sigmoid_preserves_tiny_values(
    sm90_front, warps, projection, mask_kind
):
    """No FTZ gate loss: compare exact bits, including BF16 subnormals."""
    launch, triton_front = sm90_front
    m, k, h2 = 128, 128, 64
    sentinels = [-100.0, -90.0, -89.0, -88.5, -88.0, -87.5, -87.0, -86.0,
                 -1.0, 0.0, 1.0, 86.0, 87.5, 88.0, 89.0, 100.0]
    logits = torch.linspace(-100, 100, m, device="cuda").to(torch.bfloat16)
    logits[:len(sentinels)] = torch.tensor(
        sentinels, device="cuda", dtype=torch.bfloat16
    )
    a = torch.eye(m, k, device="cuda", dtype=torch.bfloat16)
    w = a.new_empty((k, 4 * h2))
    w[:, 0::2] = logits[:, None]
    w[:, 1::2] = projection
    mask = None
    if mask_kind == "fractional":
        mask = torch.tensor(
            [.5, .75, .25, 1.0], device="cuda", dtype=torch.bfloat16
        ).repeat(m // 4)
    actual = a.new_empty((2 * h2, m))
    saved = a.new_empty((4 * h2, m))
    expected = torch.empty_like(actual)
    expected_saved = torch.empty_like(saved)
    config = {"BLOCK_M1": 64, "BLOCK_K_D": 64, "BLOCK_K_H2": 64,
              "num_warps": warps, "num_stages": 2}
    triton_front[(2,)](
        a, w, expected[:h2], expected[h2:], expected_saved, mask, m, m,
        K=k, H2=h2, shape_key=0, SAVE_PREACT=True, **config,
    )
    launch(a, w, actual, saved, mask, config)
    torch.cuda.synchronize()
    # Establish that the reference really contains the tiny values which a
    # relative norm would hide among ordinary positive-logit output rows.
    for value in (-88.5, -88.0, -87.5):
        ix = sentinels.index(value)
        reference_value = expected[0, ix].cpu().double().item()
        assert reference_value > 0
        if projection == 1.0:
            assert reference_value < torch.finfo(torch.bfloat16).tiny
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    assert torch.equal(saved.view(torch.int16), expected_saved.view(torch.int16))
