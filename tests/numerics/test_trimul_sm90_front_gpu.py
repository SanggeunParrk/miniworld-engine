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


@pytest.mark.parametrize(("m", "k", "h2", "mask_kind", "save", "bm", "bk", "bh", "stages"), [
    (144, 96, 40, "fractional", True, 64, 32, 32, 2),
    (256, 128, 64, "none", True, 64, 16, 16, 1),
    (256, 128, 64, "binary", True, 64, 64, 64, 4),
    (256, 128, 64, "fractional", False, 64, 32, 32, 3),
    (144, 96, 40, "fractional", True, 128, 32, 32, 2),
    (256, 128, 64, "binary", True, 128, 64, 64, 1),
])
def test_front_outputs_and_saved_contract(sm90_front, m, k, h2, mask_kind, save,
                                           bm, bk, bh, stages):
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
                  "num_warps": 8, "num_stages": stages}
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
