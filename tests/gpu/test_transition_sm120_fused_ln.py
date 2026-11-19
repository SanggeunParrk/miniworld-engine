"""RTX 5090 (sm_120a) Transition, bf16 inference at D >= 256: the LayerNorm folded into the dual GEMM (statistics kernel + gamma-folded weights + the epilogue's
rstd (acc - mean cs) + bias) stays inside the accuracy standard (error against an fp64 evaluation no larger than the unfused ln_fwd + dual chain's, up to rounding),
also for an input with a large mean, and is the path that serves from ``SM120_FUSED_LN_MIN_ROWS`` rows on."""

import copy

import pytest
import torch

from miniworld_engine.measure import config as C
from miniworld_engine.modules import Transition

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0), reason="needs an RTX 5090 (sm_120)"),
]


def _module(d: int, n: int) -> Transition:
    module = Transition(d_hidden=d, n=n, master_weights=True).cuda().float().train()
    C.randomize(module, torch.Generator(device="cuda").manual_seed(0))
    module.set_master_weights()
    return module


@pytest.mark.parametrize(
    ("d", "n", "rows", "offset"), [(256, 2, 40000, 0.0), (384, 2, 33000, 0.0), (512, 4, 40001, 0.0), (256, 4, 65536, 12.0)]
)
def test_fused_layernorm_within_the_unfused_error(monkeypatch: pytest.MonkeyPatch, d: int, n: int, rows: int, offset: float) -> None:
    from miniworld_engine.kernels.transition.cuda import fused_wide_sm80 as fw

    module = _module(d, n)
    kern = copy.deepcopy(module).set_backend("cuda").eval()
    ref = copy.deepcopy(module).double().eval()
    x = (torch.randn(1, rows, d, device="cuda", generator=torch.Generator(device="cuda").manual_seed(1)) * 2 + offset).to(torch.bfloat16)
    with torch.no_grad():
        want = ref(x.double())
        assert fw._fused_ln(rows, d, 2, x)
        fused = kern(x)
        monkeypatch.setattr(fw, "SM120_FUSED_LN", False)
        unfused = kern(x)
    err_f = (fused.double() - want).abs()
    err_u = (unfused.double() - want).abs()
    assert err_f.pow(2).mean().sqrt() <= err_u.pow(2).mean().sqrt() * 1.05 + 1e-6
    assert err_f.max() <= err_u.max() * 1.25 + 1e-3


def test_few_rows_and_training_keep_the_unfused_chain() -> None:
    from miniworld_engine.kernels.transition.cuda import fused_wide_sm80 as fw

    x = torch.zeros(1, 16384, 256, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        assert not fw._fused_ln(16384, 256, 2, x)
        assert fw._fused_ln(65536, 256, 2, x)
        assert not fw._fused_ln(65536, 128, 2, x)
    assert not fw._fused_ln(65536, 256, 2, x)  # grad enabled (training): xn is saved for the backward


def test_row_blocks_with_the_fused_layernorm_stay_accurate(monkeypatch: pytest.MonkeyPatch) -> None:
    """An inference call past ``SM120_INFER_MAX_ROWS`` (``forward_into`` per row block: two fused blocks and one unfused remainder here) is as accurate as the single call."""
    from miniworld_engine.kernels.transition.cuda import fused_wide_sm80 as fw

    module = _module(256, 2)
    kern = copy.deepcopy(module).set_backend("cuda").eval()
    ref = copy.deepcopy(module).double().eval()
    x = (torch.randn(1, 100000, 256, device="cuda", generator=torch.Generator(device="cuda").manual_seed(3)) * 2 + 1).to(torch.bfloat16)
    with torch.no_grad():
        want = ref(x.double())
        whole = kern(x)
        monkeypatch.setattr(fw, "SM120_INFER_MAX_ROWS", 40000)
        blocks = kern(x)
    err_w = (whole.double() - want).abs().pow(2).mean().sqrt()
    err_b = (blocks.double() - want).abs().pow(2).mean().sqrt()
    assert err_b <= err_w * 1.05 + 1e-6


def test_statistics_overlap_matches_the_serial_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    """Above ``SM120_STATS_OVERLAP_MIN_ROWS`` the statistics run on a side stream in row chunks (and under a CUDA graph): the output bits of the serial pass."""
    from miniworld_engine.kernels.transition.cuda import fused_wide_sm80 as fw

    module = _module(256, 2)
    kern = copy.deepcopy(module).set_backend("cuda").eval()
    x = torch.randn(1, 150000, 256, device="cuda", generator=torch.Generator(device="cuda").manual_seed(4)).to(torch.bfloat16)
    with torch.no_grad():
        overlapped = kern(x)
        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            kern(x)
        torch.cuda.synchronize()
        with torch.cuda.graph(graph, stream=stream):
            replay = kern(x)
        graph.replay()
        torch.cuda.synchronize()
        monkeypatch.setattr(fw, "SM120_STATS_OVERLAP_MIN_ROWS", 10**9)
        serial = kern(x)
    assert torch.equal(overlapped, serial)
    assert torch.equal(replay, serial)


@pytest.mark.parametrize(("d", "n", "rows"), [(256, 4, 70001), (384, 2, 66000), (512, 4, 65536)])
def test_big_work_tiles_give_the_same_bits(monkeypatch: pytest.MonkeyPatch, d: int, n: int, rows: int) -> None:
    """The 128-row tiles of the fused LayerNorm path (dual config 14, squeeze config 13) accumulate every output element in the order of the 64-row ones: the same bits."""
    from miniworld_engine.kernels.transition.cuda import fused_wide_sm80 as fw

    module = _module(d, n)
    kern = copy.deepcopy(module).set_backend("cuda").eval()
    x = (torch.randn(1, rows, d, device="cuda", generator=torch.Generator(device="cuda").manual_seed(6)) * 2).to(torch.bfloat16)
    outs = []
    for work in (0, 10**18):
        monkeypatch.setattr(fw, "SM120_FOLD_BIG_WORK", work)
        monkeypatch.setattr(fw, "SM120_FOLD_BIG_H", 0 if work == 0 else 10**9)
        assert fw._fold_tiles(rows, n * 2 * d // 2 * 1, 2, 2) == ((14, 13) if work == 0 else (2, 2))
        with torch.no_grad():
            outs.append(kern(x))
    assert torch.equal(outs[0], outs[1])


def test_training_forward_with_128_row_tiles_keeps_the_bits(monkeypatch: pytest.MonkeyPatch) -> None:
    """The training forward at D >= 256 takes dual config 14 / squeeze config 13 (``SM120_FUSED_LN_MIN_ROWS`` and up): outputs and gradients of the 64-row tiles' bits."""
    from miniworld_engine.kernels.transition.cuda import fused_wide_sm80 as fw

    module = _module(256, 4)
    x0 = (torch.randn(1, 40000, 256, device="cuda", generator=torch.Generator(device="cuda").manual_seed(8))).to(torch.bfloat16)
    up = torch.randn(x0.shape, device="cuda", generator=torch.Generator(device="cuda").manual_seed(9)).to(torch.bfloat16)
    results = []
    for min_rows in (10**9, 32768):
        monkeypatch.setattr(fw, "SM120_FUSED_LN_MIN_ROWS", min_rows)
        kern = copy.deepcopy(module).set_backend("cuda")
        x = x0.clone().requires_grad_(True)
        y = kern(x)
        y.backward(up)
        results.append([y.detach(), x.grad] + [p.grad for p in kern.parameters() if p.grad is not None])
    for a, b in zip(*results, strict=True):
        assert torch.equal(a, b)
