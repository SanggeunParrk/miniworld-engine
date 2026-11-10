"""Inference-only preparation changes: numerical equivalence and live graph inputs."""

import pytest
import torch
from settings_helpers import override_setting

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 0),
        reason="A100 required",
    ),
]


@pytest.mark.parametrize("splits", [2, 4, 8])
@pytest.mark.parametrize("length", [128, 768])
def test_pwa_shared_bmm_matches_reference_and_live_graph(splits, length):
    from miniworld_engine.kernels.pair_weighted_averaging.cuda import sm80

    torch.manual_seed(51)
    w = torch.randn(8, length, length, device="cuda").softmax(-1).bfloat16()
    v = torch.randn(8 * splits, length, 256, device="cuda", dtype=torch.bfloat16)
    sm80.broadcast_bmm(w, v)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = sm80.broadcast_bmm(w, v)
    for change in (False, True):
        if change:
            w.copy_(torch.randn_like(w.float()).softmax(-1))
            v.normal_()
        graph.replay()
        baseline = torch.bmm(
            w[:, None]
            .expand(8, splits, length, length)
            .reshape(8 * splits, length, length),
            v,
        )
        ref = torch.bmm(
            w.float()[:, None]
            .expand(8, splits, length, length)
            .reshape(8 * splits, length, length),
            v.float(),
        )
        err = lambda x, ref=ref: (x.float() - ref).norm() / ref.norm()
        assert err(out) <= err(baseline) * 1.05 + 1e-5
        torch.testing.assert_close(out, sm80.broadcast_bmm(w, v), atol=0, rtol=0)


def test_transition_inference_does_not_pack_backward_operands():
    from miniworld_engine.kernels.transition.cuda import fused_sm80 as k

    weights = [
        torch.randn(512, 128, device="cuda", dtype=torch.bfloat16) for _ in range(2)
    ]
    ws = torch.randn(128, 512, device="cuda", dtype=torch.bfloat16)
    gamma = torch.randn(128, device="cuda")
    beta = torch.randn_like(gamma)
    full = k._pack(gamma, beta, *weights, ws, save=True)
    lean = k._pack(gamma, beta, *weights, ws, save=False)
    torch.testing.assert_close(full[0], lean[0], atol=0, rtol=0)
    torch.testing.assert_close(full[1], lean[1], atol=0, rtol=0)
    assert lean[2].numel() == lean[3].numel() == 0


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_atom_pack_skips_unused_layouts_and_replays_live_weights(dtype, monkeypatch):
    from miniworld_engine.integrations import augattn_sm80 as k

    override_setting(monkeypatch, "a100_cleanup", "1")
    shapes = [
        (128, 128),
        (128,),
        (128, 128),
        (128, 128),
        (128, 128),
        (128, 128),
        (128, 128),
        (128,),
        (16,),
        (4, 16),
    ]
    weights = [torch.randn(shape, device="cuda", dtype=dtype) for shape in shapes]
    full = k._pack(weights, 4, dtype, save=True)
    lean = k._pack(weights, 4, dtype, save=False)
    assert lean[2] is lean[3] is lean[5] is None
    for i in (0, 1, 4):
        torch.testing.assert_close(lean[i], full[i], atol=0, rtol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = k._pack(weights, 4, dtype, save=False)
    for w in weights:
        w.add_(0.125)
    graph.replay()
    expected = k._pack(weights, 4, dtype, save=False)
    for i in (0, 1, 4):
        torch.testing.assert_close(captured[i], expected[i], atol=0, rtol=0)


@pytest.mark.parametrize("length", [33, 128, 1024])
def test_swa_inference_skips_lse_without_changing_output(length):
    from miniworld_engine.kernels.swa_dit.cuda import sm80

    q, k, v = [
        torch.randn(3, 4, length, 32, device="cuda", dtype=torch.bfloat16)
        for _ in range(3)
    ]
    lengths = torch.tensor([0, length // 2, length], device="cuda", dtype=torch.int32)
    expected, lse = sm80.attn_fwd(q, k, v, lengths)
    out, empty = sm80.attn_fwd(q, k, v, lengths, save_lse=False)
    assert empty.numel() == 0
    assert lse.numel() == 3 * 4 * length
    torch.testing.assert_close(out, expected, atol=0, rtol=0)
