"""A100 AF3 local atom DiT block on the sm_80 row stages (``integrations.local_dit_sm80_block``): inference and training -- with or without a [1, N] key mask, cross mode or not,
at any N --, its output and every gradient no worse against an fp64 PyTorch block than the bf16 module path's, the hoisted per-item tables, fp32 master parameters, the grouped
weight-gradient kernel against cuBLAS, and every call it does not serve keeps the previous path."""

import pytest
import torch
from settings_helpers import override_setting
from tests.integrations.test_b200_local_dit_gpu import (
    _block,
    _inputs,
    _mask,
    _rel,
    _train,
)

from miniworld_engine.integrations import local_dit_sm80_block as block
from miniworld_engine.kernels import _capture
from miniworld_engine.modules.exceptions import ImplementationType
from miniworld_engine.modules.local_dit import LocalDiTBlock

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 0),
        reason="A100 required",
    ),
]

DS = 128


def _spy(monkeypatch):
    calls = []
    orig_apply, orig_infer = block._RowsBlock.apply, block._infer
    monkeypatch.setattr(
        block._RowsBlock,
        "apply",
        lambda *a: calls.append("train") or orig_apply(*a),
    )
    monkeypatch.setattr(
        block, "_infer", lambda *a: calls.append("infer") or orig_infer(*a)
    )
    return calls


@pytest.mark.parametrize(
    ("A", "N", "masked", "cross"),
    [
        (4, 256, False, False),
        (3, 300, False, False),
        (3, 384, True, False),
        (2, 200, True, False),
        (5, 130, False, False),
        (1, 65, True, True),
        (3, 300, False, True),
        (2, 384, True, True),
        (4, 130, True, True),
    ],
)
def test_training_matches_the_previous_path(A, N, masked, cross, monkeypatch):
    ref, eng = _block(cross=cross)
    ins = _inputs(A, N)
    mask = _mask(N, 11) if masked else None
    w = torch.randn(A, 1, N, DS, device="cuda", dtype=torch.float64)
    truth = _train(ref, ins, torch.float64, w, mask)
    calls = _spy(monkeypatch)
    rows = _train(eng, ins, torch.bfloat16, w, mask)
    assert calls == ["train"], calls
    override_setting(monkeypatch, "local_dit_sm80_rows", "0")
    previous = _train(eng, ins, torch.bfloat16, w, mask)
    assert calls == ["train"], "the switch did not keep the previous path"
    assert set(rows) == set(truth)
    for k in truth:
        er, ep = _rel(rows[k], truth[k]), _rel(previous[k], truth[k])
        assert torch.isfinite(rows[k]).all(), k
        assert rows[k].dtype == previous[k].dtype, k
        assert er < 1.5 * ep + 3e-3, (
            f"{k}: row kernels {er:.2e} vs previous path {ep:.2e}"
        )


@pytest.mark.parametrize(
    ("A", "N", "masked", "cross"),
    [
        (5, 128, False, False),
        (1, 384, False, False),
        (2, 1024, False, False),
        (5, 1024, True, False),
        (5, 300, True, False),
        (1, 1000, True, False),
        (2, 100, False, False),
        (2, 300, True, True),
        (5, 1024, False, True),
        (1, 130, False, True),
    ],
)
def test_inference_matches_the_previous_path(A, N, masked, cross, monkeypatch):
    ref, eng = _block(2, cross)
    s, c, z = _inputs(A, N, 3)
    mask = _mask(N, 13) if masked else None
    with torch.no_grad():
        truth = ref(s.double(), c.double(), z.double(), mask)
        calls = _spy(monkeypatch)
        rows = eng(s.bfloat16(), c.bfloat16(), z.bfloat16(), mask)
        assert calls == ["infer"], calls
        override_setting(monkeypatch, "local_dit_sm80_rows", "0")
        previous = eng(s.bfloat16(), c.bfloat16(), z.bfloat16(), mask)
        assert calls == ["infer"], "the switch did not keep the previous path"
    assert rows.dtype == torch.bfloat16
    assert rows.shape == s.shape
    assert _rel(rows, truth) < 1.5 * _rel(previous, truth) + 3e-3


def test_fp32_master_parameters_under_autocast_give_fp32_gradients(monkeypatch):
    """The bf16-mixed recipe: fp32 parameters, bf16 activations. The kernels run on bf16 packs, the gradients come back in the parameters' dtype."""
    ref, eng = _block(cross=True)
    eng = eng.float()
    ins = _inputs(3, 300)
    mask = _mask(300, 11)
    w = torch.randn(3, 1, 300, DS, device="cuda", dtype=torch.float64)
    truth = _train(ref, ins, torch.float64, w, mask)
    calls = _spy(monkeypatch)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        leaves = [t.detach().clone().bfloat16().requires_grad_() for t in ins]
        out = eng(*leaves, mask)
        (out.double() * w).sum().backward()
    assert calls == ["train"], calls
    for name, p in eng.named_parameters():
        assert p.grad.dtype == torch.float32, name
        assert _rel(p.grad, truth[name]) < 3e-2, name
    assert _rel(leaves[0].grad, truth["dsingle"]) < 2e-2


def test_hoisted_tables_follow_the_inputs_and_the_weights(monkeypatch):
    """Under ``static_inputs()`` an inference call serves the conditioning table and the pair bias from the hoist; changing the conditioning, the pair or a weight in place refreshes them."""
    _, eng = _block(4, cross=True)
    s, c, z = (t.bfloat16() for t in _inputs(2, 256, 5))
    mask = _mask(256, 15)
    previous = _capture.set_static_inputs(True)
    try:
        with torch.no_grad():
            first = eng(s, c, z, mask)
            again = eng(s, c, z, mask)
            torch.testing.assert_close(first, again, rtol=0, atol=0)
            c.mul_(0.5)
            after_cond = eng(s, c, z, mask)
            assert not torch.equal(first, after_cond)
            z.add_(0.25)
            after_pair = eng(s, c, z, mask)
            assert not torch.equal(after_cond, after_pair)
            eng.attention.to_bias.weight.mul_(0.5)
            after_weight = eng(s, c, z, mask)
            assert not torch.equal(after_pair, after_weight)
            override_setting(monkeypatch, "local_dit_hoist", "0")
            fresh = eng(s, c, z, mask)
    finally:
        _capture.set_static_inputs(previous)
    torch.testing.assert_close(after_weight, fresh, rtol=0, atol=0)


@pytest.mark.parametrize("rows", [64, 1000, 4097, 20000])
def test_grouped_weight_gradient_matches_cublas(rows):
    from miniworld_engine.kernels.augmented_attention.cuda import sm80_atom_rows

    torch.manual_seed(0)
    a = torch.randn(rows, 512, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(rows, 384, device="cuda", dtype=torch.bfloat16)
    small = torch.randn(rows, 128, device="cuda", dtype=torch.bfloat16)
    problems = [(a[:, 128:384], b[:, 128:256]), (a, small), (small, b[:, :256])]
    out = [
        torch.empty(256, 128, device="cuda", dtype=torch.float32),
        torch.empty(512, 128, device="cuda", dtype=torch.bfloat16),
        torch.empty(128, 256, device="cuda", dtype=torch.float32),
    ]
    copy = torch.arange(128, device="cuda", dtype=torch.float32)
    copy_out = torch.empty(128, device="cuda", dtype=torch.bfloat16)
    segments = [
        (0, None, 0, 256, out[0]),
        (1, None, 0, 512, out[1]),
        (2, None, 0, 128, out[2]),
        (None, copy, 0, 0, copy_out),
    ]
    sm80_atom_rows.wgrad(problems, segments, rows)
    for (x, y), got in zip(problems, out, strict=True):
        want = torch.mm(x.t().double(), y.double())
        assert _rel(got, want) < (5e-3 if got.dtype == torch.bfloat16 else 5e-5)
    torch.testing.assert_close(copy_out.float(), copy, rtol=0, atol=0)


def test_wgrad_is_deterministic_and_graph_capturable():
    from miniworld_engine.kernels.augmented_attention.cuda import sm80_atom_rows

    a = torch.randn(9000, 256, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(9000, 128, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(256, 128, device="cuda", dtype=torch.float32)
    run = lambda: sm80_atom_rows.wgrad([(a, b)], [(0, None, 0, 256, out)], 9000, 7)
    run()
    first = out.clone()
    run()
    torch.testing.assert_close(out, first, rtol=0, atol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    out.zero_()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out, first, rtol=0, atol=0)


def test_module_default_implementation_without_the_kernels_is_unchanged():
    """A PYTORCH-implementation block never takes the kernels."""
    torch.manual_seed(0)
    m = (
        LocalDiTBlock(cross_attention=True, implementation=ImplementationType.PYTORCH)
        .cuda()
        .bfloat16()
    )
    s, c, z = (t.bfloat16() for t in _inputs(1, 130))
    assert not block.serves(m, s, c, z, None)
