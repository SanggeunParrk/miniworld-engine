"""The Triton window attention (``kernels/augmented_attention/triton/window.py``) against its fp32 PyTorch definition: forward and the
gradients of q, k, v and the bias, the slid and the clipped key window, a key mask, ESMFold2's 192 keys with the rank window, Chai-1's
same-conformer mask, and the two properties the backward is built for -- bitwise reproducibility (no atomics) and padding queries."""

import pytest
import torch

from miniworld_engine.kernels.augmented_attention.reference import (
    window_attention_pytorch,
)
from miniworld_engine.kernels.augmented_attention.triton.window import (
    plan,
    window_attention,
)

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]

# dtype, key window, keys, window_half, uid, N, real atoms, A, B
CASES = [
    (torch.float32, "pad", 128, None, False, 200, 200, 3, 1),
    (torch.float32, "slide", 128, None, False, 200, 170, 3, 1),
    (torch.bfloat16, "slide", 128, None, False, 200, 170, 2, 2),
    (torch.bfloat16, "pad", 128, None, False, 333, 333, 2, 1),
    (torch.bfloat16, "pad", 192, 64, False, 300, 250, 2, 1),
    (torch.float32, "slide", 192, 64, False, 300, 250, 2, 1),
    (torch.bfloat16, "slide", 128, None, True, 256, 240, 2, 1),
    (torch.float32, "slide", 128, None, False, 64, 50, 2, 1),       # under 128 real atoms: the window runs past the real atoms
    (torch.float32, "slide", 192, 64, False, 100, 100, 1, 1),       # 192-key window starting before atom 0
    (torch.float32, "slide", 128, None, False, 1000, 1000, 1, 1),
]
BAND = {torch.float32: 6e-3, torch.bfloat16: 3e-2}      # max |error| / max |reference|; fp32 runs on TF32 tensor cores


def _rel(a, b):
    return ((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-12)).item()


def _inputs(dtype, rule, keys, uid, n, real, a, b, heads=4, head_dim=32, seed=0):
    torch.manual_seed(seed)
    nwin = (n + 31) // 32
    mk = lambda: torch.randn(a, b, n, heads, head_dim, device="cuda", dtype=dtype, requires_grad=True)
    q, k, v = mk(), mk(), mk()
    bias = torch.randn(b, heads, nwin, 32, keys, device="cuda", dtype=dtype, requires_grad=True)
    keep = torch.zeros(b, n, dtype=torch.bool, device="cuda")
    keep[:, :real] = True
    ids = (torch.arange(n, device="cuda") // 40)[None].expand(b, -1).contiguous().int() if uid else None
    return q, k, v, bias, keep, ids, plan(rule, n, nwin, keys, keep, "cuda")


@pytest.mark.parametrize(("dtype", "rule", "keys", "half", "uid", "n", "real", "a", "b"), CASES)
def test_matches_the_pytorch_definition_with_gradients(dtype, rule, keys, half, uid, n, real, a, b):
    q, k, v, bias, keep, ids, starts = _inputs(dtype, rule, keys, uid, n, real, a, b)
    out = window_attention(q, k, v, bias, starts, keep, uid=ids, keys=keys, window_half=half)
    go = torch.randn_like(out)
    out.backward(go)
    got = [t.grad for t in (q, k, v, bias)]
    leaves = [t.detach().float().requires_grad_() for t in (q, k, v, bias)]
    want = window_attention_pytorch(*leaves[:3], leaves[3], starts, keep, uid=ids, keys=keys, window_half=half)
    want.backward(go.float())
    assert torch.isfinite(out).all()
    assert _rel(out, want) < BAND[dtype]
    for name, g, r in zip(("dq", "dk", "dv", "dbias"), got, (t.grad for t in leaves), strict=True):
        assert g.shape == r.shape, name
        assert _rel(g, r) < BAND[dtype], name


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_two_runs_are_bitwise_identical(dtype):
    """The key / value gradients are summed in a fixed order and the bias gradient over fixed sample groups: no atomics."""
    q, k, v, bias, keep, _, starts = _inputs(dtype, "slide", 128, False, 4096, 4000, 6, 1)
    go = torch.randn(6, 1, 4096, 4, 32, device="cuda", dtype=dtype)
    runs = []
    for _ in range(3):
        for t in (q, k, v, bias):
            t.grad = None
        out = window_attention(q, k, v, bias, starts, keep)
        out.backward(go)
        runs.append([out.detach().clone(), *(t.grad.clone() for t in (q, k, v, bias))])
    for later in runs[1:]:
        for first, other in zip(runs[0], later, strict=True):
            assert torch.equal(first, other)


def test_a_row_with_no_live_key_is_zero_and_padding_queries_are_not_written():
    q, k, v, bias, keep, _, starts = _inputs(torch.float32, "pad", 128, False, 100, 100, 1, 1)
    keep.zero_()
    out = window_attention(q, k, v, bias, starts, keep)
    assert out.shape == q.shape
    assert torch.count_nonzero(out) == 0


def test_the_inputs_may_be_views_of_a_projection():
    """The module hands over q / k / v as views of [A, B, N, H * D] projections and the bias as a permuted view of the linear output."""
    n, h, d, keys = 160, 4, 32, 128
    nwin = (n + 31) // 32
    torch.manual_seed(2)
    qkv = torch.randn(2, 1, n, 3, h * d, device="cuda")
    q, k, v = (qkv[:, :, :, i].view(2, 1, n, h, d) for i in range(3))
    z = torch.randn(1, nwin, 32, keys, h, device="cuda")
    bias = z.permute(0, 4, 1, 2, 3)
    keep = torch.ones(1, n, dtype=torch.bool, device="cuda")
    starts = plan("pad", n, nwin, keys, keep, "cuda")
    got = window_attention(q, k, v, bias, starts, keep)
    want = window_attention_pytorch(q, k, v, bias, starts, keep)
    assert _rel(got, want) < BAND[torch.float32]


def test_a_head_wider_than_the_kernel_is_refused():
    q = torch.randn(1, 1, 64, 1, 128, device="cuda")
    bias = torch.randn(1, 1, 2, 32, 128, device="cuda")
    keep = torch.ones(1, 64, dtype=torch.bool, device="cuda")
    with pytest.raises(ValueError, match="head_dim <= 64"):
        window_attention(q, q, q, bias, plan("pad", 64, 2, 128, keep, "cuda"), keep)


def _kernels():
    from miniworld_engine.kernels.augmented_attention.triton import window

    return {"fwd": window._window_fwd, "bwd": window._window_bwd, "fold": window._window_fold}


@pytest.mark.parametrize("which", ["fwd", "bwd", "fold"])
def test_every_candidate_config_is_correct_and_reproducible(which, monkeypatch):
    """The autotuner may pick any config of the default set on a card nobody has tuned: each one, alone, must give the right numbers (a
    tile that races or mis-masks shows here, not as a flaky run) and the same bits twice."""
    tuner = _kernels()[which]
    configs = list(tuner.configs)
    assert len(configs) >= 4
    for config in configs:
        monkeypatch.setattr(tuner, "configs", [config])
        monkeypatch.setattr(tuner, "cache", {})
        q, k, v, bias, keep, _, starts = _inputs(torch.bfloat16, "slide", 192, False, 333, 300, 6, 1, seed=3)
        go = torch.randn_like(q)
        runs = []
        for _ in range(2):
            for t in (q, k, v, bias):
                t.grad = None
            out = window_attention(q, k, v, bias, starts, keep, keys=192, window_half=64)
            out.backward(go)
            runs.append([out.detach().clone(), *(t.grad.clone() for t in (q, k, v, bias))])
        for first, other in zip(runs[0], runs[1], strict=True):
            assert torch.equal(first, other), config
        leaves = [t.detach().float().requires_grad_() for t in (q, k, v, bias)]
        want = window_attention_pytorch(*leaves, starts, keep, keys=192, window_half=64)
        want.backward(go.float())
        for name, got, ref in zip(("out", "dq", "dk", "dv", "dbias"), runs[0], (want.detach(), *(t.grad for t in leaves)), strict=True):
            assert _rel(got, ref) < BAND[torch.bfloat16], (config, name)
