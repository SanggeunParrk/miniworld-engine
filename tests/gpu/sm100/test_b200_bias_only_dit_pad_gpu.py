"""The bias-only token DiT inference on B200 at lengths off the cores' 128-row tile (integrations/bias_only_dit.py ``update``: single /
cond padded with zero rows, the pair or the hoisted bias with zero rows and columns, the key mask with False, the first L rows
returned). Against fp32 within the PyTorch block's own error in the same regime (bf16: 1.05x as test_b200_bias_only_dit_gpu.py;
fp32: the TF32 bar of test_b200_bias_only_dit_tf32_gpu.py, 1.5x + 2^-10), the P cache keyed on the caller's pair, and the CUDA
graph replay."""

import pytest
import torch
from portable_helpers import randomize, relative
from portable_helpers import tf32_matmul_flag as tf32
from settings_helpers import policy_blackwell_portable_off as policy  # noqa: F401

from miniworld_engine.integrations import bias_only_dit
from miniworld_engine.modules.bias_only_dit import BiasOnlyDiTBlock
from miniworld_engine.modules.exceptions import ImplementationType

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0), reason="Blackwell (sm_100) required"
    ),
]


def blocks(dtype, seed=811, n_head=16, d_head=None):
    torch.manual_seed(seed)
    ref = randomize(BiasOnlyDiTBlock(n_head=n_head, d_head=d_head, implementation=ImplementationType.PYTORCH)).cuda().eval()
    fast = BiasOnlyDiTBlock(n_head=n_head, d_head=d_head, implementation=ImplementationType.MINIWORLD).cuda()
    fast.load_state_dict(ref.state_dict())
    base = BiasOnlyDiTBlock(n_head=n_head, d_head=d_head, implementation=ImplementationType.PYTORCH).cuda()
    base.load_state_dict(ref.state_dict())
    return ref, fast.to(dtype).eval(), base.to(dtype).eval()


def data(L, S, shared, masked, seed=3):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(S, 1, L, 768, device="cuda", generator=g)
    c = (torch.randn(1, 1, L, 384, device="cuda", generator=g).expand(S, 1, L, 384) if shared
         else torch.randn(S, 1, L, 384, device="cuda", generator=g))
    p = torch.randn(1, L, L, 128, device="cuda", generator=g)
    mask = None
    if masked:
        mask = torch.rand(1, L, device="cuda", generator=g) > 0.2
        mask[0, 0] = True
    return x, c, p, mask


@pytest.mark.parametrize(("n_head", "d_head"), [(16, None), (24, 32), (12, 64)])
@pytest.mark.parametrize("L", [8, 100, 200, 333, 700, 1000])
@pytest.mark.parametrize("S", [1, 5])
@pytest.mark.parametrize("shared", [True, False], ids=["shared", "own"])
@pytest.mark.parametrize("masked", [False, True], ids=["nomask", "mask"])
def test_padded_bf16_block_matches_the_pytorch_block(L, S, shared, masked, n_head, d_head):
    ref, fast, base = blocks(torch.bfloat16, n_head=n_head, d_head=d_head)
    x, c, p, mask = data(L, S, shared, masked)
    with torch.no_grad(), tf32(False):
        want = ref(x, c, p, mask)
        args = (x.bfloat16(), c.bfloat16(), p.bfloat16(), mask)
        assert bias_only_dit.serves(fast, *args)
        got = fast(*args)
        bf = base(*args)
    assert got.shape == x.shape and torch.isfinite(got).all()
    assert relative(got, want) <= 1.05 * relative(bf, want) + 1e-4, (relative(got, want), relative(bf, want))


@pytest.mark.parametrize("L", [100, 333, 700])
@pytest.mark.parametrize("masked", [False, True], ids=["nomask", "mask"])
def test_padded_fp32_block_matches_the_pytorch_block(L, masked):
    ref, fast, _ = blocks(torch.float32)
    x, c, p, mask = data(L, 2, True, masked)
    with torch.no_grad():
        gold = ref.double()(x.double(), c.double(), p.double(), mask)
        with tf32(True):
            assert bias_only_dit.serves(fast, x, c, p, mask)
            got = fast(x, c, p, mask)
            base = ref.float()(x, c, p, mask)
    assert relative(got, gold) <= 1.5 * relative(base, gold) + 2**-10, (relative(got, gold), relative(base, gold))


def test_padded_hoisted_bias_matches_the_pair_call():
    """A block given its hoisted bias [1, H, L, L] at an L off the tile: padded with the mask, the same step as the pair call."""
    from miniworld_engine.modules.bias_only_dit.hoist import pair_bias_all

    _, fast, _ = blocks(torch.bfloat16)
    x, c, p, mask = data(200, 3, True, True)
    xb, cb, pb = x.bfloat16(), c.bfloat16(), p.bfloat16()
    with torch.no_grad():
        bias = pair_bias_all([fast], pb)[0]
        assert bias_only_dit.serves(fast, xb, cb, None, mask, bias=bias)
        got = fast(xb, cb, None, mask, bias=bias)
        want = fast(xb, cb, pb, mask)
    assert relative(got, want) < 1e-2


def test_padded_cache_follows_the_pair_and_replays_in_a_graph():
    """The attention weights stay cached on the caller's pair across padded calls (a changed pair misses), and a CUDA graph
    of the padded call replays the eager result bitwise."""
    _, fast, base = blocks(torch.bfloat16)
    x, c, p, mask = data(200, 2, True, True)
    xb, cb, pb = x.bfloat16(), c.bfloat16(), p.bfloat16()
    with torch.no_grad():
        y = fast(xb, cb, pb, mask)
        assert torch.equal(y, fast(xb, cb, pb, mask))
        pb.add_(0.5 * torch.randn_like(pb))
        y2 = fast(xb, cb, pb, mask)
        assert not torch.equal(y, y2)
        assert relative(y2, base(xb, cb, pb, mask)) < 1e-2
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            fast(xb, cb, pb, mask)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            yg = fast(xb, cb, pb, mask)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(yg, y2)
