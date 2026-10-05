"""The fused bias-only token DiT inference path on A100 (integrations/bias_only_dit.py, kernels/bias_only_dit/cuda with the mma.sync core of cuda/sm80): against the PyTorch block in the same bf16
regime and against fp32, the attention core and the softmax against their references, the caches (weight pack, hoisted attention weights) against live changes, CUDA-graph capture."""

import pytest
import torch

from miniworld_engine import settings
from miniworld_engine.integrations import bias_only_dit
from miniworld_engine.kernels.bias_only_dit import cuda as C
from miniworld_engine.kernels.bias_only_dit import reference as R
from miniworld_engine.kernels.bias_only_dit.cuda import sm80
from miniworld_engine.modules.bias_only_dit import BiasOnlyDiTBlock
from miniworld_engine.modules.exceptions import ImplementationType

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 0), reason="A100 (sm_80)"),
]
LAYOUTS = [(16, 48), (24, 32), (12, 64), (16, 64)]


def relative(a, b):
    return float((a.detach().float() - b.detach().float()).norm() / b.detach().float().norm().clamp_min(1e-12))


def randomize(module):
    with torch.no_grad():
        for name, p in module.named_parameters():
            if p.ndim == 2:
                p.normal_(std=p.shape[-1] ** -0.5)
            elif "weight" in name:
                p.copy_(1 + 0.1 * torch.randn_like(p))
            else:
                p.normal_(std=0.05)
    return module


@pytest.fixture(autouse=True)
def policy():
    old = settings.configure(engine_backend="auto")
    try:
        yield
    finally:
        settings.configure(**vars(old))


def blocks(seed=811, n_head=16, d_head=None):
    torch.manual_seed(seed)
    ref = randomize(BiasOnlyDiTBlock(n_head=n_head, d_head=d_head, implementation=ImplementationType.PYTORCH)).cuda().eval()
    fast = BiasOnlyDiTBlock(n_head=n_head, d_head=d_head, implementation=ImplementationType.MINIWORLD).cuda()
    fast.load_state_dict(ref.state_dict())
    ref_bf = BiasOnlyDiTBlock(n_head=n_head, d_head=d_head, implementation=ImplementationType.PYTORCH).cuda()
    ref_bf.load_state_dict(ref.state_dict())
    return ref, fast.to(torch.bfloat16).eval(), ref_bf.to(torch.bfloat16).eval()


@pytest.mark.parametrize(("n_head", "d_head"), LAYOUTS)
@pytest.mark.parametrize(("L", "S"), [(128, 1), (384, 5), (256, 3), (1024, 2)])
@pytest.mark.parametrize("shared", [True, False])
@pytest.mark.parametrize("masked", [False, True])
def test_block_matches_the_pytorch_block(L, S, shared, masked, n_head, d_head):
    ref, fast, ref_bf = blocks(n_head=n_head, d_head=d_head)
    x = torch.randn(S, 1, L, 768, device="cuda")
    c = torch.randn(1, 1, L, 384, device="cuda").expand(S, 1, L, 384) if shared else torch.randn(S, 1, L, 384, device="cuda")
    p = torch.randn(1, L, L, 128, device="cuda")
    mask = (torch.rand(1, L, device="cuda") > 0.2) if masked else None
    torch.backends.cuda.matmul.allow_tf32 = False
    with torch.no_grad():
        want = ref(x, c, p, mask)
        args = (x.bfloat16(), c.bfloat16(), p.bfloat16(), mask)
        assert bias_only_dit.serves(fast, *args)
        got = fast(*args)
        base = ref_bf(*args)
    # against fp32: within the PyTorch bf16 block's own error
    assert relative(got, want) <= 1.05 * relative(base, want) + 1e-4, (relative(got, want), relative(base, want))
    assert relative(got, base) < 0.01


def test_caches_follow_live_weights_and_pair():
    _, fast, ref_bf = blocks(seed=5)
    L = 256
    x = torch.randn(5, 1, L, 768, device="cuda", dtype=torch.bfloat16)
    c = torch.randn(5, 1, L, 384, device="cuda", dtype=torch.bfloat16)
    p = torch.randn(1, L, L, 128, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        got = fast(x, c, p)
        assert relative(got, ref_bf(x, c, p)) < 0.01
        compiled = torch.compile(fast, fullgraph=True, options={"triton.cudagraphs": False})
        assert relative(compiled(x, c, p), got) < 1e-5
        # a weight updated in place bumps its version: the pack misses
        fast.transition.squeeze.weight.mul_(0.8)
        ref_bf.transition.squeeze.weight.mul_(0.8)
        new = fast(x, c, p)
        assert not torch.equal(got, new)
        assert relative(new, ref_bf(x, c, p)) < 0.01
        # a pair changed in place bumps its version: the hoisted attention weights miss
        p.add_(0.5 * torch.randn_like(p))
        assert relative(fast(x, c, p), ref_bf(x, c, p)) < 0.01


def test_declines_what_it_does_not_serve():
    _, fast, _ = blocks()
    x = torch.randn(5, 1, 384, 768, device="cuda", dtype=torch.bfloat16)
    c = torch.randn(5, 1, 384, 384, device="cuda", dtype=torch.bfloat16)
    p = torch.randn(1, 384, 384, 128, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        assert bias_only_dit.serves(fast, x, c, p)
        assert not bias_only_dit.serves(fast, x.float(), c.float(), p.float())             # bf16 only
        assert not bias_only_dit.serves(fast, x[:, :, :200], c[:, :, :200], p[:, :200, :200])   # L % 128
        assert not bias_only_dit.serves(fast, x, c, p, torch.ones(5, 384, device="cuda", dtype=torch.bool))  # a mask per sample
    assert not bias_only_dit.serves(fast, x, c, p)                                          # autograd on


@pytest.mark.parametrize(("n_head", "d_head"), LAYOUTS)
@pytest.mark.parametrize("L", [128, 256, 384, 512, 768, 1024])
@pytest.mark.parametrize("S", [1, 2, 5, 8])
def test_core_and_softmax_match_their_references(L, S, n_head, d_head):
    torch.manual_seed(L + S)
    M, DA = S * L, n_head * d_head
    vg = torch.randn(M, 2 * DA, device="cuda", dtype=torch.bfloat16)
    bias = 2 * torch.randn(n_head * L, L, device="cuda", dtype=torch.bfloat16)
    mask = torch.rand(L, device="cuda") > 0.3
    P = torch.empty_like(bias)
    C.softmax_rows(bias, P, mask)
    assert relative(P, R.attention_weights(bias, mask)) < 1e-3
    a = torch.empty(M, DA, device="cuda", dtype=torch.bfloat16)
    sm80.PvGate(n_head, d_head)(vg[:, :DA], P, a, S, g=vg[:, DA:])
    assert relative(a, R.pv_gate(vg, P, S)) < 1e-3
    for stages, sg in ((2, 1), (2, 3), (3, 2), (3, 4)):                                       # every schedule (a short last group of samples included) computes the same function
        b = torch.empty_like(a)
        sm80.PvGate(n_head, d_head)(vg[:, :DA], P, b, S, g=vg[:, DA:], stages=stages, sg=sg)
        assert relative(b, a) < 1e-6, (stages, sg)
    plain = torch.empty(M, DA, device="cuda", dtype=torch.bfloat16)                          # without the gate: the product alone
    sm80.PvGate(n_head, d_head)(vg[:, :DA], P, plain, S)
    want = torch.einsum("hij,sjhd->sihd", P.float().view(n_head, L, L), vg[:, :DA].float().view(S, L, n_head, d_head)).reshape(M, DA)
    assert relative(plain, want.bfloat16()) < 1e-3                                           # the same single rounding to bf16 as the kernel's


def test_fully_masked_rows_are_uniform():
    bias = torch.randn(32, 256, device="cuda", dtype=torch.bfloat16)
    P = torch.empty_like(bias)
    C.softmax_rows(bias, P, torch.zeros(256, device="cuda", dtype=torch.bool))
    assert torch.allclose(P.float(), torch.full_like(P.float(), 1 / 256), rtol=1e-2)


def test_inference_is_cuda_graph_capturable_and_deterministic():
    _, fast, _ = blocks(seed=9)
    L = 256
    x = torch.randn(5, 1, L, 768, device="cuda", dtype=torch.bfloat16)
    c = torch.randn(1, 1, L, 384, device="cuda", dtype=torch.bfloat16).expand(5, 1, L, 384)
    p = torch.randn(1, L, L, 128, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        eager = fast(x, c, p)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            fast(x, c, p)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = fast(x, c, p)
        graph.replay()
        torch.cuda.synchronize()
        again = fast(x, c, p)
    torch.testing.assert_close(out, eager, atol=0, rtol=0)
    torch.testing.assert_close(again, eager, atol=0, rtol=0)
