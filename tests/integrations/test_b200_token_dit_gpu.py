"""The fused token DiT inference path on B200 (integrations/token_dit.py): the same contract as on H100, with the
step's attention on the sm_100a gated core (kernels/augmented_attention/cuda/sm100; bf16, or TF32 for fp32)."""

import pytest
import torch

from miniworld_engine import settings
from miniworld_engine.integrations import token_dit
from miniworld_engine.modules.dit import DiTBlock
from miniworld_engine.modules.exceptions import ImplementationType

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


def relative(a, b):
    return float(
        (a.detach().float() - b.detach().float()).norm()
        / b.detach().float().norm().clamp_min(1e-12)
    )


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
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("Blackwell (sm_100) required")
    old = settings.configure(engine_backend="auto")
    try:
        yield
    finally:
        settings.configure(**vars(old))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_token_dit_inference_live_inputs_weights_and_mask(dtype):
    torch.manual_seed(811)
    m = randomize(
        DiTBlock(implementation=ImplementationType.MINIWORLD).cuda().to(dtype)
    ).eval()
    ref = DiTBlock(implementation=ImplementationType.PYTORCH).cuda().to(dtype).eval()
    ref.load_state_dict(m.state_dict())
    x = torch.randn(1, 1, 384, 768, device="cuda", dtype=dtype).transpose(-1, -2).contiguous().transpose(-1, -2)
    c = torch.randn(1, 1, 384, 384, device="cuda", dtype=dtype)
    p = torch.randn(1, 384, 384, 256, device="cuda", dtype=dtype)[..., ::2]
    mask = torch.rand(1, 384, device="cuda") > 0.2
    with torch.no_grad():
        assert token_dit.serves(m, x, c, p)
        got = m(x, c, p, mask)
        want = ref(x, c, p, mask)
        assert relative(got, want) < 0.025
        compiled = torch.compile(
            m, fullgraph=True, options={"triton.cudagraphs": False}
        )
        assert relative(compiled(x, c, p, mask), got) < 1e-5
        old = got.clone()
        m.transition.squeeze.weight.mul_(0.8)
        new = m(x, c, p, mask)
        assert not torch.equal(old, new)
        # No stale pair-bias cache: a live changed pair must change the output.
        # (not p * 0.7: ln_pair is scale-invariant up to eps, and the fp32 path's TF32 projection rounds that away)
        assert not torch.equal(new, m(x, c, p * 0.7 + 0.3 * torch.randn_like(p), mask))
        # An in-place change of the pair bumps its version: the pair-bias cache misses.
        before = m(x, c, p, mask)
        p.add_(0.3 * torch.randn_like(p))
        assert not torch.equal(before, m(x, c, p, mask))
        # Inference-only contract: a replay reads the weights' pack and the pair bias it was captured with, and the
        # live single / cond.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = m(x, c, p, mask)
        x.mul_(.9)
        graph.replay()
        assert relative(captured, m(x, c, p, mask)) < 1e-5


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_token_dit_step_runs_the_sm100_core(dtype, monkeypatch):
    """The step on B200 launches the sm_100a gated core (bf16, or TF32 for fp32), and matches the Triton gated core it
    replaces (the runner cache is dropped between the two, so the second call really rebuilds without the sm100 core)."""
    from miniworld_engine.kernels.augmented_attention.cuda import sm100

    torch.manual_seed(812)
    m = randomize(DiTBlock(implementation=ImplementationType.MINIWORLD).cuda().to(dtype)).eval()
    x = torch.randn(5, 1, 384, 768, device="cuda", dtype=dtype)
    c = torch.randn(1, 1, 384, 384, device="cuda", dtype=dtype).expand(5, 1, 384, 384)
    p = torch.randn(1, 384, 384, 128, device="cuda", dtype=dtype)
    calls = []
    orig = sm100.GatedInferenceCore.__call__
    monkeypatch.setattr(sm100.GatedInferenceCore, "__call__", lambda self, *a: calls.append(1) or orig(self, *a))
    token_dit._RUNNERS.clear()
    with torch.no_grad():
        got = m(x, c, p)
        assert calls, "the step did not take the sm_100a core"
        n = len(calls)
        monkeypatch.setenv("MINIWORLD_AUGATTN_BF16_SM100", "0")
        token_dit._RUNNERS.clear()
        tri = m(x, c, p)
        assert len(calls) == n, "the switch did not take the core out"
    token_dit._RUNNERS.clear()
    assert relative(got, tri) < 5e-3


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_token_dit_per_sample_conditioning(dtype):
    """A different conditioning per sample takes the fused step too (S L table rows), matches the reference, and each
    sample reads its own conditioning (sample i's output equals a one-sample call with cond i)."""
    torch.manual_seed(813)
    m = randomize(DiTBlock(implementation=ImplementationType.MINIWORLD).cuda().to(dtype)).eval()
    ref = DiTBlock(implementation=ImplementationType.PYTORCH).cuda().to(dtype).eval()
    ref.load_state_dict(m.state_dict())
    S, L = 4, 256
    x = torch.randn(S, 1, L, 768, device="cuda", dtype=dtype)
    c = torch.randn(S, 1, L, 384, device="cuda", dtype=dtype)
    p = torch.randn(1, L, L, 128, device="cuda", dtype=dtype)
    mask = torch.rand(1, L, device="cuda") > 0.2
    with torch.no_grad():
        assert token_dit.serves(m, x, c, p)
        got = m(x, c, p, mask)
        assert relative(got, ref(x, c, p, mask)) < 0.025
        for i in (0, S - 1):
            one = m(x[i:i + 1], c[i:i + 1], p, mask)
            assert relative(got[i:i + 1], one) < 1e-2


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize(("S", "L", "shared"), [(5, 136, True), (4, 200, False), (3, 333, True), (5, 517, True), (2, 700, False)])
def test_token_dit_any_length(dtype, S, L, shared):
    """Lengths that are not a multiple of 128 (and 333 / 517, not of 8: padded to one) take the fused step on B200 and match the
    reference, key mask on, shared or per-sample conditioning; a CUDA-graph replay reads the live single."""
    torch.manual_seed(814)
    m = randomize(DiTBlock(implementation=ImplementationType.MINIWORLD).cuda().to(dtype)).eval()
    ref = DiTBlock(implementation=ImplementationType.PYTORCH).cuda().to(dtype).eval()
    ref.load_state_dict(m.state_dict())
    x = torch.randn(S, 1, L, 768, device="cuda", dtype=dtype)
    c = (torch.randn(1, 1, L, 384, device="cuda", dtype=dtype).expand(S, 1, L, 384) if shared
         else torch.randn(S, 1, L, 384, device="cuda", dtype=dtype))
    p = torch.randn(1, L, L, 128, device="cuda", dtype=dtype)
    mask = torch.rand(1, L, device="cuda") > 0.2
    token_dit._RUNNERS.clear()
    with torch.no_grad():
        assert token_dit.serves(m, x, c, p)
        got = m(x, c, p, mask)
        assert got.shape == x.shape
        assert relative(got, ref(x, c, p, mask)) < 0.025
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = m(x, c, p, mask)
        x.mul_(.9)
        graph.replay()
        assert relative(captured, m(x, c, p, mask)) < 1e-5
    token_dit._RUNNERS.clear()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize(("S", "L", "shared"), [(5, 384, True), (4, 200, False), (3, 333, True)])
def test_token_dit_qk_norm(dtype, S, L, shared, monkeypatch):
    """QK-norm blocks take the fused step on B200 (one in-place CUDA RMSNorm pass over the q / k heads, the logit scale in the
    q norm's weight), match the reference, and the norm weights are live (a change reaches the output)."""
    from miniworld_engine.kernels.conditioned_transition import cuda as cuda_rows

    torch.manual_seed(815)
    m = randomize(DiTBlock(use_qk_norm=True, implementation=ImplementationType.MINIWORLD).cuda().to(dtype)).eval()
    ref = DiTBlock(use_qk_norm=True, implementation=ImplementationType.PYTORCH).cuda().to(dtype).eval()
    ref.load_state_dict(m.state_dict())
    x = torch.randn(S, 1, L, 768, device="cuda", dtype=dtype)
    c = (torch.randn(1, 1, L, 384, device="cuda", dtype=dtype).expand(S, 1, L, 384) if shared
         else torch.randn(S, 1, L, 384, device="cuda", dtype=dtype))
    p = torch.randn(1, L, L, 128, device="cuda", dtype=dtype)
    mask = torch.rand(1, L, device="cuda") > 0.2
    calls = []
    orig = cuda_rows.qknorm_rows
    monkeypatch.setattr(cuda_rows, "qknorm_rows", lambda *a: calls.append(1) or orig(*a))
    token_dit._RUNNERS.clear()
    with torch.no_grad():
        assert token_dit.serves(m, x, c, p)
        got = m(x, c, p, mask)
        assert calls, "the QK-norm pass did not run"
        assert relative(got, ref(x, c, p, mask)) < 0.025
        m.attention.norm_key.weight.mul_(1.5)
        assert not torch.equal(got, m(x, c, p, mask)), "a changed k norm weight must change the output"
    token_dit._RUNNERS.clear()
