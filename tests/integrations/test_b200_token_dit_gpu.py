"""The fused token DiT inference path on B200 (integrations/token_dit.py): the same contract as on H100, with the
bf16 step's attention on the sm_100a gated core (kernels/augmented_attention/cuda/sm100)."""

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


def test_token_dit_bf16_step_runs_the_sm100_core(monkeypatch):
    """The bf16 step on B200 launches the sm_100a gated core, and matches the Triton gated core it replaces."""
    from miniworld_engine.kernels.augmented_attention.cuda import sm100

    torch.manual_seed(812)
    m = randomize(DiTBlock(implementation=ImplementationType.MINIWORLD).cuda().to(torch.bfloat16)).eval()
    x = torch.randn(5, 1, 384, 768, device="cuda", dtype=torch.bfloat16)
    c = torch.randn(1, 1, 384, 384, device="cuda", dtype=torch.bfloat16).expand(5, 1, 384, 384)
    p = torch.randn(1, 384, 384, 128, device="cuda", dtype=torch.bfloat16)
    calls = []
    orig = sm100.GatedInferenceCore.__call__
    monkeypatch.setattr(sm100.GatedInferenceCore, "__call__", lambda self, *a: calls.append(1) or orig(self, *a))
    with torch.no_grad():
        got = m(x, c, p)
        assert calls, "the bf16 step did not take the sm_100a core"
        monkeypatch.setenv("MINIWORLD_AUGATTN_BF16_SM100", "0")
        tri = m(x, c, p)
    assert relative(got, tri) < 5e-3
