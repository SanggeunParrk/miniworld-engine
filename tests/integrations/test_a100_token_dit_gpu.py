"""The fused token DiT inference path on A100 (integrations/token_dit.py): the H100 runner -- Triton gated attention core and row
passes, cuBLAS GEMMs -- serves the block on sm_80 under the same contract as on H100 (inference only, bf16 or fp32, 16 heads x 48,
L a multiple of 128)."""

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
    if torch.cuda.get_device_capability() != (8, 0):
        pytest.skip("Ampere A100 (sm_80) required")
    old = settings.configure(engine_backend="auto")
    try:
        yield
    finally:
        settings.configure(**vars(old))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_token_dit_inference_live_inputs_weights_and_mask(dtype):
    torch.manual_seed(811)
    m = randomize(DiTBlock(implementation=ImplementationType.MINIWORLD).cuda().to(dtype)).eval()
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
        compiled = torch.compile(m, fullgraph=True, options={"triton.cudagraphs": False})
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
        # Inference-only contract: a replay reads the weights' pack and the pair bias it was captured with, and the live single / cond.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = m(x, c, p, mask)
        x.mul_(0.9)
        graph.replay()
        assert relative(captured, m(x, c, p, mask)) < 1e-5


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_token_dit_per_sample_conditioning(dtype):
    """A different conditioning per sample takes the fused step too (S L table rows), matches the reference, and each sample reads its
    own conditioning (sample i's output equals a one-sample call with cond i)."""
    torch.manual_seed(813)
    m = randomize(DiTBlock(implementation=ImplementationType.MINIWORLD).cuda().to(dtype)).eval()
    ref = DiTBlock(implementation=ImplementationType.PYTORCH).cuda().to(dtype).eval()
    ref.load_state_dict(m.state_dict())
    s, length = 4, 256
    x = torch.randn(s, 1, length, 768, device="cuda", dtype=dtype)
    c = torch.randn(s, 1, length, 384, device="cuda", dtype=dtype)
    p = torch.randn(1, length, length, 128, device="cuda", dtype=dtype)
    mask = torch.rand(1, length, device="cuda") > 0.2
    with torch.no_grad():
        assert token_dit.serves(m, x, c, p)
        got = m(x, c, p, mask)
        assert relative(got, ref(x, c, p, mask)) < 0.025
        for i in (0, s - 1):
            one = m(x[i:i + 1], c[i:i + 1], p, mask)
            assert relative(got[i:i + 1], one) < 1e-2


@pytest.mark.parametrize(("s", "length", "shared"), [(1, 128, True), (5, 256, True), (8, 384, True), (5, 512, False), (2, 640, True)])
def test_token_dit_samples_and_lengths(s, length, shared):
    """The sampling shapes: S samples with one shared conditioning (a stride-0 expand) or one per sample, L a multiple of 128; no mask."""
    torch.manual_seed(814)
    m = randomize(DiTBlock(implementation=ImplementationType.MINIWORLD).cuda().bfloat16()).eval()
    ref = DiTBlock(implementation=ImplementationType.PYTORCH).cuda().bfloat16().eval()
    ref.load_state_dict(m.state_dict())
    x = torch.randn(s, 1, length, 768, device="cuda", dtype=torch.bfloat16)
    c = (torch.randn(1, 1, length, 384, device="cuda", dtype=torch.bfloat16).expand(s, 1, length, 384) if shared
         else torch.randn(s, 1, length, 384, device="cuda", dtype=torch.bfloat16))
    p = torch.randn(1, length, length, 128, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        assert token_dit.serves(m, x, c, p)
        assert relative(m(x, c, p), ref(x, c, p)) < 0.025


def test_the_gate_serves_only_what_the_runner_was_built_for():
    """Everything the fused step cannot take keeps the module path: autograd, a length off the 128-row tiles, another head layout,
    QK-norm blocks (the QK-norm pass is a B200 row kernel), another conditioning width, another dtype."""
    m = randomize(DiTBlock(implementation=ImplementationType.MINIWORLD).cuda().bfloat16()).eval()
    x = torch.randn(2, 1, 256, 768, device="cuda", dtype=torch.bfloat16)
    c = torch.randn(2, 1, 256, 384, device="cuda", dtype=torch.bfloat16)
    p = torch.randn(1, 256, 256, 128, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        assert token_dit.serves(m, x, c, p)
        assert not token_dit.serves(m, x.half(), c, p)
        assert not token_dit.serves(m, x[:, :, :200], c[:, :, :200], p[:, :200, :200])
        assert not token_dit.serves(m, x, c[:, :, :, :256], p)
        assert not token_dit.serves(m, x, c[:1].expand(3, 1, 256, 384), p)
    assert not token_dit.serves(m, x, c, p), "autograd: the runner is inference-only"
    qk = DiTBlock(implementation=ImplementationType.MINIWORLD, use_qk_norm=True).cuda().bfloat16().eval()
    wide = DiTBlock(d_single=768, n_head=24, implementation=ImplementationType.MINIWORLD).cuda().bfloat16().eval()
    with torch.no_grad():
        assert not token_dit.serves(qk, x, c, p)
        assert not token_dit.serves(wide, x, c, p)


def test_fp32_runs_the_tf32_attention_core(monkeypatch):
    """fp32 inference takes the hand-CUDA TF32 core (``sm80.GatedInferenceCore`` on ``ops_tf32.cu``), not the Triton gated core, and stays within TF32's error of the fp32 PyTorch block."""
    from miniworld_engine.kernels.augmented_attention.cuda import sm80

    torch.manual_seed(7)
    m = randomize(DiTBlock(implementation=ImplementationType.MINIWORLD).cuda().float()).eval()
    ref = DiTBlock(implementation=ImplementationType.PYTORCH).cuda().float().eval()
    ref.load_state_dict(m.state_dict())
    x = torch.randn(3, 1, 256, 768, device="cuda")
    c = torch.randn(1, 1, 256, 384, device="cuda").expand(3, 1, 256, 384)
    p = torch.randn(1, 256, 256, 128, device="cuda")
    mask = torch.rand(1, 256, device="cuda") > 0.15
    kinds = []
    orig = sm80.GatedInferenceCore.__call__
    monkeypatch.setattr(sm80.GatedInferenceCore, "__call__", lambda self, *a, **k: kinds.append(self.dtype) or orig(self, *a, **k))
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        with torch.no_grad():
            want = ref(x, c, p, mask)
            got = m(x, c, p, mask)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32
    assert kinds
    assert all(kind is torch.float32 for kind in kinds)
    assert relative(got, want) < 6e-3
