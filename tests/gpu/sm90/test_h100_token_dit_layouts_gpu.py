"""Token DiT on H100 at every head layout (16 x 48, 24 x 32, 12 x 64 at d 768; 16 x 64 at d 1024): inference and training gradients
against the PyTorch module in fp32 IEEE matmuls, on non-trivial random weights."""

import pytest
import torch

from miniworld_engine.integrations import token_dit as TD
from miniworld_engine.modules.dit import DiTBlock
from miniworld_engine.modules.exceptions import ImplementationType

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0), reason="Hopper (sm_90a) required"
    ),
]

LAYOUTS = [(16, 768), (24, 768), (12, 768), (16, 1024)]


def rel(a, b):
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def blocks(nh, d, qk=False):
    torch.manual_seed(0)
    fast = DiTBlock(d_single=d, d_cond=384, d_pair=128, n_head=nh, use_qk_norm=qk, implementation=ImplementationType.TRITON)
    with torch.no_grad():
        for p in fast.parameters():
            if p.ndim == 2:
                p.normal_(std=p.shape[1] ** -0.5)
            elif p.ndim == 1 and p.numel() > 1:
                p.add_(torch.randn_like(p) * 0.1)
    fast = fast.cuda().float()
    ref = DiTBlock(d_single=d, d_cond=384, d_pair=128, n_head=nh, use_qk_norm=qk, implementation=ImplementationType.PYTORCH).cuda().float()
    ref.load_state_dict(fast.state_dict())
    return fast, ref


def inputs(L, A, d, dtype, shared=True):
    torch.manual_seed(L + A)
    single = torch.randn(A, 1, L, d, device="cuda", dtype=dtype)
    cond = torch.randn(1 if shared else A, 1, L, 384, device="cuda", dtype=dtype)
    cond = cond.expand(A, 1, L, 384) if shared else cond
    pair = torch.randn(1, L, L, 128, device="cuda", dtype=dtype)
    mask = torch.rand(1, L, device="cuda") > 0.15
    return single, cond, pair, mask


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize(("nh", "d"), LAYOUTS)
@pytest.mark.parametrize(("L", "A", "shared"), [(256, 3, True), (384, 5, False), (768, 5, True)])
def test_inference_against_the_fp32_oracle(dtype, nh, d, L, A, shared):
    fast, ref = blocks(nh, d)
    single, cond, pair, mask = inputs(L, A, d, dtype, shared)
    torch.backends.cuda.matmul.allow_tf32 = False
    with torch.no_grad():
        want = ref(single.float(), cond.float(), pair.float(), mask)
    torch.backends.cuda.matmul.allow_tf32 = True
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=dtype is torch.bfloat16):
        assert TD.serves(fast, single, cond, pair), "the native path must serve this layout"
        got = fast(single, cond, pair, mask)
    assert rel(got, want) < (6e-3 if dtype is torch.bfloat16 else 1.5e-3)


@pytest.mark.parametrize(("nh", "d"), LAYOUTS)
@pytest.mark.parametrize("L", [256, 384])
def test_fp32_training_gradients_against_the_fp32_oracle(nh, d, L):
    A = 6
    fast, ref = blocks(nh, d)
    single, cond, pair, mask = inputs(L, A, d, torch.float32)
    dy = torch.randn_like(single)

    def run(module, tf32):
        torch.backends.cuda.matmul.allow_tf32 = tf32
        ins = [t.detach().clone().requires_grad_() for t in (single, cond, pair)]
        module.zero_grad(set_to_none=True)
        out = module(*ins, mask)
        out.backward(dy)
        return out.detach(), [t.grad for t in ins], {n: p.grad.clone() for n, p in module.named_parameters()}

    out_f, in_f, g_f = run(fast, True)
    out_r, in_r, g_r = run(ref, False)
    assert rel(out_f, out_r) < 2.5e-3
    for a, b in zip(in_f, in_r, strict=True):
        assert rel(a, b) < 5e-3
    worst = max(rel(g_f[n], g_r[n]) for n in g_r)
    assert worst < 1e-2, f"worst parameter gradient error {worst:.2e}"
    out_f2, in_f2, _ = run(fast, True)
    assert torch.equal(out_f, out_f2)
    assert all(torch.equal(a, b) for a, b in zip(in_f, in_f2, strict=True))
