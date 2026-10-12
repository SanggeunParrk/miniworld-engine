"""Fused Q/K must preserve strided inputs, partial RoPE and input gradients."""
from dataclasses import asdict

import pytest
import torch

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("shape", [(2, 37, 3, 32, 16), (2, 17, 2, 30, 11)])
@pytest.mark.parametrize("q_only", [False, True])
def test_strided_qk_forward_backward(dtype, shape, q_only):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from miniworld_engine import settings
    from miniworld_engine.kernels.rope.interface import qk_norm_rope_3d
    from miniworld_engine.modules.swa_atom_attention.module import apply_rotary_emb_3d

    previous = settings.configure(run_autotune=False, capture=False)
    try:
        torch.manual_seed(24)
        n, seq, heads, width, half = shape
        qkv = torch.randn(n, seq, 3, heads, width, device="cuda", dtype=dtype, requires_grad=True)
        q, k, _ = qkv.unbind(2)
        angles = torch.randn(1, seq, half * 2, device="cuda")
        cos, sin = angles[..., ::2], angles[..., 1::2]
        actual = qk_norm_rope_3d(q, k, cos, sin)
        expected = tuple(apply_rotary_emb_3d(torch.nn.functional.rms_norm(
            x.float(), (width,), eps=torch.finfo(torch.float32).eps).to(dtype), cos, sin)
            for x in (q, k))
        grads = tuple(torch.randn_like(x) for x in actual)
        count = 1 if q_only else 2
        da = torch.autograd.grad(actual[:count], qkv, grads[:count], retain_graph=True)[0]
        de = torch.autograd.grad(expected[:count], qkv, grads[:count])[0]
        for a, e in [*zip(actual, expected, strict=True), (da, de)]:
            assert bool(torch.isfinite(a).all())
            relative = (a.float() - e.float()).norm() / e.float().norm().clamp_min(1e-12)
            assert relative < (8e-3 if dtype == torch.bfloat16 else 3e-6)
        assert torch.count_nonzero(da[:, :, 2]) == 0
        if q_only:
            assert torch.count_nonzero(da[:, :, 1]) == 0
    finally:
        settings.configure(**asdict(previous))
