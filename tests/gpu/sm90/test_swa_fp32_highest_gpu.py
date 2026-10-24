"""fp32 SWA atom DiT under torch's "highest" matmul precision on H100 / B200: the native H100 stages serve it (exact-fp32 cuBLAS
projections); a call they do not take runs the PyTorch statements, never the TF32 or Triton fp32 kernels
(docs/gpus/h100/swa_atom_dit/swa_atom_dit.md). The engine SWADiTBlock's refused-contract leg and SwiGLUFFN run the PyTorch statements
on these cards too, not the Triton RMS-adaLN / gated-residual / SwiGLU ops."""
from __future__ import annotations

import collections

import pytest
import torch

from miniworld_engine import settings
from miniworld_engine.architecture.blocks.swa_atom_transformer import SWAAtomTransformer
from miniworld_engine.modules.swa_atom_attention import build_attention_params

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0), reason="H100 only"),
]


@pytest.fixture
def watch(monkeypatch):
    """Counts Triton launches and fused-block calls; restores the matmul precision and the settings."""
    import triton.runtime.jit as jit

    seen = collections.Counter()
    original = jit.JITFunction.run

    def run(self, *args, **kwargs):
        seen[f"triton:{getattr(self.fn, '__name__', '?')}"] += 1
        return original(self, *args, **kwargs)

    fused = SWAAtomTransformer._forward_fused

    def forward_fused(self, *args, **kwargs):
        seen["fused"] += 1
        return fused(self, *args, **kwargs)

    monkeypatch.setattr(jit.JITFunction, "run", run)
    monkeypatch.setattr(SWAAtomTransformer, "_forward_fused", forward_fused)
    precision = torch.get_float32_matmul_precision()
    previous = settings.configure(triton_on_hopper_blackwell=False, force_triton_kernels=False)
    yield seen
    torch.set_float32_matmul_precision(precision)
    settings.configure(**{f: getattr(previous, f) for f in ("triton_on_hopper_blackwell", "force_triton_kernels", "swa_dit_h100_native")})


def _atom_inputs(model, dtype, grad, s=256, a=2):
    torch.manual_seed(1)
    q = torch.randn(a, s, 128, device="cuda", dtype=dtype, requires_grad=grad)
    c = torch.randn(a, s, 128, device="cuda", dtype=dtype)
    cos, sin = model.build_rope(torch.randn(1, s, 3, device="cuda"), torch.arange(s, device="cuda")[None])
    return q, c, build_attention_params(cos, sin, torch.ones(a, s, dtype=torch.bool, device="cuda"), a)


def _rel(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm())


@pytest.mark.parametrize("grad", [False, True], ids=["inference", "training"])
@pytest.mark.parametrize("variant", ["esmfold2", "global"])
def test_fp32_highest_native_stages_or_pytorch(watch, variant, grad):
    config = SWAAtomTransformer.Config(n_block=2, d_cond=128, fused_triton=True, swa_window_size=10**6 if variant == "global" else 128)
    torch.manual_seed(0)
    model = SWAAtomTransformer(config).cuda()
    for p in model.parameters():
        p.data.normal_(0, 0.05)
    q, c, params = _atom_inputs(model, torch.float32, grad)
    torch.set_float32_matmul_precision("highest")
    outs = {}
    for native in (None, False):  # the default native H100 stages, then the opt-out (which would be the TF32 / Triton kernels)
        settings.configure(swa_dit_h100_native=native)
        watch.clear()
        with torch.set_grad_enabled(grad):
            out = model(q, c, params)
            if grad:
                out.sum().backward()
        torch.cuda.synchronize()
        assert out.dtype == torch.float32
        assert torch.isfinite(out).all()
        triton = {k: v for k, v in watch.items() if k.startswith("triton:")}
        assert not triton, triton
        # global attention on H100 always takes the native stages (FlashAttention-4 between them), whatever the opt-out says
        assert (watch["fused"] > 0) == (native is None or variant == "global"), (native, dict(watch))
        outs[native] = out.detach()
    assert _rel(outs[None], outs[False]) < 2e-2


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
def test_engine_block_refused_leg_launches_no_triton(watch, dtype):
    from miniworld_engine.modules.exceptions import ImplementationType
    from miniworld_engine.modules.swa_dit import SWADiTBlock

    torch.manual_seed(0)
    block = SWADiTBlock(128, 128, 4, implementation=ImplementationType.MINIWORLD).cuda().to(dtype)
    block.attn.register_forward_hook(lambda *a: None)  # a hook refuses the fused block
    rope = SWAAtomTransformer(SWAAtomTransformer.Config(n_block=1)).cuda()
    q, c, params = _atom_inputs(rope, dtype, True)
    out = block(q, c, params)
    out.float().sum().backward()
    torch.cuda.synchronize()
    assert torch.isfinite(out).all()
    assert out.dtype == dtype
    assert not watch, dict(watch)
