"""SWA output fusion must preserve BF16 rounding and projection semantics."""

from dataclasses import asdict

import pytest
import torch

from miniworld_engine.modules.exceptions import ImplementationType

pytestmark = pytest.mark.gpu


@pytest.fixture
def gpu_settings():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from miniworld_engine import settings

    previous = settings.configure(
        run_autotune=False, capture=False, autotune_miss_cap=1
    )
    try:
        yield
    finally:
        settings.configure(**asdict(previous))


@pytest.mark.parametrize("width", [127, 128])
def test_fused_output_matches_rounded_reference(gpu_settings, width):
    from miniworld_engine.kernels.gated_projection.triton.swa import (
        swa_gate_out_inference,
    )

    torch.manual_seed(33)
    gate = torch.randn(2, 37, width, device="cuda", dtype=torch.bfloat16)
    out = torch.randn_like(gate)
    weight = torch.randn(130, width, device="cuda", dtype=gate.dtype)
    with torch.no_grad():
        actual = swa_gate_out_inference(gate, out, weight)
        expected = (torch.sigmoid(gate.float()) * out.float()).to(gate.dtype) @ weight.T
    relative = (actual.float() - expected.float()).norm() / expected.float().norm()
    assert bool(torch.isfinite(actual).all())
    assert relative < 0.013
    with pytest.raises(RuntimeError, match="gradients"):
        swa_gate_out_inference(gate.requires_grad_(), out, weight)


@pytest.mark.parametrize(
    "variant", ["plain", "training", "mp", "hook", "pytorch", "fp32"]
)
def test_projection_contract_is_preserved(gpu_settings, variant, monkeypatch):
    from miniworld_engine.kernels.gated_projection.triton import swa
    from miniworld_engine.modules.swa_atom_attention.module import (
        SWA3DRoPEAttention,
        build_attention_params,
    )

    dtype = torch.float32 if variant == "fp32" else torch.bfloat16
    model = (
        SWA3DRoPEAttention(
            128,
            4,
            implementation=ImplementationType.PYTORCH if variant == "pytorch" else ImplementationType.MINIWORLD,
            mp_full=variant == "mp",
        )
        .cuda()
        .to(dtype)
    )
    calls = []

    def fuse(gate, out, weight):
        calls.append("fused")
        return (torch.sigmoid(gate.float()) * out.float()).to(dtype) @ weight.T

    monkeypatch.setattr(swa, "swa_gate_out_inference", fuse)
    model.out_proj.register_forward_pre_hook(
        lambda *a: calls.append("projection")
    ) if variant == "hook" else None
    x = torch.randn(
        2, 64, 128, device="cuda", dtype=dtype, requires_grad=variant == "training"
    )
    valid = torch.ones(2, 64, device="cuda", dtype=torch.bool)
    angle = torch.randn(1, 64, 16, device="cuda")
    ap = build_attention_params(angle.cos(), angle.sin(), valid, num_aug=2)
    with torch.set_grad_enabled(variant == "training"):
        result = model(x, ap)
        if variant == "training":
            result.sum().backward()
            assert x.grad is not None
            assert x.grad is not None
            assert torch.isfinite(x.grad).all()
            assert model.out_proj.weight.grad is not None
    assert torch.isfinite(result).all()
    assert ("fused" in calls) == (variant == "plain")
    if variant == "hook":
        assert "projection" in calls
