"""RMSNorm validation must not quantize its independent reference a second time."""
import pytest
import torch

from miniworld_engine.autotune import run_all
from miniworld_engine.kernels.checks import rmsnorm as checks
from miniworld_engine.kernels.rmsnorm import interface
from miniworld_engine.kernels.rmsnorm.reference import rmsnorm_reference


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("backward", [False, True])
@pytest.mark.parametrize("corrupt", [False, True])
def test_rmsnorm_checker_keeps_reference_unrounded_and_rejects_errors(monkeypatch, dtype, backward, corrupt):
    torch.manual_seed(15)
    x = torch.randn(1, 257, 32, dtype=dtype)
    weight = torch.randn(32, dtype=dtype)
    monkeypatch.setattr(checks, "_D", 32)
    monkeypatch.setattr(checks, "_x", lambda d: x.clone())
    monkeypatch.setattr(checks, "vec", lambda d: weight.clone())

    def implementation(x, w, eps):
        y = rmsnorm_reference(x, w, eps)
        return y * 1.1 if corrupt else y

    # Exercise the actual checker with a CPU stand-in for the GPU kernel.
    monkeypatch.setattr(interface, "triton_rmsnorm", implementation)
    check = checks.rmsnorm_bwd_triton if backward else checks.rmsnorm_fwd_triton
    pairs = check()
    assert set(pairs) == ({"dx_aff", "dweight_aff", "dx_plain"} if backward else {"y_aff", "y_plain"})
    assert all(a.dtype == dtype and e.dtype == torch.float32 for a, e in pairs.values())
    if dtype == torch.bfloat16:
        assert all(not torch.equal(e, e.bfloat16().float()) for _, e in pairs.values())
    monkeypatch.setattr(run_all, "run_checker", lambda path: pairs)
    # BF16 output rounding requires 4e-3; FP32 retains its much tighter bands.
    band = .004 if dtype == torch.bfloat16 else (4e-6 if backward else 8e-7)
    passed, detail = run_all.check_one("rmsnorm", band)
    assert passed == (not corrupt), detail


def test_reference_leaf_promotion_is_required_for_gradients():
    torch.manual_seed(19)
    x = torch.randn(2, 32, dtype=torch.bfloat16, requires_grad=True)
    xf = x.detach().float().requires_grad_(True)
    dy = torch.randn_like(x)
    rmsnorm_reference(x).backward(dy)
    rmsnorm_reference(xf).backward(dy.float())
    assert x.grad is not None
    assert xf.grad is not None
    assert x.grad.dtype == torch.bfloat16
    assert xf.grad.dtype == torch.float32
    assert not torch.equal(x.grad.float(), xf.grad)


def test_correct_bf16_rounding_can_exceed_the_old_forward_band():
    # Exact BF16 inputs. Even the correctly rounded FP64 formula exceeds .0032;
    # asking the kernel to meet that band would require a less accurate output.
    x = torch.tensor([[1.0, .9921875]], dtype=torch.bfloat16)
    oracle = x.double() * torch.rsqrt(x.double().square().mean(-1, keepdim=True) + 1e-5)
    rounded = oracle.bfloat16().double()
    error = float((rounded - oracle).abs().max() / oracle.abs().max())
    assert .0032 < error < .004
