"""The family is chosen by the CALLER: the hand-CUDA stages that used to hide in Triton files are reached through the routing layer, and not under
``force_portable_kernels``.

Default (non-forced) dispatch is what it was: the same stage is taken on the same card, now from ``_ops_routed`` / ``transition.interface`` /
``triangle_attention.interface`` / ``integrations.token_dit``. Forced, none of them is entered (the Triton files hold no way to).
"""

import pytest
import torch

from miniworld_engine import kernels, settings


@pytest.fixture(autouse=True)
def restore_policy():
    previous = settings.current()
    yield
    settings.configure(**vars(previous))


def _cap():
    return torch.cuda.get_device_capability()


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _masked_ln(width=128, rows=512):
    x = torch.randn(1, rows, width, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w = torch.randn(width, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    b = torch.randn(width, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    rs = (torch.rand(rows, device="cuda") > 0.2).to(torch.bfloat16)
    return x, w, b, rs


@pytest.mark.parametrize("forced", [False, True])
def test_masked_layernorm_backward_is_chosen_by_ops_layer_norm(monkeypatch, forced):
    from miniworld_engine import ops
    from miniworld_engine.kernels.layernorm import cuda as ln_cuda

    settings.configure(force_portable_kernels=forced)
    real = ln_cuda.layer_norm_bwd_cuda
    calls = []
    monkeypatch.setattr(ln_cuda, "layer_norm_bwd_cuda", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    x, w, b, rs = _masked_ln()
    y = ops.layer_norm(x, w, b, 1e-5, rs)
    y.sum().backward()
    ref = torch.nn.functional.layer_norm(x.detach().float(), (x.shape[-1],), w.float(), b.float(), 1e-5) * rs.float()[None, :, None]
    assert (y.float() - ref).abs().max() < 0.1
    assert bool(calls) is (not forced)


def test_triton_layernorm_file_is_pure_even_when_not_forced(monkeypatch):
    from miniworld_engine.kernels.layernorm import cuda as ln_cuda
    from miniworld_engine.kernels.layernorm.triton.main import triton_layernorm

    settings.configure(force_portable_kernels=False)
    monkeypatch.setattr(ln_cuda, "layer_norm_bwd_cuda", lambda *a, **k: (_ for _ in ()).throw(AssertionError("CUDA LN backward from a Triton file")))
    x, w, b, rs = _masked_ln()
    triton_layernorm(x, w, b, 1e-5, rs).sum().backward()


@pytest.mark.skipif(_cap()[0] != 9 if torch.cuda.is_available() else True, reason="H100")
@pytest.mark.parametrize("forced", [False, True])
def test_transition_training_stages_are_chosen_by_the_interface(monkeypatch, forced):
    from miniworld_engine.kernels.transition import cuda as tcuda

    settings.configure(force_portable_kernels=forced)
    seen = []
    for name in ("transition_b2b_fwd", "transition_expand_gatebwd_wgmma"):
        real = getattr(tcuda, name)
        monkeypatch.setattr(tcuda, name, (lambda real, name: lambda *a, **k: (seen.append(name), real(*a, **k))[1])(real, name))
    for width, want in ((128, {"transition_b2b_fwd"}), (256, {"transition_b2b_fwd", "transition_expand_gatebwd_wgmma"})):
        seen.clear()
        x = torch.randn(2, 128, width, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        ln_w, ln_b = (torch.ones(width, device="cuda", dtype=torch.bfloat16) for _ in range(2))
        wa, wb = (torch.randn(4 * width, width, device="cuda", dtype=torch.bfloat16) * 0.05 for _ in range(2))
        ws = torch.randn(width, 4 * width, device="cuda", dtype=torch.bfloat16) * 0.05
        kernels.triton_transition_fused(x, ln_w, ln_b, wa, wb, ws, 4).sum().backward()
        assert set(seen) == (set() if forced else want), (width, seen)


@pytest.mark.skipif(_cap() != (9, 0) if torch.cuda.is_available() else True, reason="H100")
@pytest.mark.parametrize("forced", [False, True])
def test_triangle_attention_cuda_forward_is_chosen_by_the_interface(monkeypatch, forced):
    from miniworld_engine.kernels.triangle_attention.cuda import training_forward

    if not training_forward._artifact_ready():
        pytest.skip("the prebuilt training forward is not available")
    settings.configure(force_portable_kernels=forced)
    real = training_forward.forward
    calls = []
    monkeypatch.setattr(training_forward, "forward", lambda *a: (calls.append(1), real(*a))[1])
    L = 384
    base = torch.randn(1, L, L, 4, 32, device="cuda", dtype=torch.bfloat16)
    q, k, v = (base.clone().permute(0, 3, 1, 2, 4).requires_grad_() for _ in range(3))
    bias = torch.randn(1, 4, L, L, device="cuda", dtype=torch.bfloat16)
    out = kernels.triton_triangle_attention_pair_bias(q, k, v, bias)
    out.sum().backward()
    assert bool(calls) is (not forced)


def test_the_token_dit_runner_class_is_picked_by_the_caller():
    from miniworld_engine.kernels.conditioned_transition.cuda.token_dit_composed import FusedTokenDiTComposed
    from miniworld_engine.kernels.conditioned_transition.triton.token_dit_runner import FusedTokenDiT

    assert issubclass(FusedTokenDiTComposed, FusedTokenDiT)
    # the Triton runner takes no CUDA hook at all
    dummy = object.__new__(FusedTokenDiT)
    assert FusedTokenDiT._rows(torch.device("cuda")) is __import__("miniworld_engine.kernels.conditioned_transition.triton.token_dit_kernels", fromlist=["x"])
    assert dummy._gemm_swiglu(torch.device("cuda"), None) is None
    assert dummy._cuda_core(torch.device("cuda"), 384, 768, 16) is None
    assert dummy._tf32_core(torch.device("cuda"), 384) is None
