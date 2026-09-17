"""Strided Transition inputs/weights/gradients must agree with dense equivalents."""
import pytest
import torch
import triton

pytestmark = pytest.mark.gpu


@pytest.fixture(params=[(16, 32, 32, 1), (32, 64, 64, 4), (16, 64, 32, 8)])
def tiles(request, monkeypatch):
    from miniworld_engine.kernels.transition.triton import fused, main
    # Different cuBLAS operand layouts may otherwise choose BF16 partial reductions.
    # Isolate addressing correctness from that independent precision setting.
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_bf16_reduced_precision_reduction", False)
    bm, bn, bk, group = request.param
    cfg = triton.Config({"BLOCK_M1": bm, "BLOCK_N": bn, "BLOCK_K": bk, "GROUP_M": group},
                        num_warps=4, num_stages=2)
    for kernel in (main.transition_fwd_kernel, fused._transition_expand_gatebwd_kernel):
        monkeypatch.setattr(kernel, "configs", [cfg])
        monkeypatch.setattr(kernel, "cache", {})


def _weight(nd, d, dtype, layout):
    if layout == "transpose":
        return (torch.randn(d, nd, device="cuda", dtype=dtype) / d**.5).T
    return (torch.randn(nd, d * 2, device="cuda", dtype=dtype) / d**.5)[:, ::2]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("shape", [(3, 5, 96), (2, 7, 128)])
def test_transition_strided_forward_and_all_gradients(tiles, dtype, shape):
    from miniworld_engine.kernels.transition.triton.main import triton_transition
    torch.manual_seed(37)
    d = shape[-1]
    # Non-ratio width exercises ND tails as well as row/K tails.
    nd = 2 * d + 17
    x = torch.randn(shape, device="cuda", dtype=dtype).transpose(0, 1)
    args = [x, _weight(nd, d, dtype, "transpose"), _weight(nd, d, dtype, "slice"),
            _weight(d, nd, dtype, "transpose")]
    args = [v.detach().requires_grad_() for v in args]
    dense = [v.detach().contiguous().requires_grad_() for v in args]
    go = torch.randn(shape, device="cuda", dtype=dtype).transpose(0, 1)
    actual = triton_transition(*args)
    expected = triton_transition(*dense)
    ga = torch.autograd.grad(actual, args, go)
    ge = torch.autograd.grad(expected, dense, go.contiguous())
    for label, a, e in zip(("y", "dx", "dwa", "dwb", "dws"), (actual, *ga), (expected, *ge), strict=True):
        torch.testing.assert_close(a, e, rtol=0.01 if dtype == torch.bfloat16 else 2e-4,
                                   atol=0.01 if dtype == torch.bfloat16 else 2e-5, msg=label)


@pytest.mark.parametrize("variant", ["saved_stacked", "saved_split", "saved_no_h",
                                    "recompute_stacked", "recompute_split"])
def test_gate_backward_uses_independent_input_and_output_strides(tiles, variant):
    from miniworld_engine.kernels.transition.triton import fused
    torch.manual_seed(51)
    m, d, nd = 35, 96, 211
    dtype = torch.bfloat16
    x = torch.randn(m, d * 2, device="cuda", dtype=dtype)[:, ::2]
    wa, wb = _weight(nd, d, dtype, "transpose"), _weight(nd, d, dtype, "slice")
    dh = torch.randn(m, nd * 2, device="cuda", dtype=dtype)[:, ::2]
    gamma = torch.randn(d, device="cuda", dtype=torch.float32)
    beta = torch.randn_like(gamma)
    mean = x.float().mean(-1)
    rstd = torch.rsqrt(x.float().var(-1, unbiased=False) + 1e-5)
    def run(x, wa, wb, dh):
        if variant == "saved_stacked":
            return fused._transition_expand_gatebwd_savedxn_stacked(x, wa, wb, dh)
        if variant.startswith("saved_"):
            return fused._transition_expand_gatebwd_savedxn(x, wa, wb, dh,
                                                           store_h=variant != "saved_no_h")
        f = (fused._transition_expand_gatebwd_stacked if variant.endswith("stacked")
             else fused._transition_expand_gatebwd)
        return f(x, rstd, mean * rstd, gamma, beta, wa, wb, dh)
    actual = run(x, wa, wb, dh)
    expected = run(x.contiguous(), wa.contiguous(), wb.contiguous(), dh.contiguous())
    for a, e in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, e, rtol=0.01, atol=0.01)
