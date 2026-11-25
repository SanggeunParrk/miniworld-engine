"""B200 e4m3 expand + SwiGLU kernel (``gemm_swiglu_f8_sm100.cu``) and the quantising row passes on sm_100a: against a torch reference on the SAME e4m3 operands."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0), reason="B200 (sm_100a) only")

D, H = 768, 1536


def _quant_rows(x):
    s = x.float().abs().amax(1).clamp_min(1e-30) / 448.0
    return (x.float() / s[:, None]).to(torch.float8_e4m3fn), s


@pytest.mark.parametrize("M", [128, 640, 1280, 3840, 5120])
def test_expand_kernel_matches_reference(M):
    from foldjet_engine.kernels.conditioned_transition.cuda import token_dit_fp8 as f8

    torch.manual_seed(M)
    x = (torch.randn(M, D, device="cuda") * (1 + 3 * torch.rand(M, 1, device="cuda"))).to(torch.bfloat16)
    w = (torch.randn(2 * H, D, device="cuda") * 0.05).to(torch.bfloat16)
    x8, sx = f8.rows_q8(x)
    a8, sw = f8.quant_weight(w)
    h = torch.empty(M, H, device="cuda", dtype=torch.bfloat16)
    f8.ext().gemm_swiglu_f8(x8, a8, h, sx, sw, 148, 128, False, False)
    torch.cuda.synchronize()
    ab = (x8.float() * sx.view(-1, 1)) @ (a8.float() * sw[:, None]).t()
    a, b = ab[:, :H], ab[:, H:]
    ref = torch.nn.functional.silu(a) * b
    err = (h.float() - ref).abs().max() / ref.abs().max()
    assert err < 1e-2, float(err)  # bf16 rounding of h only (same e4m3 operands)
    assert torch.isfinite(h.float()).all()
