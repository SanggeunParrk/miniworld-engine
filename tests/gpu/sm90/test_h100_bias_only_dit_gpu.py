"""Bias-only token DiT on H100 (sm_90a): the wgmma cores, the bias gradient, the fp32 (TF32) step and the module dispatch.

Every check is against an fp64 / fp32-IEEE PyTorch oracle on non-trivial (randomised) weights, with the tolerance of the dtype
(bf16 outputs round at 2^-8; TF32 drops 13 mantissa bits of the operands). Kernels that carry a TMA epilogue or a software
pipeline are also checked for bitwise run-to-run determinism.
"""

import pytest
import torch

from miniworld_engine.kernels.bias_only_dit import cuda as C
from miniworld_engine.modules.bias_only_dit import BiasOnlyDiTBlock
from miniworld_engine.modules.exceptions import ImplementationType

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
        reason="Hopper (sm_90a) required",
    ),
]

H, DH = 16, 48
LENGTHS = [(128, 1), (256, 3), (384, 5), (640, 2), (768, 5), (768, 48)]


def rel(a, b):
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def probs(L, device="cuda"):
    return torch.softmax(torch.randn(H, L, L, device=device) * 2, -1)


# ------------------------------------------------------------------------------------------------------ PV + gate core
@pytest.mark.parametrize(("L", "S"), LENGTHS)
def test_pvx_gate_matches_fp64_and_is_deterministic(L, S):
    torch.manual_seed(L + S)
    vg = torch.randn(S * L, 2 * H * DH, device="cuda", dtype=torch.bfloat16)
    P = probs(L).to(torch.bfloat16).view(H * L, L)
    out = torch.empty(S * L, H * DH, device="cuda", dtype=torch.bfloat16)
    core = C.PvGateCore(0, nh=H, dh=DH)
    core(vg[:, : H * DH], P, out, S, g=vg[:, H * DH :])
    first = out.clone()
    core(vg[:, : H * DH], P, out, S, g=vg[:, H * DH :])
    assert torch.equal(out, first), "pvx is not bitwise repeatable"
    ref = torch.einsum(
        "hlk,skhd->slhd", P.view(H, L, L).double(), vg[:, : H * DH].double().view(S, L, H, DH)
    ) * torch.sigmoid(vg[:, H * DH :].double().view(S, L, H, DH))
    assert rel(out.view(S, L, H, DH), ref) < 4e-3


@pytest.mark.parametrize(("L", "S"), [(384, 5), (768, 48)])
def test_pvx_without_gate_is_the_plain_product(L, S):
    torch.manual_seed(5)
    v = torch.randn(S * L, H * DH, device="cuda", dtype=torch.bfloat16)
    P = probs(L).to(torch.bfloat16).view(H * L, L)
    out = torch.empty_like(v)
    C.PvGateCore(0, nh=H, dh=DH)(v, P, out, S, g=None)
    ref = torch.einsum("hlk,skhd->slhd", P.view(H, L, L).double(), v.double().view(S, L, H, DH))
    assert rel(out.view(S, L, H, DH), ref) < 4e-3


@pytest.mark.parametrize(("L", "S"), [(256, 3), (384, 5), (768, 5), (768, 48)])
def test_pvx32_fp32_matches_fp64(L, S):
    torch.manual_seed(L)
    v = torch.randn(S * L, H * DH, device="cuda")
    g = torch.randn(S * L, H * DH, device="cuda")
    P = probs(L).view(H * L, L).contiguous()
    out = torch.empty_like(v)
    C.PvGateCoreH100TF32(0, nh=H, dh=DH)(v.t().contiguous(), P, out, S, g)
    ref = torch.einsum("hlk,skhd->slhd", P.view(H, L, L).double(), v.double().view(S, L, H, DH)) * torch.sigmoid(
        g.double().view(S, L, H, DH)
    )
    assert rel(out.view(S, L, H, DH), ref) < 2e-3  # TF32 operands


# ------------------------------------------------------------------------------------------------------ bias gradient
@pytest.mark.parametrize(("L", "A"), [(256, 5), (384, 48), (640, 6), (768, 5), (768, 48)])
def test_dbias_matches_fp64_and_is_deterministic(L, A):
    torch.manual_seed(L * 7 + A)
    do = torch.randn(A * L, H * DH, device="cuda", dtype=torch.bfloat16)
    vg = torch.randn(A * L, 2 * H * DH, device="cuda", dtype=torch.bfloat16)
    P = probs(L).to(torch.bfloat16).view(H * L, L)
    dd = torch.randn(A, H, L, device="cuda")
    out = torch.empty(H * L, L, device="cuda", dtype=torch.bfloat16)
    kern = C.DpbKernel(0, nh=H, dh=DH)
    kern(do, vg[:, : H * DH], P, dd, out, A)
    first = out.clone()
    kern(do, vg[:, : H * DH], P, dd, out, A)
    assert torch.equal(out, first), "dpb is not bitwise repeatable"
    dp = torch.einsum("slhd,skhd->hlk", do.double().view(A, L, H, DH), vg[:, : H * DH].double().view(A, L, H, DH))
    ref = (dp - dd.double().sum(0).unsqueeze(-1)) * P.view(H, L, L).double()
    assert rel(out.view(H, L, L), ref) < 4e-3


# ------------------------------------------------------------------------------------------------------ the block
def make_block(dtype):
    torch.manual_seed(0)
    block = BiasOnlyDiTBlock(implementation=ImplementationType.TRITON)
    with torch.no_grad():
        for p in block.parameters():
            if p.ndim == 2:
                p.normal_(std=p.shape[1] ** -0.5)
            elif p.ndim == 1 and p.numel() > 1:
                p.add_(torch.randn_like(p) * 0.1)
    return block.cuda().float()


def reference(block, single, cond, pair, mask):
    ref = BiasOnlyDiTBlock(implementation=ImplementationType.PYTORCH).cuda().float()
    ref.load_state_dict(block.state_dict())
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        with torch.no_grad():
            return ref(single.float(), cond.float(), pair.float(), mask)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize(("L", "A", "shared"), [(128, 2, True), (256, 3, True), (384, 5, True), (512, 4, False), (640, 3, True), (768, 5, False)])
def test_block_inference_against_the_fp32_oracle(dtype, L, A, shared):
    block = make_block(dtype)
    torch.manual_seed(L)
    single = torch.randn(A, 1, L, 768, device="cuda", dtype=dtype)
    cond = torch.randn(1 if shared else A, 1, L, 384, device="cuda", dtype=dtype)
    cond = cond.expand(A, 1, L, 384) if shared else cond
    pair = torch.randn(1, L, L, 128, device="cuda", dtype=dtype)
    mask = torch.rand(1, L, device="cuda") > 0.2
    expected = reference(block, single, cond, pair, mask)
    torch.backends.cuda.matmul.allow_tf32 = True
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=dtype is torch.bfloat16):
        got = block(single, cond, pair, mask)
    # bf16 rounds at 2^-8 per tensor (the PyTorch bf16 block measures 3.9e-3 on this recipe); TF32 at 2^-11 (~3e-4 measured)
    assert rel(got, expected) < (6e-3 if dtype is torch.bfloat16 else 1.5e-3)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("L", [128, 256, 512, 640])
def test_block_training_gradients_against_the_fp32_oracle(dtype, L):
    A = 4
    block = make_block(dtype)
    ref = BiasOnlyDiTBlock(implementation=ImplementationType.PYTORCH).cuda().float()
    ref.load_state_dict(block.state_dict())
    torch.manual_seed(1)
    inputs = [
        torch.randn(A, 1, L, 768, device="cuda", dtype=dtype),
        torch.randn(A, 1, L, 384, device="cuda", dtype=dtype),
        torch.randn(1, L, L, 128, device="cuda", dtype=dtype),
    ]
    mask = torch.rand(1, L, device="cuda") > 0.15
    dy = torch.randn(A, 1, L, 768, device="cuda", dtype=dtype)

    def run(module, tf32, dt):
        torch.backends.cuda.matmul.allow_tf32 = tf32
        ins = [t.detach().to(dt).clone().requires_grad_() for t in inputs]
        module.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dt is torch.bfloat16):
            out = module(*ins, mask)
        out.backward(dy.to(dt))
        grads = {n: p.grad.clone() for n, p in module.named_parameters()}
        return out.detach(), [t.grad for t in ins], grads

    out_o, in_o, g_o = run(block, True, dtype)
    out_r, in_r, g_r = run(ref, False, torch.float32)
    tol = 8e-3 if dtype is torch.bfloat16 else 2.5e-3
    assert rel(out_o, out_r) < tol
    for a, b in zip(in_o, in_r, strict=True):
        assert rel(a, b) < 2 * tol
    worst = max(rel(g_o[n], g_r[n]) for n in g_r)
    assert worst < 4 * tol, f"worst parameter gradient error {worst:.2e}"
    # the step is bitwise repeatable
    out_o2, in_o2, g_o2 = run(block, True, dtype)
    assert torch.equal(out_o, out_o2)
    assert all(torch.equal(a, b) for a, b in zip(in_o, in_o2, strict=True))
