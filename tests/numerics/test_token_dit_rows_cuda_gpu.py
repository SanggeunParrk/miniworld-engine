"""The CUDA token DiT row kernels (kernels/conditioned_transition/cuda) against the Triton ones they replace, on the
operand layouts the fused runner hands them (per-token tables as strided column slices of one GEMM output)."""
import pytest
import torch

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


def _rel(a, b):
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def _mods():
    from miniworld_engine.kernels.conditioned_transition import cuda as C
    from miniworld_engine.kernels.conditioned_transition.triton import token_dit_kernels as T
    return C, T


def _tables(L, nb, D, dtype, g):
    """g1 [L, nb, 4, D] and g2 [L, nb, 2, D] as the runner's conditioning GEMMs lay them out."""
    g1 = torch.randn(L, nb * 4 * D, device="cuda", generator=g).to(dtype).view(L, nb, 4, D)
    g2 = torch.randn(L, nb * 2 * D, device="cuda", generator=g).to(dtype).view(L, nb, 2, D)
    return g1, g2


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_adaln_and_resgate_rows_match_triton(dtype):
    C, T = _mods()
    g = torch.Generator(device="cuda").manual_seed(0)
    S, L, D, nb = 5, 384, 768, 2
    M = S * L
    g1, g2 = _tables(L, nb, D, dtype, g)
    x0 = torch.randn(M, D, device="cuda", generator=g) * 3 + 1
    y = torch.randn(M, D, device="cuda", generator=g).to(dtype)
    outs = {}
    for name, K in (("cuda", C), ("triton", T)):
        x = x0.clone()
        a = torch.empty(M, D, device="cuda", dtype=dtype)
        K.adaln_rows(x, g1[:, 0, 0], g1[:, 0, 1], a, L, 1e-5)
        b = torch.empty(M, D, device="cuda", dtype=dtype)
        K.resgate_adaln_rows(x, y, g2[:, 0, 0], g1[:, 0, 2], g1[:, 0, 3], b, L, 1e-5)
        K.resgate_adaln_rows(x, y, g2[:, 1, 1], None, None, y, L, 1e-5)        # last half-block: residual only
        outs[name] = (a, b, x)
    tol = 1e-2 if dtype is torch.bfloat16 else 1e-5
    for i, what in enumerate(("adaln", "resgate+adaln", "residual")):
        e = _rel(outs["cuda"][i], outs["triton"][i])
        assert e < (1e-6 if what == "residual" else tol), f"{what}: {e:.2e}"


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_gate_and_swiglu_rows_match_triton(dtype):
    C, T = _mods()
    g = torch.Generator(device="cuda").manual_seed(1)
    M, D, N = 1920, 768, 1536
    qkvg = torch.randn(M, 4 * D, device="cuda", generator=g).to(dtype)
    ab = torch.randn(M, 2 * N, device="cuda", generator=g).to(dtype)
    res = {}
    for name, K in (("cuda", C), ("triton", T)):
        o = torch.empty(M, D, device="cuda", dtype=dtype)
        K.gate_rows(qkvg[:, :D], qkvg[:, 3 * D:], o)
        h = torch.empty(M, N, device="cuda", dtype=dtype)
        K.swiglu_rows(ab, h)
        res[name] = (o, h)
    tol = 8e-3 if dtype is torch.bfloat16 else 1e-6
    for i, what in enumerate(("gate", "swiglu")):
        e = _rel(res["cuda"][i], res["triton"][i])
        assert e < tol, f"{what}: {e:.2e}"


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_pair_bias_all_matches_triton(dtype):
    C, T = _mods()
    g = torch.Generator(device="cuda").manual_seed(2)
    L, nb, H = 256, 3, 16
    z = torch.randn(L * L, 128, device="cuda", generator=g).to(dtype)
    wt = (torch.randn(128, nb * H, device="cuda", generator=g) * 128 ** -0.5).to(dtype)
    res = {}
    for name, K in (("cuda", C), ("triton", T)):
        out = torch.empty(nb * H, L, L, device="cuda", dtype=dtype)
        K.pair_bias_all(z, wt, out, L, 1e-5)
        res[name] = out
    e = _rel(res["cuda"], res["triton"])
    assert e < (1.5e-2 if dtype is torch.bfloat16 else 2e-3), f"pair bias: {e:.2e}"


@pytest.mark.skipif(not (torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 0)), reason="B200")
def test_runner_takes_the_cuda_rows_on_b200():
    from types import SimpleNamespace

    from miniworld_engine.kernels.conditioned_transition import cuda as C
    from miniworld_engine.kernels.conditioned_transition.triton.token_dit_runner import _row_kernels

    assert _row_kernels(torch.device("cuda")) is C
    del SimpleNamespace


@pytest.mark.skipif(not (torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 0)), reason="B200")
@pytest.mark.parametrize("M", [1920, 3840, 200])
def test_gemm_swiglu_sm100_matches_fp32(M, monkeypatch):
    """The sm_100a expand GEMM with the SwiGLU epilogue against silu(x Wa^T) (x Wb^T) in fp32 on the same bf16 operands;
    M = 200 exercises the partial row tile (TMA clips the store)."""
    from miniworld_engine.kernels.conditioned_transition.cuda import gemm_swiglu

    g = torch.Generator(device="cuda").manual_seed(M)
    K, H = 768, 1536
    x = torch.randn(M, K, device="cuda", generator=g).bfloat16()
    wa, wb = ((torch.randn(H, K, device="cuda", generator=g) * K ** -0.5).bfloat16() for _ in range(2))
    w_ab = torch.cat([wa, wb]).contiguous()
    monkeypatch.setenv("MINIWORLD_TOKEN_DIT_GEMM_SWIGLU", "1")
    assert gemm_swiglu.supported(x, w_ab)
    out = torch.full((M, H), float("nan"), device="cuda", dtype=torch.bfloat16)
    gemm_swiglu.GemmSwiglu(torch.cuda.current_device())(x, w_ab, out)
    a, b = x.float() @ wa.float().t(), x.float() @ wb.float().t()
    ref = torch.nn.functional.silu(a) * b
    assert torch.isfinite(out).all()
    e = _rel(out.float(), ref)
    assert e < 6e-3, f"gemm_swiglu: {e:.2e}"


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("nb", [1, 2])
def test_narrow_pair_bias_with_mask_matches_triton(dtype, nb):
    """The fused one-pass pair bias (nb * 16 <= 32 columns) with the key mask folded in, against the Triton projection +
    the runner's own -inf fill."""
    C, T = _mods()
    g = torch.Generator(device="cuda").manual_seed(3 + nb)
    L, H = 384, 16
    z = torch.randn(L * L, 128, device="cuda", generator=g).to(dtype)
    wt = (torch.randn(128, nb * H, device="cuda", generator=g) * 128 ** -0.5).to(dtype)
    mask = torch.rand(L, device="cuda", generator=g) > 0.2
    got = torch.empty(nb * H, L, L, device="cuda", dtype=dtype)
    C.pair_bias_all(z, wt, got, L, 1e-5, mask=mask)
    want = torch.empty_like(got)
    T.pair_bias_all(z, wt, want, L, 1e-5)
    want[:, :, ~mask] = float("-inf")
    assert torch.equal(torch.isinf(got), torch.isinf(want))
    fin = torch.isfinite(want)
    e = _rel(got[fin], want[fin])
    assert e < (1.5e-2 if dtype is torch.bfloat16 else 2e-3), f"pair bias: {e:.2e}"
