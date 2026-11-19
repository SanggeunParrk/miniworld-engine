"""The bias-only token DiT's pair-bias logits ``to_bias(ln_pair(pair))`` and their softmax are fp32 (user rule 2026-11-19), training and
inference, on every backend (PyTorch reference, Triton, H100 cuda).

Why: the logits are 1e3..2e5 with a key spread of ~500; bf16 resolves 32..1024 there, the attention picks other keys, the peptide C-N
bond grows 0.1 A and FoldBench lDDT halves. This is NOT the registry harness (tolerances there are PyTorch-relative): a pair whose
logits are huge (mean ~4e3, max ~7.5e3: "huge"; ~2e5: "extreme", where the fp32 rounding of the logits itself bounds the error) and the
fp64 module as reference, so a path that rounds the logits to bf16 / TF32 / a split product fails by O(1).

Checked against fp64, per backend and dtype (bf16 module, fp32 module): the attention distribution P (max-row total variation), the
block output ``y`` and, in training, the gradients of the pair, ``to_bias.weight`` and ``ln_pair.weight`` (max error of the max).
"""

# torch's Tensor.pow / mean overloads are not resolved by the checker in the error-measure helpers:
# pyright: reportCallIssue=false
from __future__ import annotations

import copy

import pytest
import torch
import torch.nn.functional as F

from miniworld_engine.modules.token_dit import TokenDiTMiniWorldBiasOnly, pair_bias_all

pytestmark = [pytest.mark.gpu]

requires_h100 = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0), reason="H100 (sm_90)")

L, A, H, DP = 256, 2, 16, 128
EPS = 1e-5
#: name -> (to_bias weight std, constant channel-0 weight, channel-0 offset of the pair, floor of the max-row TV bound of P)
CASES = {"huge": (45.0, 870.0, 6.0, 1e-3), "extreme": (900.0, 17000.0, 6.0, 5e-2), "wide": (45.0, 0.0, 0.0, 1e-3)}
GRAD_TOL = 5e-2  # gradients of pair / to_bias.weight / ln_pair.weight against fp64 (bf16 logits: O(1))
Y_TOL = {torch.bfloat16: 1e-2, torch.float32: 5e-3}  # rms-relative, floors of the allowed output error
#: P is rounded to the activation dtype for the fused block, so bf16 P carries its own 2^-9 rounding in the total variation
TV_SLACK = {torch.bfloat16: 4e-3, torch.float32: 0.0}


@pytest.fixture(autouse=True)
def _tf32_on():
    """The callers' default (``set_float32_matmul_precision("medium")``, TF32 allowed): the logits' products must not inherit it."""
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    yield
    torch.backends.cuda.matmul.allow_tf32 = old


def _build(case: str, dtype: torch.dtype, seed: int = 0):
    sw, w0, off, _ = CASES[case]
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(seed)
    m = TokenDiTMiniWorldBiasOnly(backend="pytorch").to(dev).float()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if n == "attention.to_bias.weight":
                w = torch.randn(p.shape, device=dev, generator=g) * sw
                w[:, 0] = w0
                p.copy_(w)
            elif n == "attention.to_out.weight":
                p.copy_(torch.randn(p.shape, device=dev, generator=g))
            elif n == "attention.to_scale.bias":
                p.fill_(3.0)
            elif "ln" in n and n.endswith("weight"):
                p.copy_(1 + 0.1 * torch.randn(p.shape, device=dev, generator=g))
            else:
                p.copy_(torch.randn(p.shape, device=dev, generator=g) * (0.05 if p.ndim == 2 else 0.1))
    single = torch.randn(A, 1, L, 768, device=dev, generator=g)
    cond = torch.randn(A, 1, L, 384, device=dev, generator=g)
    pair = torch.randn(1, L, L, DP, device=dev, generator=g)
    pair[..., 0] += off
    mask = torch.ones(1, L, dtype=torch.bool, device=dev)
    mask[:, L - 24 :] = False
    gy = torch.randn(A, 1, L, 768, device=dev, generator=g)
    mq = copy.deepcopy(m).bfloat16() if dtype is torch.bfloat16 else copy.deepcopy(m)
    ins = [t.to(dtype) for t in (single, cond, pair)]
    m64 = copy.deepcopy(mq).double().set_backend("pytorch")  # fp64 of the dtype-rounded values
    return mq, m64, ins, [t.double() for t in ins], mask, gy.to(dtype), gy.double()


def _logits64(m64, pair64) -> torch.Tensor:
    a = m64.attention
    ln = F.layer_norm(pair64, (DP,), None, None, EPS) * a.ln_pair.weight
    return (ln @ a.to_bias.weight.t()).permute(0, 3, 1, 2)


def _softmax(logits, mask) -> torch.Tensor:
    return torch.softmax(logits.masked_fill(~mask[:, None, None, :], float("-inf")), -1)


def _tv(p, p64) -> float:
    return 0.5 * (p.double() - p64).abs().sum(-1).max().item()


def _rel(a, b) -> float:
    return ((a.double() - b.double()).abs().max() / b.double().abs().max()).item()


def _run(m, single, cond, pair, mask, gy, mode, hoist=False):
    """y, and (pair, to_bias.weight, ln_pair.weight) gradients in training."""
    a = m.attention
    if mode == "eval":
        with torch.no_grad():
            if hoist:
                return m(single, cond, None, mask, bias=pair_bias_all([m], pair)[0]), None
            return m(single, cond, pair, mask), None
    single = single.detach().clone().requires_grad_(True)
    pair = pair.detach().clone().requires_grad_(True)
    for p in m.parameters():
        p.grad = None
    y = m(single, cond, None, mask, bias=pair_bias_all([m], pair)[0]) if hoist else m(single, cond, pair, mask)
    (y.float() * gy.float()).sum().backward()
    return y.detach(), (pair.grad, a.to_bias.weight.grad, a.ln_pair.weight.grad)


DTYPES = [torch.bfloat16, torch.float32]
IDS = ["bf16", "fp32"]


@requires_h100
@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("dtype", DTYPES, ids=IDS)
def test_attention_distribution_fp32(case: str, dtype: torch.dtype) -> None:
    """P = softmax(logits) of every path against fp64: max-row TV (a bf16 / TF32 logit gives 0.5..1)."""
    mq, m64, ins, i64, mask, _, _ = _build(case, dtype)
    p64 = _softmax(_logits64(m64, i64[2]), mask)
    # the fp32 floor of this data (to_bias products of ~1e4 summed in fp32): the PyTorch fp32 reference's own error, 1.2e-3 on "huge"; every
    # backend must stay within 1.5x of it (a bf16 / TF32 logit is 0.5..1)
    bound = max(CASES[case][3], 1.5 * _tv(mq.attention._attention_weights(ins[2], mask, None, torch.float32), p64))
    out = {}
    # PyTorch reference (module dtype P): fp32 P is the logits' own error
    out["pytorch"] = (mq.attention._attention_weights(ins[2], mask, None, torch.float32), bound)
    # PyTorch pair_bias_all fold
    mp = copy.deepcopy(mq).set_backend("pytorch")
    out["pytorch fold"] = (_softmax(pair_bias_all([mp], ins[2])[0], mask), bound)
    # Triton one-pass pair bias
    from miniworld_engine.kernels.bias_only_attention.triton.pair_bias import pair_bias_logits

    a = mq.attention
    out["triton"] = (_softmax(pair_bias_logits(ins[2], a.ln_pair.weight, a.to_bias.weight, EPS), mask), bound)
    # H100 cuda: the inference hoist (P of the fused block, rounded to the dtype) and the hoisted bias
    from miniworld_engine.kernels.bias_only_dit.cuda.runner import FusedBiasOnlyDiT

    mc = copy.deepcopy(mq).set_backend("cuda")
    P = FusedBiasOnlyDiT([mc], dtype=dtype).hoist(ins[2].contiguous(), mask).view(1, H, L, L)
    out["cuda runner.hoist"] = (P, bound + TV_SLACK[dtype])
    with torch.no_grad():
        b = pair_bias_all([mc], ins[2])[0]
    assert b.dtype is torch.float32, f"pair_bias_all(cuda) bias must be fp32, got {b.dtype}"
    out["cuda pair_bias_all"] = (_softmax(b, mask), bound)
    bad = {k: _tv(p, p64) for k, (p, tol) in out.items() if _tv(p, p64) > tol}
    print(case, dtype, {k: f"{_tv(p, p64):.2e}" for k, (p, _) in out.items()})
    assert not bad, f"{case} {dtype}: max-row TV of P against fp64 above its bound: {bad}"


@requires_h100
@pytest.mark.parametrize("case", ["huge", "wide"])
@pytest.mark.parametrize("dtype", DTYPES, ids=IDS)
@pytest.mark.parametrize("backend", ["pytorch", "triton", "cuda"])
@pytest.mark.parametrize("mode", ["eval", "train"])
@pytest.mark.parametrize("hoist", [False, True], ids=["pair", "hoist"])
def test_block_fp32_logits(case: str, dtype: torch.dtype, backend: str, mode: str, hoist: bool) -> None:
    """Block output (and, in training, the pair / to_bias.weight / ln_pair.weight gradients) against fp64."""
    if hoist and backend == "triton":
        pytest.skip("the Triton backend has no hoisted-bias path of its own (pair_bias_all is the PyTorch fold)")
    mq, m64, ins, i64, mask, gyq, gy64 = _build(case, dtype)
    y64, _ = _run(m64, *i64, mask, gy64, "eval")
    _, g64 = _run(m64, *i64, mask, gy64, "train")
    mm = copy.deepcopy(mq).set_backend(backend)
    y, gr = _run(mm, *ins, mask, gyq, mode, hoist)
    rms = lambda t: (((t.double() - y64) ** 2).mean().sqrt() / (y64**2).mean().sqrt()).item()
    err_y = rms(y)
    # the output's own error is the activation dtype's (the bf16 module's residual / gate rounding, the same on every backend): a backend
    # must stay within 3x of the PyTorch reference's (the fused bf16 block is 2.1x on this data, identically on 3.0.4-final: 3.217e-2 on 1.54e-2);
    # the logits / gradients below are what this file is about
    yp, _ = _run(copy.deepcopy(mq).set_backend("pytorch"), *ins, mask, gyq, "eval")
    assert err_y <= max(Y_TOL[dtype], 3.0 * rms(yp)), f"y rms-rel {err_y:.3e} (pytorch {rms(yp):.3e})"
    if mode == "train":
        errs = [_rel(a, b) for a, b in zip(gr, g64, strict=True)]
        print(case, dtype, backend, mode, hoist, "y", f"{err_y:.2e}", "grads(pair, Wb, gamma)", [f"{e:.2e}" for e in errs])
        assert all(e <= GRAD_TOL for e in errs), f"gradient errors (pair, to_bias.weight, ln_pair.weight) {errs} > {GRAD_TOL}"


@requires_h100
@pytest.mark.parametrize("dp", [128, 256])
@pytest.mark.parametrize("dtype", DTYPES, ids=IDS)
@pytest.mark.parametrize("case", ["huge", "extreme", "wide"])
def test_logits_tc_error_class_of_fma(case: str, dtype: torch.dtype, dp: int) -> None:
    """``hoist.logits_tc`` (the inference hoist's one-pass logits: LN0 in fp32 on chip, TF32 mma with the split fp32 operands, all four
    products, a fresh partial per k-step added in fp32) against fp64: its logits' max-abs error and the max-row TV of softmax stay within
    the error class of the exact fp32 FMA-pipe GEMM it replaces (max-abs within 2x, TV within 1.5x, plus the floors of the fp32 data)."""
    from miniworld_engine.kernels.bias_only_dit.cuda import hoist as HK

    sw, w0, off, tv_floor = CASES[case]
    g = torch.Generator(device="cuda").manual_seed(1)
    n = 256 * 256
    pair = torch.randn(n, dp, device="cuda", generator=g)
    pair[:, 0] += off
    pair = pair.to(dtype)
    gam = 1 + 0.1 * torch.randn(dp, device="cuda", generator=g)
    w = torch.randn(16, dp, device="cuda", generator=g) * sw
    w[:, 0] = w0
    wf = (w * gam).contiguous()
    ref = ((F.layer_norm(pair.double(), (dp,), None, None, EPS) * gam.double()) @ w.double().t()).t().contiguous()
    ln, _ = HK.ln0_fp32(pair, EPS)
    with torch.autocast("cuda", enabled=False):
        fma = HK.exact_logits(wf, ln)
    tc = torch.empty(16, n, device="cuda")
    HK.logits_tc(pair, wf, tc)

    def tv(lg: torch.Tensor) -> float:
        return (
            0.5
            * (torch.softmax(lg.double().view(16 * 256, 256), -1) - torch.softmax(ref.view(16 * 256, 256), -1)).abs().sum(-1).max().item()
        )

    e_fma, e_tc = (fma.double() - ref).abs().max().item(), (tc.double() - ref).abs().max().item()
    print(case, dtype, dp, f"max-abs fma {e_fma:.2e} tc {e_tc:.2e}; TV fma {tv(fma):.2e} tc {tv(tc):.2e}")
    assert e_tc <= 2.0 * e_fma, f"logits max-abs error {e_tc:.3e} > 2x the FMA path's {e_fma:.3e}"
    assert tv(tc) <= max(1.5 * tv(fma), tv_floor), f"max-row TV {tv(tc):.3e} above 1.5x the FMA path's {tv(fma):.3e}"
    # the row statistics (mean, rstd) the training backward reads: the same fp32 values as the LN0 rows kernel's (different summation order)
    pst = torch.empty(n, 2, device="cuda")
    HK.logits_tc(pair, wf, tc, pst)
    _, pst_ref = HK.ln0_fp32(pair, EPS, stats=True)
    assert pst_ref is not None
    torch.testing.assert_close(pst[:, 1], pst_ref[:, 1], rtol=1e-5, atol=0)
    torch.testing.assert_close(pst[:, 0], pst_ref[:, 0], rtol=1e-5, atol=1e-5)


@requires_h100
@pytest.mark.parametrize("dp", [128, 256])
@pytest.mark.parametrize("dtype", DTYPES, ids=IDS)
@pytest.mark.parametrize("off", [0.0, 6.0])
def test_hoist_bwd_tc_error_class_of_fma(dtype: torch.dtype, dp: int, off: float) -> None:
    """``hoist.hoist_bwd_tc`` (d LN0, the LN0 backward and dW' of one block in one pass, error-compensated TF32 mma) against fp64: d pair and dW'
    stay within 2x of the exact fp32 FMA / GEMM path's error (max-abs relative to the max)."""
    from miniworld_engine.kernels.bias_only_dit.cuda import hoist as HK
    from miniworld_engine.kernels.integrations.bias_only_dit_hoist import hoist_bwd_exact

    g = torch.Generator(device="cuda").manual_seed(2)
    n = 256 * 256
    x = torch.randn(n, dp, device="cuda", generator=g)
    x[:, 0] += off
    x = x.to(dtype)
    gam = 1 + 0.1 * torch.randn(dp, device="cuda", generator=g)
    wf = ((torch.randn(16, dp, device="cuda", generator=g) * 0.5) * gam).contiguous()
    db = torch.randn(16, n, device="cuda", generator=g).contiguous()
    xd = x.double().requires_grad_(True)
    y = F.layer_norm(xd, (dp,), None, None, EPS)
    (y @ wf.double().t()).backward(db.double().t())
    dx_ref, dw_ref = xd.grad, db.double() @ y.detach()
    assert dx_ref is not None
    old = HK.OPT["bwd_tc"]
    try:
        HK.OPT["bwd_tc"] = False
        dx0, dw0 = hoist_bwd_exact(x, wf, db, EPS, 64)
    finally:
        HK.OPT["bwd_tc"] = old
    dx1, dw1 = HK.hoist_bwd_tc(x, db, wf)

    def rel(a: torch.Tensor, b: torch.Tensor) -> float:
        return ((a.double() - b).abs().max() / b.abs().max()).item()

    assert rel(dx1, dx_ref) <= 2.0 * rel(dx0, dx_ref), f"d pair {rel(dx1, dx_ref):.3e} vs FMA path {rel(dx0, dx_ref):.3e}"
    assert rel(dw1, dw_ref) <= 2.0 * rel(dw0, dw_ref), f"dW' {rel(dw1, dw_ref):.3e} vs FMA path {rel(dw0, dw_ref):.3e}"
