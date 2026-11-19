"""The bias-only token DiT keeps its pair-bias logits ``to_bias(LN(pair))`` and their softmax in fp32 on the sm_80 family (A100, RTX 5090), inference and
training (USER RULE 2026-11-19): the logits are 1e3 .. 2e5 with a key spread of ~500, bf16 (resolution 32 .. 1024) changes which tokens attend (peptide C-N
+0.1 A, FoldBench lDDT halved).

Huge-logit pair (mean |logit| ~4e3, max ~7.5e3; "spiky": channel-0 outliers of the pair, max ~9e3 and a much heavier tail of the gradients), the cuda backend against the same module in fp64 (PyTorch) on the SAME rounded
inputs, bf16 / master (fp32 weights under bf16 autocast) / fp32:

- the attention distribution of the block (max-row total variation of P vs the fp64 softmax of the exact logits) < 1e-3,
- eval output, and in training the gradients of the pair, ``to_bias.weight`` and ``ln_pair.weight``: relative error (max abs / max abs) < 5e-2.
"""

# pyright: reportCallIssue=false  (torch's Tensor.pow / mean overloads are not resolved by the checker in the error-measure helpers)
from __future__ import annotations

import copy
import typing

import pytest
import torch
import torch.nn.functional as F

pytestmark = [pytest.mark.gpu]

L, A, H, DP = 384, 2, 16, 128
#: max-row total variation of the softmax of ours against fp64: bf16 logits give 0.75 here; fp32 logits: the sequential FFMA accumulation of the 128-channel projection at logits ~4e3 (ulp 5e-4: a logit is off by ~1e-3, up to 1e-2) gives 1e-3 .. 2e-3 (a compensated TwoSum variant was no better at the maximum: the error floor is the first chunk and rsqrt)
TV_FP32 = {"huge": 3e-3, "spiky": 5e-3}
CASES = {"huge": (45.0, 870.0, 6.0, 0.0), "spiky": (45.0, 870.0, 6.0, 2000.0)}  # (to_bias weight scale, channel-0 weight, channel-0 pair offset, spike scale)


def _card() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() in ((8, 0), (12, 0))


@pytest.fixture(autouse=True)
def _need_card():
    if not _card():
        pytest.skip("sm_80 / sm_120 only")
    torch.backends.cuda.matmul.allow_tf32 = False


def _build(case: str):
    from miniworld_engine.modules.token_dit import TokenDiTMiniWorldBiasOnly

    sw, w0, off, spike = CASES[case]
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(0)
    m = TokenDiTMiniWorldBiasOnly(backend="pytorch").to(dev).float()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if n.endswith("to_bias.weight"):
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
    if spike:  # a few huge entries of channel 0: logits ~2e5 on those (query, key) pairs
        idx = torch.randint(0, L, (2, 400), device=dev, generator=g)
        pair[0, idx[0], idx[1], 0] += spike
    mask = torch.ones(1, L, dtype=torch.bool, device=dev)
    mask[:, L - 24 :] = False
    gy = torch.randn(A, 1, L, 768, device=dev, generator=g)
    return m, single, cond, pair, mask, gy


def _logits64(m, pair):
    a = m.attention
    ln = F.layer_norm(pair.double(), (DP,), None, None, 1e-5) * a.ln_pair.weight.double()
    return (ln @ a.to_bias.weight.double().t()).permute(0, 3, 1, 2)


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.double() - b.double()).abs().max() / b.double().abs().max()).item()


def _run(m, single, cond, pair, mask, gy, train: bool, autocast: bool = False):
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
        if not train:
            with torch.no_grad():
                return m(single, cond, pair, mask).detach(), None
        single = single.detach().clone().requires_grad_(True)
        pair = pair.detach().clone().requires_grad_(True)
        for p in m.parameters():
            p.grad = None
        y = m(single, cond, pair, mask)
        (y.float() * gy.float()).sum().backward()
    a = m.attention
    return y.detach(), (pair.grad, a.to_bias.weight.grad, a.ln_pair.weight.grad)


@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("dtype", ["bf16", "master", "fp32"])
def test_logits_softmax_fp32(case: str, dtype: str):
    m, single, cond, pair, mask, gy = _build(case)
    act = torch.float32 if dtype == "fp32" else torch.bfloat16
    mq: typing.Any = copy.deepcopy(m)
    if dtype == "bf16":
        mq = mq.bfloat16()
    ins = [t.to(act) for t in (single, cond, pair)]
    mq.set_backend("cuda")
    m64: typing.Any = copy.deepcopy(mq).double().set_backend("pytorch")
    i64 = [t.double() for t in ins]
    gy_q = gy.to(act)
    # 1. the attention distribution of the block, from the kernels' own P: ours (eval) vs the fp64 softmax of the exact logits
    from miniworld_engine.kernels.augmented_attention.cuda import sm80 as aa
    from miniworld_engine.kernels.integrations import bias_only_dit_sm80 as bo

    lg = _logits64(m64, i64[2])
    ref = torch.softmax(lg.masked_fill(~mask[:, None, None, :], float("-inf")), -1)
    wcm = (mq.attention.to_bias.weight.float() * mq.attention.ln_pair.weight.float()[None]).t().contiguous()
    bias = aa.pair_bias_generic(ins[2][0].contiguous(), wcm, None, L, 1e-5, oscale=1.0, out_dtype=torch.float32)
    assert bias.dtype is torch.float32
    # fp32 logits -> fp32 softmax (the rule's quantity), and the P the core reads (bf16: <= 2^-9 relative per entry, so a peaked row can move by ~1e-3)
    P32, _ = bo._softmax(bias, mask, False, torch.float32)
    tv32 = 0.5 * (P32.double().view(1, H, L, L) - ref).abs().sum(-1).max().item()
    tv5090 = None
    if dtype == "fp32" and torch.cuda.get_device_capability() == (12, 0):  # the RTX 5090's fp32 inference path: the pair bias and the softmax in one pass (pair_softmax_f32)
        from miniworld_engine.kernels.bias_only_dit.cuda import sm80 as core

        P5 = core.pair_softmax_f32(ins[2][0].contiguous(), wcm, mask, L, 1e-5, 0)
        tv5090 = 0.5 * (P5.double().view(1, H, L, L) - ref).abs().sum(-1).max().item()
    P, _ = bo._softmax(bias, mask, False, act)
    tv = 0.5 * (P.double().view(1, H, L, L) - ref).abs().sum(-1).max().item()
    # 2. the whole block, eval and training, against fp64; the reference fix (the PyTorch module with fp32 logits + softmax, P rounded to the dtype) sets the floor of
    #    the rest of the network's rounding (bf16: ~6e-2 of the pair gradient at these logits)
    ac = dtype == "master"
    y64, _ = _run(m64, *i64, mask, gy.double(), False)
    _, g64 = _run(m64, *i64, mask, gy.double(), True)
    y, _ = _run(mq, *ins, mask, gy_q, False, ac)
    yt, gr = _run(mq, *ins, mask, gy_q, True, ac)
    mf: typing.Any = copy.deepcopy(mq).set_backend("pytorch")

    def fixed(self, pair, mask, bias, dtype_):
        with torch.autocast("cuda", enabled=False):
            ln = F.layer_norm(pair.float(), (DP,), None, None, 1e-5) * self.ln_pair.weight.float()
            lg_ = (ln @ self.to_bias.weight.float().t()).permute(0, 3, 1, 2)
            if mask is not None:
                lg_ = lg_.masked_fill(~mask[:, None, None, :], float("-inf"))
            return torch.softmax(lg_, -1).to(dtype_)
    mf.attention._attention_weights = fixed.__get__(mf.attention)
    _, gf = _run(mf, *ins, mask, gy_q, True, ac)
    upd = ((y.double() - y64).pow(2).mean().sqrt() / (y64 - i64[0]).pow(2).mean().sqrt()).item()
    assert gr is not None and g64 is not None and gf is not None
    errs = [_rel(a, b) for a, b in zip(gr, g64, strict=True)]
    refs = [_rel(a, b) for a, b in zip(gf, g64, strict=True)]
    print(f"[{case} {dtype}] TV fp32 softmax {tv32:.3e} (pair_softmax_f32 {tv5090}), TV P({act}) {tv:.3e}; eval update rms-rel {upd:.3e}; train grad rel (pair, to_bias.weight, ln_pair.weight) ours "
          f"{errs[0]:.2e} {errs[1]:.2e} {errs[2]:.2e}, reference fix {refs[0]:.2e} {refs[1]:.2e} {refs[2]:.2e}; logits mean |.| {lg.abs().mean():.2e} max {lg.abs().max():.2e}")
    tv_max = TV_FP32[case]
    if tv5090 is not None:
        assert tv5090 < tv_max, f"{case} {dtype}: max-row TV of pair_softmax_f32 (RTX 5090 fp32 inference) {tv5090:.3e}"
    assert tv32 < tv_max, f"{case} {dtype}: max-row TV of the fp32 softmax {tv32:.3e} (bf16 logits: ~0.75)"
    assert tv < (2 * tv_max if act is torch.bfloat16 else tv_max), f"{case} {dtype}: max-row TV of P ({act}) {tv:.3e}"
    assert upd < 5e-2, f"{case} {dtype}: eval update rms-rel {upd:.3e}"
    assert torch.isfinite(yt).all() and all(torch.isfinite(g).all() for g in gr)
    # gradients (previous bf16 logits: O(1) in all three).  bf16 / master: the pair and the to_bias weight within the rounding floor of the bf16 network (the reference fix: 1.5e-1 at
    # 'huge'), the ln_pair weight -- a sum of 1e3-large terms of both signs -- at 'huge' within 5e-1 (5e-2 in fp32); fp32 (TF32 GEMMs, fp32 logits and bias gradient): all three < 5e-2
    lim = 5e-2 if dtype == "fp32" else 1.5e-1
    assert errs[0] < lim, f"{case} {dtype}: train grad pair rel {errs[0]:.3e} (reference fix {refs[0]:.3e})"
    assert errs[1] < lim, f"{case} {dtype}: train grad to_bias.weight rel {errs[1]:.3e} (reference fix {refs[1]:.3e})"
    if case == "huge" or dtype == "fp32":
        assert errs[2] < (5e-2 if dtype == "fp32" else 5e-1), f"{case} {dtype}: train grad ln_pair.weight rel {errs[2]:.3e} (reference fix {refs[2]:.3e})"
