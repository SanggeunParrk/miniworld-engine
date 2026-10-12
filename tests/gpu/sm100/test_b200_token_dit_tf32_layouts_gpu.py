"""fp32 (TF32 tensor cores) token DiT on B200 at every head layout: 16 x 48, 24 x 32, 12 x 64 (d 768), 16 x 64 (d 1024).

  * the TF32 attention cores built per layout (``attn_fwd_tf32``, ``attn_dkv_tf32`` + ``attn_dqb_tf32``, ``attn_inf_tf32``;
    ``sm100._tdit_defs``) against fp64;
  * the fused inference step (integrations/token_dit.py) and the fused training block (integrations/token_dit_train.py) in fp32
    against the fp32 IEEE PyTorch block, no worse than the engine's own module path in the same TF32 regime, and the path
    actually taken.
"""

import math

import pytest
import torch

from miniworld_engine import settings
from miniworld_engine.modules.dit import DiTBlock
from miniworld_engine.modules.exceptions import ImplementationType

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]
HEADS = [(16, 48), (24, 32), (12, 64), (16, 64)]                 # (heads, head dim)
LAYOUTS = [(h, h * dh) for h, dh in HEADS]                       # (heads, d_single)


@pytest.fixture(autouse=True)
def policy():
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("Blackwell (sm_100) required")
    old = settings.configure(engine_backend="auto")
    try:
        yield
    finally:
        settings.configure(**vars(old))


def _rel(a, b):
    a, b = a.detach().double(), b.detach().double()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def _ref(q, k, v, bias, mask=None):
    """softmax(q k^T / sqrt(dh) + bias) v per sample and head in fp64; q, k, v [A, L, H, dh], bias [H, L, L]."""
    dh = q.shape[-1]
    s = torch.einsum("alhd,amhd->ahlm", q.double(), k.double()) / math.sqrt(dh) + bias.double()
    if mask is not None:
        s = s.masked_fill(~mask, float("-inf"))
    return torch.einsum("ahlm,amhd->alhd", s.softmax(-1), v.double()), s


# ------------------------------------------------------------------------------------------------------------- cores
@pytest.mark.parametrize("heads", HEADS)
@pytest.mark.parametrize("L", [128, 384])
@pytest.mark.parametrize("masked", [False, True])
def test_training_core_tf32(heads, L, masked):
    """attn_fwd_tf32, then attn_dkv_tf32 + attn_dqb_tf32, built for the layout: O, LSE, dq, dk, dv, dbias against fp64."""
    from miniworld_engine.kernels.augmented_attention.cuda import sm100

    H, dh = heads
    A, W = 4, H * dh
    g = torch.Generator(device="cuda").manual_seed(L + H + masked)
    q, k, v = (torch.randn(A, L, H, dh, device="cuda", generator=g) for _ in range(3))
    bias = torch.randn(H, L, L, device="cuda", generator=g)
    if masked:
        keep = torch.rand(L, device="cuda", generator=g) > 0.2
        keep[0] = True
        bias[:, :, ~keep] = float("-inf")
    q2, k2, v2 = (t.reshape(A * L, W).contiguous() for t in (q, k, v))
    O, LSE = sm100.forward_tf32(q2, k2, v2, bias.contiguous(), A, L, H, dh)
    qd, kd, vd = (t.double().requires_grad_() for t in (q, k, v))
    bd = bias.double().requires_grad_()
    want, s = _ref(qd, kd, vd, bd)
    # LSE: a TF32 MMA keeps 10 of q's and k's 23 mantissa bits, so each product q_i k_i moves by at most 2 * 2^-10 of its
    # magnitude and a logit s_j by at most 2^-9 sum_i |q_i k_ij| / sqrt(dh); LSE is 1-Lipschitz in the max norm, so a row's
    # LSE moves by at most that bound's max over the keys (/ ln 2 in log2 units). A fixed absolute bound does not hold across
    # layouts: the shift is systematic (truncation), it scales with the logits and the max runs over A H L rows (24 x 32,
    # L384: 6.1e-3 measured against the 5e-3 that held at 16 x 48).
    lse_err = (LSE.double() - torch.logsumexp(s, -1) / math.log(2)).abs()                   # [A, H, L]
    qk_abs = torch.einsum("alhd,amhd->ahlm", q.double().abs(), k.double().abs()).amax(-1)    # max_j sum_i |q_i k_ij|
    lse_bound = qk_abs * 2.0 ** -9 / math.sqrt(dh) / math.log(2) + 1e-4                      # + fp32 accumulation / ex2.approx
    dO = torch.randn(A, L, H, dh, device="cuda", generator=g)
    Dd = (dO * O.view(A, L, H, dh)).sum(-1).permute(0, 2, 1).contiguous()                 # [A, H, L]
    DQ, DK, DV, DB = sm100.backward_tf32(q2, k2, v2, dO.reshape(A * L, W).contiguous(), bias.contiguous(), LSE, Dd, A, L, H, dh)
    torch.cuda.synchronize()
    want.backward(dO.double())
    bgrad = bd.grad.masked_fill(torch.isinf(bias.double()), 0.0)
    errs = {"O": _rel(O.view(A, L, H, dh), want), "dq": _rel(DQ.view(A, L, H, dh), qd.grad),
            "dk": _rel(DK.view(A, L, H, dh), kd.grad), "dv": _rel(DV.view(A, L, H, dh), vd.grad),
            "dbias": _rel(DB.masked_fill(torch.isinf(bias), 0.0), bgrad)}
    print(f"{H} x {dh} L{L} masked={masked}:", {k: f"{v:.2e}" for k, v in errs.items()},
          f"LSE max {float(lse_err.max()):.2e}, max err / bound {float((lse_err / lse_bound).max()):.2f}")
    assert bool((lse_err <= lse_bound).all()), float((lse_err / lse_bound).max())
    assert errs["O"] < 2e-3, errs
    for name in ("dq", "dk", "dv", "dbias"):
        assert errs[name] < 3e-3, errs


@pytest.mark.parametrize("heads", HEADS)
@pytest.mark.parametrize("L", [128, 200, 384])
def test_inference_core_tf32(heads, L):
    """attn_inf_tf32 built for the layout: q | k | g [S L, 3 W] with pre-scaled logits, v^T [W, S L], block 1 of a two-block
    hoisted bias; sigmoid(g) * o written over q, against fp64."""
    from miniworld_engine.kernels.augmented_attention.cuda import sm100

    H, dh = heads
    S, W, nb = 3, H * dh, 2
    c = math.log2(math.e)
    g = torch.Generator(device="cuda").manual_seed(L * H + 1)
    q, k, v, gt = (torch.randn(S, L, H, dh, device="cuda", generator=g) for _ in range(4))
    bias = torch.randn(nb * H, L, L, device="cuda", generator=g)
    mask = torch.rand(L, device="cuda", generator=g) > 0.2
    mask[0] = True
    qkg = torch.cat([(q * (c / math.sqrt(dh))).reshape(S * L, W), k.reshape(S * L, W), gt.reshape(S * L, W)], 1).contiguous()
    vt = v.reshape(S * L, W).t().contiguous()                                              # [W, S L]
    b2 = (bias * c).masked_fill(~mask, float("-inf")).contiguous()
    core = sm100.GatedInferenceCore(torch.cuda.current_device(), torch.float32, H, dh)
    core(qkg, b2, 1, S, vt)
    torch.cuda.synchronize()
    o, _ = _ref(q, k, v, bias[H:], mask)
    want = torch.sigmoid(gt.double()) * o
    assert _rel(qkg[:, :W].view(S, L, H, dh), want) < 2e-3, _rel(qkg[:, :W].view(S, L, H, dh), want)


# ------------------------------------------------------------------------------------------------------------- blocks
def _randomize(m):
    with torch.no_grad():
        for name, p in m.named_parameters():
            if p.ndim == 2:
                p.normal_(std=p.shape[-1] ** -0.5 * 0.5)
            elif "weight" in name:
                p.copy_(1 + 0.1 * torch.randn_like(p))
            else:
                p.normal_(std=0.05)
    return m


def _pair_of_blocks(layout, qk, seed):
    H, d = layout
    torch.manual_seed(seed)
    ref = _randomize(DiTBlock(d_single=d, n_head=H, use_qk_norm=qk, implementation=ImplementationType.PYTORCH).cuda())
    eng = DiTBlock(d_single=d, n_head=H, use_qk_norm=qk, implementation=ImplementationType.MINIWORLD).cuda()
    eng.load_state_dict(ref.state_dict())
    return ref, eng


class _TF32:
    """allow_tf32 for the module path's GEMMs (the fused fp32 path forces TF32 on its own), restored after."""

    def __init__(self, on):
        self.on = on

    def __enter__(self):
        self.old = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = self.on

    def __exit__(self, *exc):
        torch.backends.cuda.matmul.allow_tf32 = self.old


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize(("S", "L", "qk"), [(5, 384, False), (4, 200, True), (3, 333, False)])
def test_fused_inference_fp32(layout, S, L, qk, monkeypatch):
    from miniworld_engine.integrations import token_dit

    H, d = layout
    ref, eng = _pair_of_blocks(layout, qk, seed=11)
    ref.eval(); eng.eval()
    g = torch.Generator(device="cuda").manual_seed(12)
    x = torch.randn(S, 1, L, d, device="cuda", generator=g)
    c = torch.randn(1, 1, L, 384, device="cuda", generator=g).expand(S, 1, L, 384)
    p = torch.randn(1, L, L, 128, device="cuda", generator=g)
    mask = torch.rand(1, L, device="cuda", generator=g) > 0.2
    token_dit._RUNNERS.clear()
    with torch.no_grad():
        assert token_dit.serves(eng, x, c, p)
        with _TF32(False):
            truth = ref(x, c, p, mask)
        got = eng(x, c, p, mask)
        monkeypatch.setattr(token_dit, "serves", lambda *a, **k: False)
        with _TF32(True):
            module = eng(x, c, p, mask)
    token_dit._RUNNERS.clear()
    assert got.dtype is torch.float32 and torch.isfinite(got).all()
    ef, em = _rel(got, truth), _rel(module, truth)
    assert ef < 1.5 * em + 2e-3, f"fused {ef:.2e} vs module path {em:.2e}"


def _run(m, single, cond, pair, mask, w):
    ins = [t.detach().clone().requires_grad_() for t in (single, cond, pair)]
    out = m(*ins, mask)
    (out.float() * w).sum().backward()
    res = {"out": out.detach().float(), "dsingle": ins[0].grad.float(), "dcond": ins[1].grad.float(), "dpair": ins[2].grad.float()}
    res.update({n: p.grad.float() for n, p in m.named_parameters() if p.grad is not None})
    m.zero_grad(set_to_none=True)
    return res


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize(("A", "L", "qk"), [(4, 256, True), (2, 384, False), (3, 200, True)])
def test_fused_training_fp32(layout, A, L, qk, monkeypatch):
    """The fused fp32 training block (TF32 kernels; L = 200 and odd A through the internal padding) against the fp32 IEEE
    block: output and every gradient within 1.5x the module path's error in the same TF32 regime."""
    from miniworld_engine.integrations import token_dit_train

    H, d = layout
    ref, eng = _pair_of_blocks(layout, qk, seed=21)
    g = torch.Generator(device="cuda").manual_seed(22)
    single = torch.randn(A, 1, L, d, device="cuda", generator=g)
    cond = torch.randn(A, 1, L, 384, device="cuda", generator=g)
    pair = torch.randn(1, L, L, 128, device="cuda", generator=g)
    mask = torch.rand(1, L, device="cuda", generator=g) > 0.15
    w = torch.randn(A, 1, L, d, device="cuda", generator=g)
    with _TF32(False):
        truth = _run(ref, single, cond, pair, mask, w)
    calls = []
    orig = token_dit_train._Block.apply
    monkeypatch.setattr(token_dit_train._Block, "apply", lambda *a: calls.append(a[7]) or orig(*a))
    with _TF32(True):
        fused = _run(eng, single, cond, pair, mask, w)
        assert calls and all(calls), "the fp32 training call did not take the fused fp32 path"
        monkeypatch.setenv("MINIWORLD_TOKEN_DIT_TRAIN", "0")
        module = _run(eng, single, cond, pair, mask, w)
    assert set(fused) == set(truth), sorted(set(truth) ^ set(fused))
    worst = []
    for k in truth:
        ef, em = _rel(fused[k], truth[k]), _rel(module[k], truth[k])
        worst.append((ef / max(em, 1e-6), k, ef, em))
        assert torch.isfinite(fused[k]).all(), k
        assert ef < 1.5 * em + 3e-3, f"{k}: fused {ef:.2e} vs module path {em:.2e}"
    print("worst ratios:", sorted(worst, reverse=True)[:4])


@pytest.mark.parametrize("layout", [(16, 768), (24, 768)])
def test_fused_training_fp32_under_bf16_autocast(layout, monkeypatch):
    """bf16-mixed training runs the model under bf16 autocast: an fp32 block still takes the fp32 (TF32) kernels -- the
    precision follows the tensors -- with the same result as without autocast."""
    from miniworld_engine.integrations import token_dit_train

    H, d = layout
    _, eng = _pair_of_blocks(layout, True, seed=31)
    A, L = 4, 256
    g = torch.Generator(device="cuda").manual_seed(32)
    single = torch.randn(A, 1, L, d, device="cuda", generator=g)
    cond = torch.randn(A, 1, L, 384, device="cuda", generator=g)
    pair = torch.randn(1, L, L, 128, device="cuda", generator=g)
    w = torch.randn(A, 1, L, d, device="cuda", generator=g)
    plain = _run(eng, single, cond, pair, None, w)
    calls = []
    orig = token_dit_train._Block.apply
    monkeypatch.setattr(token_dit_train._Block, "apply", lambda *a: calls.append(a[7]) or orig(*a))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        amp = _run(eng, single, cond, pair, None, w)
    assert calls == [True]
    # the same kernels on the same inputs; the attention core's dQ reductions and the ln_cond weight gradient's sums are
    # atomic, so the order of fp32 additions differs run to run (ln_cond.weight: 2.2-2.3e-5 measured on B200) -- 1e-4 bounds
    # that order noise and stays far below a precision change (bf16 operands would show ~1e-2)
    for k in plain:
        assert _rel(amp[k], plain[k]) < 1e-4, (k, _rel(amp[k], plain[k]))
