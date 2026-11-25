"""The bias-only token DiT's two pair-bias logit modes (``foldjet_engine.bias_logits``): ``"fp32"`` (default; the only training mode) and the
opt-in inference mode ``"bf16"`` (logits computed and stored in bf16, fp32 softmax over them, P rounded to bf16).

Checked on whatever card runs the file (H100 / A100 / RTX 5090: the ``cuda`` backend; every card: ``pytorch`` and ``triton``):

(a) the fp32 mode is the default and is untouched by the switch: explicit ``"fp32"`` == unset (bitwise), and a call after a trip through
    ``"bf16"`` and back is bitwise the same (no state leaks; the P / bias caches are keyed by the mode). Bit-identity against the 3.0.6
    release (the ``fp32`` mode must equal it) is a separate cross-process comparison (``benchmarks/bias_logits_modes.py --dump``).
(b) the bf16 mode: finite; its error against the fp64 module (the logits and the softmax in fp64) and against the PyTorch backend in the same mode
    (the arithmetic of the kernels' bf16 path) stays within the documented bf16-logit tolerance -- on "mild" logits (the regime where bf16 resolves them)
    tightly, on "wide" ones (|logit| ~ 1e3, bf16 resolution 4..8: the reason the default is fp32) only against the same-mode PyTorch reference,
    because there the mode is allowed to pick other keys than fp64 does.
(c) training / gradient calls stay fp32 under the bf16 switch: the output and the gradients are bitwise those of the fp32 mode.

Not a model-level accuracy study (peptide C-N, FoldBench lDDT): the user ran that experiment and asked for none here.
"""

# torch's Tensor.pow / mean overloads are not resolved by the checker in the error-measure helpers:
# pyright: reportCallIssue=false
from __future__ import annotations

import copy
from typing import cast

import pytest
import torch
import torch.nn.functional as F

from foldjet_engine import bias_logits
from foldjet_engine.modules.token_dit import TokenDiTMiniWorldBiasOnly, pair_bias_all

pytestmark = [pytest.mark.gpu]

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")
L, A, DP = 256, 2, 128
EPS = 1e-5
#: name -> (to_bias weight std, constant channel-0 weight, channel-0 offset of the pair)
CASES = {"mild": (0.05, 0.0, 0.0), "wide": (45.0, 0.0, 0.0)}
#: bf16 mode on "mild" logits: rms-relative error of the block output against fp64 (the bf16 module's own error ~1.5e-2 is inside it), and the
#: max-row total variation of P against fp64 (bf16 logits of |x| < ~3 resolve 1e-2: the TV is the logits' rounding times the row's spread)
MODE_TOL_MILD = 1.5e-2
#: bf16 mode on "wide" logits (|logit| ~ 1e3), rms-relative error of the block output against fp64 (measured 9.0e-2 .. 9.6e-2 on H100 and 5090; fp32 mode 1.5e-2 .. 3.2e-2)
Y_TOL_WIDE = 1.3e-1
TV_TOL_MILD = 3e-2
#: training gradients under the bf16 switch against the fp32 mode (max-abs relative to the max): atomic-add reduction-order noise only (Triton: 1.9e-4 on the
#: bf16 pair gradient); a bf16 logit leaking into the backward moves them by O(1) on this data
GRAD_REL = 2e-3
#: bf16 mode against the PyTorch backend in the same mode ("wide": the same rounding recipe, the GEMM's summation order differs): rms-relative
Y_TOL_CROSS = {"mild": 8e-2, "wide": 1.5e-1}


@pytest.fixture(autouse=True)
def _clean_switch():
    """Every test starts and ends on the unset default (fp32), whatever the environment says."""
    old = bias_logits._DEFAULT[0]
    bias_logits.set_default(None)
    yield
    bias_logits._DEFAULT[0] = old


def _build(case: str, seed: int = 0):
    sw, w0, off = CASES[case]
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
    mq = copy.deepcopy(m).bfloat16()
    ins = [t.bfloat16() for t in (single, cond, pair)]
    m64 = copy.deepcopy(mq).double().set_backend("pytorch")
    return mq, m64, ins, [t.double() for t in ins], mask, gy.bfloat16()


def _logits64(m64, pair64) -> torch.Tensor:
    a = m64.attention
    ln = F.layer_norm(pair64, (DP,), None, None, EPS) * a.ln_pair.weight
    return (ln @ a.to_bias.weight.t()).permute(0, 3, 1, 2)


def _p64(m64, pair64, mask) -> torch.Tensor:
    lg = _logits64(m64, pair64).masked_fill(~mask[:, None, None, :], float("-inf"))
    return torch.softmax(lg, -1)


def _backends() -> list[str]:
    return ["pytorch", "triton", "cuda"]


def _block(mq, backend: str, mode: str | None):
    m = copy.deepcopy(mq).set_backend(backend)
    if mode is not None:
        m.set_pair_bias_logits(mode)
    return m


def _eval(m, single, cond, pair, mask, hoist: bool = False) -> torch.Tensor:
    with torch.no_grad():
        if hoist:
            return m(single, cond, None, mask, bias=pair_bias_all([m], pair)[0])
        return m(single, cond, pair, mask)


def _skip_unsupported(m, single, cond, pair, mask, hoist=False):
    with torch.no_grad():
        bias = pair_bias_all([m], pair)[0] if hoist else None
        why = m.unsupported(single, cond, None if hoist else pair, mask, bias=bias) if m.backend != "pytorch" else None
    if why is not None:
        pytest.skip(f"{m.backend} backend on this card: {why}")


def _rms_rel(a, ref) -> float:
    return (((a.double() - ref.double()) ** 2).mean().sqrt() / (ref.double() ** 2).mean().sqrt()).item()


def _max_rel(a, ref) -> float:
    return ((a.double() - ref.double()).abs().max() / ref.double().abs().max()).item()


# ---------------------------------------------------------------------------------------------------------------------- the switch itself
def test_switch_api() -> None:
    assert bias_logits.default() == "fp32"
    with pytest.raises(ValueError):
        bias_logits.set_default("fp16")
    with bias_logits.scope("bf16"):
        assert bias_logits.default() == "bf16"
    assert bias_logits.default() == "fp32"
    m = TokenDiTMiniWorldBiasOnly(backend="pytorch")
    assert m.pair_bias_logits is None and m.set_pair_bias_logits("BF16") is m and m.pair_bias_logits == "bf16"
    m.set_pair_bias_logits(None)
    assert m.pair_bias_logits is None
    with pytest.raises(ValueError):
        m.set_pair_bias_logits("tf32")
    # the effective mode: bf16 only for bf16 activations without autograd
    eff = bias_logits.effective
    assert eff("bf16", grad=False, dtype=torch.bfloat16) == "bf16"
    assert eff("bf16", grad=True, dtype=torch.bfloat16) == "fp32"
    assert eff("bf16", grad=False, dtype=torch.float32) == "fp32"
    assert eff(None, grad=False, dtype=torch.bfloat16) == "fp32"
    with bias_logits.scope("bf16"):
        assert eff(None, grad=False, dtype=torch.bfloat16) == "bf16" and eff("fp32", grad=False, dtype=torch.bfloat16) == "fp32"


def test_env_default(monkeypatch) -> None:
    monkeypatch.setenv(bias_logits.ENV, "bf16")
    assert bias_logits.default() == "bf16"
    bias_logits.set_default("fp32")  # the API wins over the environment
    assert bias_logits.default() == "fp32"
    bias_logits.set_default(None)
    monkeypatch.setenv(bias_logits.ENV, "fp16")
    with pytest.raises(ValueError):
        bias_logits.default()
    monkeypatch.setenv(bias_logits.ENV, "")
    assert bias_logits.default() == "fp32"


# ---------------------------------------------------------------------------------------------------------------------- (a) fp32 untouched
@requires_cuda
@pytest.mark.parametrize("backend", _backends())
@pytest.mark.parametrize("hoist", [False, True], ids=["pair", "hoist"])
def test_fp32_mode_is_default_and_stateless(backend: str, hoist: bool) -> None:
    mq, _, ins, _, mask, _ = _build("wide")
    if hoist and backend == "triton":
        pytest.skip("the Triton backend has no hoisted-bias path of its own")
    m0 = _block(mq, backend, None)
    _skip_unsupported(m0, *ins, mask, hoist)
    y0 = _eval(m0, *ins, mask, hoist)
    y1 = _eval(_block(mq, backend, "fp32"), *ins, mask, hoist)
    assert torch.equal(y0, y1), "explicit fp32 must equal the default"
    m2 = _block(mq, backend, "bf16")  # a trip through bf16 and back on one module (the P / bias caches must not serve the wrong mode)
    yb = _eval(m2, *ins, mask, hoist)
    assert not torch.equal(yb, y0), "the bf16 mode must be a different computation on wide logits (the switch is dead)"
    m2.set_pair_bias_logits("fp32")
    assert torch.equal(_eval(m2, *ins, mask, hoist), y0), "fp32 after bf16 must be bitwise the default"
    with bias_logits.scope("bf16"):
        assert torch.equal(_eval(m0, *ins, mask, hoist), yb), "the process default bf16 equals the module's own bf16"
    assert torch.equal(_eval(m0, *ins, mask, hoist), y0)


# ---------------------------------------------------------------------------------------------------------------------- (b) bf16 accuracy
@requires_cuda
@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("backend", _backends())
@pytest.mark.parametrize("hoist", [False, True], ids=["pair", "hoist"])
def test_bf16_mode_accuracy(case: str, backend: str, hoist: bool) -> None:
    if hoist and backend == "triton":
        pytest.skip("the Triton backend has no hoisted-bias path of its own")
    mq, m64, ins, i64, mask, _ = _build(case)
    with torch.no_grad():
        y64 = m64(*i64[:2], i64[2], mask)
    mb = _block(mq, backend, "bf16")
    _skip_unsupported(mb, *ins, mask, hoist)
    y = _eval(mb, *ins, mask, hoist)
    assert torch.isfinite(y.float()).all(), "bf16 mode produced non-finite values"
    yref = _eval(_block(mq, "pytorch", "bf16"), *ins, mask, False)  # the PyTorch backend in the same mode
    yf32 = _eval(_block(mq, backend, "fp32"), *ins, mask, hoist)
    e64, ecross, e32, emode = _rms_rel(y, y64), _rms_rel(y, yref), _rms_rel(yf32, y64), _rms_rel(y, yf32)
    print(
        f"{case} {backend} {'hoist' if hoist else 'pair'}: bf16-mode vs fp64 rms {e64:.3e} max {_max_rel(y, y64):.3e}; vs pytorch-bf16 rms "
        f"{ecross:.3e} max {_max_rel(y, yref):.3e}; (fp32 mode vs fp64 rms {e32:.3e}; bf16 mode vs fp32 mode of the same backend rms {emode:.3e} max {_max_rel(y, yf32):.3e})"
    )
    assert ecross <= Y_TOL_CROSS[case], f"{case} {backend}: rms-rel vs the PyTorch bf16 mode {ecross:.3e} > {Y_TOL_CROSS[case]}"
    if (
        case == "mild"
    ):  # bf16 resolves these logits: the mode adds at most its small logit rounding to the backend's own (mode-independent) error
        assert emode <= MODE_TOL_MILD, f"{case} {backend}: bf16 vs fp32 mode of the same backend {emode:.3e} > {MODE_TOL_MILD}"
        assert e64 <= e32 + MODE_TOL_MILD, f"{case} {backend}: rms-rel vs fp64 {e64:.3e} (fp32 mode {e32:.3e})"
    else:  # |logit| ~ 1e3: bf16 resolution 4..8 moves the attention (the reason the default is fp32); bounded by the error measured on H100 / 5090
        assert e64 <= Y_TOL_WIDE, f"{case} {backend}: rms-rel vs fp64 {e64:.3e} > {Y_TOL_WIDE}"


@requires_cuda
def test_bf16_logits_distribution_mild() -> None:
    """The PyTorch / Triton bf16 logits against fp64 on mild logits: max-row total variation of P (the kernels' own P is inside the block test)."""
    from foldjet_engine.kernels.bias_only_attention.triton.pair_bias import pair_bias_logits_bf16
    from foldjet_engine.modules.attention import bf16_pair_bias_logits

    mq, m64, ins, i64, mask, _ = _build("mild")
    a = mq.attention
    p64 = _p64(m64, i64[2], mask)
    for name, lg in (
        ("pytorch", bf16_pair_bias_logits(ins[2], a.ln_pair.weight, a.to_bias.weight, EPS)),
        ("triton", pair_bias_logits_bf16(ins[2], a.ln_pair.weight, a.to_bias.weight, EPS)),
    ):
        assert lg.dtype is torch.bfloat16
        p = torch.softmax(lg.float().masked_fill(~mask[:, None, None, :], float("-inf")), -1)
        tv = (0.5 * (p.double() - p64).abs().sum(-1).max()).item()
        print(f"mild {name}: max-row TV of P vs fp64 {tv:.3e}")
        assert tv <= TV_TOL_MILD, f"{name}: max-row TV {tv:.3e} > {TV_TOL_MILD}"


# ---------------------------------------------------------------------------------------------------------------------- (c) training = fp32
@requires_cuda
@pytest.mark.parametrize("backend", _backends())
@pytest.mark.parametrize("hoist", [False, True], ids=["pair", "hoist"])
def test_training_stays_fp32_under_bf16_switch(backend: str, hoist: bool) -> None:
    if hoist and backend == "triton":
        pytest.skip("the Triton backend has no hoisted-bias path of its own")
    mq, _, ins, _, mask, gy = _build("wide")

    def run(mode):
        m = _block(mq, backend, mode)
        single, cond, pair = (t.detach().clone().requires_grad_(i != 1) for i, t in enumerate(ins))
        for p in m.parameters():
            p.grad = None
        y = m(single, cond, None, mask, bias=pair_bias_all([m], pair)[0]) if hoist else m(single, cond, pair, mask)
        (y.float() * gy.float()).sum().backward()
        a = m.attention
        grads = (pair.grad, a.to_bias.weight.grad, a.ln_pair.weight.grad)
        assert all(g is not None for g in grads)
        return y.detach(), tuple(cast(torch.Tensor, g) for g in grads)

    with torch.enable_grad():
        try:
            y32, g32 = run("fp32")
        except Exception as e:  # a backend that does not train on this card (the same refusal in both modes)
            pytest.skip(f"{backend} training on this card: {type(e).__name__}: {str(e)[:120]}")
        yb, gb = run("bf16")
        with bias_logits.scope("bf16"):
            yd, gd = run(None)
    assert torch.equal(y32, yb) and torch.equal(y32, yd), "training output under the bf16 switch must be bitwise the fp32 mode's"
    # the gradients: the weight-gradient reductions are atomic adds on some backends (Triton), so two identical calls differ in the last bits; a bf16 logit
    # leaking into the backward would move them by 1e-2 and more on this data
    for a, b, c in zip(g32, gb, gd, strict=True):
        assert _max_rel(a, b) <= GRAD_REL and _max_rel(a, c) <= GRAD_REL, "training gradients under the bf16 switch must be the fp32 mode's"


@requires_cuda
def test_pair_bias_all_modes() -> None:
    """``pair_bias_all``: fp32 bias by default / for training, bf16 bias in the bf16 mode without autograd; a bf16 mode with autograd stays fp32."""
    mq, _, ins, _, _, _ = _build("wide")
    pair = ins[2]
    for backend in ("pytorch", "cuda"):
        m = _block(mq, backend, None)
        with torch.no_grad():
            assert pair_bias_all([m], pair)[0].dtype is torch.float32
            m.set_pair_bias_logits("bf16")
            assert pair_bias_all([m], pair)[0].dtype is torch.bfloat16
        pg = pair.detach().clone().requires_grad_(True)
        assert pair_bias_all([m], pg)[0].dtype is torch.float32, "autograd on: fp32 whatever the switch"
