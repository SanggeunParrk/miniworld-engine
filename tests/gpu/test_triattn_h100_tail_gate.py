"""H100 TriangleAttention inference at d_pair 128 x 4 heads: the tail projects the gate from x (``h100_sm90.INFERENCE_TAIL_GATE``,
``tri_tail(wg=...)`` with the front writing q | k | v only).  Against the gate the front stores in bf16 (the previous form) it may move
a bf16 rounding step, never more; against the fp32 PyTorch reference it is as accurate; run to run it is bitwise; the starting and
ending nodes, AF3's flip, RoseTTAFold3's gate / out biases and a key mask included.  At d_pair 256 x 8 heads the wide front's CTA
groups run in step (``FW_LAG``): bitwise run to run."""

import copy

import pytest
import torch

from miniworld_engine.kernels.triangle_attention.cuda import h100_sm90 as hs
from miniworld_engine.modules.triangle import TriangleAttention

pytestmark = [pytest.mark.gpu, pytest.mark.h100]


def _module(d: int, h: int, model: str | None, starting: bool) -> TriangleAttention:
    torch.manual_seed(0)
    kw = {} if model is None else {"model": model}
    m = TriangleAttention(d, h, starting=starting, master_weights=True, backend="cuda", **kw).cuda().eval()
    with torch.no_grad():
        for p in m.parameters():
            if p.dim() == 2:
                p.normal_(0, p.shape[1] ** -0.5)
            elif p is not m.ln_pair.weight:
                p.normal_(0, 0.3)
        m.ln_pair.weight.uniform_(0.5, 1.5)
    return m


def _needs_h100() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("needs an H100 (sm_90)")


@pytest.mark.parametrize("model", [None, "af3", "rf3"])
@pytest.mark.parametrize("starting", [True, False])
def test_tail_gate_matches_stored_gate(model: str | None, starting: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    _needs_h100()
    m = _module(128, 4, model, starting)
    ref = copy.deepcopy(m).float()
    ref.set_backend("pytorch")
    L = 256
    g = torch.Generator(device="cuda").manual_seed(1)
    x = torch.randn(1, L, L, 128, device="cuda", generator=g).bfloat16()
    mask = torch.rand(1, L, device="cuda", generator=g) > 0.1
    with torch.no_grad():
        r = ref(x.float(), mask) - x.float()
        outs = {}
        for flag in (False, True):
            monkeypatch.setattr(hs, "INFERENCE_TAIL_GATE", flag)
            outs[flag] = m(x, mask).float() - x.float()
        again = m(x, mask).float() - x.float()
    assert torch.equal(outs[True], again)  # bitwise run to run
    # the two gate forms differ by the stored gate's bf16 rounding, seen through the output's bf16 rounding (at x's scale): two bf16
    # steps of the largest output at most
    top = float((x.float() + r).abs().max())
    assert float((outs[True] - outs[False]).abs().max()) <= 2**-6 * top
    rms = {f: float(((o - r) ** 2).mean().sqrt()) for f, o in outs.items()}
    assert rms[True] <= 1.05 * rms[False]


def test_wide_front_in_step_bitwise() -> None:
    _needs_h100()
    m = _module(256, 8, None, True)
    L = 384
    g = torch.Generator(device="cuda").manual_seed(2)
    x = torch.randn(1, L, L, 256, device="cuda", generator=g).bfloat16()
    with torch.no_grad():
        a = m(x)
        b = m(x)
    assert torch.isfinite(a).all()
    assert torch.equal(a, b)


@pytest.mark.parametrize("model", [None, "af3"])
@pytest.mark.parametrize("starting", [True, False])
def test_wide_front_folded_layernorm(model: str | None, starting: bool) -> None:
    """d_pair 256 x 8 heads: the LayerNorm folded into the wide front's pack (the GEMM reads x as loaded, ``r (x Wf^T - mu S1) + S0``)
    against the fp32 reference, on rows with a mean several times their spread (the fold's cancellation) and all-zero rows."""
    _needs_h100()
    m = _module(256, 8, model, starting)
    ref = copy.deepcopy(m).float()
    ref.set_backend("pytorch")
    L = 384
    g = torch.Generator(device="cuda").manual_seed(3)
    x = torch.randn(1, L, L, 256, device="cuda", generator=g) + 4.0 * torch.randn(1, L, L, 1, device="cuda", generator=g)
    x[0, 5] = 0.0
    x[0, :, 7] = 0.0
    x = x.bfloat16()
    mask = torch.rand(1, L, device="cuda", generator=g) > 0.1
    with torch.no_grad():
        r = ref(x.float(), mask) - x.float()
        a = m(x, mask)
        b = m(x, mask)
    assert torch.isfinite(a).all()
    assert torch.equal(a, b)
    o = a.float() - x.float()
    assert float(((o - r) ** 2).mean().sqrt() / (r**2).mean().sqrt()) < 5e-2


def test_ending_tail_reads_o_transposed_bitwise() -> None:
    """The inference tail of the ending node without a dropout scale takes rows of the pair as stored and reads o transposed
    (``tr = 2``); with a dropout scale of ones it takes rows of the transposed pair (``tr = 1``): the same values bit for bit."""
    _needs_h100()
    L, C = 256, 128
    g = torch.Generator(device="cuda").manual_seed(4)
    bf = lambda *s: torch.randn(*s, device="cuda", generator=g).bfloat16()
    x, o = bf(L * L, C), bf(L * L, C)
    wo, wg = bf(C, C) * C**-0.5, bf(C, C) * C**-0.5
    lnw, lnb = torch.rand(C, device="cuda", generator=g) + 0.5, torch.randn(C, device="cuda", generator=g) * 0.1
    mod = hs._mod()
    empty = torch.empty(0, device="cuda", dtype=torch.bfloat16)
    ones = torch.ones(1, L, C, device="cuda", dtype=torch.bfloat16)
    kw = {"wg": wg, "lnw": lnw, "lnb": lnb, "eps": 1e-5}
    a = mod.tri_tail(empty, o, x, wo, empty, L, True, None, **kw)
    b = mod.tri_tail(empty, o, x, wo, ones, L, True, None, **kw)
    assert torch.equal(a, b)
    s = mod.tri_tail(empty, o, x, wo, empty, L, False, None, **kw)  # the starting node on the transposed operands
    xt = x.view(L, L, C).transpose(0, 1).reshape(L * L, C).contiguous()
    st = mod.tri_tail(empty, o, xt, wo, empty, L, False, None, **kw).view(L, L, C).transpose(0, 1).reshape(L * L, C)
    assert torch.equal(a, st)
    assert not torch.equal(a, s)


def test_af3_compact_front_past_768(monkeypatch: pytest.MonkeyPatch) -> None:
    """AF3 d_pair 64 (4 heads x 16) beyond L768: the one-kernel front with compact heads (``_FRONT_L_COMPACT``) against the LN + GEMM
    + bias-glue front it replaced there -- the same update within bf16 rounding, bitwise run to run."""
    _needs_h100()
    from miniworld_engine.kernels.triangle_attention.cuda import h100_sm90_generic as gen
    from miniworld_engine.modules.triangle import TriangleAttentionAF3

    torch.manual_seed(0)
    m = TriangleAttentionAF3(64, 4, starting=False, master_weights=True, backend="cuda").cuda().eval()
    with torch.no_grad():
        for p in m.parameters():
            if p.dim() == 2:
                p.normal_(0, p.shape[1] ** -0.5)
        m.ln_pair.weight.uniform_(0.5, 1.5)
    L = 1024
    g = torch.Generator(device="cuda").manual_seed(5)
    x = torch.randn(1, L, L, 64, device="cuda", generator=g).bfloat16()
    with torch.no_grad():
        a = m(x)
        again = m(x)
        monkeypatch.setattr(gen, "_FRONT_L_COMPACT", {})
        b = m(x)
    assert torch.equal(a, again)
    assert not torch.equal(a, b)  # two fronts
    upd = b.float() - x.float()
    assert float(((a.float() - b.float()) ** 2).mean().sqrt() / (upd**2).mean().sqrt()) < 3e-2
