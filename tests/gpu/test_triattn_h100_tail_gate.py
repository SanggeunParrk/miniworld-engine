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
