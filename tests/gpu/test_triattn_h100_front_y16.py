"""H100 TriangleAttention inference at d_pair 64, 32-channel heads, beyond the one-kernel bf16 front's length (``h100_sm90_generic.FRONT_Y16``):
the fp16-exact fused front (``front_fused3_sm90.cu`` built with ``MW_F3_Y16``) replaces LayerNorm + cuBLAS fp16 GEMMs + bias kernel and
its module output is BIT-IDENTICAL to the unfused route (starting / ending node, AF3's flip, RoseTTAFold3's biases, a key mask).
``FRONT_Y16_SHORT`` (default off, labeled) also takes it below that length: bits move to the fp16 route's (bf16 y and W of the one-kernel
front become fp16); against the fp32 PyTorch reference it must be no less accurate."""

import copy

import pytest
import torch

from foldjet_engine.kernels.triangle_attention.cuda import h100_sm90_generic as gen
from foldjet_engine.modules.triangle import TriangleAttention

pytestmark = [pytest.mark.gpu, pytest.mark.h100]


def _module(d: int, h: int, model: str | None, starting: bool) -> TriangleAttention:
    torch.manual_seed(0)
    kw = {} if model is None else {"model": model}
    m = TriangleAttention(d, h, d_hidden=h * 32, starting=starting, master_weights=True, backend="cuda", **kw).cuda().eval()
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


@pytest.mark.parametrize("heads", [2, 4])
@pytest.mark.parametrize("model", [None, "af3", "rf3"])
@pytest.mark.parametrize("starting", [True, False])
def test_y16_front_bit_identical_to_unfused(heads: int, model: str | None, starting: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    _needs_h100()
    m = _module(64, heads, model, starting)
    L = 896
    g = torch.Generator(device="cuda").manual_seed(1)
    x = torch.randn(1, L, L, 64, device="cuda", generator=g).bfloat16()
    mask = torch.rand(1, L, device="cuda", generator=g) > 0.1
    outs = {}
    with torch.no_grad():
        for flag in (False, True):
            monkeypatch.setattr(gen, "FRONT_Y16", flag)
            outs[flag] = m(x, mask)
    assert torch.equal(outs[False], outs[True])


@pytest.mark.parametrize("heads", [2, 4])
def test_y16_short_not_less_accurate(heads: int, monkeypatch: pytest.MonkeyPatch) -> None:
    _needs_h100()
    m = _module(64, heads, "af3", True)
    ref = copy.deepcopy(m).float()
    ref.set_backend("pytorch")
    L = 512
    g = torch.Generator(device="cuda").manual_seed(1)
    x = torch.randn(1, L, L, 64, device="cuda", generator=g).bfloat16()
    mask = torch.rand(1, L, device="cuda", generator=g) > 0.1
    with torch.no_grad():
        r = ref(x.float(), mask) - x.float()
        rms = {}
        for flag in (False, True):
            monkeypatch.setattr(gen, "FRONT_Y16_SHORT", flag)
            rms[flag] = float(((m(x, mask).float() - x.float() - r) ** 2).mean().sqrt())
    print(f"rms vs fp32 reference, heads {heads}: bf16 front {rms[False]:.3e}, fp16 front {rms[True]:.3e}")
    assert rms[True] <= 1.05 * rms[False]
