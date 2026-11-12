"""FoldForge 5I28 (OpenDDE n = 16): the structural-token refiner runs TriangleMultiplication at d_pair 384 on S = 3968 tokens, an
11.26 GiB pair. The H100 / A100 TriMul inference kept its a | b planes (two pair sizes) alive while it allocated the output, so one
call peaked at x + planes + contraction + y = 5 pair sizes and the refiner's second update ran out of an 80 GB card. The planes are dead
once contracted and are now freed before the output is allocated: the call's own peak (above the input) is planes + contraction
= 3 pair sizes (it was 4). The arithmetic is unchanged; the outputs are compared with the float32 reference as the accuracy smoke does.
"""

import copy

import pytest
import torch

from miniworld_engine.measure import config as C
from miniworld_engine.modules import TriangleMultiplication

pytestmark = [pytest.mark.gpu]


def _card() -> tuple[int, int] | None:
    return torch.cuda.get_device_capability() if torch.cuda.is_available() else None


@pytest.mark.skipif(_card() not in ((9, 0), (8, 0)), reason="H100 / A100")
@pytest.mark.parametrize(("backend", "limit"), [("cuda", 3.2), ("triton", 4.2)])
@pytest.mark.parametrize("outgoing", [True, False])
@pytest.mark.parametrize(
    ("d", "length"),
    # token-flat path (L % 128 == 0) and, on H100, the row-padded path (L % 128 != 0) at d_pair 384 / 256
    [(384, 1024), (384, 1000), (256, 1024)],
)
def test_trimul_inference_peak_is_planes_plus_contraction(backend: str, limit: float, d: int, length: int, outgoing: bool) -> None:
    m = TriangleMultiplication(d, outgoing=outgoing).cuda().float().eval()
    C.randomize(m, torch.Generator(device="cuda").manual_seed(0))
    m.set_master_weights()
    kern = copy.deepcopy(m).set_backend(backend)
    gen = torch.Generator(device="cuda").manual_seed(1)
    inputs = m.make_inputs(length=length, samples=1, depth=None, dtype=torch.bfloat16, device="cuda", generator=gen)
    pair, mask = inputs["pair"], inputs["mask"]
    why = kern.unsupported(pair, mask)
    if why is not None:
        pytest.skip(f"not served on this card: {why}")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        kern(pair, mask)  # first call: weight forms, kernel loads
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        out = kern(pair, mask)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() - before
    plane = pair.numel() * pair.element_size()  # one pair size (bf16): hidden == d_pair, so one plane set of the contraction too
    # planes (2) + contraction (1) + the LN statistics and folded weights; the output is allocated after the planes are freed
    # (triton: the LN'd input x_n is a pair size more; its left / right projections were alive with y: 5)
    assert peak <= limit * plane, f"{backend}: peak {peak / plane:.2f} pair sizes above the input (expected <= {limit})"
    with torch.no_grad():
        gold = copy.deepcopy(m).float()(pair.float(), mask).float()
    err = float((out.float() - gold).abs().max() / gold.abs().max())
    assert err <= 8e-3, f"max error {err:.3e} vs the float32 reference"


@pytest.mark.skipif(_card() != (9, 0), reason="H100 (the wide sm90a Transition)")
def test_wide_transition_row_chunks_are_bitwise(monkeypatch: pytest.MonkeyPatch) -> None:
    """The H100 wide Transition (D >= 384) runs a call whose hidden h passes ``_H_ONE_LAUNCH_BYTES`` in launches of whole 128-row tiles
    (OpenDDE's structural pair: h = 22.5 GiB in one launch). Every row is the same tile computation: the chunked output is bitwise the
    one-launch output. Forced here at a small size by lowering the bounds."""
    from miniworld_engine.kernels.transition.cuda import fused_wide_sm90a as fw
    from miniworld_engine.modules import Transition

    m = Transition(384, 2).cuda().float().eval()
    C.randomize(m, torch.Generator(device="cuda").manual_seed(0))
    m.set_master_weights()
    m.set_backend("cuda")
    x = torch.randn(1, 200, 200, 384, device="cuda", generator=torch.Generator(device="cuda").manual_seed(1)).bfloat16()
    why = m.unsupported(x)
    if why is not None:
        pytest.skip(f"not served: {why}")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        one = m(x)
        monkeypatch.setattr(fw, "_H_ONE_LAUNCH_BYTES", 1 << 20)
        monkeypatch.setattr(fw, "_H_CHUNK_BYTES", 128 * 768 * 2 * 37)  # 37 tiles per launch: a ragged last launch
        assert fw._inference_chunk_rows(200 * 200, 384, 768) < 200 * 200
        chunked = m(x)
    assert torch.equal(one, chunked)
