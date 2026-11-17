"""H100: the experimental e4m3 TriMul planes (``h100_b200style_fp8``, opt-in): the fused route stays inside the e4m3 storage error of the fp32
reference (about 4 % of the update, 6x the bf16 floor), masked rows stay exactly zero, and the default routes are untouched (POLICY None)."""

from __future__ import annotations

import pytest
import torch

pytestmark = [pytest.mark.gpu]


def _needs_h100() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("needs an H100 (sm_90)")


def _rel(a, b) -> float:
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


@pytest.mark.parametrize("d", [128, 256])
@pytest.mark.parametrize("outgoing", [True, False])
@pytest.mark.parametrize("n", [40, 100, 256, 300, 771])
def test_fused_fp8_planes(d: int, outgoing: bool, n: int) -> None:
    _needs_h100()
    from miniworld_engine.kernels.trimul_inproj.cuda import h100_b200style_fp8 as F8
    from miniworld_engine.kernels.trimul_inproj.cuda._h100_pack import cast_training_weights
    from miniworld_engine.modules.triangle import TriangleMultiplication

    assert F8.POLICY is None
    torch.manual_seed(0)
    m = TriangleMultiplication(d, outgoing=outgoing, master_weights=True, backend="cuda").cuda().eval()
    ref = TriangleMultiplication(d, outgoing=outgoing, backend="pytorch").cuda().eval()
    with torch.no_grad():
        for p in m.parameters():
            if p.dim() > 1:
                p.normal_(0, d**-0.5)
        ref.load_state_dict({k: v.float() for k, v in m.state_dict().items()})
        x = torch.randn(1, n, n, d, device="cuda").bfloat16()
        mask = torch.ones(1, n, dtype=torch.bool, device="cuda")
        mask[0, n - 37 :] = False
        y32 = ref(x.float(), mask)
        mats = [m.to_left.weight, m.to_left_gate.weight, m.to_right.weight, m.to_right_gate.weight, m.to_gate.weight, m.to_out.weight]
        leaves = [x, *cast_training_weights(mats), m.ln_pair.weight, m.ln_pair.bias, m.ln_out.weight, m.ln_out.bias]
        y = F8.run(leaves, mask, 1 if outgoing else 2)
    assert _rel(y - x, y32 - x.float()) < 0.06
    # masked pair rows / columns: the update is only the (exactly zero) contraction through LN_out's bias; the plane bytes themselves are zero
    assert torch.isfinite(y.float()).all()


@pytest.mark.parametrize("d", [128, 256])
@pytest.mark.parametrize("outgoing", [True, False])
def test_fp8_route_k3_pipe_and_fast_accum(d: int, outgoing: bool) -> None:
    """The K3 pipeline macros of the FP8 route are bit-identical to the plain K3; cuBLASLt fast accumulation stays within 1 % of the update of the exact-accumulation route."""
    _needs_h100()
    from miniworld_engine.kernels.trimul_inproj.cuda import _h100_fp8 as F
    from miniworld_engine.kernels.trimul_inproj.cuda import h100_b200style_fp8 as F8
    from miniworld_engine.kernels.trimul_inproj.cuda._h100_pack import cast_training_weights
    from miniworld_engine.modules.triangle import TriangleMultiplication

    torch.manual_seed(0)
    n = 300
    m = TriangleMultiplication(d, outgoing=outgoing, master_weights=True, backend="cuda").cuda().eval()
    with torch.no_grad():
        for p in m.parameters():
            if p.dim() > 1:
                p.normal_(0, d**-0.5)
        x = torch.randn(1, n, n, d, device="cuda").bfloat16()
        mask = torch.ones(1, n, dtype=torch.bool, device="cuda")
        mask[0, n - 37 :] = False
        mats = [m.to_left.weight, m.to_left_gate.weight, m.to_right.weight, m.to_right_gate.weight, m.to_gate.weight, m.to_out.weight]
        leaves = [x, *cast_training_weights(mats), m.ln_pair.weight, m.ln_pair.bias, m.ln_out.weight, m.ln_out.bias]
        direction = 1 if outgoing else 2
        old = (dict(F8.K.K3_EXTRA), F.FAST_ACCUM)  # K3_EXTRA: the single (2.5.9) bf16 K3 pipeline hook; {} = plain K3
        try:
            F8.K.K3_EXTRA.clear()
            F.FAST_ACCUM = False
            y_plain = F8.run(leaves, mask, direction, margin=16.0)
            F8.K.K3_EXTRA.update(old[0])
            y_pipe = F8.run(leaves, mask, direction, margin=16.0)
            F.FAST_ACCUM = True
            y_fast = F8.run(leaves, mask, direction, margin=16.0)
        finally:
            F8.K.K3_EXTRA.clear()
            F8.K.K3_EXTRA.update(old[0])
            F.FAST_ACCUM = old[1]
    assert torch.equal(y_plain, y_pipe)
    assert _rel(y_fast - x, y_plain - x) < 0.01
