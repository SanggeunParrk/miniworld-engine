"""RTX 5090 (sm_120a): the fp32 SWA atom block's training stages on chunks of samples (``composed.CHUNK_ROWS_SM120``: the stage-to-stage tensors stay in the L2)
give the output and the activation gradients of the one call over every sample to the last bits, and its weight gradients up to the order of the fp32 sums."""

from __future__ import annotations

import copy

import pytest
import torch

from miniworld_engine.modules.atom_dit import AtomDiTMiniWorld

pytestmark = [pytest.mark.gpu]


def _rel(x: torch.Tensor, gold: torch.Tensor) -> float:
    return float((x.double() - gold.double()).abs().max() / gold.double().abs().max().clamp_min(1e-30))


@pytest.mark.parametrize("half_window,samples,length,chunk_rows", [(64, 6, 1024, 2048), (64, 5, 1024, 2048), (1000000, 4, 512, 1024)])  # a global window stays on one call
def test_chunked_fp32_training_equals_the_single_call(half_window: int, samples: int, length: int, chunk_rows: int, monkeypatch: pytest.MonkeyPatch) -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("the RTX 5090 (sm_120) path")
    from miniworld_engine.kernels.swa_dit.cuda.sm80 import composed

    torch.manual_seed(0)
    base = AtomDiTMiniWorld(half_window=half_window).cuda()
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for p in base.parameters():
            r = torch.randn(p.shape, generator=g)
            p.copy_(1.0 + 0.1 * r if p.ndim == 1 else r * p.shape[-1] ** -0.5)
    inputs = base.make_inputs(
        length=length, samples=samples, depth=None, dtype=torch.float32, device="cuda", generator=torch.Generator("cuda").manual_seed(0)
    )
    up = torch.randn(samples, length, 128, generator=torch.Generator().manual_seed(1)).cuda()
    results = []
    if half_window == 1000000:
        assert composed._chunk_samples(inputs["x"], half_window) is None
        monkeypatch.setattr(composed, "CHUNK_ROWS_SM120", chunk_rows)
        assert composed._chunk_samples(inputs["x"], half_window) is None
        return
    for rows in (0, chunk_rows):  # 0: one call over every sample
        monkeypatch.setattr(composed, "CHUNK_ROWS_SM120", rows)
        monkeypatch.setattr(composed, "CHUNK_MIN_ROWS_SM120", 0)
        m = copy.deepcopy(base).set_backend("cuda")
        x, cond = (inputs[k].detach().clone().requires_grad_(True) for k in ("x", "cond"))
        out = m(x, cond, inputs["attention_params"])
        out.backward(up.to(out.dtype))
        results.append((out.detach(), x.grad, cond.grad, [p.grad for p in m.parameters()]))
    (o1, dx1, dc1, g1), (o2, dx2, dc2, g2) = results
    assert _rel(o2, o1) < 5e-4  # (cuBLAS picks its kernels by the row count: the fp32 sums can differ in the last bits)
    assert _rel(dx2, dx1) < 5e-4
    assert _rel(dc2, dc1) < 1e-3  # dmod sums the samples' rows of dc through the modulation product: per-sample rows are the same, the sums are cuBLAS's
    for a, b in zip(g1, g2, strict=True):
        if a is not None:
            assert _rel(b, a) < 1e-3
