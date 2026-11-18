"""H100 Transition D 384 / 512 inference past the one-launch h budget: the blocks write into their rows of one output (``fwd_into``, no copy);
the output is bitwise the one-launch output."""

import pytest
import torch

from miniworld_engine.kernels.transition.cuda import fused_wide_sm90a as W

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0), reason="H100 (sm_90) only")


@pytest.mark.parametrize("d", [384, 512])
@pytest.mark.parametrize("rows", [2048, 3968])
def test_chunked_inference_bitwise_one_launch(d: int, rows: int, monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(0)
    x = (torch.randn(rows, d, device="cuda") * 2).bfloat16()
    ws = [(torch.randn(*s, device="cuda") * s[-1] ** -0.5).bfloat16() for s in ((2 * d, d), (2 * d, d), (d, 2 * d))]
    g, b = torch.ones(d, device="cuda"), torch.zeros(d, device="cuda")
    with torch.no_grad():
        ref = W.transition_wide_sm90a(x, g, b, *ws, 1e-5)
        monkeypatch.setattr(W, "_H_ONE_LAUNCH_BYTES", 1 << 20)
        monkeypatch.setattr(W, "_H_CHUNK_BYTES", 1 << 20)
        assert W._inference_chunk_rows(rows, d, 2 * d) < rows
        got = W.transition_wide_sm90a(x, g, b, *ws, 1e-5)
    assert torch.equal(got, ref)
