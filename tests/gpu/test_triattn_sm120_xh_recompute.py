"""RTX 5090 (sm_120a) TriangleAttention bf16 training, rows engine (d_pair 128, 4 heads of 32): the forward does not save the normalised input ``xh``
[T, D + 8] (38 MiB at L 384); the backward rebuilds it with the forward's own ``ln_rows`` launch.  The saved bytes drop by one such tensor and every
gradient is bitwise the saved-xh path's."""

import copy

import pytest
import torch

from foldjet_engine.measure import config as C
from foldjet_engine.modules import TriangleAttention

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0), reason="needs an RTX 5090 (sm_120)"),
]

LENGTH = 128


def _step(module, inputs, saved_bytes: list[int]):
    storages: dict[int, int] = {}

    def pack(t: torch.Tensor) -> torch.Tensor:
        storages[t.untyped_storage().data_ptr()] = t.untyped_storage().nbytes()
        return t

    params = {p.data_ptr() for p in module.parameters()}
    with torch.autocast("cuda", dtype=torch.bfloat16), torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        z = inputs["pair"].detach().clone().requires_grad_(True)
        y = module(**{**inputs, "pair": z})
    saved_bytes.append(sum(n for p, n in storages.items() if p not in params))
    up = torch.randn(y.shape, device="cuda", dtype=torch.float32, generator=torch.Generator(device="cuda").manual_seed(9)).to(y.dtype)
    y.backward(up)
    return {"y": y.detach(), "dz": z.grad, **{"d_" + n: p.grad for n, p in module.named_parameters() if p.grad is not None}}


@pytest.mark.parametrize("starting", [True, False])
def test_xh_recompute_saves_one_tensor_and_is_bitwise(starting: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    from foldjet_engine.kernels.triangle_attention.cuda import sm80_wide

    module = TriangleAttention(d_pair=128, n_head=4, starting=starting, p_drop=0.0, master_weights=True).cuda().float().train()
    C.randomize(module, torch.Generator(device="cuda").manual_seed(0))
    module.set_master_weights()
    inputs = module.make_inputs(
        length=LENGTH, samples=1, depth=None, dtype=torch.bfloat16, device="cuda", generator=torch.Generator(device="cuda").manual_seed(1)
    )
    kern = copy.deepcopy(module).set_backend("cuda")
    saved: list[int] = []
    new = _step(kern, inputs, saved)
    kern.zero_grad(set_to_none=True)
    with monkeypatch.context() as m:
        m.setattr(sm80_wide, "_recompute_xh", lambda x, fused: False)
        old = _step(kern, inputs, saved)
    xh_bytes = LENGTH * LENGTH * (128 + 8) * 2
    assert saved[1] - saved[0] == xh_bytes, (saved, xh_bytes)
    for name, a in new.items():
        assert torch.equal(a, old[name]), name
