"""RTX 5090 (sm_120a) Transition bf16 training (d 128, n 4): the forward does not save the LayerNorm output ``xn`` [T, D] (36 MiB at L 384); the backward
launch rebuilds it from x, (mean, rstd), gamma and beta (the TriMul build's ``ln_apply``).  The saved bytes drop by one [T, D] bf16 tensor; the gradients are
those of the saved-xn path."""

import copy

import pytest
import torch

from foldjet_engine.measure import config as C
from foldjet_engine.modules import Transition

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0), reason="needs an RTX 5090 (sm_120)"),
]

LENGTH = 128


def _step(module, x, saved_bytes: list[int]):
    storages: dict[int, int] = {}

    def pack(t: torch.Tensor) -> torch.Tensor:
        storages[t.untyped_storage().data_ptr()] = t.untyped_storage().nbytes()
        return t

    params = {p.data_ptr() for p in module.parameters()}
    with torch.autocast("cuda", dtype=torch.bfloat16), torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        z = x.detach().clone().requires_grad_(True)
        y = module(z)
    saved_bytes.append(sum(n for p, n in storages.items() if p not in params))
    up = torch.randn(y.shape, device="cuda", dtype=torch.float32, generator=torch.Generator(device="cuda").manual_seed(9)).to(y.dtype)
    y.backward(up)
    return {"y": y.detach(), "dz": z.grad, **{"d_" + n: p.grad for n, p in module.named_parameters() if p.grad is not None}}


def test_xn_recompute_saves_one_tensor_and_is_bitwise(monkeypatch: pytest.MonkeyPatch) -> None:
    from foldjet_engine.kernels.transition.cuda import fused_wide_sm80 as fused_sm80

    module = Transition(d_hidden=128, n=4, master_weights=True).cuda().float().train()
    C.randomize(module, torch.Generator(device="cuda").manual_seed(0))
    module.set_master_weights()
    g = torch.Generator(device="cuda").manual_seed(1)
    x = torch.randn(1, LENGTH, LENGTH, 128, device="cuda", generator=g).to(torch.bfloat16)
    kern = copy.deepcopy(module).set_backend("cuda")
    saved: list[int] = []
    new = _step(kern, x, saved)
    kern.zero_grad(set_to_none=True)
    with monkeypatch.context() as m:
        m.setattr(fused_sm80, "_recompute_xn", lambda x, ln, h: False)
        old = _step(kern, x, saved)
    assert saved[1] - saved[0] == LENGTH * LENGTH * 128 * 2, saved
    for name, a in new.items():
        assert torch.equal(a, old[name]), name
