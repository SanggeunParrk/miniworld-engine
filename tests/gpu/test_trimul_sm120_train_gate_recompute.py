"""RTX 5090 (sm_120a) TriMul bf16 training, wide path (d_pair 128, hidden 128): the forward saves p' but not the output gate's pre-activation g' (36 MiB at L 384);
the backward recomputes g' = x_n W_g^T / 2 from the LayerNorm-ed input.  The saved bytes drop by one [T, D] bf16 tensor and the gradients stay inside the bf16
noise of the saved-g' path (they are not bit-identical: x_n is rounded to bf16 before the GEMM)."""

import copy

import pytest
import torch

from miniworld_engine.measure import config as C
from miniworld_engine.modules import TriangleMultiplication

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


@pytest.mark.parametrize("outgoing", [True, False])
def test_gate_recompute_saves_one_gate_tensor_and_matches(outgoing: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    from miniworld_engine.kernels.trimul_inproj.cuda import sm80_wide

    module = TriangleMultiplication(d_pair=128, d_hidden=128, outgoing=outgoing, p_drop=0.0).cuda().float().train()
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
        m.setattr(sm80_wide, "_recompute_gate", lambda z: False)
        old = _step(kern, inputs, saved)
    gate_bytes = LENGTH * LENGTH * 128 * 2
    assert saved[1] - saved[0] == gate_bytes, (saved, gate_bytes)
    for name, a in new.items():
        b = old[name].double()
        err = (a.double() - b).abs().max().item() / max(b.abs().max().item(), 1e-30)
        assert err <= 2.5e-2, (name, err)


@pytest.mark.parametrize("mode", [0, 1])
@pytest.mark.parametrize("length", [128, 256])
def test_repicked_plane_gemm_tile_is_bitwise_the_old_tile(mode: int, length: int) -> None:
    """Hidden 128, L 128 / 256 (modes 0 / 1): the 64 x 64 tile the table names now gives the bits of the 128 x 128 tile it replaced (the k order per element is the same)."""
    from miniworld_engine.kernels.trimul_inproj.cuda import contract_sm80

    assert contract_sm80.SM120_TILES[(mode, 128, length)] == 4
    g = torch.Generator(device="cuda").manual_seed(0)
    a = torch.randn(128, length, length, device="cuda", dtype=torch.bfloat16, generator=g)
    b = torch.randn(128, length, length, device="cuda", dtype=torch.bfloat16, generator=g)
    new, old = torch.empty_like(a), torch.empty_like(a)
    contract_sm80.planes(mode, a, b, new)
    contract_sm80._ext120().run(a, b, old, 0, mode, 1, 0)
    assert torch.equal(new, old)
