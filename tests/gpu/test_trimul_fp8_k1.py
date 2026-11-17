"""H100: the experimental FP8-compute K1 (``h100_b200style_fp8.K1_F8C``, opt-in): K1 projects the e4m3-cast LayerNorm output with e4m3 weights. The module
stays inside the e4m3 error of the fp32 reference (planes plus operands: about 6 % of the update), the planes inside 6 % of the fp32 planes, and the flag off
is the plain e4m3-plane route."""

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
def test_k1_planes_fp8_compute(d: int) -> None:
    from miniworld_engine.kernels import _device  # noqa: F401
    from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
    from miniworld_engine.kernels.trimul_inproj.cuda import h100_b200style_direct as Dm

    _needs_h100()
    torch.manual_seed(0)
    n = 256
    ws = [(torch.randn(d, d, device="cuda") * d**-0.5).bfloat16() for _ in range(4)]
    gi = (1 + 0.3 * torch.randn(d, device="cuda")).abs()
    bi = 0.2 * torch.randn(d, device="cuda")
    x = (torch.randn(1, n, n, d, device="cuda") * 3 + 2).bfloat16()
    tok = torch.ones(n, dtype=torch.bool, device="cuda")
    tok[n - 5 :] = False
    z = x.reshape(n * n, d).float()
    xn = torch.nn.functional.layer_norm(z, (d,), gi, bi, 1e-5)
    m2 = (tok[:, None] & tok[None, :]).reshape(n * n, 1).float()

    def plane(g, p):
        return (torch.sigmoid(xn @ g.float().T) * (xn @ p.float().T) * m2).reshape(n, n, d).permute(2, 0, 1)

    ref = torch.cat([plane(ws[1], ws[0]), plane(ws[3], ws[2])])
    qs = 2.0 ** torch.floor(torch.log2(torch.tensor(28.0 / ref.abs().amax().item())))
    qinv = torch.full((2 * d,), float(qs), device="cuda")
    f8c = Dm.f8c_prepare(ws, gi, bi, qinv, d, d)
    q = torch.zeros(2 * d, n, n, device="cuda", dtype=torch.float8_e4m3fn)
    amax = torch.zeros(1, device="cuda")
    xs = torch.empty(n * n, 2, device="cuda")
    with T.native_context(x.device):
        Dm.front(x, ws, tok, gi, bi, q, xs, qinv=qinv, amax=amax, f8c=f8c)
    torch.cuda.synchronize()
    got = q.float() / qs
    assert _rel(got, ref) < 0.06
    assert bool((got[:, n - 5 :, :] == 0).all()) and bool((got[:, :, n - 5 :] == 0).all())  # masked rows / columns stay exact zeros
    assert amax.item() <= 448.0


@pytest.mark.parametrize("d", [128, 256])
@pytest.mark.parametrize("outgoing", [True, False])
def test_module_fp8_compute(d: int, outgoing: bool) -> None:
    _needs_h100()
    from miniworld_engine.kernels.trimul_inproj.cuda import h100_b200style_fp8 as F8
    from miniworld_engine.kernels.trimul_inproj.cuda._h100_pack import cast_training_weights
    from miniworld_engine.modules.triangle import TriangleMultiplication

    torch.manual_seed(0)
    n = 256
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
        direction = 1 if outgoing else 2
        old = F8.K1_F8C
        try:
            F8.K1_F8C = False
            y_planes = F8.run(leaves, mask, direction, margin=16.0)
            F8.K1_F8C = True
            y_k1 = F8.run(leaves, mask, direction, margin=16.0)
        finally:
            F8.K1_F8C = old
    assert _rel(y_k1 - x, y32 - x.float()) < 0.09
    assert _rel(y_planes - x, y32 - x.float()) < 0.06
    assert torch.isfinite(y_k1.float()).all()
