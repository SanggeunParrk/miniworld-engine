"""FoldForge 5I28 (A100, OpenDDE n = 12): the structural-token refiner's pair is [S, S, 384] with S = 2944 (3.3e9 elements > 2**31). The
Triton TriMul's fused back half indexed its channel-major contraction in int32 (``channel * L * L``): an illegal address on A100, a
silently wrong result on H100. Both inference and training now index in 64 bits above the bound (the training LN_out kernels over the
m-major contraction view took a WIDE constexpr too, so the training refusal past 2**31 elements was lifted).

The reference of a plane that large does not fit next to the kernel's, so the check uses a symmetry instead: tiling the input R x R over
the token axes leaves every (P x P) block of TriMul's output equal to the output at P (the contraction sums R equal copies, a constant
factor the output LayerNorm removes). Each block of the big output is compared with the float32 reference at P.
"""

import copy

import pytest
import torch

from miniworld_engine.measure import config as C
from miniworld_engine.modules import BidirectionalTriangleMultiplication, TriangleMultiplication

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

P, R, D = 300, 8, 384  # L = 2400: 2.2e9 pair elements (> 2**31), ~27 GB peak


def _card() -> tuple[int, int] | None:
    return torch.cuda.get_device_capability() if torch.cuda.is_available() else None


def _module(outgoing: bool) -> TriangleMultiplication:
    m = TriangleMultiplication(D, outgoing=outgoing).cuda().float().eval()
    C.randomize(m, torch.Generator(device="cuda").manual_seed(0))
    m.set_master_weights()
    return m


@pytest.mark.skipif(_card() not in ((9, 0), (8, 0)), reason="H100 / A100")
@pytest.mark.parametrize("outgoing", [True, False])
def test_triton_inference_past_2_31_elements(outgoing: bool) -> None:
    base = _module(outgoing)
    gen = torch.Generator(device="cuda").manual_seed(1)
    small = base.make_inputs(length=P, samples=1, depth=None, dtype=torch.bfloat16, device="cuda", generator=gen)
    with torch.no_grad():
        gold = copy.deepcopy(base).float()(small["pair"].float(), small["mask"]).float()
        kern = copy.deepcopy(base).set_backend("triton")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            at_p = kern(**small).float()
        pair = small["pair"].repeat(1, R, R, 1).contiguous()
        mask = small["mask"].repeat(1, R)
        assert pair.numel() >= 2**31
        assert kern.unsupported(pair, mask) is None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            big = kern(pair, mask)
    scale = gold.abs().max()
    err_p = float((at_p - gold).abs().max() / scale)
    for i in range(R):
        for j in (0, R - 1):
            block = big[:, i * P : (i + 1) * P, j * P : (j + 1) * P].float()
            err = float((block - gold).abs().max() / scale)
            assert err <= max(1.5 * err_p, 8e-3), f"block ({i}, {j}): {err:.3e} (at P: {err_p:.3e})"


@pytest.mark.skipif(_card() not in ((9, 0), (8, 0)), reason="H100 / A100")
@pytest.mark.parametrize(
    ("bidirectional", "rt"),
    # L = P * rt; the m-major LN_out view of the contraction is (L * L, width) and passes 2**31 elements: uni d_pair 64 / d_hidden 128
    # (protenix1's row) at L 4200 (2.26e9). Bidirectional (width 2 * d_hidden) is refused past the bound: d_pair 128 at L 3000 runs out of
    # 80 GB in training, as does d 384 at L 2400 unidirectional.
    [(False, 14)],
)
def test_triton_training_past_2_31_elements(bidirectional: bool, rt: int) -> None:
    """Training past 2**31 elements of the widest m-major tensor: the output, the pair gradient (an
    upstream gradient tiled like the input gives every block the gradient at P: the rt-fold contraction sum is undone by LN_out's backward)
    and the parameter gradients (rt**2 equal blocks summed) against the float32 reference at P."""
    if bidirectional:
        base, width = BidirectionalTriangleMultiplication(128, p_drop=0.0), 256
    else:
        base, width = TriangleMultiplication(64, d_hidden=128, outgoing=True, p_drop=0.0), 128
    base = base.cuda().float().eval()
    C.randomize(base, torch.Generator(device="cuda").manual_seed(0))
    base.set_master_weights()
    gen = torch.Generator(device="cuda").manual_seed(1)
    small = base.make_inputs(length=P, samples=1, depth=None, dtype=torch.bfloat16, device="cuda", generator=gen)
    dy = torch.randn(small["pair"].shape, device="cuda", generator=gen).to(
        torch.bfloat16
    )  # bf16 values: the same upstream gradient on every run
    names = ("ln_pair.weight", "to_left.weight", "ln_out.weight", "to_out.weight")

    def run(mod, pair, mask, grad, fp32):
        mod.zero_grad(set_to_none=True)
        x = pair.detach().clone().requires_grad_(True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=not fp32):
            out = mod(x, mask)
        out.backward(grad.to(out.dtype))
        params = dict(mod.named_parameters())
        assert x.grad is not None
        return out.detach(), x.grad, {n: params[n].grad.float().clone() for n in names}  # the big plane stays bf16: no float copies

    gold_mod = copy.deepcopy(base).float().train()
    g_out, g_dx, g_dw = run(gold_mod, small["pair"].float(), small["mask"], dy, True)
    del gold_mod
    kern = copy.deepcopy(base).set_backend("triton").train()
    p_out, p_dx, p_dw = run(kern, small["pair"], small["mask"], dy, False)
    pair = small["pair"].repeat(1, rt, rt, 1).contiguous()
    mask = small["mask"].repeat(1, rt)
    assert pair.shape[1] * pair.shape[2] * width >= 2**31
    assert kern.unsupported(pair, mask) is None
    b_out, b_dx, b_dw = run(kern, pair, mask, dy.repeat(1, rt, rt, 1), False)
    del pair
    for what, big, at_p, gold in (("out", b_out, p_out, g_out), ("dpair", b_dx, p_dx, g_dx)):
        gold = gold.float()
        scale = gold.abs().max()
        err_p = float((at_p.float() - gold).abs().max() / scale)
        for i in range(rt):
            for j in (0, rt - 1):
                block = big[:, i * P : (i + 1) * P, j * P : (j + 1) * P].float()
                err = float((block - gold).abs().max() / scale)
                assert err <= max(1.5 * err_p, 1e-2), f"{what} block ({i}, {j}): {err:.3e} (at P: {err_p:.3e})"
    for n in names:
        scale = g_dw[n].abs().max()
        err_p = float((p_dw[n] - g_dw[n]).abs().max() / scale)
        err = float((b_dw[n] / rt**2 - g_dw[n]).abs().max() / scale)
        assert err <= max(1.5 * err_p, 1e-2), f"d{n}: {err:.3e} (at P: {err_p:.3e})"


@pytest.mark.skipif(_card() not in ((9, 0), (8, 0)), reason="H100 / A100")
def test_triton_bidirectional_training_past_2_31_elements_is_refused() -> None:
    kern = BidirectionalTriangleMultiplication(128).cuda().set_backend("triton").train()
    pair = torch.empty(1, 3000, 3000, 128, device="cuda", dtype=torch.bfloat16)
    why = kern.unsupported(pair, None)
    assert why is not None and "2**31" in why
    with torch.no_grad():
        assert kern.eval().unsupported(pair, None) is None
