"""FoldForge R21: ``MSAPairWeightedAveraging``'s cuda backend TRAINS past 1024 tokens (1536 / 2048) at every requirement width on H100 and A100:
H100 ``pwa_train`` (d_msa 64, 8 x 32: pair3.cu / pair_wide.cu's pair backward reads the w / dw rows from global memory past what its tile
ring stages) and ``pwa_wide`` (d_msa 128 at 8 x 8 / 16, the BIG build), A100 ``pwa_sm80`` (the pair backward's LONG variant streams the row)
and ``pwa_wide_sm80`` (d_msa 256, d_pair 512). Checked: the forward output and every gradient against a float32 reference (float64 does not
fit at 2048 x d_pair 512) no worse than 5x the ``pytorch`` backend's error (no RED) or 8e-3, bitwise reproducible across two runs, and the
peak memory of the cuda step printed."""

import copy

import pytest
import torch

from miniworld_engine.measure import config as C
from miniworld_engine.modules import MSAPairWeightedAveraging

pytestmark = [pytest.mark.gpu]

FIXED = 8e-3


def _card() -> tuple[int, int] | None:
    return torch.cuda.get_device_capability() if torch.cuda.is_available() else None


def _run(module, inp: dict, dtype: torch.dtype, dy: torch.Tensor, autocast: bool) -> list[torch.Tensor]:
    """The forward output and the gradients of msa, pair and every parameter (fp32 copies on the host: at 2048 x d_pair 512 three sets of
    pair gradients do not fit next to the reference runs on the card)."""
    msa = inp["msa"].detach().to(dtype).requires_grad_(True)
    pair = inp["pair"].detach().to(dtype).requires_grad_(True)
    module.zero_grad(set_to_none=True)
    with torch.enable_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
        out = module(msa, pair, inp["mask"])
    out.backward(dy.to(out.dtype))
    res = [out.detach().float().cpu(), msa.grad.float().cpu(), pair.grad.float().cpu()]
    del out, msa, pair
    res += [p.grad.float().cpu() for p in module.parameters()]
    module.zero_grad(set_to_none=True)
    return res


WIDTHS = [  # requirement rows 49-55: (d_msa, d_pair, d_hidden)
    (64, 128, 8),
    (64, 128, 32),
    (64, 256, 32),
    (128, 256, 8),
    (128, 256, 16),
    (128, 384, 8),
    (256, 512, 32),
]


@pytest.mark.skipif(_card() not in ((9, 0), (8, 0)), reason="H100 / A100")
@pytest.mark.parametrize("length", [1536, 2048])
@pytest.mark.parametrize(("d_msa", "d_pair", "d_hidden"), WIDTHS)
def test_long_training_on_cuda(d_msa: int, d_pair: int, d_hidden: int, length: int) -> None:
    gen = torch.Generator(device="cuda").manual_seed(0)
    base = MSAPairWeightedAveraging(d_msa, d_pair, 8, d_hidden=d_hidden, p_drop=0.25).cuda().eval()  # eval: no dropout draw (numerics only)
    C.randomize(base, gen)
    inp = base.make_inputs(length=length, samples=1, depth=32, dtype=torch.bfloat16, device="cuda", generator=gen)
    dy = torch.randn(inp["msa"].shape, device="cuda", generator=gen, dtype=torch.float32)
    kern = copy.deepcopy(base).set_backend("cuda")
    with torch.enable_grad():
        why = kern.unsupported(**inp)
    assert why is None, why

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    got = _run(kern, inp, torch.bfloat16, dy, True)
    torch.cuda.synchronize()
    peak = (torch.cuda.max_memory_allocated() - before) / 2**30
    again = _run(kern, inp, torch.bfloat16, dy, True)
    same = all(torch.equal(a, b) for a, b in zip(got, again, strict=True))
    del again
    ref = _run(base, inp, torch.bfloat16, dy, True)
    gold = _run(copy.deepcopy(base).float(), inp, torch.float32, dy, False)

    names = ["out", "d_msa", "d_pair", *[n for n, _ in base.named_parameters()]]
    worst, lines = 0.0, []
    for name, g, r, k in zip(names, gold, ref, got, strict=True):
        scale = float(g.abs().max()) or 1.0
        err_ref = float((r - g).abs().max()) / scale
        err = float((k - g).abs().max()) / scale
        tol = max(err_ref, FIXED)
        worst = max(worst, err / tol)
        lines.append(f"{name} {err:.2e} (pytorch {err_ref:.2e})")
    print(
        f"\nL{length} d_msa {d_msa} d_pair {d_pair} 8x{d_hidden}: peak {peak:.2f} GiB, worst err/tol {worst:.2f}, deterministic {same}: "
        + "; ".join(lines)
    )
    assert same, "two cuda training steps differ"
    assert worst <= 5.0, f"RED: worst err/tol {worst:.2f}: " + "; ".join(lines)
