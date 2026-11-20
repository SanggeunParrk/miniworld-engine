"""A100 ``pwa_sm80`` inference past L 2048 (FoldForge AF3 n >= 17: L 2176 .. 3200+): ``pair_fwd`` stages the bf16 logits in the output rows when the
shared-memory row does not fit.  The cuda backend must serve these lengths (no CUDA hole) and match a row-chunked float64 reference as closely as the
``pytorch`` backend does (the engine accuracy standard, not bit-identity)."""

import copy

import pytest
import torch
import torch.nn.functional as F

from miniworld_engine.measure import config as C
from miniworld_engine.modules import MSAPairWeightedAveraging

pytestmark = [pytest.mark.gpu]


def _card() -> tuple[int, int] | None:
    return torch.cuda.get_device_capability() if torch.cuda.is_available() else None


def gold_f64(mod: MSAPairWeightedAveraging, msa: torch.Tensor, pair: torch.Tensor, mask: torch.Tensor | None, rows: int = 128) -> torch.Tensor:
    """The module's math in float64 with the pair side (LN, bias projection, softmax) taken ``rows`` query rows at a time (a full float64 pair does not fit)."""
    m = copy.deepcopy(mod).double()
    h = m.n_head
    x = msa[0].double()  # [S, L, D]
    length = x.shape[1]
    w = torch.empty(length, length, h, dtype=torch.float64, device=x.device)
    lnb = m.ln_pair.bias
    for i0 in range(0, length, rows):
        zz = pair[0, i0 : i0 + rows].double()
        lg = F.linear(F.layer_norm(zz, (zz.shape[-1],), m.ln_pair.weight, lnb, m.ln_pair.eps), m.to_bias.weight)  # [r, L, h]
        if mask is not None:
            lg = lg.masked_fill(~mask[0][None, :, None], float("-inf"))
        w[i0 : i0 + rows] = torch.softmax(lg, dim=1)
    xn = m.ln_msa(x)
    v = m.to_value(xn).unflatten(-1, (h, -1))  # [S, L, h, c]
    o = torch.einsum("ijh,sjhc->sihc", w, v).flatten(-2)
    return (x + m.to_out(torch.sigmoid(m.to_gate(xn)) * o))[None]


@pytest.mark.skipif(_card() != (8, 0), reason="A100")
@pytest.mark.parametrize(
    ("d_msa", "d_pair", "d_hidden"),
    [(64, 128, 8), (64, 128, 32), (64, 256, 32), (128, 256, 16), (128, 384, 8)],
)
@pytest.mark.parametrize("length", [2176, 3072, 4096])
def test_pwa_sm80_inference_past_2048(d_msa: int, d_pair: int, d_hidden: int, length: int) -> None:
    gen = torch.Generator(device="cuda").manual_seed(0)
    base = MSAPairWeightedAveraging(d_msa, d_pair, 8, d_hidden=d_hidden, p_drop=0.25).cuda().eval()
    C.randomize(base, gen)
    inp = base.make_inputs(length=length, samples=1, depth=32, dtype=torch.bfloat16, device="cuda", generator=gen)
    kern = copy.deepcopy(base).set_backend("cuda")
    with torch.no_grad():
        assert kern.unsupported(**inp) is None
        gold = gold_f64(base, inp["msa"], inp["pair"], inp["mask"])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ref = base(**inp).double()
            got = kern(**inp).double()
    scale = gold.abs().max()
    err_ref = float((ref - gold).abs().max() / scale)
    err = float((got - gold).abs().max() / scale)
    rms_ref = float(((ref - gold) ** 2).mean().sqrt() / gold.pow(2).mean().sqrt())
    rms = float(((got - gold) ** 2).mean().sqrt() / gold.pow(2).mean().sqrt())
    assert err <= max(err_ref, 8e-3) and rms <= max(rms_ref, 4e-3), f"cuda max {err:.3e} rms {rms:.3e}; pytorch max {err_ref:.3e} rms {rms_ref:.3e}"
