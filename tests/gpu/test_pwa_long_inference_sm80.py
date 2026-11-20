"""A100 ``pwa_sm80`` inference past L 2048 (FoldForge AF3 n >= 17: L 2176 .. 3200+): ``pair_fwd`` stages the bf16 logits in the output rows when the
shared-memory row does not fit.  The cuda backend must serve these lengths (no CUDA hole) and match a row-chunked float64 reference as closely as the
``pytorch`` backend does (the engine accuracy standard, not bit-identity)."""

import copy

import pytest
import torch
import torch.nn.functional as F

from miniworld_engine.measure import config as C
from miniworld_engine.modules import MSAPairWeightedAveraging

pytestmark = [pytest.mark.gpu, pytest.mark.a100]


def _card() -> tuple[int, int] | None:
    return torch.cuda.get_device_capability() if torch.cuda.is_available() else None


def gold_f64(
    mod: MSAPairWeightedAveraging, msa: torch.Tensor, pair: torch.Tensor, mask: torch.Tensor | None, rows: int = 128
) -> torch.Tensor:
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


_NARROW = [(64, 128, 8), (64, 128, 32), (64, 256, 32), (128, 256, 16), (128, 384, 8)]
_CASES = [(*w, n) for w in _NARROW for n in (2176, 3072, 4096)]
# the BIG width (pwa_wide_sm80: cuBLAS composition, no length bound but the pair itself): not at L 4096, where the pytorch reference (fp32 LayerNorm of the pair, 32 GiB) does not fit next to the pair
_CASES += [(256, 512, 32, 2176), (256, 512, 32, 3072)]


@pytest.mark.skipif(_card() != (8, 0), reason="A100")
@pytest.mark.parametrize(("d_msa", "d_pair", "d_hidden", "length"), _CASES)
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
    assert err <= max(err_ref, 8e-3) and rms <= max(rms_ref, 4e-3), (
        f"cuda max {err:.3e} rms {rms:.3e}; pytorch max {err_ref:.3e} rms {rms_ref:.3e}"
    )


@pytest.mark.skipif(_card() != (8, 0), reason="A100")
@pytest.mark.parametrize(("d_msa", "d_pair", "d_hidden"), _NARROW)
def test_pwa_sm80_inference_gate(d_msa: int, d_pair: int, d_hidden: int) -> None:
    """The narrow kernels take L a multiple of 16 up to MAX_LEN_INFER (8192) and refuse the rest by a reason (the call then falls to another backend); shapes only (no memory)."""
    mod = MSAPairWeightedAveraging(d_msa, d_pair, 8, d_hidden=d_hidden, p_drop=0.25).cuda().eval().set_backend("cuda")

    def refusal(length: int) -> str | None:
        msa = torch.empty(1, 1, 1, d_msa, dtype=torch.bfloat16, device="cuda").expand(1, 4, length, d_msa)
        pair = torch.empty(1, 1, 1, d_pair, dtype=torch.bfloat16, device="cuda").expand(
            1, length, length, d_pair
        )  # stride 0: no allocation
        with torch.no_grad():
            return mod.unsupported(msa=msa, pair=pair, mask=None)

    assert refusal(8192) is None
    assert refusal(8208) is not None  # past the bound
    assert refusal(2056) is not None  # not a multiple of 16
