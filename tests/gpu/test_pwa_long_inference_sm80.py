"""``pwa_sm80`` inference past L 2048 on the A100 and the RTX 5090 (FoldForge AF3 n >= 17: L 2176 .. 3200+): ``pair_fwd`` stages the bf16 logits in the output rows when the
shared-memory row does not fit.  The cuda backend must serve these lengths (no CUDA hole) and match a row-chunked float64 reference as closely as the
``pytorch`` backend does (the engine accuracy standard, not bit-identity)."""

import copy
import gc

import pytest
import torch
import torch.nn.functional as F

from miniworld_engine.measure import config as C
from miniworld_engine.modules import MSAPairWeightedAveraging
from miniworld_engine.modules.primitives import key_mask, random_input

pytestmark = [pytest.mark.gpu, pytest.mark.a100]


def _card() -> tuple[int, int] | None:
    return torch.cuda.get_device_capability() if torch.cuda.is_available() else None


def gold_f64(
    mod: MSAPairWeightedAveraging, msa: torch.Tensor, pair: torch.Tensor, mask: torch.Tensor | None, rows: int | None = None
) -> torch.Tensor:
    """The module's math in float64 with the pair side (LN, bias projection, softmax) taken ``rows`` query rows at a time (a full float64 pair does not fit)."""
    m = copy.deepcopy(mod).double()
    h = m.n_head
    x = msa[0].double()  # [S, L, D]
    length = x.shape[1]
    if rows is None:  # a float64 chunk of the pair stays near 64 MiB (the 32 GB card holds a pair of 28 GiB at the longest L)
        rows = max(4, min(128, 2**23 // (length * pair.shape[-1])))
    lnb = m.ln_pair.bias
    xn = m.ln_msa(x)
    v = m.to_value(xn).unflatten(-1, (h, -1))  # [S, L, h, c]
    o = torch.empty(x.shape[0], length, h, v.shape[-1], dtype=torch.float64, device=x.device)
    for i0 in range(0, length, rows):
        zz = pair[0, i0 : i0 + rows].double()
        lg = F.linear(F.layer_norm(zz, (zz.shape[-1],), m.ln_pair.weight, lnb, m.ln_pair.eps), m.to_bias.weight)  # [r, L, h]
        if mask is not None:
            lg = lg.masked_fill(~mask[0][None, :, None], float("-inf"))
        o[:, i0 : i0 + rows] = torch.einsum(
            "ijh,sjhc->sihc", torch.softmax(lg, dim=1), v
        )  # w is never held whole: [L, L, h] float64 is 2.6 GB at L 6336
    return (x + m.to_out(torch.sigmoid(m.to_gate(xn)) * o.flatten(-2)))[None]


def make_inputs_chunked(mod: MSAPairWeightedAveraging, length: int, depth: int, gen: torch.Generator, rows: int = 32) -> dict:
    """``mod.make_inputs`` with the pair drawn ``rows`` rows at a time: its one fp32 draw of [L, L, d_pair] needs 3x the bf16 pair (more than the 32 GB card has at the longest L)."""
    d_msa, d_pair = mod.ln_msa.normalized_shape[0], mod.ln_pair.normalized_shape[0]
    pair = torch.empty(1, length, length, d_pair, dtype=torch.bfloat16, device="cuda")
    for i0 in range(0, length, rows):
        pair[0, i0 : i0 + rows] = random_input((min(rows, length - i0), length, d_pair), dtype=torch.bfloat16, device="cuda", generator=gen)
    return {
        "msa": random_input((1, depth, length, d_msa), dtype=torch.bfloat16, device="cuda", generator=gen),
        "pair": pair,
        "mask": key_mask(1, length, device="cuda"),
    }


@pytest.fixture(autouse=True)
def _free_cuda() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def _longest(d_pair: int) -> int:
    from miniworld_engine.kernels.pair_weighted_averaging.cuda import sm80

    return sm80.max_len(d_pair, False)


_NARROW = [(64, 128, 8), (64, 128, 32), (64, 256, 32), (128, 256, 16), (128, 384, 8)]
_CASES = [(*w, n) for w in _NARROW for n in (2176, 3072, 4096, "max")]
# the BIG width (pwa_wide_sm80: cuBLAS composition, no length bound but the pair itself): not at L 4096, where the pytorch reference (fp32 LayerNorm of the pair, 32 GiB) does not fit next to the pair
_CASES += [(256, 512, 32, 2176), (256, 512, 32, 3072)]


@pytest.mark.skipif(_card() not in ((8, 0), (12, 0)), reason="A100 / RTX 5090")
@pytest.mark.parametrize(("d_msa", "d_pair", "d_hidden", "want"), _CASES)
def test_pwa_sm80_inference_past_2048(d_msa: int, d_pair: int, d_hidden: int, want: int | str) -> None:
    longest = _longest(d_pair) if d_pair < 512 else 8192  # the BIG width has no length bound of its own
    length = longest if want == "max" else int(want)
    if length > longest:
        pytest.skip(f"L {length} is past this card's limit {longest} at d_pair {d_pair}")
    gen = torch.Generator(device="cuda").manual_seed(0)
    base = MSAPairWeightedAveraging(d_msa, d_pair, 8, d_hidden=d_hidden, p_drop=0.25).cuda().eval()
    C.randomize(base, gen)
    # at the card's longest L the bf16 pair alone is ~28 GiB of the 32 GB: the MSA depth shrinks to 8 so the float64 gold (v, o, x: ~S*L*h*c*8 bytes each) fits next to it
    inp = make_inputs_chunked(base, length, 8 if want == "max" else 32, gen)
    kern = copy.deepcopy(base).set_backend("cuda")
    with torch.no_grad():
        assert kern.unsupported(**inp) is None
        gold = gold_f64(base, inp["msa"], inp["pair"], inp["mask"])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            got = kern(**inp).double()
            try:
                ref = base(**inp).double()  # the pytorch backend's own float32 intermediates may not fit the 32 GB card at the longest L
            except torch.OutOfMemoryError:
                ref = None
                torch.cuda.empty_cache()
    scale = gold.abs().max()
    rms_gold = gold.pow(2).mean().sqrt()
    err = float((got - gold).abs().max() / scale)
    rms = float(((got - gold) ** 2).mean().sqrt() / rms_gold)
    err_ref = 0.0 if ref is None else float((ref - gold).abs().max() / scale)
    rms_ref = 0.0 if ref is None else float(((ref - gold) ** 2).mean().sqrt() / rms_gold)
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


@pytest.mark.skipif(_card() != (12, 0), reason="RTX 5090")
@pytest.mark.parametrize("d_pair", [128, 256, 384])
def test_pwa_sm120_refuses_past_its_limit(d_pair: int) -> None:
    """Past the 5090's limit the cuda backend refuses cleanly (another backend serves it, no OOM): shapes only, nothing is allocated for the pair."""
    longest = _longest(d_pair)
    base = MSAPairWeightedAveraging(64, d_pair, 8, d_hidden=8, p_drop=0.25).cuda().eval().set_backend("cuda")
    length = longest + 16
    msa = torch.empty(1, 1, 1, 64, dtype=torch.bfloat16, device="cuda").expand(
        1, 4, length, 64
    )  # expanded views: the shape without the memory
    pair = torch.empty(1, 1, 1, d_pair, dtype=torch.bfloat16, device="cuda").expand(1, length, length, d_pair)
    from miniworld_engine.kernels.integrations import pwa_sm80

    assert pwa_sm80.refusal(base, msa, pair, None) is not None
