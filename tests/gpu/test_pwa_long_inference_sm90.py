"""H100 ``pwa_train`` / ``pwa_wide`` inference past N 2048 (FoldForge AF3 n >= 17: N 2176 .. 3200+): ``pair_fwd3`` stages the bf16 logits in the output rows
(``csrc/pair3.cu`` / ``pair_wide.cu`` GLOB) when the shared-memory row does not fit.  The cuda backend must serve these lengths (no CUDA hole) and match a
row-chunked float64 reference as closely as the ``pytorch`` backend does (the engine accuracy standard, not bit-identity); N <= 2048 keeps the shared-memory
path, and the output-row path reproduces it bit for bit there (wide pair kernels)."""

import copy
import gc

import pytest
import torch
import torch.nn.functional as F

from foldjet_engine.measure import config as C
from foldjet_engine.modules import MSAPairWeightedAveraging

pytestmark = [pytest.mark.gpu]


def _card() -> tuple[int, int] | None:
    return torch.cuda.get_device_capability() if torch.cuda.is_available() else None


WIDTHS = [(64, 128, 8), (64, 128, 32), (64, 256, 32), (128, 256, 8), (128, 256, 16), (128, 384, 8), (256, 512, 32)]


def gold_f64(
    mod: MSAPairWeightedAveraging, msa: torch.Tensor, pair: torch.Tensor, mask: torch.Tensor | None, rows: int = 64
) -> torch.Tensor:
    """The module's math in float64 with the pair side (LN, bias projection, softmax) taken ``rows`` query rows at a time (a full float64 pair does not fit)."""
    m = copy.deepcopy(mod).double()
    h = m.n_head
    x = msa[0].double()  # [S, L, D]
    length = x.shape[1]
    w = torch.empty(length, length, h, dtype=torch.float64, device=x.device)
    for i0 in range(0, length, rows):
        zz = pair[0, i0 : i0 + rows].double()
        lg = F.linear(F.layer_norm(zz, (zz.shape[-1],), m.ln_pair.weight, m.ln_pair.bias, m.ln_pair.eps), m.to_bias.weight)  # [r, L, h]
        if mask is not None:
            lg = lg.masked_fill(~mask[0][None, :, None], float("-inf"))
        w[i0 : i0 + rows] = torch.softmax(lg, dim=1)
    xn = m.ln_msa(x)
    v = m.to_value(xn).unflatten(-1, (h, -1))  # [S, L, h, c]
    o = torch.einsum("ijh,sjhc->sihc", w, v).flatten(-2)
    return (x + m.to_out(torch.sigmoid(m.to_gate(xn)) * o))[None]


def _errs(out: torch.Tensor, gold: torch.Tensor) -> tuple[float, float]:
    d = out.double() - gold
    return float(d.abs().max() / gold.abs().max()), float(d.pow(2).mean().sqrt() / gold.pow(2).mean().sqrt())


def _check(d_msa: int, d_pair: int, d_hidden: int, length: int) -> None:
    gc.collect()
    torch.cuda.empty_cache()  # the previous case's cached blocks: make_inputs draws the pair through an fp32 temporary
    gen = torch.Generator(device="cuda").manual_seed(0)
    base = MSAPairWeightedAveraging(d_msa, d_pair, 8, d_hidden=d_hidden, p_drop=0.25).cuda().eval()
    C.randomize(base, gen)
    inp = base.make_inputs(length=length, samples=1, depth=32, dtype=torch.bfloat16, device="cuda", generator=gen)
    kern = copy.deepcopy(base).set_backend("cuda")
    with torch.no_grad():
        assert kern.unsupported(**inp) is None
        gold = gold_f64(base, inp["msa"], inp["pair"], inp["mask"])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            err, rms = _errs(kern(**inp), gold)
            try:  # the pytorch backend materialises a [8, L, L] fp32 logit tensor: it does not always fit next to the pair at the long lengths
                err_ref, rms_ref = _errs(base(**inp), gold)
            except torch.OutOfMemoryError:
                err_ref = rms_ref = 0.0
            finally:
                torch.cuda.empty_cache()
    assert err <= max(err_ref, 8e-3) and rms <= max(rms_ref, 4e-3), (
        f"cuda max {err:.3e} rms {rms:.3e}; pytorch max {err_ref:.3e} rms {rms_ref:.3e}"
    )


@pytest.mark.skipif(_card() != (9, 0), reason="H100")
@pytest.mark.parametrize(("d_msa", "d_pair", "d_hidden"), WIDTHS)
@pytest.mark.parametrize("length", [2176, 3072])
def test_pwa_sm90_inference_past_2048(d_msa: int, d_pair: int, d_hidden: int, length: int) -> None:
    _check(d_msa, d_pair, d_hidden, length)


@pytest.mark.slow
@pytest.mark.skipif(_card() != (9, 0), reason="H100")
@pytest.mark.parametrize(("d_msa", "d_pair", "d_hidden"), WIDTHS)
def test_pwa_sm90_inference_4096(d_msa: int, d_pair: int, d_hidden: int) -> None:
    _check(d_msa, d_pair, d_hidden, 4096)


@pytest.mark.skipif(_card() != (9, 0), reason="H100")
@pytest.mark.parametrize("d_pair", [128, 256, 384, 512])
def test_pair_fwd3_output_rows_equal_shared_memory_row(d_pair: int) -> None:
    """At N <= 2048 (both paths build) the logits staged in the output rows give the shared-memory path's weights bit for bit."""
    from foldjet_engine.kernels.integrations import pwa_wide

    n = 1024
    gen = torch.Generator(device="cuda").manual_seed(1)
    z = (torch.randn(n, n, d_pair, device="cuda", generator=gen) * 2).bfloat16()
    wb = (torch.randn(8, d_pair, device="cuda", generator=gen) / d_pair**0.5).bfloat16()
    lw = (1 + 0.1 * torch.randn(d_pair, device="cuda", generator=gen)).float()
    pm = (torch.rand(n, device="cuda", generator=gen) > 0.1).bfloat16()
    k = pwa_wide._pair(d_pair)
    args = (z, pm, lw, torch.zeros_like(lw), 1e-5, wb)
    assert torch.equal(k.pair_fwd3(*args, 0), k.pair_fwd3(*args, 1))


@pytest.mark.skipif(_card() != (9, 0), reason="H100")
def test_pwa_sm90_refuses_past_the_bound_cleanly() -> None:
    mod = MSAPairWeightedAveraging(64, 128, 8, d_hidden=8).cuda().eval().set_backend("cuda")
    msa = torch.empty(1, 2, 8208, 64, device="cuda", dtype=torch.bfloat16)
    pair = torch.empty(1, 1, 1, 128, device="cuda", dtype=torch.bfloat16).expand(1, 8208, 8208, 128)
    with torch.no_grad():
        why = mod.unsupported(msa=msa, pair=pair, mask=None)
    assert why is not None and "8192" in why
