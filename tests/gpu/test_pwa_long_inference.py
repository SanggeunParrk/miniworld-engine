"""E3 (FoldForge 5I28): ``MSAPairWeightedAveraging``'s cuda backend serves inference past 1024 tokens at the 8 x 8 / 8 x 16 widths (AF3 / OpenFold3 /
Protenix-v1, Protenix-v2, ESMFold2, OpenDDE) -- H100 ``pwa_wide`` up to 2048, A100 ``pwa_sm80`` up to 2048 (the pair forward's row of
logits in bf16 past what fp32 holds) -- and matches the float64 reference as closely as the ``pytorch`` backend does; training past 1024 is refused with a reason."""

import copy

import pytest
import torch

from miniworld_engine.measure import config as C
from miniworld_engine.modules import MSAPairWeightedAveraging

pytestmark = [pytest.mark.gpu]


def _card() -> tuple[int, int] | None:
    return torch.cuda.get_device_capability() if torch.cuda.is_available() else None


def _module(d_msa: int, d_pair: int, d_hidden: int, gen: torch.Generator) -> MSAPairWeightedAveraging:
    m = MSAPairWeightedAveraging(d_msa, d_pair, 8, d_hidden=d_hidden, p_drop=0.25).cuda().eval()
    C.randomize(m, gen)
    return m


@pytest.mark.skipif(_card() not in ((9, 0), (8, 0)), reason="H100 / A100")
@pytest.mark.parametrize(
    ("d_msa", "d_pair", "d_hidden", "length"), [(64, 128, 8, 2048), (128, 256, 16, 1536), (128, 384, 8, 1536), (128, 384, 8, 2048)]
)
def test_long_inference_on_cuda(d_msa: int, d_pair: int, d_hidden: int, length: int) -> None:
    gen = torch.Generator(device="cuda").manual_seed(0)
    base = _module(d_msa, d_pair, d_hidden, gen)
    inp = base.make_inputs(length=length, samples=1, depth=32, dtype=torch.bfloat16, device="cuda", generator=gen)
    kern = copy.deepcopy(base).set_backend("cuda")
    with torch.no_grad():
        why = kern.unsupported(**inp)
    assert why is None, why
    with torch.no_grad():
        gold = copy.deepcopy(base).double()(inp["msa"].double(), inp["pair"].double(), inp["mask"]).double()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ref = base(**inp).double()
            got = kern(**inp).double()
    scale = gold.abs().max()
    err_ref = float((ref - gold).abs().max() / scale)
    err = float((got - gold).abs().max() / scale)
    assert err <= max(err_ref, 8e-3), f"cuda err {err:.3e}, pytorch {err_ref:.3e}"


@pytest.mark.skipif(_card() not in ((9, 0), (8, 0)), reason="H100 / A100")
def test_long_training_is_refused() -> None:
    gen = torch.Generator(device="cuda").manual_seed(0)
    kern = _module(64, 128, 8, gen).set_backend("cuda")
    inp = kern.make_inputs(length=1536, samples=1, depth=32, dtype=torch.bfloat16, device="cuda", generator=gen)
    with torch.enable_grad():
        why = kern.unsupported(**inp)
    assert why is not None and "training" in why, why
