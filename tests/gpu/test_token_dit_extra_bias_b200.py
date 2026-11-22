"""B200 (sm_100a): OpenDDE's ``TokenDiTProtenix`` + ``extra_pair_bias`` [L, L] at inference is served by the cuda backend (it was refused with "inference: outside
the fused H100 inference block's range" and ran on Triton).  Accuracy against float64: rms / max error, bounded by PyTorch's own error at the same dtype (x 2)
plus a floor; a masked key stays masked; the bias is not dropped."""

from __future__ import annotations

import copy

import pytest
import torch

from miniworld_engine.measure import config as C
from miniworld_engine.modules.token_dit import TokenDiTProtenix

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not (torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 0)), reason="B200 (sm_100a)")]


def _built(dtype: str) -> TokenDiTProtenix:
    m = TokenDiTProtenix().cuda().float()
    C.randomize(m, torch.Generator(device="cuda").manual_seed(0))
    return C.cast(m, dtype).eval()


def _err(y: torch.Tensor, r: torch.Tensor) -> tuple[float, float]:
    d = y.double() - r
    return float(d.pow(2).mean().sqrt() / r.pow(2).mean().sqrt()), float(d.abs().max() / r.abs().max())


def _run(m, inp, backend: str, autocast: bool):
    m.set_backend(backend)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
        return m(**inp)


@pytest.mark.parametrize("dtype", ["master", "float32"])
@pytest.mark.parametrize("samples", [1, 5])
@pytest.mark.parametrize("length", [28, 100, 256, 384])
def test_extra_pair_bias_b200(dtype: str, samples: int, length: int) -> None:
    m = _built(dtype)
    act = torch.bfloat16 if dtype == "master" else torch.float32
    g = torch.Generator(device="cuda").manual_seed(1)
    inp = m.make_inputs(length=length, samples=samples, depth=None, dtype=act, device="cuda", generator=g)
    inp["extra_pair_bias"] = torch.randn(length, length, device="cuda", generator=g)  # fp32 [L, L], as OpenDDE hands it
    ac = dtype == "master"
    with torch.no_grad():
        assert m._supports_cuda(inp["single"], inp["cond"], inp["pair"], inp["mask"], extra_pair_bias=inp["extra_pair_bias"]) is None
    ref = copy.deepcopy(m).double().eval()
    ref.set_backend("pytorch")
    with torch.no_grad():
        r = ref(**{k: (v.double() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in inp.items()})
    y = _run(m, inp, "cuda", ac)
    p = _run(m, inp, "pytorch", ac)
    (rms, mx), (prms, pmx) = _err(y, r), _err(p, r)
    print(f"{dtype} S={samples} L={length}: cuda rms {rms:.2e} max {mx:.2e} | pytorch rms {prms:.2e} max {pmx:.2e}")
    assert rms < max(2 * prms, 4e-3) and mx < max(2 * pmx, 3e-2), (rms, mx, prms, pmx)
    y0 = _run(m, {**inp, "extra_pair_bias": None}, "cuda", ac)
    assert _err(y0, r)[0] > 3 * rms or _err(y0, r)[0] > 5e-3  # not dropped


@pytest.mark.parametrize("dtype", ["master", "float32"])
@pytest.mark.parametrize("length", [100, 256])
def test_extra_pair_bias_masked_keys_b200(dtype: str, length: int) -> None:
    m = _built(dtype)
    act = torch.bfloat16 if dtype == "master" else torch.float32
    pad = 17
    g = torch.Generator(device="cuda").manual_seed(3)
    inp = m.make_inputs(length=length, samples=2, depth=None, dtype=act, device="cuda", generator=g)
    mask = torch.ones(1, length, dtype=torch.bool, device="cuda")
    mask[:, length - pad :] = False
    inp["mask"] = mask
    extra = torch.randn(length, length, device="cuda", generator=g)
    huge = extra.clone()
    huge[:, length - pad :] = 1e3
    ac = dtype == "master"
    ref = copy.deepcopy(m).double().eval()
    ref.set_backend("pytorch")
    with torch.no_grad():
        r = ref(**{k: (v.double() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in {**inp, "extra_pair_bias": extra}.items()})
    y = _run(m, {**inp, "extra_pair_bias": extra}, "cuda", ac)
    yh = _run(m, {**inp, "extra_pair_bias": huge}, "cuda", ac)
    assert _err(y[..., : length - pad, :], r[..., : length - pad, :])[0] < 5e-3
    assert _err(yh[..., : length - pad, :], y[..., : length - pad, :].double())[0] < 5e-3  # inert on masked keys
