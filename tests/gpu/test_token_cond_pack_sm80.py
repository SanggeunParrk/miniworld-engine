"""A100: the token DiT block's six conditioning projections as two small GEMMs on the distinct conditioning rows (``kernels/integrations/token_cond_sm80``).

A sampler step gives every block the same conditioning rows for all of its samples (``cond`` expanded over the samples, stride 0). The pack must (a) engage for that call and not
for a per-sample conditioning-free fp32 / training call, (b) stay as close to the fp32 PyTorch block as the unpacked sm_80 path does (the LayerNorm weight is folded into the bf16
projection weights: another rounding of the same size), (c) be keyed on the weights (an in-place weight change is read).
"""

from __future__ import annotations

import copy
import typing

import pytest
import torch

from foldjet_engine.kernels.integrations import token_cond_sm80
from foldjet_engine.measure import config as C

pytestmark = [pytest.mark.gpu]


def _a100() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (8, 0)


def _models(d_cond: int) -> tuple[typing.Any, typing.Any]:
    from foldjet_engine.modules.token_dit import TOKEN_DIT_MODELS

    m = TOKEN_DIT_MODELS["protenix"](d_single=768, d_cond=d_cond, d_pair=128, n_head=16, n=2).cuda().float()
    C.randomize(m, torch.Generator(device="cuda").manual_seed(0))
    ref = copy.deepcopy(m).set_backend("pytorch").eval()
    built: typing.Any = C.cast(m, "bfloat16")
    return built.set_backend("cuda").eval(), ref


def _rms(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.double() - b.double()).pow(2).mean().sqrt())


@pytest.mark.skipif(not _a100(), reason="A100 (sm_80)")
@pytest.mark.parametrize("d_cond", [384, 768])
@pytest.mark.parametrize("L", [128, 256])
@pytest.mark.parametrize("samples", [5, 1])
def test_pack_matches_the_unpacked_path_in_accuracy(d_cond: int, L: int, samples: int, monkeypatch: pytest.MonkeyPatch) -> None:
    m, ref = _models(d_cond)
    inp = m.make_inputs(length=L, samples=samples, depth=None, dtype=torch.bfloat16, device="cuda", generator=torch.Generator(device="cuda").manual_seed(1))
    cond = inp["cond"][:1].expand_as(inp["cond"])  # one conditioning shared by the samples: stride 0
    mask = torch.ones(1, L, dtype=torch.bool, device="cuda")
    mask[0, L - L // 8 :] = False
    kw = dict(inp, cond=cond, mask=mask)
    if m.unsupported(**kw) is not None:
        pytest.skip(f"not served: {m.unsupported(**kw)}")
    assert token_cond_sm80.serves(m, kw["single"], kw["cond"], False)
    with torch.no_grad():
        monkeypatch.setenv("MINIWORLD_TOKEN_COND_PACK", "0")
        old = m(**kw).clone()
        monkeypatch.setenv("MINIWORLD_TOKEN_COND_PACK", "1")
        new = m(**kw).clone()
        refkw = {k: (v.float() if isinstance(v, torch.Tensor) and v.is_floating_point() else v) for k, v in kw.items()}
        want = ref(**refkw)
    assert torch.isfinite(new).all()
    e_old, e_new = _rms(old, want), _rms(new, want)
    assert e_new <= 1.5 * e_old + 1e-6, (e_old, e_new)
    assert float((new.double() - want.double()).abs().max()) <= 1.5 * float((old.double() - want.double()).abs().max()) + 1e-5


@pytest.mark.skipif(not _a100(), reason="A100 (sm_80)")
def test_pack_reads_a_changed_weight(monkeypatch: pytest.MonkeyPatch) -> None:
    m, _ = _models(384)
    inp = m.make_inputs(length=128, samples=5, depth=None, dtype=torch.bfloat16, device="cuda", generator=torch.Generator(device="cuda").manual_seed(1))
    kw = dict(inp, cond=inp["cond"][:1].expand_as(inp["cond"]), mask=torch.ones(1, 128, dtype=torch.bool, device="cuda"))
    with torch.no_grad():
        monkeypatch.setenv("MINIWORLD_TOKEN_COND_PACK", "1")
        a = m(**kw).clone()
        m.transition.to_scale.bias.mul_(0.5)  # the gate of the transition, in place
        b = m(**kw).clone()
        m.attention.ada_ln_in.ln_cond.weight.mul_(1.5)
        c = m(**kw).clone()
        monkeypatch.setenv("MINIWORLD_TOKEN_COND_PACK", "0")
        c0 = m(**kw).clone()
    assert not torch.equal(a, b) and not torch.equal(b, c)
    assert _rms(c, c0) <= 0.05 * float(c0.double().pow(2).mean().sqrt()) + 1e-6
