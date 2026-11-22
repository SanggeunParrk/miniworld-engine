"""B200 TriangleAttention inference with a logit far above the row's first 32 keys: the ``cuda`` backend must stay finite and as accurate as the
``pytorch`` backend (engine accuracy standard against a float64 gold).

Why this test exists: the B200 inference cores fix a row's softmax offset from its first 32 keys and raise it only in a guard after the tile's exponentials
(sum past 2^64, inf or NaN: the tile is then recomputed exactly). The exponentials that run on the FMA pipe (``ex2_poly2``: Cody-Waite integer part into the
exponent bits) had no upper clamp: a logit >= 2^129 above the offset carried out of the exponent field -> a tiny finite or NaN P that the guard did not see
(real AF3 / Chai-1 targets on the B200: 'non-finite AF3 coordinates', 2026-11-22; scores reach 160 log2 units, a 143-unit spike over the first 32 keys).
N(0, 1) judge inputs never get there. The test scales the q / k projection weights until the largest (row max - first-32-key max) is 130 .. 160 log2 units
(measured on the float64 gold's own scores), for every B200 inference core that has the FMA-pipe exponential:

* ``af3_d128_L256``: head dim 32, the fused front + ``triattn_sm100`` core (the AF3 / RF3 d_pair 128 route);
* ``af3_d128_L128``: the L = 128 fused kernel (``tri_l128_sm100``; its softmax takes the exact row maximum first, so it was never reachable -- kept as a
  guard for the clamp this file's kernel shares);
* ``af3_d64_h16``: head dim 16 (TA_D16: no per-tile guard, the row sum is the PV GEMM's channel 16; a non-finite l / O sends the row to ``d16_exact``);
* ``chai1_d256_dh64``: head dim 64 (``triattn_dh64_sm100``).
"""

from __future__ import annotations

import copy
import gc

import pytest
import torch

from miniworld_engine.measure import config as C

pytestmark = [pytest.mark.gpu]


def _needs_b200() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("needs a B200 (sm_100)")


L2E = 1.4426950408889634
#: (id, module, kwargs, length)
CASES = [
    ("af3_d128_L256", "TriangleAttentionAF3", {"d_pair": 128, "n_head": 4, "starting": True, "p_drop": 0.25}, 256),
    ("af3_d128_L128", "TriangleAttentionAF3", {"d_pair": 128, "n_head": 4, "starting": True, "p_drop": 0.25}, 128),
    ("af3_d64_h16", "TriangleAttentionAF3", {"d_pair": 64, "n_head": 4, "starting": True, "p_drop": 0.25}, 256),
    ("chai1_d256_dh64", "TriangleAttentionChai1", {"d_pair": 256, "n_head": 4, "starting": True, "p_drop": 0.25}, 256),
]


def _build(module: str, kwargs: dict, seed: int = 0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    base = C.construct(C.module_class(module), kwargs).to("cuda").float()
    C.randomize(base, gen)
    return C.cast(base, "master").eval(), gen


def _gold(base, inputs: dict) -> torch.Tensor:
    gm = copy.deepcopy(base).double()
    gi = {k: v.double() if isinstance(v, torch.Tensor) and v.is_floating_point() else v for k, v in inputs.items()}
    with torch.no_grad():
        out = gm(**gi)
    return out[0] if isinstance(out, tuple) else out


def _spike(base, inputs: dict) -> float:
    """The largest (row max - max of the row's first 32 keys) of the float64 gold's logits, in log2 units (the quantity the offset logic sees)."""
    gm = copy.deepcopy(base).double()
    cap: dict[str, torch.Tensor] = {}
    for name in ("to_query", "to_key", "to_bias"):
        getattr(gm, name).register_forward_hook(lambda m, i, o, n=name: cap.__setitem__(n, o.detach()))
    gi = {k: v.double() if isinstance(v, torch.Tensor) and v.is_floating_point() else v for k, v in inputs.items()}
    with torch.no_grad():
        gm(**gi)
    B, L, _, _ = cap["to_query"].shape
    H = gm.n_head
    q = cap["to_query"].view(B, L, L, H, -1).permute(0, 3, 1, 2, 4)
    k = cap["to_key"].view(B, L, L, H, -1).permute(0, 3, 1, 2, 4)
    bias = cap["to_bias"].permute(0, 3, 1, 2)  # [B, H, j, k], starting node
    s = (torch.einsum("bhijd,bhikd->bhijk", q * q.shape[-1] ** -0.5, k) + bias[:, :, None]) * L2E
    return float((s.amax(-1) - s[..., :32].amax(-1)).max())


#: the spike lands in an FMA-pipe slot (2 .. 4 of every 16 keys) only by chance: several draws per case
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5])
@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_exp_clamp_spike_over_first_keys_b200(case, seed: int) -> None:
    _needs_b200()
    _, module, kwargs, length = case
    gc.collect()
    torch.cuda.empty_cache()
    base, gen = _build(module, kwargs, seed)
    inputs = base.make_inputs(length=length, samples=1, depth=None, dtype=torch.bfloat16, device=torch.device("cuda"), generator=gen)
    spike = _spike(base, inputs)
    for _ in range(12):  # q.k is bilinear in the q / k weights, the bias is not: iterate on the measured spike
        if 135.0 <= spike <= 150.0:
            break
        factor = (140.0 / max(spike, 1.0)) ** 0.5
        with torch.no_grad():
            base.to_query.weight.mul_(factor)
            base.to_key.weight.mul_(factor)
        spike = _spike(base, inputs)
    assert 130.0 <= spike <= 160.0, f"calibration: spike {spike:.1f} log2 units (wanted 130 .. 160)"
    gold = _gold(base, inputs)
    kern = copy.deepcopy(base).set_backend("cuda")
    ref = copy.deepcopy(base).set_backend("pytorch")
    assert kern.backend_name == "cuda"
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        why = kern.unsupported(**inputs)
        assert why is None, f"the cuda backend refuses this call: {why}"
        out, ref_out = kern(**inputs), ref(**inputs)
    out, ref_out = (o[0] if isinstance(o, tuple) else o for o in (out, ref_out))
    bad = int((~torch.isfinite(out)).sum())
    assert bad == 0, f"{bad} of {out.numel()} entries not finite at a {spike:.0f} log2-unit spike (the float64 gold is finite)"

    def err(o: torch.Tensor) -> tuple[float, float]:
        d = o.double() - gold
        return float(d.pow(2).mean().sqrt() / gold.pow(2).mean().sqrt()), float(d.abs().max() / gold.abs().max())

    rms, mx = err(out)
    rms_ref, mx_ref = err(ref_out)
    assert rms <= max(2.0 * rms_ref, 8e-3) and mx <= max(3.0 * mx_ref, 2e-2), (
        f"cuda rms {rms:.3e} max {mx:.3e}; pytorch rms {rms_ref:.3e} max {mx_ref:.3e}"
    )
