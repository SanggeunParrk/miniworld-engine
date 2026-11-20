"""H100 inference with large-magnitude activations: the ``cuda`` backend must stay finite and as accurate as the ``pytorch`` backend when the residual
streams carry |x| of 1e3 and spikes of 6.5e4 / 1e5 / 1e6 (AF3 MSA activations reach ~65536, past the fp16 maximum 65504).

Why this test exists: the engine's judge and tests draw N(0, 1) inputs, where no intermediate leaves the fp16 range; a kernel that converts a residual or a
raw activation to fp16 (the sm80 Transition's fp16 GEMM2 accumulator initialised with the residual: inf at 65536, the 5090 5I28 run) is invisible to them.
Every H100 kernel family on an AF3 / Protenix / Boltz inference path is driven here with:

* ``scale1e3``: the stressed inputs times 1e3;
* ``spike65536`` / ``spike1e5`` / ``spike1e6``: 1 % of the entries of the stressed inputs replaced by +-65536 / 1e5 / 1e6 (the rest N(0, 1)).

The stressed inputs are the residual-carrying streams (``x`` / ``single`` / ``msa`` / ``pair`` / ``q``); masks, conditioning and the pair bias of the
DiT blocks are left as they are (a 1e5 conditioning drives the softmax to a one-hot in every backend: not a numerics question).  The checks, per
(module, mode): no Triton launch and the ``cuda`` backend connected; every output of the ``cuda`` backend finite; its error against the float64 gold
(the ``pytorch`` backend with float64 parameters and inputs), taken over the outputs that are not themselves spikes, within the ``pytorch`` backend's own
error (the engine accuracy standard, not bit-identity).  The float64 gold does not overflow, so a non-finite ``cuda`` output is always a bug.
"""

from __future__ import annotations

import copy
import gc
from typing import Any

import pytest
import torch
from torch.utils import _pytree as pytree

from miniworld_engine.measure import config as C

pytestmark = [pytest.mark.gpu]


def _card() -> tuple[int, int] | None:
    return torch.cuda.get_device_capability() if torch.cuda.is_available() else None


needs_h100 = pytest.mark.skipif(_card() != (9, 0), reason="H100")

#: (id, module, constructor kwargs, stressed input names, length, samples, depth)
CASES = [
    ("transition_d64n2", "Transition", {"d_hidden": 64, "n": 2}, ("x",), 256, 1, None),
    ("transition_d128n4", "Transition", {"d_hidden": 128, "n": 4}, ("x",), 256, 1, None),
    ("transition_d256n2", "Transition", {"d_hidden": 256, "n": 2}, ("x",), 256, 1, None),
    ("transition_d512n2", "Transition", {"d_hidden": 512, "n": 2}, ("x",), 256, 1, None),
    ("triattn_h16_d128", "TriangleAttention", {"d_pair": 128, "n_head": 4, "starting": False, "p_drop": 0.25}, ("pair",), 256, 1, None),
    (
        "triattn_h16_d128_start",
        "TriangleAttention",
        {"d_pair": 128, "n_head": 4, "starting": True, "p_drop": 0.25},
        ("pair",),
        256,
        1,
        None,
    ),
    ("triattn_af3_d64", "TriangleAttentionAF3", {"d_pair": 64, "n_head": 4, "starting": False, "p_drop": 0.25}, ("pair",), 256, 1, None),
    ("triattn_d256h8", "TriangleAttention", {"d_pair": 256, "n_head": 8, "starting": False, "p_drop": 0.25}, ("pair",), 256, 1, None),
    (
        "triattn_rf3_dh64",
        "TriangleAttentionRF3",
        {"d_pair": 64, "n_head": 4, "d_hidden": 256, "starting": False, "p_drop": 0.25},
        ("pair",),
        256,
        1,
        None,
    ),
    ("trimul_d128_out", "TriangleMultiplication", {"d_pair": 128, "outgoing": True, "p_drop": 0.25}, ("pair",), 256, 1, None),
    ("trimul_d128_in", "TriangleMultiplication", {"d_pair": 128, "outgoing": False, "p_drop": 0.25}, ("pair",), 256, 1, None),
    ("trimul_d256", "TriangleMultiplication", {"d_pair": 256, "outgoing": True, "p_drop": 0.25}, ("pair",), 256, 1, None),
    ("opm_d128", "OuterProductMean", {"d_msa": 128, "d_pair": 256, "d_hidden": 32, "normalize_before_proj": False}, ("msa",), 128, 1, 1024),
    (
        "opm_d64_norm",
        "OuterProductMean",
        {"d_msa": 64, "d_pair": 128, "d_hidden": 32, "normalize_before_proj": True},
        ("msa",),
        128,
        1,
        1024,
    ),
    (
        "pwa_d128",
        "MSAPairWeightedAveraging",
        {"d_msa": 128, "d_pair": 256, "n_head": 8, "d_hidden": 16, "p_drop": 0.25},
        ("msa", "pair"),
        128,
        1,
        1024,
    ),
    (
        "pwa_d256",
        "MSAPairWeightedAveraging",
        {"d_msa": 256, "d_pair": 512, "n_head": 8, "d_hidden": 32, "p_drop": 0.25},
        ("msa", "pair"),
        128,
        1,
        1024,
    ),
    ("apb_d384", "AttentionPairBias", {"d_single": 384, "d_pair": 128, "n_head": 16}, ("single", "pair"), 256, 1, None),
    (
        "token_dit_protenix",
        "TokenDiTProtenix",
        {"d_single": 768, "d_cond": 384, "d_pair": 128, "n_head": 16, "n": 2},
        ("single",),
        256,
        5,
        None,
    ),
    (
        "token_dit_biasonly",
        "TokenDiTMiniWorldBiasOnly",
        {"d_single": 768, "d_cond": 384, "d_pair": 128, "n_head": 16, "n": 2},
        ("single",),
        256,
        5,
        None,
    ),
    (
        "atom_dit_protenix",
        "AtomDiTProtenix",
        {"d_single": 128, "d_cond": 128, "d_pair": 16, "n_head": 4, "n": 2},
        ("single",),
        1024,
        1,
        None,
    ),
    ("atom_dit_af3", "AtomDiTOpenFold3", {"d_single": 128, "d_cond": 128, "d_pair": 16, "n_head": 4, "n": 2}, ("single",), 1024, 1, None),
    (
        "swa_atom",
        "SWAAtomTransformer",
        {"d_atom": 128, "d_cond": 128, "n_head": 4, "n_block": 3, "expansion_ratio": 2, "swa_window_size": 128},
        ("q",),
        1024,
        5,
        None,
    ),
]
MODES = ["scale1e3", "spike65536", "spike1e5", "spike1e6"]


def _stress(x: torch.Tensor, mode: str, gen: torch.Generator) -> torch.Tensor:
    xf = x.float()
    if mode == "scale1e3":
        return (xf * 1e3).to(x.dtype)
    value = {"spike65536": 65536.0, "spike1e5": 1e5, "spike1e6": 1e6}[mode]
    spike = torch.rand(xf.shape, device=x.device, generator=gen) < 0.01
    sign = torch.randint(0, 2, xf.shape, device=x.device, generator=gen) * 2 - 1
    return torch.where(spike, value * sign, xf).to(x.dtype)


def _build(module: str, kwargs: dict) -> tuple[Any, torch.Generator]:
    gen = torch.Generator(device="cuda").manual_seed(0)
    base = C.construct(C.module_class(module), kwargs).to("cuda").float()
    C.randomize(base, gen)
    return C.cast(base, "master").eval(), gen


def _outputs(out) -> list[torch.Tensor]:
    return [t for t in pytree.tree_leaves(out) if isinstance(t, torch.Tensor) and t.is_floating_point()]


def _errors(out: torch.Tensor, gold: torch.Tensor) -> tuple[float, float]:
    """(rms, max) error relative to the gold's rms / max, over the entries of the gold that are not spikes (all entries when there are none)."""
    d, g = out.double() - gold, gold
    keep = g.abs() < 1e3
    if int(keep.sum()) > 0:
        d, g = d[keep], g[keep]
    return float(d.pow(2).mean().sqrt() / g.pow(2).mean().sqrt()), float(d.abs().max() / g.abs().max())


class _Triton:
    """Counts Triton kernel launches (a Triton kernel in a ``cuda`` run is a CUDA hole)."""

    def __enter__(self):
        import triton.runtime.jit as jit

        self.jit, self.orig, self.n = jit.JITFunction, jit.JITFunction.run, 0
        outer = self

        def run(self_, *a, **k):
            outer.n += 1
            return outer.orig(self_, *a, **k)

        jit.JITFunction.run = run  # pyright: ignore[reportAttributeAccessIssue]
        return self

    def __exit__(self, *exc):
        self.jit.run = self.orig  # pyright: ignore[reportAttributeAccessIssue]


def _gold(base: Any, inputs: dict) -> list[torch.Tensor]:
    gm = copy.deepcopy(base).double()
    gi = {k: v.double() if isinstance(v, torch.Tensor) and v.is_floating_point() else v for k, v in inputs.items()}
    with torch.no_grad():
        return _outputs(gm(**gi))


def _run(module: Any, inputs: dict) -> list[torch.Tensor]:
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        return _outputs(module(**inputs))


@needs_h100
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_large_magnitude_inference_sm90(case, mode: str) -> None:
    _, module, kwargs, keys, length, samples, depth = case
    gc.collect()
    torch.cuda.empty_cache()
    base, gen = _build(module, kwargs)
    inputs = base.make_inputs(length=length, samples=samples, depth=depth, dtype=torch.bfloat16, device=torch.device("cuda"), generator=gen)
    gen = torch.Generator(device="cuda").manual_seed(1)
    inputs = {k: _stress(v, mode, gen) if k in keys else v for k, v in inputs.items()}
    gold = _gold(base, inputs)
    ref = copy.deepcopy(base).set_backend("pytorch")
    kern = copy.deepcopy(base).set_backend("cuda")
    assert kern.backend_name == "cuda"
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        why = kern.unsupported(**inputs)
    assert why is None, f"the cuda backend refuses this inference call: {why}"
    with _Triton() as triton_launches:
        out = _run(kern, inputs)
    ref_out = _run(ref, inputs)
    assert triton_launches.n == 0, f"{triton_launches.n} Triton launches in the cuda run"
    assert len(out) == len(gold) == len(ref_out)
    for i, (o, g, r) in enumerate(zip(out, gold, ref_out, strict=True)):
        bad = int((~torch.isfinite(o)).sum())
        assert bad == 0, f"output {i}: {bad} of {o.numel()} entries are not finite (the float64 gold is finite)"
        rms, mx = _errors(o, g)
        rms_ref, mx_ref = _errors(r, g)
        assert rms <= max(2.0 * rms_ref, 8e-3) and mx <= max(3.0 * mx_ref, 2e-2), (
            f"output {i}: cuda rms {rms:.3e} max {mx:.3e}; pytorch rms {rms_ref:.3e} max {mx_ref:.3e}"
        )


# --------------------------------------------------------------------------------------------------------------------------------- TriAttn logit range
@needs_h100
@pytest.mark.parametrize("starting", [False, True])
@pytest.mark.parametrize(
    ("module", "kwargs"),
    [
        ("TriangleAttention", {"d_pair": 128, "n_head": 4, "p_drop": 0.25}),  # the fp16 D = 32 cores
        ("TriangleAttentionRF3", {"d_pair": 64, "n_head": 4, "d_hidden": 256, "p_drop": 0.25}),  # D = 64
    ],
)
def test_triattn_fp16_logit_headroom_sm90(module: str, kwargs: dict, starting: bool) -> None:
    """The H100 TriAttn cores keep the raw scores q.k + bias / scale in fp16 (documented range |s| < 65504).  LayerNorm bounds q / k / bias whatever the
    activation magnitude, so only the pair LayerNorm gain can push the scores: a gain of 10 (scores x 100, far above a trained model's) must stay finite."""
    base, gen = _build(module, dict(kwargs, starting=starting))
    with torch.no_grad():
        for name, p in base.named_parameters():
            if name.startswith("ln_pair"):
                p.mul_(10.0)
    inputs = base.make_inputs(length=256, samples=1, depth=None, dtype=torch.bfloat16, device=torch.device("cuda"), generator=gen)
    g2 = torch.Generator(device="cuda").manual_seed(1)
    inputs = {k: _stress(v, "spike1e5", g2) if k == "pair" else v for k, v in inputs.items()}
    gold = _gold(base, inputs)
    out = _run(copy.deepcopy(base).set_backend("cuda"), inputs)
    ref = _run(copy.deepcopy(base).set_backend("pytorch"), inputs)
    for o, g, r in zip(out, gold, ref, strict=True):
        assert bool(torch.isfinite(o).all()), f"{int((~torch.isfinite(o)).sum())} non-finite entries"
        rms, mx = _errors(o, g)
        rms_ref, mx_ref = _errors(r, g)
        assert rms <= max(2.0 * rms_ref, 8e-3) and mx <= max(3.0 * mx_ref, 2e-2), (
            f"cuda rms {rms:.3e} max {mx:.3e}; pytorch {rms_ref:.3e} {mx_ref:.3e}"
        )


# ----------------------------------------------------------------------------------------------------------------------------------- engine-fp8
FP8_CASES = [
    c for c in CASES if c[0] in ("transition_d128n4", "triattn_h16_d128", "trimul_d128_out", "opm_d128", "pwa_d128", "token_dit_protenix")
]


@needs_h100
@pytest.mark.parametrize("case", FP8_CASES, ids=[c[0] for c in FP8_CASES])
def test_engine_fp8_large_magnitude_sm90(case) -> None:
    """The opt-in ``engine-fp8`` backend (static e4m3 scales taken on the first call): a normal first call, then a call with 1e5 spikes -- finite (after
    the lazy guard's re-run when a route saturated: ``fp8.check()``) and within the e4m3 routes' own error of the float64 gold."""
    from miniworld_engine import fp8

    _, module, kwargs, keys, length, samples, depth = case
    base, gen = _build(module, kwargs)
    normal = base.make_inputs(length=length, samples=samples, depth=depth, dtype=torch.bfloat16, device=torch.device("cuda"), generator=gen)
    g2 = torch.Generator(device="cuda").manual_seed(1)
    inputs = {k: _stress(v, "spike1e5", g2) if k in keys else v for k, v in normal.items()}
    gold = _gold(base, inputs)
    kern = copy.deepcopy(base).set_backend("engine-fp8")
    _run(kern, normal)  # calibrates the static scales
    out = _run(kern, inputs)
    for _ in range(3):
        if not fp8.check():
            break
        out = _run(kern, inputs)
    for o, g in zip(out, gold, strict=True):
        assert bool(torch.isfinite(o).all()), f"{int((~torch.isfinite(o)).sum())} non-finite entries"
        rms, _ = _errors(o, g)
        assert rms <= 8e-2, f"engine-fp8 rms {rms:.3e}"


@needs_h100
@pytest.mark.xfail(
    strict=True,
    reason="known limit, not an activation-magnitude bug: the H100 fp16 TriAttn cores accumulate the raw scores q.k + bias / scale in fp16 (|s| < 65504); "
    "a pair LayerNorm gain of ~30 (scores x 900; trained gains are far smaller) overflows them to inf -> NaN. Remove the xfail when the cores range-protect the scores.",
)
def test_triattn_fp16_logit_limit_documented_sm90() -> None:
    base, gen = _build("TriangleAttention", {"d_pair": 128, "n_head": 4, "starting": False, "p_drop": 0.25})
    with torch.no_grad():
        for name, p in base.named_parameters():
            if name.startswith("ln_pair"):
                p.mul_(60.0)
    inputs = base.make_inputs(length=256, samples=1, depth=None, dtype=torch.bfloat16, device=torch.device("cuda"), generator=gen)
    out = _run(copy.deepcopy(base).set_backend("cuda"), inputs)
    assert all(bool(torch.isfinite(o).all()) for o in out)
