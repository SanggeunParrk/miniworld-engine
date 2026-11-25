"""Large-magnitude inference on the sm_80 family (A100, RTX 5090): every requirements module with a ``cuda`` backend, fed activations far beyond the usual N(0, 1).

Why: the judge and the unit tests draw random NORMAL inputs, so an intermediate that is stored or accumulated in a type with less range than bf16 (fp16: 65504;
fp8; an fp16 accumulator initialised with the residual) is invisible to them.  AF3's MSA activations reach |x| ~ 65536 (FoldForge 5I28 n=1 on the 5090: the one-kernel
bf16 Transition's f16 accumulator started at x and went to -inf: 'non-finite AF3 coordinates', engine 3.0.5).  This file is the audit's regression net.

Per case (the requirements' eval cases of ``measure/config.py``: module x variant x dtype, smallest length / depth / samples; ``MW_LM_LENGTHS=a,b`` replaces the length sweep)
and per input profile (every floating INPUT tensor ``x`` becomes ``1e3 x`` and, for the spike profiles, 1% of its entries are replaced by +-S):

    scale1e3     whole-tensor scale 1e3                  (LayerNorm inputs; every residual stream)
    offset1e5    a constant +1e5 added (no scale): LayerNorm statistics must not cancel.  Finiteness is asserted; the accuracy is REPORTED only (known limit: the
                 fp32 pair-bias LayerNorm of the token DiT / AttentionPairBias loses digits with a 1e5 common offset: rms 5e-6 .. 0.2 relative to the gold, no inf)
    spike6.5e4   1% entries +-6.5e4  (just above the fp16 max 65504)
    spike1e5     1% entries +-1e5
    spike1e6     1% entries +-1e6

the ``cuda`` backend is compared with the float64 gold (the module's ``pytorch`` backend in float64 on the same rounded inputs, as ``measure/accuracy.py``) and with the
``pytorch`` backend at the case dtype:

* finite: no non-finite output element unless the ``pytorch`` backend is non-finite there too (the true result overflowing is then not ours to fix),
* rms error relative to the gold rms (and max error / max |gold|, the judge's metric) within the module's usual tolerance: ``max(3 x the pytorch backend's own error, the
  judge's fixed floor)``.

The ``cuda`` run is profiled once: a Triton kernel in a ``cuda`` forward is a bug (a CUDA hole) and fails the case; ATen kernels are printed (glue).

``MW_LM_MODULES=Transition,LayerNorm`` selects classes, ``MW_LM_PROFILES=spike1e6`` profiles, ``MW_LM_DTYPES=bfloat16`` dtypes, ``MW_LM_ALLVARIANTS=1`` every constructor variant (default: 3 per
(module, dtype, stream)), ``MW_LM_OUT=file.jsonl`` writes one record per (case, profile, tensor).  Every ``master`` case also runs as a plain ``bfloat16`` model (no autocast): the
requirements measure bf16 as ``master`` only, and the 5I28 failure was on the bf16-model route.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import math
import os

import pytest
import torch

from foldjet_engine.measure import accuracy as acc
from foldjet_engine.measure import config as cfgmod
from foldjet_engine.modules.primitives import BackendUnsupported

pytestmark = [pytest.mark.gpu]  # the sm_80 family only: A100 and RTX 5090 (skipped elsewhere below)

PROFILES: dict[str, float | None] = {
    "scale1e3": None,
    "spike6.5e4": 6.5e4,
    "spike1e5": 1e5,
    "spike1e6": 1e6,
    "offset1e5": -1.0,
}  # -1: a constant +1e5 on top (no 1e3 scale)
#: floating inputs that are not activations (rotary tables, ...): left as they are
KEEP = ("cos", "sin", "inv_freq")
#: conditioning inputs (the AdaLN modulation source): scaled by 1e3 in every profile, never spiked or offset
SCALE_ONLY = ("cond", "c")
#: classes whose floating inputs are geometry / noise / schedule values, not activations (a 1e3 coordinate or a spiked sigma is not a stress of a kernel's range, it is another model input)
NOT_ACTIVATIONS = frozenset(
    {
        "AlignEuler",
        "AugmentNoise",
        "SmoothLddtLoss",
        "PoolAtoms",
        "InputFeatureEmbedder",
        "PairEmbedAdaLN",
        "PairEmbedCA",
        "PairEmbedGeom",
        "TokenPairInit",
        "ProteinMPNN",
    }
)
#: the error floor of the judge (measure/config.py FIXED): the tolerance is max(3 x the pytorch backend's error, this)
FLOOR = {"bfloat16": 8e-3, "master": 8e-3, "float32": 1e-5}
RATIO = 3.0
VARIANTS = 3


def _sm80_family() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() in ((8, 0), (12, 0))


def _selected() -> list[cfgmod.Case]:
    names = tuple(n for n in os.environ.get("MW_LM_MODULES", "").split(",") if n)
    lengths = tuple(int(v) for v in os.environ.get("MW_LM_LENGTHS", "").split(",") if v) or "min"
    cfg = cfgmod.Config(
        modules=names or ("all",), backends=("pytorch", "cuda"), lengths=lengths, depths="min", samples="min", modes=("eval",)
    )
    seen: dict[tuple, cfgmod.Case] = {}
    for c in cfgmod.cases(cfg):
        if "cuda" in c.cls.backends and c.module not in NOT_ACTIVATIONS:
            seen.setdefault((c.module, json.dumps(c.kwargs, sort_keys=True, default=str), c.dtype, c.stream, c.length, c.depth), c)
    # the requirements measure bf16 models as ``master`` (fp32 weights under autocast); a bf16 model (``module.bfloat16()``, no autocast: FoldForge's AF3 run) takes other kernel
    # routes (the 5I28 Transition failure was one) and is not a requirements dtype: every ``master`` case also runs as ``bfloat16``
    for key, c in list(seen.items()):
        if c.dtype == "master" and "fp32" not in c.stream:  # (an fp32-conditioning stream mixes dtypes: no bf16 model twin)
            seen.setdefault((*key[:2], "bfloat16", *key[3:]), dataclasses.replace(c, dtype="bfloat16"))
    dts = tuple(d for d in os.environ.get("MW_LM_DTYPES", "").split(",") if d)  # e.g. bfloat16: only the bf16-model twins
    if dts:
        seen = {k: c for k, c in seen.items() if c.dtype in dts}
    if os.environ.get("MW_LM_ALLVARIANTS"):
        return list(seen.values())
    groups: dict[tuple, list[cfgmod.Case]] = {}
    for c in seen.values():
        groups.setdefault((c.module, c.dtype, c.stream), []).append(c)
    picked: list[cfgmod.Case] = []
    for g in (
        groups.values()
    ):  # at most VARIANTS evenly spaced constructor variants of a (module, dtype, stream) (MW_LM_ALLVARIANTS=1: all of them)
        n = min(len(g), VARIANTS)
        picked += [g[round(i * (len(g) - 1) / max(n - 1, 1))] for i in range(n)]
    return picked


def _case_id(c: cfgmod.Case) -> str:
    kw = ",".join(f"{k}={v}" for k, v in c.kwargs.items())
    return f"{c.module}({kw})-{c.dtype}-{c.stream}-L{c.length}" + (f"D{c.depth}" if c.depth else "")


CASES = _selected()


def _perturb(inputs: dict, spike: float | None, seed: int) -> dict:
    """``1e3 x`` of every floating activation input, with 1% of its entries replaced by +-spike (a seeded draw, identical for every backend)."""
    g = torch.Generator(device="cuda").manual_seed(seed)

    def one(name: str, t: torch.Tensor) -> torch.Tensor:
        if name in KEEP or not t.is_floating_point() or t.numel() == 0:
            return t
        if (
            name in SCALE_ONLY
        ):  # conditioning is not a residual stream: only the whole-tensor 1e3 (a spiked raw conditioning of 1e5 makes AdaLN scales of 1e5: true outputs of 1e9 and more)
            return (t.float() * 1e3).to(t.dtype)
        if spike == -1.0:  # a large common offset: LayerNorm statistics must not cancel (two-pass / shifted sums)
            return (t.float() + 1e5).to(t.dtype)
        y = t.float() * 1e3
        if spike is not None:
            hit = torch.rand(t.shape, device=t.device, generator=g) < 0.01
            sign = torch.randint(0, 2, t.shape, device=t.device, generator=g) * 2 - 1
            y = torch.where(hit, spike * sign.float(), y)
        return y.to(t.dtype)

    return {k: one(k, v) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}


def _errors(x: torch.Tensor, gold: torch.Tensor) -> tuple[float, float]:
    """(rms error / rms gold, max error / max |gold|) over the elements where both are finite (float64)."""
    ok = torch.isfinite(x) & torch.isfinite(gold)
    d, g = (x.double() - gold.double())[ok], gold.double()[ok]
    if d.numel() == 0:
        return float("nan"), float("nan")
    rg, mg = g.pow(2).mean().sqrt().item(), g.abs().max().item()
    return (d.pow(2).mean().sqrt().item() / rg if rg > 0 else d.pow(2).mean().sqrt().item()), (
        d.abs().max().item() / mg if mg > 0 else d.abs().max().item()
    )


def _kernel_classes(module, call: dict, autocast_dtype) -> dict[str, int]:
    """CUDA kernels launched by one forward of ``module``, by class (profiler kernel names): triton / aten (PyTorch glue) / other (engine, cuBLAS)."""
    from torch.profiler import ProfilerActivity, profile

    ctx = torch.autocast("cuda", dtype=autocast_dtype) if autocast_dtype is not None else contextlib.nullcontext()
    with torch.no_grad(), ctx:
        module(**call)  # warm
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            module(**call)
            torch.cuda.synchronize()
    out = {"triton": 0, "aten": 0, "other": 0}
    for e in prof.events() or []:
        if "CUDA" not in str(e.device_type):
            continue
        n = e.name
        out[
            "triton" if n.startswith("triton") or "triton_" in n else "aten" if "at::native" in n or n.startswith("void at::") else "other"
        ] += 1
    return out


@pytest.fixture(autouse=True)
def _need_card():
    if not _sm80_family():
        pytest.skip("sm_80 / sm_120 only (A100, RTX 5090)")


@pytest.mark.parametrize("case", CASES, ids=[_case_id(c) for c in CASES])
def test_large_magnitude_inference(case: cfgmod.Case) -> None:
    only = tuple(p for p in os.environ.get("MW_LM_PROFILES", "").split(",") if p)
    out_path = os.environ.get("MW_LM_OUT")
    dev = "cuda"
    base = cfgmod.build(case, device=dev)
    acc._quiet(base)
    inputs = cfgmod.inputs(case, base, device=dev)
    autocast = cfgmod.autocast(case)
    fixed = FLOOR[case.dtype]
    module = cfgmod.clone(base).set_backend("cuda")
    with torch.autocast("cuda", dtype=autocast) if autocast is not None else contextlib.nullcontext():
        why = module.unsupported(**inputs)
    if why is not None:
        pytest.skip(f"cuda backend declines: {why}")
    print(f"\n[backend] {_case_id(case)}: module backend = {module.backend_name}")
    failures: list[str] = []
    for pi, (pname, spike) in enumerate(PROFILES.items()):
        if only and pname not in only:
            continue
        x = _perturb(inputs, spike, seed=1000 + pi)
        acc._reference_precision()
        gold_in = {k: acc._map_floats(v, lambda t: t.double()) for k, v in x.items()}
        gold, _ = acc._run(cfgmod.clone(base).double(), gold_in, train=False, autocast=None, upstream=None, generator=None)
        del gold_in
        ref, _ = acc._run(cfgmod.clone(base), x, train=False, autocast=autocast, upstream=None, generator=None)
        acc._reference_precision() if "cuda" in cfgmod.TF32_BACKENDS else acc._backend_precision()
        try:
            got, _ = acc._run(module, x, train=False, autocast=autocast, upstream=None, generator=None)
        except BackendUnsupported as exc:
            pytest.skip(f"cuda backend declines: {exc}")
        classes = _kernel_classes(module, x, autocast) if pi == 0 else None
        if classes is not None:
            print(f"[kernels] {_case_id(case)}: {classes}")
            if classes["triton"]:
                failures.append(f"{pname}: {classes['triton']} Triton kernel(s) in the cuda forward (a CUDA hole)")
        for name, g in gold.items():
            if g is None or not torch.isfinite(g).all():
                continue  # the exact result is outside fp64's range: nothing to compare
            c, r = got[name], ref[name]
            assert c is not None and r is not None
            bad = ~torch.isfinite(c)
            ref_bad = ~torch.isfinite(r)
            rec = {
                "case": _case_id(case),
                "profile": pname,
                "tensor": name,
                "cuda_nonfinite": int(bad.sum()),
                "ref_nonfinite": int(ref_bad.sum()),
            }
            if bad.any() and not bool((ref_bad | ~bad).all()):
                rec["status"] = "NONFINITE"
                failures.append(f"{pname} {name}: {int(bad.sum())} non-finite of {bad.numel()} (pytorch: {int(ref_bad.sum())})")
            else:
                (rms_c, max_c), (rms_r, max_r) = _errors(c, g), _errors(r, g)
                tol_rms, tol_max = max(RATIO * rms_r, fixed), max(RATIO * max_r, fixed)
                rec.update(rms_cuda=rms_c, rms_pytorch=rms_r, max_cuda=max_c, max_pytorch=max_r)
                ok = math.isfinite(rms_c) and rms_c <= tol_rms and max_c <= tol_max
                rec["status"] = "ok" if ok else "ACCURACY"
                if not ok and pname == "offset1e5":
                    rec["status"] = "ACCURACY-known-limit"  # reported, not asserted (docstring)
                elif not ok:
                    failures.append(
                        f"{pname} {name}: rms {rms_c:.3e} (pytorch {rms_r:.3e}, tol {tol_rms:.3e}), max {max_c:.3e} (pytorch {max_r:.3e}, tol {tol_max:.3e})"
                    )
            if out_path:
                with open(out_path, "a") as fh:
                    fh.write(json.dumps(rec) + "\n")
        del gold, ref, got
        torch.cuda.empty_cache()
    assert not failures, f"{_case_id(case)}:\n  " + "\n  ".join(failures)
