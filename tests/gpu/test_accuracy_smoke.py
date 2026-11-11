"""Small-L accuracy smoke of the engine backends (cuda, triton) against the float64 gold, through the measurer itself
(``measure.accuracy.measure_case``: the same gold, reference, inputs, tolerances and light rule the judge's tokens come from).

Per module class (the major ones first, then every other class with a cuda or triton backend): its first requirements row at the row's smallest length / depth / sample count, every dtype and mode of the
row (forward outputs; in train also every input and parameter gradient). A backend passes a case when the measurer says GREEN or YELLOW
(``requirements/README.md``: RED = inaccurate); a measurer ``error`` fails; ``unsupported`` / ``skipped`` (out of memory) skip with the
measurer's reason. Run on an H100: ``pytest tests/gpu -m gpu``.
"""

from __future__ import annotations

import functools

import pytest

from miniworld_engine.measure import accuracy
from miniworld_engine.measure import config as cfgmod

pytestmark = [pytest.mark.gpu, pytest.mark.h100]

#: the major classes first (the issue list), then every other requirements class with an engine backend
MAJOR = ("TriangleMultiplication", "TriangleAttention", "AttentionPairBias", "Transition", "OuterProductMean", "MSAPairWeightedAveraging",
         "LayerNorm", "TokenDiTProtenix", "AtomDiTProtenix", "SWAAtomTransformer", "InputFeatureEmbedder")
_EVERY = dict.fromkeys(r["module"] for _, r in cfgmod.rows(cfgmod.from_dict({"modules": "all"})))
MODULES = (*MAJOR, *(m for m in _EVERY if m not in MAJOR and {"cuda", "triton"} & set(cfgmod.module_class(m).backends)))
BACKENDS = ("cuda", "triton")
#: (class, tensor) whose exact gradient is identically zero, so the relative error max|x - gold| / max|gold| divides by the float64
#: round-off of the gold and is meaningless (the reference itself is ~1e8 off). The H100 tokens record the same RED (r114 / r115
#: L1024 float32 train, cuda and triton). An issue of the error measure on a zero gradient, not of the kernels.
ZERO_GRADIENTS = {
    ("AtomDiTRoseTTAFold3", "grad:attention.norm_key.bias"):
        "RF3 FlatQKNorm key-LayerNorm bias: q_h . b_h shifts every logit of a query row equally and softmax is shift-invariant, so the "
        "gold gradient is fp64 round-off and max|x - gold| / max|gold| explodes (err_pytorch ~6e8); measure-rule degeneracy, see report",
}


def _config(module: str) -> cfgmod.Config:
    first = next(n for n, r in cfgmod.rows(cfgmod.from_dict({"modules": [module]})))
    return cfgmod.from_dict({"modules": [module], "backends": list(BACKENDS), "rows": [first], "lengths": "min", "depths": "min",
                             "samples": "min", "device": "cuda", "seed": 0})


CASES = [(module, case, backend) for module in MODULES for case in cfgmod.cases(_config(module))
         for backend in BACKENDS if backend in case.cls.backends]


def _kw(case: cfgmod.Case) -> str:
    return ",".join(f"{k}={v}" for k, v in case.kwargs.items())


@functools.cache
def _records(module: str, label: str) -> dict[str, dict]:
    """backend -> record, one measurement per case for all its backends."""
    cfg = _config(module)
    case = next(c for c in cfgmod.cases(cfg) if c.label == label)
    return {r["backend"]: r for r in accuracy.measure_case(case, cfg, device="cuda")}


@pytest.mark.parametrize(("module", "case", "backend"), CASES, ids=[f"{c.module}-r{c.row}-{_kw(c)}-L{c.length}-{c.dtype}-{c.mode}-{b}"
                                                                     for _, c, b in CASES])
def test_accuracy(module: str, case: cfgmod.Case, backend: str):
    rec = _records(module, case.label)[backend]
    status = rec["status"]
    if status in ("unsupported", "skipped"):
        pytest.skip(f"{status}: {rec.get('reason')}")
    assert status == "measured", f"{status}: {rec.get('reason')}"
    if rec["light"] == "RED" and (case.module, rec["tensor"]) in ZERO_GRADIENTS:
        pytest.xfail(ZERO_GRADIENTS[(case.module, rec["tensor"])])
    assert rec["light"] != "RED", (f"{case.label} {backend}: {rec['light']} on {rec['tensor']}: err {rec['err_backend']:.3e}, "
                                   f"pytorch {rec['err_pytorch']:.3e}, tol {rec['tol']:.3e}")
