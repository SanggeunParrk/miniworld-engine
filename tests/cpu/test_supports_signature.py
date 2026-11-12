"""A backend's range check takes exactly what its forward takes (FoldForge 5I28 run, A100: ``AtomDiTChai1.unsupported(single, cond,
pair, mask, uid)`` raised ``TypeError`` -- the inherited ``_supports_triton`` had no positional ``uid``).

FoldForge asks :meth:`~miniworld_engine.modules.primitives.BackendModule.unsupported` with the forward's own arguments, positionally,
before each call; a range check that cannot bind them is a crash where a refusal (or ``None``) was due. For every requirements class and
every backend it lists:

- the static contract: ``_supports_<backend>`` (and an overridden ``unsupported``) binds every argument list ``_forward_<backend>``
  binds, positional parameters positionally (the reference forward's list where a backend forward is a ``*args`` pass-through);
- the call: ``unsupported(*make_inputs in forward order)`` returns a reason or ``None`` (never raises) on CPU.
"""

from __future__ import annotations

import csv
import inspect

import pytest

from miniworld_engine.measure import config as cfgmod
from miniworld_engine.modules.primitives import BackendModule

with open(cfgmod.REQUIREMENTS, newline="") as _fh:
    FIRST: dict[str, tuple[int, dict]] = {}
    for _n, _r in enumerate(csv.DictReader(_fh), 1):
        FIRST.setdefault(_r["module"], (_n, _r))
CLASSES = sorted(FIRST)
#: a tiny length per stream (as tests/cpu/test_modules_cpu.py)
SMALL = {"token_single": 16, "token_pair": 16, "msa_token": 16, "atom_single": 128, "protein_node": 64}

_POS = (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
PAIRS = [(m, b) for m in CLASSES for b in cfgmod.module_class(m).backends]


def _call_signature(m: BackendModule, backend: str) -> inspect.Signature:
    """The argument list a call on ``backend`` takes: its ``_forward_<backend>``'s, or the reference forward's where that one is a
    ``*args`` pass-through (a cooperative mixin)."""
    sig = inspect.signature(getattr(m, f"_forward_{backend}"))
    if any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in sig.parameters.values()):
        sig = inspect.signature(m._forward_pytorch)  # pyright: ignore[reportAttributeAccessIssue]  -- every BackendModule has the reference
    return sig


def _full_call(sig: inspect.Signature) -> tuple[list[object], dict[str, object]]:
    """Every parameter given: positional ones positionally, keyword-only ones by keyword (placeholders: binding only)."""
    args = [object() for p in sig.parameters.values() if p.kind in _POS]
    kwargs = {p.name: object() for p in sig.parameters.values() if p.kind is inspect.Parameter.KEYWORD_ONLY}
    return args, kwargs


def _checks(m: BackendModule, backend: str) -> list[tuple[str, object]]:
    out = []
    check = getattr(m, f"_supports_{backend}", None)
    if check is not None:
        out.append((f"_supports_{backend}", check))
    if type(m).unsupported is not BackendModule.unsupported:
        out.append(("unsupported", m.unsupported))
    return out


@pytest.mark.parametrize(("module", "backend"), PAIRS, ids=[f"{m}-{b}" for m, b in PAIRS])
def test_supports_binds_the_forward_arguments(module: str, backend: str):
    n, row = FIRST[module]
    m = cfgmod.construct(cfgmod.module_class(module), cfgmod._variants(row["dims"], row["options"])[0]).set_backend(backend)
    sig = _call_signature(m, backend)
    args, kwargs = _full_call(sig)
    for name, check in _checks(m, backend):
        try:
            inspect.signature(check).bind(*args, **kwargs)  # pyright: ignore[reportArgumentType]
        except TypeError as e:
            pytest.fail(f"{module}.{name} does not take the arguments of _forward_{backend}{sig}: {e}")


@pytest.mark.parametrize(("module", "backend"), PAIRS, ids=[f"{m}-{b}" for m, b in PAIRS])
def test_unsupported_takes_make_inputs_positionally(module: str, backend: str):
    """``unsupported`` with the measurement inputs passed positionally in forward order (FoldForge's call) answers, never raises."""
    n, row = FIRST[module]
    m, inputs = _build(n, row, backend)
    sig = _call_signature(m, backend)
    names = [p.name for p in sig.parameters.values() if p.kind in _POS]
    assert set(inputs) <= set(sig.parameters), f"make_inputs keys {sorted(set(inputs) - set(sig.parameters))} not in _forward_{backend}{sig}"
    last = max((names.index(k) for k in inputs if k in names), default=-1)
    args = [inputs[k] if k in inputs else sig.parameters[k].default for k in names[: last + 1]]
    kwargs = {k: v for k, v in inputs.items() if k not in names}
    reason = m.unsupported(*args, **kwargs)
    assert reason is None or (isinstance(reason, str) and reason.strip())


def _build(n: int, row: dict, backend: str) -> tuple[BackendModule, dict]:
    """The row's first case, shrunk (one sample, a tiny length / depth, fp32, eval), on CPU."""
    lengths, depths = cfgmod._ints(row["lengths"]), cfgmod._ints(row["depths"])
    case = cfgmod.Case(
        row=n,
        module=row["module"],
        model=row["model"],
        stream=row["stream"],
        kwargs=cfgmod._variants(row["dims"], row["options"])[0],
        length=min(min(lengths), SMALL.get(row["stream"], min(lengths))),
        depth=min(min(depths), 4) if depths else None,
        samples=1,
        dtype="float32",
        mode="eval",
        seed=0,
    )
    module = cfgmod.build(case, device="cpu")
    return module.set_backend(backend), cfgmod.inputs(case, module, device="cpu")
