"""Every requirements module class constructs with the reference backend, runs a tiny forward on CPU, and keeps the
:class:`~miniworld_engine.modules.primitives.BackendModule` contract (primitives.py module docstring):

- an unknown backend is a ``ValueError``;
- a call outside a backend's range raises :class:`BackendUnsupported` with a reason, never a silent fallback (on CPU every engine kernel
  backend is out of range, so each one must refuse a CPU call that way);
- the state dict is the same whichever backend is selected or constructed (a checkpoint loads into any backend).
"""

from __future__ import annotations

import csv
import dataclasses
import inspect

import pytest
import torch

from miniworld_engine.measure import config as cfgmod
from miniworld_engine.modules.primitives import BackendModule, BackendUnsupported

with open(cfgmod.REQUIREMENTS, newline="") as _fh:
    ROWS = list(enumerate(csv.DictReader(_fh), 1))
FIRST = {}
for _n, _r in ROWS:
    FIRST.setdefault(_r["module"], (_n, _r))
CLASSES = sorted(FIRST)
#: a tiny length per stream (an atom row needs whole 32-atom query windows; MPNN takes k nearest neighbours)
SMALL = {"token_single": 16, "token_pair": 16, "msa_token": 16, "atom_single": 128, "protein_node": 64}


def _case(n: int, row: dict, *, mode: str = "eval", dtype: str = "float32") -> cfgmod.Case:
    """The row's first case, shrunk: a tiny length / depth, one sample, fp32 (CPU has no bf16 autocast in the measurement)."""
    kwargs = cfgmod._variants(row["dims"], row["options"])[0]
    lengths = cfgmod._ints(row["lengths"])
    depths = cfgmod._ints(row["depths"])
    length = min(min(lengths), SMALL.get(row["stream"], min(lengths)))
    return cfgmod.Case(
        row=n,
        module=row["module"],
        model=row["model"],
        stream=row["stream"],
        kwargs=kwargs,
        length=length,
        depth=min(min(depths), 4) if depths else None,
        samples=1,
        dtype=dtype,
        mode=mode,
        seed=0,
    )


def _build(case: cfgmod.Case, backend: str = "pytorch") -> tuple[BackendModule, dict]:
    module = cfgmod.build(case, device="cpu")
    inputs = cfgmod.inputs(case, module, device="cpu")
    return module.set_backend(backend), inputs


def _floats(tree) -> list[torch.Tensor]:
    if isinstance(tree, torch.Tensor):
        return [tree] if tree.is_floating_point() else []
    if isinstance(tree, (tuple, list)):
        return [t for x in tree for t in _floats(x)]
    if isinstance(tree, dict):
        return [t for x in tree.values() for t in _floats(x)]
    if dataclasses.is_dataclass(tree) and not isinstance(tree, type):
        return [t for f in dataclasses.fields(tree) for t in _floats(getattr(tree, f.name))]
    return []


# ------------------------------------------------------------------------------------------------------------------- construct
@pytest.mark.slow
@pytest.mark.parametrize(("n", "row"), ROWS, ids=[f"r{n}-{r['module']}" for n, r in ROWS])
def test_every_row_constructs(n: int, row: dict):
    """Every constructor variant of every requirements row builds (the kwargs are real constructor arguments)."""
    for kwargs in cfgmod._variants(row["dims"], row["options"]):
        module = cfgmod.construct(cfgmod.module_class(row["module"]), kwargs)
        assert isinstance(module, BackendModule)
        assert module.backend == "pytorch"
        assert "pytorch" in module.backends


@pytest.mark.parametrize("module", CLASSES)
def test_tiny_forward_on_cpu(module: str):
    n, row = FIRST[module]
    case = _case(n, row)
    m, inputs = _build(case)
    with torch.no_grad():
        out = m(**inputs)
    tensors = _floats(out)
    assert tensors, "the forward returned no floating tensor"
    for t in tensors:
        assert t.device.type == "cpu"
        assert torch.isfinite(t).all(), f"non-finite values in an output of {module}"


#: the OuterProductMean rows with AlphaFold 3's divisor n + count_offset (FoldForge R2: served by the kernels, eval and train)
COUNT_OFFSET = [(n, r) for n, r in ROWS if r["module"] == "OuterProductMean" and "count_offset" in r["options"]]


@pytest.mark.parametrize("module", ["LayerNorm", "Transition", "TriangleMultiplication", "AttentionPairBias", "OuterProductMean", "PoolAtoms"])
def test_tiny_backward_on_cpu(module: str):
    _tiny_backward(*FIRST[module])


@pytest.mark.parametrize(("n", "row"), COUNT_OFFSET, ids=[f"r{n}" for n, _ in COUNT_OFFSET])
def test_count_offset_rows_train_on_cpu(n: int, row: dict):
    """FoldForge R2: the count_offset rows exist for train, build with the offset, and the reference trains through them."""
    assert "train" in row["mode"].split("|")
    m, _ = _build(_case(n, row, mode="train"))
    assert getattr(m, "count_offset", None) == pytest.approx(1e-3)
    _tiny_backward(n, row)


def test_count_offset_rows_exist():
    assert COUNT_OFFSET, "FoldForge R2: no OuterProductMean requirement row with count_offset"


def test_pool_atoms_rows_train():
    """FoldForge R17: PoolAtoms is served in training (the gradient of the atom features)."""
    assert any("train" in r["mode"].split("|") for _, r in ROWS if r["module"] == "PoolAtoms")


def _tiny_backward(n: int, row: dict):
    case = _case(n, row, mode="train")
    m, inputs = _build(case)
    floats = {k: v.requires_grad_(True) for k, v in inputs.items() if isinstance(v, torch.Tensor) and v.is_floating_point()}
    out = _floats(m(**inputs))
    torch.autograd.backward(out, [torch.ones_like(t) for t in out])
    assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in floats.values())
    assert all(p.grad is not None for p in m.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------------------------------------------- the contract
@pytest.mark.parametrize("module", CLASSES)
def test_unknown_backend_is_a_value_error(module: str):
    cls = cfgmod.module_class(module)
    m = cfgmod.construct(cls, cfgmod._variants(FIRST[module][1]["dims"], FIRST[module][1]["options"])[0])
    with pytest.raises(ValueError, match="no backend 'no-such-backend'"):
        m.set_backend("no-such-backend")
    assert m.backend == "pytorch"  # unchanged by the refused switch
    if "backend" in inspect.signature(cls.__init__).parameters:
        with pytest.raises(ValueError, match="no backend"):
            cfgmod.construct(
                cls, {**cfgmod._variants(FIRST[module][1]["dims"], FIRST[module][1]["options"])[0], "backend": "no-such-backend"}
            )


ENGINE_BACKENDS = [(m, b) for m in CLASSES for b in cfgmod.module_class(m).backends if b not in ("pytorch", *cfgmod.EXTERNAL)]


@pytest.mark.parametrize(("module", "backend"), ENGINE_BACKENDS, ids=[f"{m}-{b}" for m, b in ENGINE_BACKENDS])
def test_cpu_call_is_refused_with_a_reason(module: str, backend: str):
    """The engine's kernel backends run on CUDA only: a CPU call raises BackendUnsupported naming why (never a PyTorch fallback, never
    another exception from inside a kernel launcher)."""
    n, row = FIRST[module]
    m, inputs = _build(_case(n, row), backend)
    reason = m.unsupported(**inputs)
    assert isinstance(reason, str) and reason.strip(), f"{module}/{backend} claims a CPU call: {reason!r}"
    with pytest.raises(BackendUnsupported, match=backend):
        m(**inputs)


@pytest.mark.parametrize("module", CLASSES)
def test_state_dict_is_backend_independent(module: str):
    cls = cfgmod.module_class(module)
    kwargs = cfgmod._variants(FIRST[module][1]["dims"], FIRST[module][1]["options"])[0]
    torch.manual_seed(0)
    ref = cfgmod.construct(cls, kwargs)
    want = {k: (tuple(v.shape), v.dtype) for k, v in ref.state_dict().items()}
    takes_backend = "backend" in inspect.signature(cls.__init__).parameters
    for backend in cls.backends:
        switched = {k: (tuple(v.shape), v.dtype) for k, v in ref.set_backend(backend).state_dict().items()}
        assert switched == want, f"set_backend({backend!r}) changed the state dict"
        if takes_backend:
            built = cfgmod.construct(cls, {**kwargs, "backend": backend})
            got = {k: (tuple(v.shape), v.dtype) for k, v in built.state_dict().items()}
            assert got.keys() == want.keys(), f"{backend}: keys differ: {sorted(got.keys() ^ want.keys())[:10]}"
            assert got == want, f"{backend}: shapes / dtypes differ"
            built.load_state_dict(ref.state_dict())  # a checkpoint of one backend loads into another
    ref.set_backend("pytorch")
