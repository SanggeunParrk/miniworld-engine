"""The ``*_shared_cond`` requirement streams (FoldForge's conditioning shared across the samples as a stride-0 ``expand``): ``make_inputs``
gives the shared layout, the measurers keep it expanded, and every other input is handled exactly as before. CPU only."""

from __future__ import annotations

import csv

import pytest
import torch

from miniworld_engine.measure import accuracy, speed
from miniworld_engine.measure import config as cfgmod
from miniworld_engine.modules.primitives import shared_cond_stream

with open(cfgmod.REQUIREMENTS, newline="") as _fh:
    SHARED_ROWS = [(n, r) for n, r in enumerate(csv.DictReader(_fh), 1) if shared_cond_stream(r["stream"])[1]]
SMALL = {"atom_single": 128, "token_single": 16, "atom_folds": 128, "token_folds": 16}


def _old_run_inputs(inputs: dict, *, train: bool) -> dict:
    """The accuracy measurer's input copy before the shared streams (``_run``'s first lines, kept verbatim)."""
    inputs = {k: accuracy._map_floats(v, lambda t: t.detach().clone()) for k, v in inputs.items()}
    if train:
        for v in inputs.values():
            if isinstance(v, torch.Tensor) and v.is_floating_point():
                v.requires_grad_(True)
    return inputs


def _inputs(samples: int = 3) -> dict:
    g = torch.Generator().manual_seed(0)
    base = torch.randn(1, 2, 8, 4, generator=g)
    return {
        "single": torch.randn(samples, 2, 8, 4, generator=g),
        "cond": base.expand(samples, -1, -1, -1),
        "mask": torch.ones(1, 8, dtype=torch.bool).expand(2, 8),
        "rope": (torch.randn(1, 8, 2, generator=g), torch.randn(1, 8, 2, generator=g)),
    }


class _Toy(torch.nn.Module):
    """``single * cond * w`` (+ the rope cos as a constant), recording the strides of the ``cond`` it is called with."""

    backends = ("pytorch",)

    def __init__(self) -> None:
        super().__init__()
        self.w = torch.nn.Parameter(torch.full((4,), 0.5))
        self.seen: list[tuple[int, ...]] = []

    def set_backend(self, backend: str = "pytorch"):
        return self

    def forward(self, single, cond, mask=None, rope=None):
        return self._forward_pytorch(single, cond, mask, rope)

    def _forward_pytorch(self, single, cond, mask=None, rope=None):
        self.seen.append(tuple(cond.stride()))
        return single * cond * self.w + (0 if rope is None else rope[0].sum())


# ----------------------------------------------------------------------------------------------------------------- the stream rule
def test_shared_cond_stream():
    assert shared_cond_stream("atom_single_shared_cond") == ("atom_single", True)
    assert shared_cond_stream("token_folds_shared_cond") == ("token_folds", True)
    assert shared_cond_stream("atom_single") == ("atom_single", False)
    assert shared_cond_stream(None) == (None, False)


def test_keeps_expanded_only_for_shared_streams():
    def case(stream):
        return cfgmod.Case(1, "X", "", stream, {}, 128, None, 5, "master", "eval", 0)

    assert cfgmod.keeps_expanded(case("atom_single_shared_cond"))
    for stream in ("atom_single", "atom_folds", "token_folds", "token_single_fp32_cond", "token_pair"):
        assert not cfgmod.keeps_expanded(case(stream))


def test_expanded_base_restores_the_strides():
    t = _inputs()["cond"]
    assert cfgmod.expanded_dims(t) == (0,)
    base = cfgmod.expanded_base(t)
    assert base.shape == (1, 2, 8, 4) and base.data_ptr() != t.data_ptr()
    again = base.expand(t.shape)
    assert again.stride() == t.stride() and torch.equal(again, t)
    assert cfgmod.expanded_base(t, torch.float64).dtype == torch.float64
    assert cfgmod.expanded_dims(torch.zeros(3, 4)) == ()
    assert cfgmod.expanded_dims(torch.zeros(1, 4).expand(1, 4)) == ()  # size 1: nothing expanded


# ------------------------------------------------------------------------------------------------------------- the accuracy measurer
@pytest.mark.parametrize("train", [False, True])
def test_ordinary_inputs_are_copied_as_before(train: bool):
    """Without ``keep_expanded`` (every existing case) the call inputs and the measured leaves are exactly the old ones."""
    inputs = _inputs()
    old = _old_run_inputs(inputs, train=train)
    new, leaves = accuracy._call_inputs(inputs, train=train, keep_expanded=False)
    assert list(new) == list(old)
    for k in old:
        o, n = old[k], new[k]
        if isinstance(o, torch.Tensor):
            assert torch.equal(o, n) and o.stride() == n.stride() and o.dtype == n.dtype and o.requires_grad == n.requires_grad
            assert n.data_ptr() != inputs[k].data_ptr() or not n.is_floating_point()
        else:
            assert all(torch.equal(a, b) and a.stride() == b.stride() for a, b in zip(o, n, strict=True))
    assert new["cond"].stride(0) != 0  # the old dense copy of an expanded input
    assert list(leaves) == ["single", "cond"]


@pytest.mark.parametrize("train", [False, True])
def test_shared_cond_stays_expanded(train: bool):
    inputs = _inputs()
    call, leaves = accuracy._call_inputs(inputs, train=train, keep_expanded=True)
    cond = call["cond"]
    assert cond.stride() == inputs["cond"].stride() and cond.stride(0) == 0 and torch.equal(cond, inputs["cond"])
    assert leaves["cond"].shape == (1, 2, 8, 4) and leaves["cond"].requires_grad == train
    assert cond.data_ptr() == leaves["cond"].data_ptr() != inputs["cond"].data_ptr()  # a copy of the base, expanded again
    # every other input exactly as without keep_expanded
    plain, _ = accuracy._call_inputs(inputs, train=train, keep_expanded=False)
    for k in ("single", "mask"):
        assert torch.equal(call[k], plain[k]) and call[k].stride() == plain[k].stride() and call[k].requires_grad == plain[k].requires_grad


def test_run_keeps_the_shared_layout_and_sums_its_gradient():
    inputs = _inputs()
    gen = torch.Generator().manual_seed(1)
    toy = _Toy()
    values, upstream = accuracy._run(toy, inputs, train=True, autocast=None, upstream=None, generator=gen, keep_expanded=True)
    assert toy.seen[-1][0] == 0  # the module saw the stride-0 conditioning
    assert upstream is not None
    grad = values["grad:cond"]
    assert grad is not None
    up = upstream["out"].float()
    expect = (up * inputs["single"] * toy.w.detach()).sum(0, keepdim=True)  # the expand's backward: the sum over the samples
    assert grad.shape == (1, 2, 8, 4)
    torch.testing.assert_close(grad, expect)
    # the gold (float64) path keeps it expanded too
    gold_inputs: dict = {k: accuracy._map_floats(v, lambda t: accuracy._keep_expanded(t, torch.float64)) for k, v in inputs.items()}
    assert gold_inputs["cond"].stride(0) == 0 and gold_inputs["cond"].dtype == torch.float64
    assert gold_inputs["single"].stride() == inputs["single"].stride()
    toy64 = _Toy().double()
    gold, _ = accuracy._run(toy64, gold_inputs, train=True, autocast=None, upstream=upstream, generator=None, keep_expanded=True)
    assert toy64.seen[-1][0] == 0
    out, gold_out = values["out"], gold["out"]
    assert out is not None and gold_out is not None
    torch.testing.assert_close(gold_out, out.double(), rtol=1e-5, atol=1e-5)


def test_run_without_keep_is_unchanged():
    """An ordinary case: the module sees a dense copy and the per-sample gradient, as before."""
    inputs = _inputs()
    toy = _Toy()
    values, upstream = accuracy._run(toy, inputs, train=True, autocast=None, upstream=None, generator=torch.Generator().manual_seed(1))
    assert toy.seen[-1][0] != 0
    grad = values["grad:cond"]
    assert grad is not None and upstream is not None
    assert grad.shape == inputs["cond"].shape
    torch.testing.assert_close(grad, upstream["out"].float() * inputs["single"] * toy.w.detach())


# ---------------------------------------------------------------------------------------------------------------- the speed measurer
@pytest.mark.parametrize("train", [False, True])
def test_speed_step_inputs(train: bool):
    for keep in (False, True):
        inputs = _inputs()
        toy = _Toy()
        step = speed.make_step(toy, inputs, backend="pytorch", train=train, autocast=None, seed=0, compile_mode=None, keep_expanded=keep)
        step()
        assert toy.seen[-1] == tuple(inputs["cond"].stride())  # both: the module is called with the stride-0 layout
        if not keep:  # as before: the expanded input is used as given (a leaf view; per-sample gradient in train)
            if train:
                assert inputs["cond"].grad is not None and inputs["cond"].grad.shape == inputs["cond"].shape
        else:  # held as its base (a copy), expanded inside the call: the caller's tensor is untouched
            assert inputs["cond"].grad is None and not inputs["cond"].requires_grad


# ------------------------------------------------------------------------------------------------------------ requirements + inputs
def test_shared_rows_exist_for_the_dit_classes():
    classes = {r["module"] for _, r in SHARED_ROWS}
    assert {"AtomDiTProtenix", "AtomDiTOpenFold3", "AtomDiTBoltz2", "AtomDiTChai1", "TokenDiTProtenix", "TokenDiTChai1"} <= classes
    for _, r in SHARED_ROWS:
        assert r["source"].startswith("FoldForge shared cond"), r["source"]
        assert all(int(s) > 1 for s in r["eval_samples"].split("|")), "a shared conditioning needs S > 1 samples"


@pytest.mark.parametrize(("n", "row"), SHARED_ROWS, ids=[f"r{n}-{r['module']}-{r['stream']}" for n, r in SHARED_ROWS])
def test_shared_rows_make_the_shared_layout(n: int, row: dict):
    base_stream = shared_cond_stream(row["stream"])[0]
    assert base_stream is not None
    depths = cfgmod._ints(row["depths"])
    kw = {
        "row": n,
        "module": row["module"],
        "model": row["model"],
        "kwargs": cfgmod._variants(row["dims"], row["options"])[0],
        "length": min(min(cfgmod._ints(row["lengths"])), SMALL[base_stream]),
        "depth": min(min(depths), 2) if depths else None,
        "samples": 3,
        "dtype": "float32",
        "mode": "eval",
        "seed": 0,
    }
    shared = cfgmod.Case(stream=row["stream"], **kw)
    plain = cfgmod.Case(stream=base_stream, **kw)
    module = cfgmod.build(shared, device="cpu")
    got, ref = cfgmod.inputs(shared, module, device="cpu"), cfgmod.inputs(plain, module, device="cpu")
    assert cfgmod.keeps_expanded(shared) and not cfgmod.keeps_expanded(plain)
    cond = got["cond"]
    assert cond.shape == ref["cond"].shape and cond.stride(0) == 0
    assert set(got) == set(ref)
    for k in got:  # every other input has the base stream's layout
        if k != "cond" and isinstance(got[k], torch.Tensor):
            assert got[k].shape == ref[k].shape and got[k].dtype == ref[k].dtype
    with torch.no_grad():
        module(**got)  # the reference runs on the shared layout
