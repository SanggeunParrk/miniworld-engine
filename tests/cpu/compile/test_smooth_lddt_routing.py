"""Smooth lDDT (``kernels/smooth_lddt``) routing, card by card -- CPU only, a fake GPU and a fake manifest directory.

H100 / B200 run the CUDA kernels; any other card runs the portable kernels once the engine holds a record for that exact card
(every ``smooth_lddt`` kernel the call launches has an ``ok`` manifest row at fp32), else one ``UnrecordedCardWarning`` and the PyTorch
reference; the A/B switch forces the portable kernels everywhere. A call that needs a backward needs the backward kernel recorded too.
"""
from __future__ import annotations

import csv
import dataclasses
import warnings
from types import SimpleNamespace

import pytest
import torch
from fake_card import fake_card, write_manifest

from miniworld_engine import settings
from miniworld_engine.autotune import records
from miniworld_engine.kernels.smooth_lddt import interface
from miniworld_engine.modules import dispatch
from miniworld_engine.modules.dispatch import UnrecordedCardWarning

A6000 = "NVIDIA RTX A6000"


@pytest.fixture(autouse=True)
def _clean():
    previous = settings.current()
    settings.configure(force_portable_kernels=False, capture=False)
    dispatch._card_cache.clear()
    dispatch._unrecorded_warned.clear()
    yield
    settings.configure(**dataclasses.asdict(previous))
    dispatch._card_cache.clear()
    dispatch._unrecorded_warned.clear()


def _x(device, dtype=torch.float32, requires_grad=False):
    """What ``_backend`` reads of the coordinates: device, dtype, requires_grad."""
    return SimpleNamespace(device=device, dtype=dtype, requires_grad=requires_grad)


def _route(x):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        backend = interface._backend(x)
    return backend, [w for w in caught if issubclass(w.category, UnrecordedCardWarning)]


def _record(directory, name, capability, *, skip=frozenset(), training=True):
    """A manifest with every smooth_lddt kernel ``ok`` at fp32 and at bf16 (what ``run_all`` writes from the two driver passes)."""
    write_manifest(directory, name, capability, ["smooth_lddt"], dtype="fp32", skip=skip, training=training)
    path = directory / f"{records.card_key(name, capability)}.csv"
    rows = list(csv.reader(path.open()))
    rows.extend([kernel, "triton", "x", "x.py", "ok", "", "bf16"]
                for kernel in records.triton_kernels("smooth_lddt", "bf16", "training" if training else "inference") if kernel not in skip)
    with path.open("w", newline="") as handle:
        csv.writer(handle).writerows(rows)


def test_the_kernels_of_the_op_are_the_three_registry_rows():
    assert set(records.triton_kernels("smooth_lddt", "fp32", "training")) == {
        "smooth_lddt_prep_triton", "smooth_lddt_fwd_triton", "smooth_lddt_bwd_triton"}
    assert set(records.triton_kernels("smooth_lddt", "fp32", "inference")) == {"smooth_lddt_prep_triton", "smooth_lddt_fwd_triton"}
    assert set(records.triton_kernels("smooth_lddt", "bf16", "training")) == {"smooth_lddt_fwd_triton", "smooth_lddt_bwd_triton"}


@pytest.mark.parametrize(("name", "capability"), [("NVIDIA H100 80GB HBM3", (9, 0)), ("NVIDIA B200", (10, 0))])
def test_h100_and_b200_take_cuda_with_no_record_and_no_warning(monkeypatch, name, capability):
    device = fake_card(monkeypatch, name, capability)
    assert _route(_x(device)) == ("cuda", [])


def test_a_card_without_a_record_warns_once_and_runs_the_reference(monkeypatch, tmp_path):
    device = fake_card(monkeypatch, A6000, (8, 6))
    with records.use_manifest_dir(tmp_path):
        backend, caught = _route(_x(device))
        assert backend == "reference"
        assert len(caught) == 1
        assert _route(_x(device)) == ("reference", [])   # once per (op, card, mode)


def test_a_recorded_card_runs_the_portable_kernels(monkeypatch, tmp_path):
    device = fake_card(monkeypatch, A6000, (8, 6))
    _record(tmp_path, A6000, (8, 6))
    with records.use_manifest_dir(tmp_path):
        assert _route(_x(device, requires_grad=True)) == ("portable", [])
        assert _route(_x(device, dtype=torch.bfloat16)) == ("portable", [])   # the bf16 rows are there too


def test_a_missing_backward_record_blocks_training_calls_only(monkeypatch, tmp_path):
    device = fake_card(monkeypatch, A6000, (8, 6))
    _record(tmp_path, A6000, (8, 6), training=False)
    with records.use_manifest_dir(tmp_path), torch.enable_grad():
        assert _route(_x(device)) == ("portable", [])
        backend, caught = _route(_x(device, requires_grad=True))
        assert backend == "reference"
        assert len(caught) == 1


def test_a_failed_kernel_is_not_evidence(monkeypatch, tmp_path):
    device = fake_card(monkeypatch, A6000, (8, 6))
    _record(tmp_path, A6000, (8, 6), skip=frozenset({"smooth_lddt_fwd_triton"}))
    with records.use_manifest_dir(tmp_path):
        assert _route(_x(device))[0] == "reference"


def test_the_ab_switch_forces_the_portable_kernels_on_every_card(monkeypatch, tmp_path):
    settings.configure(force_portable_kernels=True)
    for name, capability in (("NVIDIA H100 80GB HBM3", (9, 0)), (A6000, (8, 6))):
        device = fake_card(monkeypatch, name, capability)
        with records.use_manifest_dir(tmp_path):
            assert _route(_x(device)) == ("portable", [])


def test_no_card_at_all_runs_the_reference_without_a_warning():
    assert _route(_x(torch.device("cpu"))) == ("reference", [])


def test_refusal_says_why_only_when_the_tensors_cannot_be_served():
    x, gt = torch.zeros(2, 1, 8, 3), torch.zeros(1, 8, 3)
    flags = torch.zeros(1, 8, dtype=torch.bool)
    assert "not on a CUDA device" in interface.refusal(x, gt, flags, flags)
