"""Where the sampler glue (`kernels.sampler_ops`: random rigid motion + re-noising, Kabsch alignment + Euler update, the encoder's pooling) runs on each card
(`kernels/_pair_route.py`, `docs/dispatch.md`): CUDA on H100, the portable kernels on a card the engine holds a record for (fp32 rows for the coordinate ops, a bf16 row for the pooling),
the PyTorch reference with one `UnrecordedCardWarning` on any other card, silently on B200, and the portable kernels everywhere under the internal A/B switch. No GPU is needed: the card is
faked (`tests/fake_card.py`) and the manifests are written into a temp directory."""
from __future__ import annotations

import csv
import warnings

import pytest
import torch
from fake_card import fake_card, write_manifest
from routing_helpers import clean_routing_unrecorded as _clean  # noqa: F401
from settings_helpers import (
    forced_triton as portable_switch,  # the internal A/B switch: the one place the tests name it
)

from miniworld_engine.autotune import records
from miniworld_engine.kernels import _pair_route as R
from miniworld_engine.kernels.sampler_ops import interface, reference
from miniworld_engine.modules import dispatch

CAPS = interface._CUDA_CAPS
A6000 = ("NVIDIA RTX A6000", (8, 6))
COORD_KERNELS = ["sampler_ops_align_triton", "sampler_ops_augment_triton", "sampler_ops_cov_triton", "sampler_ops_stats_triton"]


def _record(directory, *, pool: bool = True, coords: bool = True):
    """A6000 manifest with the fp32 rows of the coordinate kernels and/or the bf16 row of the pooling."""
    path = write_manifest(directory, *A6000, ["sampler_ops"], dtype="fp32")
    with path.open() as handle:
        rows = [r for r in csv.reader(handle) if r and (coords or r[0].startswith("#") or r[0] == "kernel")]
    if pool:
        rows.append(["sampler_ops_pool_triton", "triton", "x", "x.py", "ok", "", "bf16"])
    with path.open("w", newline="") as handle:
        csv.writer(handle).writerows(rows)


def test_the_kernels_of_the_op_by_dtype():
    assert sorted(records.triton_kernels("sampler_ops", "fp32", "inference")) == COORD_KERNELS
    assert records.triton_kernels("sampler_ops", "bf16", "inference") == ("sampler_ops_pool_triton",)
    assert sorted(records.triton_kernels("sampler_ops", "fp32", "training")) == COORD_KERNELS   # no backward: a gradient never blocks the op


@pytest.mark.parametrize("dtype", ["fp32", "bf16"])
def test_h100_runs_cuda_and_b200_the_reference_without_a_warning(monkeypatch, dtype):
    dev = fake_card(monkeypatch, "NVIDIA H100 80GB HBM3", (9, 0))
    assert interface._where(dev, dtype) == R.CUDA
    dev = fake_card(monkeypatch, "NVIDIA B200", (10, 0))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert interface._where(dev, dtype) == R.REFERENCE
    assert R.restricted_reason(dev, CAPS) is not None and not interface._card_ok(dev)


@pytest.mark.parametrize("dtype", ["fp32", "bf16"])
def test_the_ab_switch_puts_every_card_on_the_portable_kernels(monkeypatch, dtype):
    for name, cap in (("NVIDIA H100 80GB HBM3", (9, 0)), ("NVIDIA B200", (10, 0)), A6000):
        dev = fake_card(monkeypatch, name, cap)
        with portable_switch():
            assert interface._where(dev, dtype) == R.PORTABLE
            assert interface._card_ok(dev)
        assert R.cuda_served(dev, CAPS) == (cap == (9, 0))


@pytest.mark.parametrize("dtype", ["fp32", "bf16"])
def test_an_unrecorded_card_warns_once_and_runs_the_reference(monkeypatch, dtype, tmp_path):
    dev = fake_card(monkeypatch, *A6000)
    with records.use_manifest_dir(tmp_path):                                  # no manifest at all
        with pytest.warns(dispatch.UnrecordedCardWarning, match="sampler_ops"):
            assert interface._where(dev, dtype) == R.REFERENCE
        with warnings.catch_warnings():
            warnings.simplefilter("error")                                    # once per (op, card, mode)
            assert interface._where(dev, dtype) == R.REFERENCE
    assert interface._card_ok(dev)                                            # an unrecorded card is handed to the glue: it warns and runs the reference


def test_a_recorded_card_runs_the_portable_kernels(monkeypatch, tmp_path):
    dev = fake_card(monkeypatch, *A6000)
    _record(tmp_path)
    with records.use_manifest_dir(tmp_path), warnings.catch_warnings():
        warnings.simplefilter("error")
        assert interface._where(dev, "fp32") == R.PORTABLE
        assert interface._where(dev, "bf16") == R.PORTABLE


def test_the_pooling_and_the_coordinate_ops_are_recorded_separately(monkeypatch, tmp_path):
    dev = fake_card(monkeypatch, *A6000)
    _record(tmp_path, pool=False)
    with records.use_manifest_dir(tmp_path):
        assert interface._where(dev, "fp32") == R.PORTABLE
        with pytest.warns(dispatch.UnrecordedCardWarning, match="sampler_ops_pool_triton"):
            assert interface._where(dev, "bf16") == R.REFERENCE
    dispatch._unrecorded_warned.clear()
    _record(tmp_path, coords=False)
    with records.use_manifest_dir(tmp_path):
        assert interface._where(dev, "bf16") == R.PORTABLE
        with pytest.warns(dispatch.UnrecordedCardWarning, match="sampler_ops_"):
            assert interface._where(dev, "fp32") == R.REFERENCE


def test_one_failed_kernel_blocks_the_coordinate_ops(monkeypatch, tmp_path):
    dev = fake_card(monkeypatch, *A6000)
    write_manifest(tmp_path, *A6000, ["sampler_ops"], dtype="fp32", skip=frozenset({"sampler_ops_cov_triton"}))
    with records.use_manifest_dir(tmp_path), pytest.warns(dispatch.UnrecordedCardWarning, match="sampler_ops_cov_triton"):
        assert interface._where(dev, "fp32") == R.REFERENCE


def test_a_tensor_that_is_not_on_a_card_is_the_reference(monkeypatch):
    fake_card(monkeypatch, "NVIDIA H100 80GB HBM3", (9, 0))
    with portable_switch():
        assert interface._where(torch.device("cpu"), "fp32") == R.REFERENCE


def test_the_entry_points_are_the_reference_on_cpu_tensors():
    g = torch.Generator().manual_seed(0)
    b, n = 2, 37
    coords = 10 * torch.randn(b, n, 3, generator=g) + 3
    mask = (torch.arange(n) < 30).float()
    rv, tr, noise = torch.randn(b, 2, 3, generator=g), torch.randn(b, 3, generator=g), torch.randn(b, n, 3, generator=g)
    sigma, sigma_next = torch.tensor(2.0), torch.tensor(1.5)
    cfg = dict(window=0, gamma=0.8, gamma_min=1.0, sigma_min=0.0, sigma_max=float("inf"), noise_scale=1.003, variance_floor=0.0, clamp=False)
    noisy, sc = interface.augment_noise(coords, mask, rv, tr, noise, sigma, sigma_next, **cfg)
    want, want_sc = reference.augment_noise(coords, mask, rv, tr, noise, sigma, sigma_next, **cfg)
    assert torch.equal(noisy, want) and torch.equal(sc, want_sc)
    denoised = coords + 0.1 * torch.randn(b, n, 3, generator=g)
    assert torch.equal(interface.align_euler(noisy, denoised, mask, sc, sigma_next, 1.5), reference.align_euler(noisy, denoised, mask, sc, sigma_next, 1.5))
    x = torch.randn(2, 64, 8, generator=g).bfloat16()
    idx = torch.randint(0, 64, (5, 4), generator=g)
    am = torch.rand(5, 4, generator=g) > 0.3
    assert torch.equal(interface.pool_atoms(x, idx, am, am), reference.pool_atoms(x, idx, am, am))
    assert not interface.serves(coords, mask) and not interface.pool_atoms_serves(x)         # a CPU tensor is never handed to the glue
