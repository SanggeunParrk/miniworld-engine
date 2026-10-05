"""The A100 MSA paths' gates, decided without a GPU: every registry row of OuterProductMean / MSAPairWeightedAveraging is inside the contract of the sm_80 kernels, and
inputs the kernels do not implement (CPU tensors here) are declined before anything is built."""

import csv
from pathlib import Path

import pytest
import torch

from miniworld_engine.integrations import opm_sm80, pwa_sm80
from miniworld_engine.kernels.outer_product_mean.cuda import sm80 as opm_kernels
from miniworld_engine.kernels.pair_weighted_averaging.cuda import sm80 as pwa_kernels
from miniworld_engine.modules import MSAPairWeightedAveraging, OuterProductMean
from miniworld_engine.modules.exceptions import ImplementationType

REGISTRY = Path(opm_kernels.__file__).resolve().parents[2] / "registry" / "registry_module.csv"


def _rows(module):
    with REGISTRY.open() as handle:
        for row in csv.DictReader(handle):
            if row["module"] == module:
                yield {k: int(v) for k, v in (kv.split("=") for kv in row["dims"].split(";"))}, row


def test_every_registry_row_of_opm_is_served():
    rows = list(_rows("outer_product_mean"))
    assert rows
    for dims, row in rows:
        assert opm_kernels.supported(dims["d_msa"], dims["d_hidden"], dims["d_pair"]), row["dims"]
        assert row["dtypes"] == "bfloat16"
        assert "train" in row["modes"]
        assert "eval" in row["modes"]


def test_every_registry_row_of_pwa_is_served():
    rows = list(_rows("msa_pair_weighted_averaging"))
    assert rows
    for dims, row in rows:
        d_hidden = dims.get("d_hidden", dims["d_msa"] // dims["n_head"])                 # the module's default per-head width
        assert pwa_kernels.supported(dims["d_msa"], dims["d_pair"], dims["n_head"], d_hidden), row["dims"]
        assert row["dtypes"] == "bfloat16"
    assert pwa_kernels.supported(64, 128, 8, 32)                                          # MiniWorld passes d_hidden_msa = 32 (the bench's 64 / 128 row)


@pytest.mark.parametrize(("args", "ok"), [
    ((64, 32, 128), True), ((128, 32, 256), True), ((128, 32, 384), True), ((64, 32, 256), True), ((128, 32, 128), True),
    ((96, 32, 128), False), ((64, 16, 128), False), ((64, 64, 128), False), ((64, 32, 192), False), ((256, 32, 128), False),
])
def test_opm_contract(args, ok):
    assert opm_kernels.supported(*args) is ok


@pytest.mark.parametrize(("args", "ok"), [
    ((64, 128, 8, 8), True), ((64, 128, 8, 16), True), ((64, 128, 8, 32), True), ((128, 256, 8, 16), True), ((128, 384, 8, 8), True), ((128, 128, 8, 8), True),
    ((128, 256, 8, 32), False), ((64, 128, 4, 16), False), ((64, 128, 16, 8), False), ((96, 128, 8, 8), False), ((64, 192, 8, 8), False), ((64, 128, 8, 64), False),
])
def test_pwa_contract(args, ok):
    assert pwa_kernels.supported(*args) is ok


def test_cpu_inputs_are_declined():
    opm = OuterProductMean(64, 128, 32, implementation=ImplementationType.MINIWORLD).bfloat16()
    pwa = MSAPairWeightedAveraging(64, 128, 8, 32, implementation=ImplementationType.MINIWORLD).bfloat16()
    msa = torch.randn(1, 8, 16, 64, dtype=torch.bfloat16)
    pair = torch.randn(1, 16, 16, 128, dtype=torch.bfloat16)
    with torch.no_grad():
        assert not opm_sm80.serves_inference(opm, msa)
        assert not pwa_sm80.serves_inference(pwa, msa, pair)
    assert not opm_sm80.serves_train(opm, msa)
    assert not pwa_sm80.serves_train(pwa, msa, pair)


def test_the_env_switches_decline(monkeypatch):
    opm = OuterProductMean(64, 128, 32, implementation=ImplementationType.MINIWORLD)
    pwa = MSAPairWeightedAveraging(64, 128, 8, 32, implementation=ImplementationType.MINIWORLD)
    monkeypatch.setenv(opm_sm80.ENV, "0")
    monkeypatch.setenv(pwa_sm80.ENV, "0")
    msa = torch.randn(1, 8, 16, 64, dtype=torch.bfloat16)
    assert not opm_sm80._eligible(opm, msa, None, None, None)
    assert not pwa_sm80._eligible(pwa, msa, torch.randn(1, 16, 16, 128, dtype=torch.bfloat16), None)
