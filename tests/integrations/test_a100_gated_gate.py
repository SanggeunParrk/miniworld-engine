"""The shape / dtype gate of the A100 hand-CUDA gated projections (``gated_projection/cuda/sm80.shape_ok`` and the tm1 / tm2 / gate wrappers' ``serves``) without a
device: every ``gated_linear`` row of the model registry is served, and what the 64-column tiles do not divide keeps the existing path."""
import csv
from pathlib import Path

import pytest
import torch

from miniworld_engine.kernels.gated_projection.cuda import sm80 as gated
from miniworld_engine.kernels.tm1.cuda import sm80 as tm1
from miniworld_engine.kernels.tm2.cuda import sm80 as tm2

BF16 = torch.bfloat16


def _gated_linear_rows():
    registry = Path(gated.__file__).resolve().parents[2] / "registry" / "registry_module.csv"
    rows = []
    for row in csv.DictReader(registry.open()):
        if row["module"] == "gated_linear":
            dims = dict(kv.split("=") for kv in row["dims"].split(";"))
            rows.append((row["stream"], int(dims["d_hidden"]), int(dims["d_out"])))
    return rows


def test_every_registered_gated_linear_row_is_served():
    rows = _gated_linear_rows()
    assert len(rows) >= 11, rows
    for stream, d_hidden, d_out in rows:
        assert gated.shape_ok(BF16, d_hidden, d_out), (stream, d_hidden, d_out)


def test_the_gate_rejects_what_it_is_not_built_for():
    assert gated.shape_ok(BF16, 128, 128)
    assert not gated.shape_ok(torch.float32, 128, 128)
    assert not gated.shape_ok(torch.float16, 128, 128)
    assert not gated.shape_ok(BF16, 96, 128)                  # the 64-column tiles must divide both widths
    assert not gated.shape_ok(BF16, 128, 100)
    assert not gated.shape_ok(BF16, 0, 128)
    assert not gated.shape_ok(BF16, 128, 0)


@pytest.mark.parametrize("d", [64, 128, 256, 384])
def test_tm1_and_tm2_need_a_cuda_tensor_of_one_dtype_and_square_weights(d):
    x = torch.zeros(4, d, dtype=BF16)
    w = torch.zeros(d, d, dtype=BF16)
    assert not tm2.serves(x, x, w, w)                         # a CPU tensor
    assert not tm1.serves(x, w, w, w, w)
    assert not tm2.serves(x, x[:2], w, w)                     # shapes differ
    assert not tm2.serves(x, x, w[:, :-1], w)                 # a non-square weight
    assert not tm1.serves(x, w, w, w)                         # four weights
    assert not tm2.serves(x.float(), x.float(), w.float(), w.float())
