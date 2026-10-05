"""The shape / dtype gate of the A100 width-generic TriMul (``sm80_wide.shape_ok``), without a device: every registered TriMul width, both modules, the lengths the
kernels tile, and the shapes that must keep the existing path."""
import csv
from pathlib import Path

import pytest
import torch

from miniworld_engine.kernels.trimul_inproj.cuda import sm80_wide

BF16 = torch.bfloat16


@pytest.mark.parametrize("dtype", [BF16, torch.float32])
@pytest.mark.parametrize("d", [64, 128, 256, 384])
@pytest.mark.parametrize("n", [16, 128, 384, 768])
def test_every_registered_width_is_taken_in_both_modules(d, n, dtype):
    assert sm80_wide.shape_ok((1, n, n, d), dtype, d)                         # one direction: hidden D
    assert sm80_wide.shape_ok((1, n, n, d), dtype, d, hs=2 * d)               # bidirectional: hidden 2 D
    assert sm80_wide.shape_ok((2, n, n, d), dtype, d, mask_shape=(2, n), mask_dtype=torch.bool)


def test_the_gate_rejects_what_it_is_not_built_for():
    ok = {"shape": (1, 128, 128, 256), "dtype": BF16, "d_hidden": 256}
    assert sm80_wide.shape_ok(**ok)
    assert not sm80_wide.shape_ok((1, 128, 128, 96), BF16, 96)                          # a width without a unit
    assert not sm80_wide.shape_ok((1, 128, 128, 256), BF16, 128)                        # d_hidden != d_pair
    assert not sm80_wide.shape_ok((1, 128, 128, 256), BF16, 256, hs=384)
    assert not sm80_wide.shape_ok((1, 40, 40, 256), BF16, 256)                          # L % 16
    assert not sm80_wide.shape_ok((1, 128, 64, 256), BF16, 256)                         # not square
    assert not sm80_wide.shape_ok((1, 128, 128, 256), torch.float16, 256)
    assert not sm80_wide.shape_ok((128, 128, 256), BF16, 256)                           # not [B, L, L, D]
    assert not sm80_wide.shape_ok((0, 128, 128, 256), BF16, 256)
    assert not sm80_wide.shape_ok((1, 128, 128, 256), BF16, 256, mask_shape=(1, 64), mask_dtype=torch.bool)
    assert not sm80_wide.shape_ok((1, 128, 128, 256), BF16, 256, mask_shape=(1, 128), mask_dtype=torch.float32)
    assert not sm80_wide.shape_ok((2, 128, 128, 256), BF16, 256, mask_shape=(1, 128), mask_dtype=torch.bool)    # a mask for another batch


def test_the_registered_triangle_multiplication_widths_all_have_a_kernel():
    """Every ``d_pair`` of a ``triangle_multiplication`` row of the model registry is a width of the unit set (a registry width without a kernel would silently run Triton)."""
    registry = Path(sm80_wide.__file__).resolve().parents[2] / "registry" / "registry_module.csv"
    widths = set()
    for row in csv.DictReader(registry.open()):
        if row["module"].startswith("triangle_multiplication"):
            widths.add(int(dict(kv.split("=") for kv in row["dims"].split(";"))["d_pair"]))
    assert widths, "no triangle_multiplication rows"
    assert widths <= set(sm80_wide.WIDTHS), widths
