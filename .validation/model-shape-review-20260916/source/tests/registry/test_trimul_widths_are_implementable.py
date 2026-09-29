"""`d_hidden` has to be a width the module actually builds, or not be accepted.

It was neither. `TriangleMultiplication` took `d_hidden` and used it in exactly one place --
`to_out = Linear(d_hidden, d_pair)` -- while the four front projections stayed
`Linear(d_pair, d_pair)`. The moment the two differed the module was internally inconsistent,
and each path failed differently:

  * the pytorch reference raised `mat1 and mat2 shapes cannot be multiplied`;
  * the fused kernels read OUT OF BOUNDS -- `trimul_back_triton` derived one `D` from tri's
    channel axis and passed it as both K and N, so it indexed `to_out.weight.T` as
    (d_pair, d_pair) when it was (d_hidden, d_pair). That read lands inside whatever the caching
    allocator holds next, so it usually returned another tensor's bytes and only trapped when it
    cleared the mapping -- an "illegal memory access" that came and went with `empty_cache()`.

These check the shapes, on CPU, so the inconsistency cannot come back quietly.
"""
from __future__ import annotations

import pytest

from miniworld_engine.modules.triangle_multiplication import TriangleMultiplication
from miniworld_engine.modules.triangle_multiplication.bidirectional import (
    BidirectionalTriangleMultiplication,
)
from miniworld_engine.modules.triangle_multiplication.reference import (
    TriangleMultiplicationReference,
)


@pytest.mark.parametrize("cls", [TriangleMultiplication, TriangleMultiplicationReference])
@pytest.mark.parametrize(("d_pair", "d_hidden"), [(128, 128), (256, 128), (128, 64)])
def test_the_front_emits_what_to_out_consumes(cls, d_pair: int, d_hidden: int) -> None:
    """to_left/to_right must produce exactly the width `to_out` takes, at any (d_pair, d_hidden)."""
    m = cls(d_pair, d_hidden=d_hidden)
    for name in ("to_left", "to_left_gate", "to_right", "to_right_gate"):
        w = getattr(m, name).weight
        assert tuple(w.shape) == (d_hidden, d_pair), f"{cls.__name__}.{name}: {tuple(w.shape)}"
    assert tuple(m.to_out.weight.shape) == (d_pair, d_hidden)
    # LN_out normalises the contraction output, so it is d_hidden wide -- not d_pair.
    assert m.ln_out.weight.numel() == d_hidden


@pytest.mark.parametrize(("d_pair", "d_hidden"), [(128, 128), (256, 128)])
def test_the_bidirectional_front_emits_two_directions(d_pair: int, d_hidden: int) -> None:
    """The bidirectional module was already right; this pins it so both stay in step."""
    m = BidirectionalTriangleMultiplication(d_pair, d_hidden)
    for name in ("to_left", "to_left_gate", "to_right", "to_right_gate"):
        w = getattr(m, name).weight
        assert tuple(w.shape) == (2 * d_hidden, d_pair), f"{name}: {tuple(w.shape)}"
    assert tuple(m.to_out.weight.shape) == (d_pair, 2 * d_hidden)
    assert m.ln_out.weight.numel() == 2 * d_hidden


def test_the_build_matrix_only_drives_widths_a_kernel_can_serve() -> None:
    """`build` drives kernel impls only, so an asymmetric dim compiles and benches nothing.

    Two such pairs sat in PAIR_HID and produced units that launched no kernel -- and, before the
    width fix, reached the back half out of bounds.
    """
    from miniworld_engine.autotune import builder

    for case in builder.cases():
        if not case.name.startswith("triangle_multiplication"):
            continue
        bad = [d for d in case.dims
               if "d_hidden" in d and "d_pair" in d and d["d_hidden"] != d["d_pair"]]
        assert not bad, (
            f"{case.name} drives asymmetric widths {bad}, which every fused trimul refuses; "
            f"those units compile and bench nothing")
