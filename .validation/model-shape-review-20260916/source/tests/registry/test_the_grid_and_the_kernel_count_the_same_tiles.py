"""A 1-D grid and the kernel that decodes it must agree on how many column tiles there are.

With a 2-D grid the two could disagree quietly. The launcher sized axis 1 from the OUTPUT width D
while the kernel read the input width K; they are equal for every transition the model runs, and
had they differed the surplus programs would have written masked-off columns -- wrong, but bounded.

``tile_order`` removes that slack. It takes one flat ``pid`` and recovers (row, column) by dividing
by ``n_n = cdiv(K, BLOCK)``. If the grid was built from a different extent, EVERY program gets the
wrong pair -- silently wrong output over the whole matrix, not a masked edge -- and ``K = 0`` makes
it a division by zero inside a kernel, which is undefined behaviour rather than an exception.

So the check moved to the host, where it is two ints and a clear message. This file pins that the
four launchers that name the two extents differently still perform it, and that they name the same
pair the kernel does.
"""
from __future__ import annotations

import ast

import pytest
from paths import ROOT

from miniworld_engine.kernels._tiles import check_tile_axes

KERNELS = ROOT / "src" / "miniworld_engine" / "kernels"

#: Launcher file -> how many launches in it build the grid from an extent the kernel does not
#: divide by. Declared, so adding a fifth such launcher without its check is a failure here rather
#: than a silent one at runtime.
GRID_KERNEL_MISMATCH = {
    "conditioned_transition/triton/inference.py": 1,
    "conditioned_transition/triton/training.py": 1,
    "transition/triton/fused.py": 2,
}


def test_the_check_accepts_equal_extents() -> None:
    check_tile_axes("k", 768, 768, "D", "K")          # must not raise


@pytest.mark.parametrize(("grid_n", "kernel_n"), [(768, 384), (384, 768), (768, 0)])
def test_the_check_refuses_a_disagreement(grid_n: int, kernel_n: int) -> None:
    with pytest.raises(ValueError, match="some_kernel") as exc:
        check_tile_axes("some_kernel", grid_n, kernel_n, "D", "K")
    said = str(exc.value)
    # The message has to carry both numbers: this fires inside a fused launcher whose caller sees
    # only a module, and "shapes disagree" would send someone to the wrong file.
    assert str(grid_n) in said, said
    assert str(kernel_n) in said, said


def _calls(tree: ast.AST, name: str) -> list[ast.Call]:
    return [n for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == name]


@pytest.mark.parametrize("rel", sorted(GRID_KERNEL_MISMATCH))
def test_each_mismatched_launcher_checks_before_it_launches(rel: str) -> None:
    tree = ast.parse((KERNELS / rel).read_text())
    n_want = GRID_KERNEL_MISMATCH[rel]
    checks = _calls(tree, "check_tile_axes")
    assert len(checks) == n_want, (
        f"{rel} makes {len(checks)} check_tile_axes call(s), expected {n_want}. If a launch was "
        f"added or removed, update GRID_KERNEL_MISMATCH and say which extents it compares.")
    for call in checks:
        assert len(call.args) == 5, ast.dump(call)
        grid, kern = call.args[1], call.args[2]
        assert isinstance(grid, ast.Name), ast.dump(call)
        assert isinstance(kern, ast.Name), ast.dump(call)
        assert (grid.id, kern.id) == ("D", "K"), (
            f"{rel} checks {grid.id} against {kern.id}; the grid is built from D and the kernel "
            f"divides by K")


def _is_cdiv(node: ast.AST) -> bool:
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "cdiv" and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "triton")


def test_no_launcher_multiplies_its_own_tile_grid() -> None:
    """One helper, so the product is computed in one place and can be checked in one place.

    Five launches used to spell ``triton.cdiv(M, BM) * triton.cdiv(N, BN)`` by hand. Identical
    arithmetic -- and identical is the problem: nothing tied them to ``tile_order``'s decode, so a
    change to its contract would have had to be found by reading.

    A bare ``triton.cdiv`` is fine and common: a 1-D grid over rows alone, or over ``numel``, has
    no second axis and nothing to order. What this looks for is the PRODUCT of two of them, which
    is a tile grid whoever wrote it.
    """
    offenders = [
        f"{path.relative_to(KERNELS)}:{node.lineno}"
        for path in sorted(KERNELS.rglob("*.py")) if path.name != "_tiles.py"
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult)
        and _is_cdiv(node.left) and _is_cdiv(node.right)]
    assert not offenders, (
        "these launches multiply two triton.cdiv calls to build a tile grid instead of calling "
        f"tile_grid, which is what tile_order decodes: {offenders}")
