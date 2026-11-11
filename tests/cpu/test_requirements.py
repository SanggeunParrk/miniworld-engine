"""The requirement lists are well-formed and consistent with the code (requirements/README.md is the spec)."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import pytest

from miniworld_engine.measure import config as cfgmod
from miniworld_engine.modules.primitives import BackendModule

REQ = Path(cfgmod.REQUIREMENTS).parent
KERNELS_DIR = REQ.parent / "kernels"
MODULE_HEADER = [
    "module",
    "model",
    "stream",
    "lengths",
    "depths",
    "dims",
    "options",
    "dtypes",
    "modes",
    "eval_samples",
    "train_samples",
    "source",
]
KERNEL_HEADER = ["name", "family", "modules", "role", "version"]
#: requirements/README.md: no row says bfloat16 (master is the only bf16 case)
DTYPES = {"float32", "master"}
MODES = {"eval", "train"}
#: a human version: a plain positive integer, never a hash
HUMAN = re.compile(r"^[1-9][0-9]*$")
INTS = re.compile(r"^[1-9][0-9]*(\|[1-9][0-9]*)*$")
#: requirements/README.md: the trunk-side classes are master only
KNOWN_BACKENDS = ("pytorch", "triton", "cuda", *cfgmod.EXTERNAL)
TRUNK = (
    "TriangleMultiplication",
    "TriangleMultiplicationRF3",
    "TriangleAttention",
    "TriangleAttentionAF3",
    "TriangleAttentionRF3",
    "TriangleAttentionChai1",
    "TriangleAttentionChai1Confidence",
    "OuterProductMean",
    "MSAPairWeightedAveraging",
    "BidirectionalTriangleAttention",
    "BidirectionalTriangleMultiplication",
    "PairformerBlock",
)


def _read(name: str) -> tuple[list[str], list[dict]]:
    with open(REQ / name, newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


MODULE_FIELDS, MODULE_ROWS = _read("requirements_module.csv")
KERNEL_FIELDS, KERNEL_ROWS = _read("requirements_kernel.csv")
CLASSES = sorted({r["module"] for r in MODULE_ROWS})


def _ids(rows: list[dict], key: str) -> list[str]:
    return [f"r{n}-{r[key]}" for n, r in enumerate(rows, 1)]


# ------------------------------------------------------------------------------------------------------------------ module rows
def test_module_header():
    assert MODULE_FIELDS == MODULE_HEADER
    assert MODULE_ROWS


@pytest.mark.parametrize("row", MODULE_ROWS, ids=_ids(MODULE_ROWS, "module"))
def test_module_row(row: dict):
    assert None not in row, "a data row with more cells than the header"
    cls = cfgmod.module_class(row["module"])
    assert issubclass(cls, BackendModule)
    assert INTS.match(row["lengths"]), row["lengths"]
    assert not row["depths"] or INTS.match(row["depths"]), row["depths"]
    dtypes, modes = row["dtypes"].split("|"), row["modes"].split("|")
    assert set(dtypes) <= DTYPES and len(set(dtypes)) == len(dtypes), row["dtypes"]
    assert set(modes) <= MODES and len(set(modes)) == len(modes), row["modes"]
    for mode in modes:  # every mode the row serves states its sample counts
        assert INTS.match(row[f"{mode}_samples"]), f"{mode}_samples = {row[f'{mode}_samples']!r}"
    assert row["stream"], "every row names its stream"
    assert row["source"].strip(), "every row says who builds it"
    variants = cfgmod._variants(row["dims"], row["options"])
    assert variants
    for kw in variants:  # identifiers (that they are constructor arguments: tests/cpu/test_modules_cpu.py builds every row)
        assert all(re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", k) for k in kw), kw


def test_module_rows_are_unique():
    """Rows with equal (module, stream, dims, options) are one row (requirements/README.md)."""
    seen: dict[tuple, int] = {}
    for n, r in enumerate(MODULE_ROWS, 1):
        key = (r["module"], r["stream"], r["dims"], r["options"])
        assert key not in seen, f"rows {seen[key]} and {n} are the same case set {key}"
        seen[key] = n


@pytest.mark.parametrize("module", TRUNK)
def test_trunk_classes_are_master_only(module: str):
    rows = [r for r in MODULE_ROWS if r["module"] == module]
    assert rows, f"{module} has no requirement row"
    assert {r["dtypes"] for r in rows} == {"master"}


# ------------------------------------------------------------------------------------------------------------------ kernel rows
def test_kernel_header():
    assert KERNEL_FIELDS == KERNEL_HEADER
    assert KERNEL_ROWS


def test_kernel_names_are_unique():
    names = [r["name"] for r in KERNEL_ROWS]
    dup = sorted({n for n in names if names.count(n) > 1})
    assert not dup, dup


@pytest.mark.parametrize("row", KERNEL_ROWS, ids=_ids(KERNEL_ROWS, "name"))
def test_kernel_row(row: dict):
    assert None not in row, "a data row with more cells than the header"
    assert HUMAN.match(row["version"]), f"version {row['version']!r} is not a plain human integer"
    assert row["role"] in cfgmod.ROLE_MODES, row["role"]
    family = row["family"]  # a family directory, or a top-level kernels/_<family>*.py (cast, tf32)
    assert (KERNELS_DIR / family).is_dir() or any(KERNELS_DIR.glob(f"_{family}*.py")), f"no kernel family {family} under kernels/"
    modules = row["modules"].split("|")
    for m in modules:
        cfgmod.module_class(m)  # raises if no such class
    # the derive rule: a kernel resolves to at least one module row in a mode its role serves
    serves = set(cfgmod.ROLE_MODES[row["role"]])
    hits = [r for r in MODULE_ROWS if r["module"] in modules and serves & set(r["modes"].split("|"))]
    assert hits, f"{row['name']} ({row['role']}) resolves to no requirements row of {modules}"


# ------------------------------------------------------------------------------------------------------------------- versions
@pytest.mark.parametrize("module", CLASSES)
def test_class_versions_are_human(module: str):
    cls = cfgmod.module_class(module)
    assert isinstance(cls.version, int) and cls.version >= 1, cls.version
    for backend, v in cls.backend_versions.items():
        name, _, card = backend.partition(":")
        # a subclass inherits its parent's entries, so a name may be a backend the subclass dropped; it must be a backend name
        assert name in KNOWN_BACKENDS, f"backend_versions names {backend!r}: no such backend"
        assert not card or card in ("h100_sm90", "a100_sm80", "b200_sm100"), backend
        assert isinstance(v, int) and v >= 1, (backend, v)


TABLES = sorted(p for p in KERNELS_DIR.glob("*/configs/*.json") if p.name != "candidates.json")


@pytest.mark.parametrize("path", TABLES, ids=[f"{p.parts[-3]}/{p.name}" for p in TABLES])
def test_launch_table_versions_are_human(path: Path):
    data = json.loads(path.read_text())
    if isinstance(data, dict) and "version" in data:
        assert HUMAN.match(str(data["version"])), f"{path}: version {data['version']!r}"


def test_versions_of_every_class_resolve():
    """``measure.config.versions`` (what every record carries) works for every class, mode and card the judge keeps."""
    from miniworld_engine import judge

    for module in CLASSES:
        for mode in ("eval", "train"):
            for gpu in judge.GPUS:
                v = cfgmod.versions(module, mode, gpu)
                assert set(v) == {"module", "kernels", "tables", "backends", "tf32"}
                assert all(HUMAN.match(str(x)) for x in [v["module"], *v["kernels"].values(), *v["tables"].values()])
