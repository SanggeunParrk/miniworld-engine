"""`dev build-wheel --split`: one kernel wheel per prebuilt variant, installing into the engine's `_prebuilt/`, and a pure
engine wheel that requires each of them at its own version -- so every file stays under PyPI's per-file limit while an
install pulls the same kernels one wheel used to carry."""
from __future__ import annotations

import zipfile
from pathlib import Path

from miniworld_engine.build import kernel_bundle


def _base_wheel(tmp_path: Path) -> Path:
    info = "miniworld_engine-9.9.9.dist-info"
    base = tmp_path / "miniworld_engine-9.9.9-py3-none-any.whl"
    with zipfile.ZipFile(base, "w") as z:
        z.writestr("miniworld_engine/__init__.py", "")
        z.writestr(f"{info}/METADATA", "Metadata-Version: 2.4\nName: miniworld-engine\nVersion: 9.9.9\n"
                                       "Requires-Dist: torch>=2.8\n\nThe description.\n")
        z.writestr(f"{info}/WHEEL", "Wheel-Version: 1.0\n")
        z.writestr(f"{info}/RECORD", "")
        z.writestr(f"{info}/entry_points.txt", "[console_scripts]\nminiworld-engine = miniworld_engine.cli:main\n")
        z.writestr(f"{info}/licenses/LICENSE", "MIT")
    return base


def test_each_variant_is_its_own_wheel_and_the_engine_requires_them(tmp_path: Path) -> None:
    out = tmp_path / "dist"
    out.mkdir()
    variants = [
        (v, [(f"miniworld_engine/_prebuilt/{v}/manifest.json", b"{}"),
             (f"miniworld_engine/_prebuilt/{v}/artifacts/cubin/k.cubin", b"\0" * 8)], f"torch 2.13.0 / CUDA {c}")
        for v, c in (("sm90-cu129", "12.9"), ("sm90-cu130", "13.0"))
    ]
    wheels = kernel_bundle.build_split_wheels(_base_wheel(tmp_path), variants, "2.13", "cp312", out)
    names = sorted(w.name for w in wheels)
    assert names == [
        "miniworld_engine-9.9.9-py3-none-any.whl",
        "miniworld_engine_kernels_sm90_cu129-9.9.9-cp312-cp312-manylinux_2_28_x86_64.whl",
        "miniworld_engine_kernels_sm90_cu130-9.9.9-cp312-cp312-manylinux_2_28_x86_64.whl",
    ]
    with zipfile.ZipFile(out / names[0]) as z:
        meta = z.read("miniworld_engine-9.9.9.dist-info/METADATA").decode()
        files = z.namelist()
    head, _, body = meta.partition("\n\n")
    for v in ("sm90-cu129", "sm90-cu130"):
        assert f"Requires-Dist: miniworld-engine-kernels-{v}==9.9.9; {kernel_bundle.KERNEL_WHEEL_MARKER}" in head
    assert body == "The description.\n"
    assert "miniworld_engine-9.9.9.dist-info/entry_points.txt" in files
    assert not any(f.startswith("miniworld_engine/_prebuilt/") for f in files)
    with zipfile.ZipFile(out / names[1]) as z:
        files = z.namelist()
        meta = z.read("miniworld_engine_kernels_sm90_cu129-9.9.9.dist-info/METADATA").decode()
        wheel = z.read("miniworld_engine_kernels_sm90_cu129-9.9.9.dist-info/WHEEL").decode()
    assert "miniworld_engine/_prebuilt/sm90-cu129/manifest.json" in files
    assert not any("sm90-cu130" in f for f in files)
    assert "Name: miniworld-engine-kernels-sm90-cu129\nVersion: 9.9.9\n" in meta
    assert "Requires-Dist: torch==2.13.*" in meta
    assert "Root-Is-Purelib: false" in wheel
    assert "Tag: cp312-cp312-manylinux_2_28_x86_64" in wheel
    assert "miniworld_engine_kernels_sm90_cu129-9.9.9.dist-info/licenses/LICENSE" in files
