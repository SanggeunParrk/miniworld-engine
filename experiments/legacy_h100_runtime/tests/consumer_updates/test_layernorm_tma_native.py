"""B4 uses the actual Triton CSV and the native build/cache contract."""

import csv, itertools
from pathlib import Path


def test_tma_grid_is_exact_triton_grid():
    from miniworld_engine.autotune.trimul_sm90_config import declared_configs
    from miniworld_engine.autotune import trimul_sm90_config

    path = (
        Path(trimul_sm90_config.__file__).parent
        / "configs/grid/layernorm_bwd_split_triton.csv"
    )
    with path.open() as f:
        rows = list(csv.DictReader(f))
    expected = [
        dict(zip([r["axis"] for r in rows], v))
        for v in itertools.product(
            *[[int(v) for v in r["values"].split()] for r in rows]
        )
    ]
    assert declared_configs("layernorm_bwd_split_sm90_cute") == expected
    assert len(expected) == 1200


def test_tma_native_builder_metadata():
    from miniworld_engine.autotune.native import (
        build_ops_for_arch,
        native_shape_supported,
    )
    from miniworld_engine.autotune.native_compile import task_for

    op = "layernorm_bwd_split_sm90_cute"
    assert op in build_ops_for_arch("sm90") and op not in build_ops_for_arch("sm80")
    assert native_shape_supported(op, 137, "float32")
    assert not native_shape_supported(op, 128, "float16")
    assert task_for(op, {}, "unused") is None


def test_tma_setting_roundtrip():
    from miniworld_engine import settings

    before = settings.current().trimul_sm90_kernels
    try:
        settings.configure(trimul_sm90_kernels={"out_ln_bwd"})
        assert settings.current().trimul_sm90_kernels == frozenset({"out_ln_bwd"})
    finally:
        settings.configure(trimul_sm90_kernels=before)
