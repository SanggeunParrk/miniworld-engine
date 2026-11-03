"""A release's sm90 prebuilt bundle keeps every artifact name the previous release carried, unless an allowlist line says why not.

2.4.2 and 2.4.3 shipped without eleven MiniWorld MSA-module families (the OPM ``tma_static`` family, the native PWA family): a replay without
``--keep`` pruned them and no coverage case reached them, so MiniWorld JIT-compiled them on its first call."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from miniworld_engine.build import bundle_coverage, kernel_bundle
from miniworld_engine.kernels import prebuilt

DATA = Path(__file__).parent / "data"
MSA_FAMILIES = (
    "miniworld_opm_epilogue_native_tma_static",
    "miniworld_opm_epilogue_native_tma_static_epi_tma_overlap_half",
    "miniworld_opm_epilogue_native_tma_static_fused",
    "miniworld_pwa_ctr",
    "miniworld_pwa_dgv_bwd_native_v1",
    "miniworld_pwa_fwd2",
    "miniworld_pwa_fwd3",
    "miniworld_pwa_gateout",
    "miniworld_pwa_glue3_native_v1",
    "miniworld_pwa_ln_vg",
    "miniworld_pwa_pair3_native_v1",
)


def _previous() -> set[str]:
    return kernel_bundle.read_name_list(DATA / "prebuilt_names_prev_release.txt")


def _allowed() -> set[str]:
    return kernel_bundle.read_name_list(DATA / "prebuilt_names_allowed_drops.txt")


def test_the_previous_release_list_names_the_msa_module_families():
    assert set(MSA_FAMILIES) <= _previous()


def test_the_dropped_names_are_found_and_the_allowlist_explains_them():
    manifest = {"artifacts": {"k1": {"name": "a"}, "k2": {"name": "b"}}}
    assert kernel_bundle.dropped_names(manifest, {"a", "b"}) == []
    assert kernel_bundle.dropped_names(manifest, {"a", "b", "c", "d"}) == ["c", "d"]
    assert kernel_bundle.dropped_names(manifest, {"a", "b", "c", "d"}, {"d"}) == ["c"]


def test_the_name_list_format(tmp_path: Path):
    f = tmp_path / "n.txt"
    f.write_text("# header\nx\n\ny  # why\n")
    assert kernel_bundle.read_name_list(f) == {"x", "y"}


def test_the_coverage_has_the_msa_module_at_every_depth_and_mode():
    cases = [c for c in bundle_coverage.cases() if c["kind"] == "msa_module"]
    for train in (False, True):
        depths = {c["S"] for c in cases if c["train"] is train}
        assert set(bundle_coverage.MSA_DEPTHS) <= depths
        assert {384, 768} <= {c["L"] for c in cases if c["train"] is train}


@pytest.mark.parametrize("variant", ["sm90-cu129", "sm90-cu130"])
def test_the_shipped_bundle_keeps_the_previous_release_names(variant: str):
    """Runs on a checkout that holds a bundle (a release tree: ``_prebuilt/`` or ``MINIWORLD_ENGINE_CHECK_BUNDLES=<root>``)."""
    root = Path(os.environ.get("MINIWORLD_ENGINE_CHECK_BUNDLES") or prebuilt.PREBUILT_DIR)
    manifest_path = root / variant / prebuilt.MANIFEST
    if not manifest_path.is_file():
        pytest.skip(f"no bundle at {manifest_path}")
    missing = kernel_bundle.dropped_names(json.loads(manifest_path.read_text()), _previous(), _allowed())
    assert not missing, f"{variant} no longer carries {missing}: cover them (bundle_coverage) or list them in prebuilt_names_allowed_drops.txt"
