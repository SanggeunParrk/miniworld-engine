"""The shipped bundle carries every artifact each ``registry_module.csv`` row is served from.

``data/registry_row_artifacts_sm90.json`` is ``<out>/row_artifacts.json`` of the last ``dev bundle-verify`` that passed (row id -> the artifact
names its calls loaded from the bundle). A row of the registry that the file does not list (a row added, a dimension changed) fails here until
``dev bundle-verify`` is re-run and the file regenerated; with ``MINIWORLD_ENGINE_CHECK_BUNDLES=<prebuilt root>`` every name of every row must also be in the
bundle manifest of each CUDA build (one name covers its keys and Python ABIs)."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from miniworld_engine.build import kernel_bundle, registry_calls

DATA = Path(__file__).parent / "data" / "registry_row_artifacts_sm90.json"


def _recorded() -> dict[str, list[str]]:
    return json.loads(DATA.read_text())["rows"]


def test_the_recorded_rows_are_exactly_the_registry_rows_of_the_arch():
    expected = {e.id for e in registry_calls.expand("sm90") if e.calls}
    recorded = set(_recorded())
    assert recorded == expected, (
        f"registry rows not in {DATA.name}: {sorted(expected - recorded)}; recorded rows the registry no longer has: {sorted(recorded - expected)}. "
        "Re-run `miniworld-engine dev bundle-verify` on the wheels and copy <out>/row_artifacts.json here.")


def test_the_kernels_are_not_empty_for_the_hand_cuda_rows():
    rows = _recorded()
    served = sum(1 for names in rows.values() if names)
    assert served > 40, "most H100 rows are served by hand-CUDA artifacts; an empty record means the recorder did not run"


@pytest.mark.parametrize("variant", ["sm90-cu129", "sm90-cu130"])
def test_the_bundle_manifest_carries_every_registry_row_artifact(variant: str):
    root = os.environ.get("MINIWORLD_ENGINE_CHECK_BUNDLES")
    if not root:
        pytest.skip("MINIWORLD_ENGINE_CHECK_BUNDLES is not set")
    manifest = Path(root) / variant / "manifest.json"
    if not manifest.is_file():
        pytest.skip(f"no bundle at {manifest}")
    have = kernel_bundle.artifact_names(json.loads(manifest.read_text()))
    missing = {row: sorted(set(names) - have) for row, names in _recorded().items() if set(names) - have}
    assert not missing, f"{variant}: registry rows whose artifacts are not in the bundle manifest: {json.dumps(missing, indent=1)}"
