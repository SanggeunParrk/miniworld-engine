"""The shipped bundle carries every artifact each ``registry_module.csv`` row is served from.

``data/registry_row_artifacts_sm90.json`` is kept by ``miniworld-engine verify``: per row key (``module/stream/dims``) the digest of what the row
declared when it ran clean (``text``), the artifact names its calls loaded from the bundle, and ``served`` (the digest of those artifacts). A row
of the registry the file does not hold for what it declares now (a row added, a length or dtype changed) fails here until ``miniworld-engine
verify`` has run it; with ``MINIWORLD_ENGINE_CHECK_BUNDLES=<prebuilt root>`` every name of every row must also be in the bundle manifest of each
CUDA build (one name covers its keys and Python ABIs)."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from miniworld_engine.build import kernel_bundle, registry_calls

DATA = Path(__file__).parent / "data" / "registry_row_artifacts_sm90.json"


def _recorded() -> dict[str, dict]:
    return json.loads(DATA.read_text())["rows"]


def test_a_row_key_names_one_row():
    keys = [e.key for e in registry_calls.expand("sm90")]
    assert len(keys) == len(set(keys)), sorted({k for k in keys if keys.count(k) > 1})


def test_every_registry_row_of_the_arch_has_a_clean_record_of_what_it_declares_now():
    expected = {e.key: e.text_digest for e in registry_calls.expand("sm90") if e.calls}
    recorded = _recorded()
    stale = sorted(k for k in expected if recorded.get(k, {}).get("text") != expected[k])
    gone = sorted(set(recorded) - set(expected))
    assert not stale, (f"registry rows without a clean record of what they declare now: {stale}. "
                       "Run `miniworld-engine bundle` then `miniworld-engine verify` (they run exactly these rows).")
    assert not gone, f"recorded rows the registry no longer has: {gone}. Run `miniworld-engine verify` (it drops them)."


def test_the_kernels_are_not_empty_for_the_hand_cuda_rows():
    served = sum(1 for r in _recorded().values() if r["artifacts"])
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
    missing = {row: sorted(set(r["artifacts"]) - have) for row, r in _recorded().items() if set(r["artifacts"]) - have}
    assert not missing, f"{variant}: registry rows whose artifacts are not in the bundle manifest: {json.dumps(missing, indent=1)}"
