"""Hardware and compiler provenance required before reusing captured timings."""
from __future__ import annotations


def provenance(gpu: str | None = None) -> dict[str, str]:
    """Describe the environment that measured a shard, not the merge destination."""
    from miniworld_engine.autotune import cache

    return {"gpu": gpu if gpu is not None else cache.gpu_key(),
            "env_identity": cache.env_identity()}


def provenance_error(data: object, gpu: str | None = None) -> str | None:
    """Reject historical/foreign timings instead of relabelling them on publication."""
    if not isinstance(data, dict) or not isinstance(data.get("_provenance"), dict):
        return "shard has no GPU/compiler provenance; remeasure it"
    actual = data["_provenance"]
    for field, expected in provenance(gpu).items():
        if actual.get(field) != expected:
            return f"shard {field} mismatch: measured {actual.get(field)!r}, expected {expected!r}"
    return None
