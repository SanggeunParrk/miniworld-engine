"""The estimator is scored against 595 real compiles, not against synthetic data.

`data_shared_a6000.csv` is the `metadata.shared` triton reported for 595 configs of
`_transition_expand_gatebwd_kernel` on an NVIDIA RTX A6000 (limit 101,376 B). Keeping it in the
tree is the point: a smem model checked only against numbers the model itself generated proves
nothing, and the one number that matters -- how often it would remove a config that WOULD have run
-- can only come from ground truth.

The build currently compiles all of these and discovers at launch that 57 of them do not fit. The
question these tests answer is whether a few dozen compiles can tell you that first, and at what
cost in configs wrongly discarded.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from miniworld_engine.autotune import viability

LIMIT = 101376                      # A6000 shared_memory_per_block_optin
AXES = ["BLOCK_K", "BLOCK_M1", "BLOCK_N"]
DATA = Path(__file__).with_name("data_shared_a6000.csv")


def _measured() -> tuple[list[dict], dict[tuple, int]]:
    configs, shared = [], {}
    for r in csv.DictReader(DATA.open()):
        if r["outcome"] != "ok":
            continue
        c = {a: int(r[a]) for a in AXES}
        c["num_warps"], c["num_stages"] = int(r["num_warps"]), int(r["num_stages"])
        configs.append(c)
        shared[(*(c[a] for a in AXES), c["num_warps"], c["num_stages"])] = int(r["shared"])
    return configs, shared


@pytest.fixture(scope="module")
def data():
    configs, shared = _measured()
    assert len(configs) > 500, f"only {len(configs)} measured configs in {DATA.name}"
    return configs, shared


def _run(configs, shared):
    """Both probe rounds, exactly as a build would run them."""
    def key(c):
        return (*(c[a] for a in AXES), c["num_warps"], c["num_stages"])
    first = viability.choose_probes(configs)
    fits = viability.fit({key(p): shared[key(p)] for p in first}, configs)
    anchors_probes = viability.choose_anchor_probes(configs, fits)
    probes = first + anchors_probes
    anchors = [p for p in probes if shared[key(p)] > LIMIT]
    return probes, viability.classify(configs, fits, LIMIT, measured_over=anchors)


def test_the_probe_set_is_small(data) -> None:
    configs, shared = data
    probes, _ = _run(configs, shared)
    assert len(probes) <= len(configs) // 4, (
        f"{len(probes)} probe compiles for {len(configs)} configs is not a probe")
    assert {p["num_stages"] for p in viability.choose_probes(configs)} == {1, 2}


def test_it_never_discards_a_config_that_would_have_run(data) -> None:
    """The only error with a cost. `cache.py`'s old static `num_warps>=16` filter made exactly
    this one and was reverted for it."""
    configs, shared = data
    _, split = _run(configs, shared)
    lost = [c for c in split["skip"]
            if shared[(*(c[a] for a in AXES), c["num_warps"], c["num_stages"])] <= LIMIT]
    assert not lost, f"{len(lost)} usable configs would be discarded, e.g. {lost[:3]}"


def test_it_catches_most_of_what_cannot_run(data) -> None:
    configs, shared = data
    probes, split = _run(configs, shared)
    over = [c for c in configs
            if shared[(*(c[a] for a in AXES), c["num_warps"], c["num_stages"])] > LIMIT]
    caught = len(split["skip"])
    assert over, "the sample must contain configs that exceed the card, or this proves nothing"
    assert caught >= 0.7 * len(over), (
        f"caught {caught} of {len(over)} unusable configs with {len(probes)} probe compiles")


def test_an_unpredictable_kernel_compiles_everything() -> None:
    """Falling back must mean 'compile it all', never 'skip it all'."""
    configs = [{"BLOCK_E": e, "num_warps": 4, "num_stages": s}
               for e in (16, 32, 64) for s in (1, 2, 3)]
    fits = viability.fit({}, configs)                      # nothing measured at all
    split = viability.classify(configs, fits, LIMIT, measured_over=[])
    assert split["skip"] == []
    assert len(split["keep"]) == len(configs)
    assert split["unpredictable_warps"] == [4]


def test_the_feature_set_does_not_name_a_kernels_axes() -> None:
    """It has to work for a kernel whose axes are not BLOCK_M/N/K."""
    names = viability.feature_names(["BLOCK_E", "BLOCK_R"])
    assert "BLOCK_E*BLOCK_R" in names, names
    assert all("BLOCK_M" not in n for n in names)
