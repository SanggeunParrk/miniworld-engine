"""Every committed autotune cache must be fresh against the CODE it ships with.

This is the CI guard for the failure mode that silently degraded a whole benchmark run: a change
that makes a cache's recorded measurements void, shipped without rebuilding, so every launch falls
back to the bounded heuristic subset -- correct, but slower, and mislabelled as the tuned kernel.
The runtime only warns, per launch, after the fact; this recomputes the fingerprints up front (no
GPU), so the commit fails instead of quietly costing performance later.

WHAT COUNTS AS STALE is a policy, and a deliberately narrow one (see ``cache.build_rev``). Three
things void a measurement:

  * ``build_rev`` bumped in registry.csv -- a person declaring that the way this kernel is MEASURED
    changed, so everything recorded under the old way is void;
  * ``key_scheme`` bumped -- the bucket string means something else, so entries are mislabelled;
  * ``env_identity`` -- a different triton/cuda/ptxas. Machine-specific, so surfaced but never
    asserted here: a cache built under another toolchain is legitimately "stale on this machine"
    without the committed code being wrong.

A config-grid edit, a kernel-source edit and a build-driver edit are NOT stale. The first is served
incrementally (`cache.configs_to_bench` benches what the grid added, `store_ranked_configs` drops
what it removed); the other two are auto-hashes of code, and code moving is a different question
from numbers being wrong -- resetting on them cost this repository 32 of 38 tuned buckets on
``cond_transition_expand_swiglu`` for a driver edit that said nothing against them.

To fix a failure: rebuild the named op(s) for the named GPU and `dev merge` the result, or run
``miniworld-engine dev cache-status --gpu <key>`` locally to see the same list.
"""
from __future__ import annotations

from miniworld_engine.autotune import cache_status

#: GPUs whose committed caches are known stale and CANNOT be refreshed from this project's
#: hardware. Asserted separately below so the gate stays meaningful for the cards we do build on,
#: rather than being switched off wholesale.
#:
#: sm86 (A5000 / A6000): 36 caches predate the ``KEY_SCHEME`` 3 bump, which re-based every
#: ``level=atom`` bucket when ``atom_key`` gained the token rungs 384 and 768. Their entries are
#: mislabelled, not merely old, so the fix is a rebuild on an sm86 card -- and this cluster has
#: only A100s. Recorded rather than ignored: the day an A5000 is available, this list is the
#: work item.
CANNOT_REBUILD_HERE = ("NVIDIA RTX A5000 (sm86)", "NVIDIA RTX A6000 (sm86)")


def _stale():
    return [r for r in cache_status.scan() if r.stale]


def test_no_committed_cache_is_stale_on_a_gpu_we_can_rebuild():
    stale = [r for r in _stale() if r.gpu not in CANNOT_REBUILD_HERE]
    assert not stale, (
        "committed autotune caches are stale against the current code -- rebuild + `dev merge` "
        "them (see `dev cache-status`):\n"
        + "\n".join(f"  {r.op} [{r.gpu}]: {r.reason}" for r in stale)
    )


def test_the_unrebuildable_list_is_still_needed_and_still_honest():
    """Guard the exemption. If an sm86 card ever refreshes those caches the entry must go, and if
    the list names a GPU that is no longer stale it is stale documentation."""
    stale_gpus = {r.gpu for r in _stale()}
    dead = [g for g in CANNOT_REBUILD_HERE if g not in stale_gpus]
    assert not dead, (
        f"CANNOT_REBUILD_HERE names GPUs whose caches are no longer stale: {dead}. Remove them -- "
        f"the exemption has been earned back and the gate should cover them again.")
