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

#: SCOPE. The engine is currently built, verified and released for the H100 (sm90) only. The caches for the other cards stay
#: committed as they were last built, and staleness there is reported (``dev cache-status``) but not asserted:
#:
#: sm80 (A100), sm86 (A5000 / A6000), sm100 (B200): out of scope for this release line. Their Triton caches go stale on every kernel
#: edit (Triton's ``cache_key`` hashes the source *and its line number*), and nothing here can rebuild them -- the cluster has no
#: A100 / B200 in the H100 queue, and the A5000 / A6000 caches are refreshed in a separate sweep. Gating on them would turn every
#: edit into a permanent red. They are not "fine": when one of those cards is taken up again, remove its arch from
#: ``OUT_OF_SCOPE_ARCHES`` and rebuild it.
OUT_OF_SCOPE_ARCHES = ("(sm80)", "(sm86)", "(sm100)")

#: sm90 Triton caches (``*_triton`` ops): the H100 launches no Triton kernel on the engine's paths -- every op has a hand-CUDA kernel or
#: the PyTorch reference (docs/gpus/h100/dispatch.md; tests/gpu/sm90/test_architecture_no_triton_gpu.py records the launches). Those caches
#: serve only explicit portable-kernel A/B runs, so a stale one costs no production launch. The native (CUDA) caches on sm90 are the
#: ones the H100 serves, and they ARE asserted.
H100_TRITON_SUFFIX = "_triton"


def _stale():
    return [r for r in cache_status.scan() if r.stale]


def _in_scope(r) -> bool:
    if any(a in r.gpu for a in OUT_OF_SCOPE_ARCHES):
        return False
    if "(sm90)" in r.gpu and r.op.endswith(H100_TRITON_SUFFIX):
        return False
    return True


def test_no_committed_cache_is_stale_on_a_gpu_we_can_rebuild():
    stale = [r for r in _stale() if _in_scope(r)]
    assert not stale, (
        "committed autotune caches are stale against the current code -- rebuild + `dev merge` "
        "them (see `dev cache-status`):\n"
        + "\n".join(f"  {r.op} [{r.gpu}]: {r.reason}" for r in stale)
    )


def test_the_out_of_scope_list_is_still_needed_and_still_honest():
    """Guard the exemption. If a card in OUT_OF_SCOPE_ARCHES no longer has any stale cache the entry must go (the gate should
    cover it again), and the H100 Triton exemption must still be hiding something."""
    stale = _stale()
    dead = [a for a in OUT_OF_SCOPE_ARCHES if not any(a in r.gpu for r in stale)]
    assert not dead, (
        f"OUT_OF_SCOPE_ARCHES names archs whose caches are no longer stale: {dead}. Remove them -- "
        f"the exemption has been earned back and the gate should cover them again.")
    assert any("(sm90)" in r.gpu and r.op.endswith(H100_TRITON_SUFFIX) for r in stale), (
        "no sm90 Triton cache is stale any more: drop the H100_TRITON_SUFFIX exemption so the gate covers them")
