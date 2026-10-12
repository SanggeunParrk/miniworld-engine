"""The wiring has to compute what the offline scoring computed. Nothing guarantees it does.

`viability` and `compile_budget` were scored by calling them directly on the fixture logs. The
build reaches them through `capture._predict_unusable`, which builds their inputs its own way --
`_config_dict` drops non-integer axes, `_cfg_sig` fixes the signature order, the probe set comes
back as config OBJECTS and has to be matched to dicts again, and the shared-memory readings arrive
by re-reading a log file at an offset. A mistake anywhere in that returns a plausible answer:
too few configs skipped, and the pass is dead weight; too many, and it removes configs that win.

So this drives the real `_predict_unusable` on the frozen fixture, with only the three things that
need a GPU or a pool replaced -- the device limit, the probe compile, and the log read -- and
checks its answer against the numbers those modules were scored on.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from miniworld_engine.autotune import capture, smem_log, viability

DATA = Path(__file__).parent / "compile_budget"
LIMIT = 101376


class _Cfg:
    """Shaped like a triton Config: kwargs plus the two knobs."""

    def __init__(self, d: dict):
        self.num_warps = d["num_warps"]
        self.num_stages = d["num_stages"]
        self.kwargs = {k: v for k, v in d.items() if k not in ("num_warps", "num_stages")}


def _parse(sig: str) -> dict | None:
    try:
        return {k: int(v) for k, v in (p.split("=") for p in sig.split(","))}
    except ValueError:
        return None


def _offline_skip(shared: dict[str, int]) -> tuple[set[str], set[str]] | None:
    """`viability`'s own answer, computed the way it was scored: the config SIGS it rules out."""
    configs, index = [], {}
    for sig, b in shared.items():
        d = _parse(sig)
        if not d or "num_warps" not in d or "num_stages" not in d:
            continue
        configs.append(d)
        index[id(d)] = (sig, b)
    if len(configs) < 150:
        return None
    axes = viability.tile_axes(configs)

    def key(d):
        return (*(d[a] for a in axes), d["num_warps"], d["num_stages"])

    measured = {key(d): index[id(d)][1] for d in configs}
    first = viability.choose_probes(configs)
    fits = viability.fit({key(p): measured[key(p)] for p in first if key(p) in measured}, configs)
    probes = first + viability.choose_anchor_probes(configs, fits)
    seen = {key(p): measured[key(p)] for p in probes if key(p) in measured}
    split = viability.classify(
        configs, fits, LIMIT,
        measured_over=[p for p in probes if measured.get(key(p), 0) > LIMIT],
        comparison_ok=viability.comparison_holds(seen, configs))
    by_key = {key(d): index[id(d)][0] for d in configs}
    return {by_key[key(d)] for d in split["skip"]}, {by_key[key(p)] for p in probes}


@pytest.fixture(scope="module")
def kernel():
    """The largest fixture kernel that has both shared-memory readings and kills."""
    shared = smem_log.read(Path(__file__).parent)      # the nine-kernel smem fixtures
    times = smem_log.compile_ms(DATA)
    killed = smem_log.killed(DATA)
    name = "_dx_swiglubwd_kernel"
    assert name in shared, sorted(shared)
    assert name in times, sorted(times)
    return name, shared[name], times[name], killed.get(name, set())


@pytest.fixture(autouse=True)
def _clean():
    capture._PREDICTED_BAD.clear()
    for k, v in list(capture._PREDICT.items()):
        capture._PREDICT[k] = type(v)()
    yield
    capture._PREDICTED_BAD.clear()


def _run(monkeypatch, name, shared, times, killed):
    """`_predict_unusable` on the fixture, with the GPU and the pool stood in for."""
    # LOG order, not sorted: `viability.choose_probes` strides its input, so the order decides
    # which configs get probed and therefore which anchors exist. Measured on the fixture,
    # `_attn_fwd` rules out 171 configs in log order and 129 sorted -- a 25% swing from ordering
    # alone. The offline scoring reads the log in order, so this has to as well or the comparison
    # is against a different experiment.
    sigs = list(dict.fromkeys([*shared, *times, *killed]))
    configs, keep_sigs = [], []
    for sig in sigs:
        d = _parse(sig)
        if d and "num_warps" in d and "num_stages" in d:
            configs.append(_Cfg(d))
            keep_sigs.append(sig)
    monkeypatch.setattr(capture, "_shared_limit", lambda: LIMIT)
    monkeypatch.setattr(capture, "_smem_log_end", lambda: 0)

    seen: set[str] = set()

    def _probes(src, target, options, probes, kname, rnd, jobs):
        got, dead = {}, set()
        for c in probes:
            sig = capture._cfg_sig(c)
            (dead.add(sig) if sig in killed else got.__setitem__(sig, times.get(sig, 0) / 1000))
        seen.update(capture._cfg_sig(c) for c in probes)
        return got, dead

    monkeypatch.setattr(capture, "_compile_probes", _probes)
    monkeypatch.setattr(capture, "_read_smem_from",
                        lambda off: {s: shared[s] for s in seen if s in shared})
    kept = capture._predict_unusable(None, None, None, configs, name, "r", jobs=4)
    return configs, kept, seen


def test_the_wiring_rules_out_exactly_what_the_model_does(monkeypatch):
    """Nine kernels, the shared-memory half, through the build's glue against the model's own
    answer. They are not the same NUMBER and should not be: the model is scored on the configs it
    can rule out, the build on the compiles it avoids, and a config already compiled as a probe is
    ruled out by the first and not avoided by the second. Subtract those and they must be equal.
    """
    shared_all = smem_log.read(Path(__file__).parent)
    checked = 0
    for name in sorted(shared_all):
        answer = _offline_skip(shared_all[name])
        if answer is None:
            continue
        model_skip, _ = answer
        configs, kept, probes = _run(monkeypatch, name, shared_all[name], {}, set())
        kept_sigs = {capture._cfg_sig(c) for c in kept}
        avoided = {capture._cfg_sig(c) for c in configs} - kept_sigs
        assert avoided == model_skip - probes, (
            f"{name}: the build avoided {len(avoided)} compiles, the model rules out "
            f"{len(model_skip)} of which {len(model_skip & probes)} were already probed")
        checked += 1
    assert checked >= 6, f"only {checked} kernels were comparable"


def test_the_answer_depends_on_the_order_the_configs_arrive_in(kernel):
    """Not a defect found, a property recorded. `choose_probes` strides its input, so the order
    decides which configs are probed and therefore which anchors exist. Measured on `_attn_fwd`:
    171 configs ruled out in log order, 129 sorted -- a 25% swing with the same 144 probes and no
    usable fit either way, because everything came from the comparison rule and the comparison
    rule can only reach what an anchor is above.

    Left alone rather than sorted, because there is no evidence a particular order is better and
    picking one would be picking it blind. What matters is that the build's order is the grid's,
    which is deterministic, so a build is reproducible.
    """
    _name, shared, _, _ = kernel
    first = _offline_skip(shared)
    second = _offline_skip(dict(sorted(shared.items())))
    assert first is not None
    assert second is not None
    assert first[0] != second[0], "the fixture no longer shows the ordering effect this pins"


def test_a_probe_is_never_ruled_out(monkeypatch, kernel):
    """It has already been compiled. Ruling it out spends the compile and keeps nothing."""
    name, shared, times, killed = kernel
    _, kept, probes = _run(monkeypatch, name, shared, times, killed)
    kept_sigs = {capture._cfg_sig(c) for c in kept}
    assert probes - killed <= kept_sigs


def test_the_probe_is_a_small_share_of_the_grid(monkeypatch, kernel):
    name, shared, times, killed = kernel
    configs, _, probes = _run(monkeypatch, name, shared, times, killed)
    assert len(probes) < len(configs) // 4, f"{len(probes)} probes for {len(configs)} configs"


def test_the_signature_the_wiring_builds_is_the_one_the_logs_use(kernel):
    """`_cfg_sig` and the log's own key have to agree or nothing matches anything."""
    _name, shared, _, _ = kernel
    sig = next(iter(shared))
    d = _parse(sig)
    assert d is not None
    assert capture._cfg_sig(_Cfg(d)) == sig


def test_the_axis_order_the_wiring_uses_is_the_models_own(kernel):
    """`_predict_unusable` builds its keys from `viability.tile_axes`; if it built them from
    `dict` order the fit would be fitting a permutation."""
    _name, shared, _, _ = kernel
    dicts = [d for s in shared if (d := _parse(s))]
    axes = viability.tile_axes(dicts)
    assert axes == sorted(axes)
    assert "num_warps" not in axes
    assert "num_stages" not in axes


def test_both_halves_run_on_a_kernel_that_has_kills(monkeypatch, kernel):
    """The compile-budget half needs killed probes; the shared-memory half needs readings. This
    is the one fixture kernel that carries both."""
    name, shared, times, killed = kernel
    configs, kept, _ = _run(monkeypatch, name, shared, times, killed)
    skipped = len(configs) - len(kept)
    assert skipped > 0.3 * len(configs), (
        f"{skipped} of {len(configs)} ruled out; the offline score on this kernel was 76% over "
        f"the shared-memory limit alone, so the wiring is not reaching the models")
    assert capture._PREDICT["skipped"] == skipped
    assert capture._PREDICT["kernels"] == 1
