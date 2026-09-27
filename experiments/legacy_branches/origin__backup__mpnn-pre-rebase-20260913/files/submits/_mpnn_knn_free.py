"""Time the model with the neighbour search hoisted out, to size what that is worth.

The neighbour graph is a function of the coordinates alone and nothing flows back
through it (``coordinate_grad`` is off), so a dataloader could hand it to the model
instead of the forward recomputing it. This measures the ceiling on that change by
caching the search after its first call: every timed step then gets the graph for free,
which is exactly what a precomputed graph would look like from inside the step.

Exact, not approximate, for this benchmark: it builds one static coordinate tensor and
``coordinate_noise`` is 0, so the search returns identical tensors every step anyway.
That is NOT true of real training, where noise perturbs the coordinates per step and the
graph genuinely changes -- there the dataloader would have to own the noise too.

  python submits/_mpnn_knn_free.py kernel=mpnn ...        # same overrides as bench.py
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

# bench.py bootstraps the repo root onto sys.path only when it is sys.path[0], which is
# true when python runs it directly and false when runpy does. Do it here instead, or
# every `from benchmarks...` import inside it fails.
_REPO_ROOT = Path(__file__).resolve().parents[1]
for entry in (str(_REPO_ROOT / "src"), str(_REPO_ROOT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from miniworld_kernels.modules.mpnn import BackboneFeatures  # noqa: E402

def _cached_nearest_neighbors(
    self,
    alpha_carbon,
    residue_mask,
    segment_lengths,
    # Bound as defaults on purpose. `runpy(run_name="__main__")` replaces this
    # module's namespace with bench.py's, so anything looked up as a global from
    # here is gone by the time Dynamo traces the guard: "module '__main__' has no
    # attribute '_CACHE'". Defaults are captured at definition time and survive.
    _original=BackboneFeatures._nearest_neighbors,
    _cache={},  # noqa: B006
):
    # Keyed by module and shape: a sweep builds one model per point, and each point
    # must not be handed another point's graph.
    key = (id(self), tuple(alpha_carbon.shape))
    if key not in _cache:
        _cache[key] = _original(self, alpha_carbon, residue_mask, segment_lengths)
    return _cache[key]


BackboneFeatures._nearest_neighbors = _cached_nearest_neighbors  # type: ignore[method-assign]

sys.argv = ["benchmarks/runners/bench.py", *sys.argv[1:]]
runpy.run_path(str(_REPO_ROOT / "benchmarks/runners/bench.py"), run_name="__main__")
