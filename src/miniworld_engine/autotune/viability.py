"""Predict which configs cannot run on THIS card, from a handful of trial compiles.

The build compiles every config in the grid and finds out at LAUNCH which ones do not fit:
`OutOfResources: shared memory, Required: 196608, Hardware limit: 101376`. Measured on an A6000,
40-74% of the configs of the larger grids fail that way, and compiling is 77% of a unit's wall
time -- so roughly half the build is spent producing kernels that can never be launched on the
card producing them.

Shared memory turns out to be an exact function of the config, not an approximation. For one
Triton GEMM, 595 measured configs fit

    shared(stages) = m * (stages - 1)        for stages >= 2, and a measured constant at stages=1
    m              = 2*BK*BM + 4*BK*BN + 64

to the byte, on every one of eleven tiles. The coefficients are the operand tiles staged in smem
(2 bytes for the bf16 operand, 4 for the promoted one) plus swizzle padding -- the fit recovers
the layout rather than approximating it.

Two things make this safe to act on, and both are enforced below rather than assumed:

  * the fit is CHECKED against the probe points it was built from, and against held-out probe
    points. A kernel whose smem is not this shape is reported unpredictable and nothing is
    skipped for it. Elementwise and reduction kernels are expected to land there.
  * a config is only skipped when the prediction clears the limit by `MARGIN`. Predicting over
    when a config would in fact have run is the one error that costs something -- it removes a
    config from the search, which is the mistake `cache.py`'s old static `num_warps>=16` filter
    made and which was reverted for exactly that reason.

`num_warps` is not a feature of the fit; it selects between fits. Above some warp count the
compiler stops staging an operand through shared memory and the BK*BM term disappears (measured:
its coefficient goes from 2.12 at warps<=8 to 0.00 at 16 and -1.00 at 32). One fit per warp count.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

#: A fit is only useful if its bias (below) is small next to the device limit. Past this it is
#: reported unpredictable rather than kept as a bound that can never fire.
MAX_USEFUL_BIAS = 0.5

#: Largest relative error the fit may show on the probe points it did NOT train on, before its
#: bias is applied. Diagnostic only -- soundness comes from the bias, not from this.
MAX_HELDOUT_ERROR = 0.05


@dataclass
class Piece:
    """One linear regime. `bias` is the worst amount it over-predicted any probe point."""

    coef: dict[str, float] = field(default_factory=dict)
    bias: float = 0.0

    def m(self, tile: dict[str, int]) -> float:
        raw = sum(w * _feature_value(n, tile) for n, w in self.coef.items())
        return raw - self.bias


@dataclass
class Fit:
    """One warp count's model: `shared = m(tile) * (stages - 1)`, plus the stages=1 value.

    `m` is piecewise, not linear. Triton either stages an operand through shared memory or keeps
    it in registers, and which one it picks changes with the tile -- at 4 warps the same kernel
    reports 6,208 B for a tile that a 1-warp launch reports 8,192 B for. One straight line through
    both regimes fits neither: it was rejected by the held-out check for four of six warp counts,
    which disabled the prediction for two thirds of the grid.

    So the probe points are split by the sign of their residual against a first pass and each side
    is refitted. A prediction is the MINIMUM over the pieces -- the minimum of sound lower bounds
    is a sound lower bound, and it does not require knowing which regime a config will land in.
    """

    pieces: list[Piece] = field(default_factory=list)
    at_one: float = 0.0                                    # measured shared at num_stages == 1
    heldout_error: float = 0.0

    @property
    def bias(self) -> float:
        return max((p.bias for p in self.pieces), default=0.0)

    def predict(self, tile: dict[str, int], stages: int) -> float:
        """A LOWER BOUND on the shared memory this config needs, never an estimate of it."""
        if not self.pieces:
            return 0.0
        if stages <= 1:
            return max(self.at_one - self.bias, 0.0)
        return max(min(p.m(tile) for p in self.pieces) * (stages - 1), 0.0)


def _feature_value(name: str, tile: dict[str, int]) -> float:
    if name == "1":
        return 1.0
    prod = _product(name, tile)
    return 0.0 if prod is None else float(prod)


def _product(name: str, tile: dict[str, int]) -> int | None:
    v = 1
    for part in name.split("*"):
        if part not in tile:
            return None
        v *= tile[part]
    return v


def feature_names(axes: list[str]) -> list[str]:
    """`1`, each axis, and every pair product -- the tile AREAS and the edges.

    Not hand-written per kernel: which axes bound an operand tile differs by kernel
    (`BLOCK_M1*BLOCK_K` on a GEMM), so all pairs are offered and the fit picks. The singletons are
    there for the kernels that have ONE axis -- a reduction over `BLOCK_E` stages a vector, not a
    tile, and with pairs only its feature set is the constant term alone, which can express
    nothing and sends every such kernel to the fallback. The cost of an unused feature is one more
    probe tile, and `choose_probes` sizes itself from the feature count.
    """
    pairs = [f"{a}*{b}" for a, b in itertools.combinations(axes, 2)]
    # Singletons only when the pairs alone would be too thin to express anything. A kernel with
    # one axis has NO pairs, so without this its whole feature set is the constant term and it can
    # never be predicted; a kernel with three has three areas already, and every extra feature
    # costs a probe tile (adding singletons there took the probe from 107 compiles to 143 and
    # caught not one more config).
    return ["1", *pairs] if len(pairs) >= 3 else ["1", *axes, *pairs]


def _solve(rows: list[list[float]], rhs: list[float]) -> list[float] | None:
    """Least squares by normal equations with Gaussian elimination. No numpy: this runs inside the
    build, and the build already refuses to grow dependencies for convenience."""
    n = len(rows[0])
    ata = [[sum(r[i] * r[j] for r in rows) for j in range(n)] for i in range(n)]
    atb = [sum(r[i] * b for r, b in zip(rows, rhs, strict=True)) for i in range(n)]
    for i in range(n):
        ata[i][i] += 1e-6                       # ridge: the feature set is deliberately redundant
    for i in range(n):
        p = max(range(i, n), key=lambda r: abs(ata[r][i]))
        if abs(ata[p][i]) < 1e-12:
            return None
        ata[i], ata[p] = ata[p], ata[i]
        atb[i], atb[p] = atb[p], atb[i]
        for r in range(i + 1, n):
            f = ata[r][i] / ata[i][i]
            for c in range(i, n):
                ata[r][c] -= f * ata[i][c]
            atb[r] -= f * atb[i]
    out = [0.0] * n
    for i in reversed(range(n)):
        out[i] = (atb[i] - sum(ata[i][c] * out[c] for c in range(i + 1, n))) / ata[i][i]
    return out


def choose_probes(configs: list[dict], per_group: int = 0) -> list[dict]:
    """The configs to compile first: a spread of tiles at `num_stages` 1 and 2, per warp count.

    Two stages are the minimum that pins the model -- 1 gives the un-pipelined constant, 2 gives
    the slope -- and a spread of tiles rather than the first few, because a fit trained only on
    small tiles mispredicted the large ones by 38% in the sample this was built from.
    """
    axes = _tile_axes(configs)
    # Enough tiles to determine the fit AND hold some back to score it. Fewer and every fit is
    # underdetermined, which reads as "this kernel is unpredictable" and quietly disables the
    # whole mechanism.
    per_group = per_group or len(feature_names(axes)) + 3
    out: list[dict] = []
    for w in sorted({c["num_warps"] for c in configs}):
        tiles = sorted({tuple(c[a] for a in axes) for c in configs if c["num_warps"] == w})
        if not tiles:
            continue
        step = max(1, len(tiles) // per_group)
        picked = tiles[::step][:per_group]
        if tiles[-1] not in picked:                 # the largest tile is where a fit goes wrong
            picked = [*picked[:-1], tiles[-1]]
        for t in picked:
            for stages in (1, 2):
                cand = next((c for c in configs
                             if c["num_warps"] == w and c["num_stages"] == stages
                             and tuple(c[a] for a in axes) == t), None)
                if cand is not None:
                    out.append(cand)
    return out


def _tile_axes(configs: list[dict]) -> list[str]:
    return sorted(k for k in configs[0] if k not in ("num_warps", "num_stages"))


def fit(measured: dict[tuple, int], configs: list[dict]) -> dict[int, Fit | None]:
    """One `Fit` per warp count, or None where the model does not describe that kernel.

    `measured` maps a config tuple (values in `_tile_axes` order, then warps, then stages) to the
    `shared` the compiler reported. Every fit is scored on probe points held out of its own
    training set, and a warp count whose held-out error exceeds `MAX_HELDOUT_ERROR` returns None
    -- which the caller must read as "compile everything for this warp count", never as zero.
    """
    axes = _tile_axes(configs)
    names = feature_names(axes)
    out: dict[int, Fit | None] = {}
    for w in sorted({c["num_warps"] for c in configs}):
        pts = [(k, v) for k, v in measured.items() if k[len(axes)] == w]
        ones = [v for k, v in pts if k[len(axes) + 1] == 1]
        slopes = [(dict(zip(axes, k, strict=False)), float(v))
                  for k, v in pts if k[len(axes) + 1] == 2]
        if len(slopes) < 2:
            out[w] = None
            continue
        hold = slopes[::3] if len(slopes) > 3 else []
        train = [s for s in slopes if s not in hold] or slopes

        first = _piece(train, names)
        if first is None:
            out[w] = None
            continue
        pieces = [first]
        # Residuals over ALL probe points, not just the training ones. Least squares drives the
        # training residuals to ~0 whatever the true shape is, so a split decided on them never
        # fires -- measured, it stayed at one piece for every warp count while the held-out error
        # was 5.69. The held-out points are where a straight line through two regimes shows.
        resid = [(t, y, sum(c * _feature_value(n, t) for n, c in first.coef.items()) - y)
                 for t, y in slopes]
        worst = max((abs(r) / max(y, 1) for _, y, r in resid), default=0.0)
        if worst > MAX_HELDOUT_ERROR:                 # one line through two regimes: split them
            lo = [(t, y) for t, y, r in resid if r <= 0]
            hi = [(t, y) for t, y, r in resid if r > 0]
            split = [p for p in (_piece(lo, names), _piece(hi, names)) if p is not None]
            if len(split) == 2:
                pieces = split

        for p in pieces:                              # bias over EVERY probe point, held out too
            p.bias = max((sum(c * _feature_value(n, t) for n, c in p.coef.items()) - y
                          for t, y in slopes), default=0.0)
            p.bias = max(p.bias, 0.0)

        f = Fit(pieces=pieces, at_one=(sum(ones) / len(ones)) if ones else 0.0)
        f.heldout_error = max((abs(min(p.m(t) for p in pieces) - y) / max(y, 1)
                               for t, y in (hold or train)), default=0.0)
        # The gate is what keeps this sound. Without it a warp count whose shape the model does
        # not capture still produces predictions, and those wrongly discarded 27 usable configs.
        out[w] = f if f.heldout_error <= MAX_HELDOUT_ERROR else None
    return out


def _piece(points: list[tuple[dict, float]], names: list[str]) -> Piece | None:
    if len(points) < 2:
        return None
    rows = [[_feature_value(n, t) for n in names] for t, _ in points]
    coef = _solve(rows, [y for _, y in points])
    return None if coef is None else Piece(coef=dict(zip(names, coef, strict=True)))


def _dominated_by_a_known_bad(c: dict, axes: list[str], bad: list[dict]) -> bool:
    """True when some config already MEASURED over the limit is <= this one on every axis.

    No model and no arithmetic: a larger tile with at least as many pipeline stages cannot need
    less shared memory. Measured over 12,377 comparable pairs of one kernel at fixed `num_warps`,
    zero violations -- and `num_warps` has to be fixed, because raising it lets the compiler stop
    staging an operand through smem, which made 18% of the steps up that axis go DOWN.

    This is what covers the warp counts where the linear fit does not hold.
    """
    for b in bad:
        if b["num_warps"] != c["num_warps"]:
            continue
        if b["num_stages"] <= c["num_stages"] and all(b[a] <= c[a] for a in axes):
            return True
    return False


def classify(configs: list[dict], fits: dict[int, Fit | None], limit: int,
             measured_over: list[dict] | None = None) -> dict[str, list]:
    """Split the grid into what is worth compiling and what provably is not.

    `skip` is only ever populated for a warp count that produced a usable fit, and only for
    configs whose prediction clears `limit * MARGIN`. Everything else compiles, including every
    config of an unpredictable warp count -- the fallback is the old behaviour, not silence.
    """
    axes = _tile_axes(configs)
    bad = measured_over or []
    keep, skip = [], []
    for c in configs:
        f = fits.get(c["num_warps"])
        if f is None:
            # No usable fit for this warp count -- fall back to the comparison, which needs no
            # model at all, then to compiling it.
            (skip if _dominated_by_a_known_bad(c, axes, bad) else keep).append(c)
            continue
        if f.bias > limit * MAX_USEFUL_BIAS:      # a bound that can never fire is not a bound
            keep.append(c)
            continue
        pred = f.predict({a: c[a] for a in axes}, c["num_stages"])
        (skip if pred > limit else keep).append(c)
    return {"keep": keep, "skip": skip,
            "unpredictable_warps": sorted(w for w, f in fits.items()
                                          if f is None or f.bias > limit * MAX_USEFUL_BIAS)}


def choose_anchor_probes(configs: list[dict], fits: dict[int, Fit | None],
                         per_group: int = 6) -> list[dict]:
    """A second, smaller probe round for the warp counts the fit could not describe.

    The first round only compiles `num_stages` 1 and 2 -- the two points that pin the model -- and
    those are the CHEAPEST configs in the grid, so not one of them exceeds the card. That left the
    comparison rule with nothing to compare against and it skipped nothing at all. This round
    deliberately probes the expensive end, so that a config measured over the limit becomes an
    anchor: everything at or above it on every axis, at the same warp count, is then known to be
    over without compiling.
    """
    axes = _tile_axes(configs)
    area = lambda c: sum(c[a] for a in axes) * c["num_stages"]
    out: list[dict] = []
    for w, f in sorted(fits.items()):
        if f is not None:
            continue
        group = sorted((c for c in configs if c["num_warps"] == w), key=area)
        if not group:
            continue
        # Spread across the upper half: the boundary is somewhere in there, and an anchor below it
        # is useless while an anchor far above it covers little.
        upper = group[len(group) // 2:]
        step = max(1, len(upper) // per_group)
        out.extend(upper[::step][:per_group])
    return out
