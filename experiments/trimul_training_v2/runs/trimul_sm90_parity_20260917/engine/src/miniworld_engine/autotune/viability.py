"""Predict which configs cannot run on THIS card, from a handful of trial compiles.

The build compiles every config in the grid and finds out at LAUNCH which ones do not fit:
`OutOfResources: shared memory, Required: 196608, Hardware limit: 101376`. Measured on an A6000,
40-74% of the configs of the larger grids fail that way, and compiling is 77% of a unit's wall
time -- so roughly half the build is spent producing kernels that can never be launched on the
card producing them.

Nothing here is written for a particular card. The limit is passed in by the caller, which reads
`shared_memory_per_block_optin` off the device; the coefficients are fitted, per kernel, from
compiles done on the machine running the build; and both the fit and the comparison are validated
against that machine's own measurements before either is used. A card with a different limit, or
a compiler that lays kernels out differently, changes the numbers the probe collects and therefore
the answer -- it does not change what has to be true for the answer to be trusted.

What IS card-specific is the EVIDENCE. Every number quoted in this module was measured on sm86
(A5000 and A6000, both 101,376 B). Whether a fit stays byte-exact on sm90 or sm100, where the
compiler has TMA and clusters and 227 KB to work with, is not known here and cannot be: no such
card is reachable from this cluster (`docs/supported.md`). The gates are what make that safe
rather than merely unknown -- a shape that stops holding stops being used, and the fallback is to
compile everything, which is what the build did before any of this existed.

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

#: Relative error at which a fit is treated as reproducing the compiler rather than approximating
#: it. Not a tolerance to tune: `shared` is deterministic, so anything above rounding is a sign
#: that the model has the wrong shape for this kernel.
EXACT = 1e-9

#: Largest relative error the fit may show on the probe points it did NOT train on, before its
#: bias is applied. Diagnostic only -- soundness comes from the bias, not from this.
MAX_HELDOUT_ERROR = 0.05


@dataclass
class Piece:
    """One linear regime. `bias` is the worst amount it over-predicted any probe point."""

    coef: dict[str, float] = field(default_factory=dict)
    bias: float = 0.0
    affine: bool = False

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

    #: Worst over-prediction on points the fit did NOT train on.
    bias: float = 0.0

    def predict(self, tile: dict[str, int], stages: int) -> float:
        """A LOWER BOUND on the shared memory this config needs, never an estimate of it."""
        if not self.pieces:
            return 0.0
        p = self.pieces[0]
        if stages <= 1:                       # un-pipelined: a measured constant, not the form
            return max(self.at_one - self.bias, 0.0)
        names = [n for n in p.coef if not n.endswith("@s")]
        return max(_raw(p, tile, stages, names, p.affine) - self.bias, 0.0)


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
    """A sample that spans EVERY axis, per warp count -- including the one being extrapolated into.

    The first version of this probed `num_stages` 1 and 2 only: the two points that pin a
    `m * (stages - 1)` model, and the two cheapest configs in the grid. The model was then applied
    out to stages 12 and scored on held-out points that were also stages 1 and 2, so the direction
    it extrapolated in was the one direction never validated. Measured across nine kernels, that
    discarded configs which would have run on four of them -- 168, 100, 81 and 14 of them.

    A validation set has to be drawn from the region the model is applied to. So the probe is a
    deterministic stride through the group ordered by every axis at once, which lands points at
    small and large stages, small and large tiles, and the corners in between.
    """
    axes = tile_axes(configs)
    # Three times the feature count: two thirds to fit on, one third to score the fit on points it
    # has not seen. Fewer and there is nothing left to validate with, which is how a model with a
    # 5.69 held-out error was still being trusted.
    # Sized against the column count, then capped at a share of the group. Validation has to
    # scale with the model -- the affine form has twice the columns and, with a probe sized for
    # the narrower one, produced fits that matched every held-out point and still discarded 27
    # configs that would have run. But a probe is only worth paying for if it is much cheaper
    # than what it saves: unbounded, this compiled 77% of one kernel's grid to predict the other
    # 23%, which is worse than not predicting at all.
    want = 6 * (2 * len(feature_names(axes)) + 1)
    cap = max(1, len({c["num_warps"] for c in configs}) and len(configs) // 10)
    per_group = per_group or max(8, min(want, cap // max(len({c["num_warps"] for c in configs}), 1)))
    out: list[dict] = []
    for w in sorted({c["num_warps"] for c in configs}):
        group = sorted((c for c in configs if c["num_warps"] == w),
                       key=lambda c: (c["num_stages"], *(c[a] for a in axes)))
        if not group:
            continue
        step = max(1, len(group) // per_group)
        picked = group[::step][:per_group]
        for corner in (group[0], group[-1]):        # both extremes, always
            if corner not in picked:
                picked.append(corner)
        # Pairs that differ in exactly ONE axis. A stride through the space almost never produces
        # them, and without them nothing can tell an axis that moves shared memory from one that
        # does not: `GROUP_M` is an L2 swizzle and never moved it by a byte across 2,538 otherwise
        # identical configs, yet it contributed four columns to the fit -- free parameters that
        # absorb residual and cost the byte-exact gate.
        base = picked[len(picked) // 2]
        for a in [*axes, "num_stages"]:
            for other in group:
                if other is base:
                    continue
                if other[a] != base[a] and all(other[b] == base[b]
                                               for b in [*axes, "num_stages"] if b != a):
                    if other not in picked:
                        picked.append(other)
                    break
        out.extend(picked)
    return out


def tile_axes(configs: list[dict]) -> list[str]:
    return sorted(k for k in configs[0] if k not in ("num_warps", "num_stages"))


def inert_axes(measured: dict[tuple, int], configs: list[dict]) -> list[str]:
    """Axes the probe shows have NO effect on shared memory, so they should not be features.

    `GROUP_M` is an L2 swizzle -- it changes the order tiles are visited, not what is held in
    shared memory -- and measured over 2,538 groups of otherwise-identical configs its value never
    moved `shared` by a byte. It still contributed four columns to the fit (itself and its three
    pair products), and those columns are free parameters that absorb residual: with them the
    kernel's fit missed the byte-exact gate for all six warp counts and the prediction was switched
    off, leaving 5,505 unusable configs to be compiled and discovered at launch.

    Detected rather than named, because which axis is inert differs by kernel and a list of axis
    names in this file would be another thing to keep in sync with 91 kernels.
    """
    axes = tile_axes(configs)
    out = []
    for i, a in enumerate(axes):
        groups: dict[tuple, set[int]] = {}
        for k, v in measured.items():
            rest = k[:i] + k[i + 1:]
            groups.setdefault(rest, set()).add(v)
        if groups and all(len(v) == 1 for v in groups.values()) and len(groups) < len(measured):
            out.append(a)
    return out


def fit(measured: dict[tuple, int], configs: list[dict]) -> dict[int, Fit | None]:
    """One `Fit` per warp count, or None where the model is not safe to use for that kernel.

    `measured` maps a config tuple (values in `tile_axes` order, then warps, then stages) to the
    `shared` the compiler reported.

    Two thirds of the probe points train the fit; the remaining third -- drawn from the same
    full-range sample, so it covers the region the fit is applied to -- does two jobs:

      * it decides whether the fit is used at all, and
      * it supplies `bias`, the worst amount the fit OVER-predicted a point it had not seen.

    Both come from held-out points on purpose. A bias measured on training points is meaningless:
    least squares drives those residuals to zero whatever the true shape is, and a bias of 0 taken
    that way discarded 37 usable configs the first time it was tried.

    There is no assumed shape beyond "a linear combination of the tile edges and areas, times the
    pipeline depth". Whether that describes a given kernel is not assumed either -- measured
    across nine kernels it describes some and not others, and shared memory does not even move in
    the same DIRECTION for all of them (a reduction's went DOWN as its tile grew, 399 times).
    """
    axes = tile_axes(configs)
    live = [a for a in axes if a not in inert_axes(measured, configs)]
    names = feature_names(live or axes)
    out: dict[int, Fit | None] = {}
    for w in sorted({c["num_warps"] for c in configs}):
        # `num_stages == 1` is a DIFFERENT form -- no pipeline, so a measured constant rather
        # than `f(tile) * (stages - 1)`, which would be zero there. Folding those points into the
        # same least-squares system pollutes it: every fit that had been exact to the byte failed
        # the exactness gate, and prediction switched off for all nine kernels at once.
        ones = [float(v) for k, v in measured.items()
                if k[len(axes)] == w and k[len(axes) + 1] == 1]
        pts = [(dict(zip(axes, k, strict=False)), k[len(axes) + 1], float(v))
               for k, v in measured.items()
               if k[len(axes)] == w and k[len(axes) + 1] > 1]
        if len(pts) < 6 * len(names):
            out[w] = None
            continue
        hold = pts[::3]
        train = [q for q in pts if q not in hold]
        # Try both shapes and keep whichever the held-out points accept. Proportional is the
        # narrower one -- half the columns, so it survives on fewer probe points -- and it is
        # exact for some kernels: `_wgrad_kernel` fits it to the byte and scores 96.7%, while the
        # affine form has too many parameters for the probe it gets and drops that kernel to
        # 47.5%. Affine is what the others need. Neither is a superset of the other in practice,
        # so the choice is measured rather than argued.
        best = None
        for affine in (False, True):
            piece = _piece_full(train, names, affine=affine)
            if piece is None:
                continue
            over = [_raw(piece, t, st, names, affine) - y for t, st, y in hold]
            err = max(abs(d) / max(y, 1) for d, (_, _, y) in zip(over, hold, strict=True))
            if best is None or err < best[0]:
                best = (err, piece, max(max(over), 0.0), affine)
        if best is None:
            out[w] = None
            continue
        err, piece, bias, affine = best
        piece.affine = affine
        f = Fit(pieces=[piece], at_one=(sum(ones) / len(ones)) if ones else 0.0)
        f.bias, f.heldout_error = bias, err
        # EXACT on the held-out third, or not used. Shared memory is not a noisy measurement --
        # the compiler computes it, and the same config always reports the same number. So a
        # linear form that reproduces unseen points to the byte is almost certainly the formula
        # the compiler is using, while one that is merely close is the wrong shape and its error
        # off the sample is unbounded. Gating on "close" (a held-out error under 5%, a bias
        # measured on held-out points) still discarded 119 configs that would have run, across two
        # of nine kernels. Gating on exact is the only version of this that has held.
        out[w] = f if f.heldout_error <= EXACT else None
    return out


def _limit_hint(pts) -> float:
    """Scale for judging a bias, without the caller having to pass the device limit in twice."""
    return max((y for *_, y in pts), default=1.0)


def _raw(piece: Piece, tile: dict, stages: int, names: list[str], affine: bool) -> float:
    """Shared memory this fit predicts, in whichever of the two shapes it was built as.

    proportional  `f(tile) * (stages - 1)` -- forced through zero at one stage
    affine        `g(tile) + h(tile) * stages`

    Neither covers every kernel. `_wgrad_kernel` is exact under the proportional form and scores
    96.7%; under the affine one it drops to 47.5%, because twice the columns need more probe
    points than it gets and the fit stops passing the exactness gate. `_dx_swiglubwd_kernel` is
    the other way round: it measures 10,240 B at two stages and 18,432 at three, so its line meets
    stages=1 at 2,048 rather than 0 and the proportional form cannot reach it at any coefficient
    -- 58.6% proportional against 96.1% affine.

    So both are fitted and the held-out points choose. Which one wins is a fact about the kernel,
    not something to settle in advance.
    """
    if not affine:
        return sum(piece.coef[n] * _feature_value(n, tile) for n in names) * (stages - 1)
    return sum(piece.coef[k] * v for k, v in _terms(tile, stages, names))


def _terms(tile: dict, stages: int, names: list[str]):
    """Every feature twice: flat, and multiplied by `num_stages`.

    `f(tile) * (stages - 1)` is proportional -- forced to zero at one stage -- and that is the
    shape of some kernels and not others. `_dx_swiglubwd_kernel` measures 10,240 B at two stages
    and 18,432 at three, a step of 8,192, so its line meets stages=1 at 2,048 rather than 0 and no
    coefficient of the proportional form reaches it. Fitted on its two-stage points it predicts
    20,480 for three.

    The extra columns are not free: they doubled the parameters and, on the first attempt, let
    fits reproduce the held-out points exactly while being wrong off-sample -- 27 configs
    discarded that would have run. The probe is sized against the column count for that reason
    (`choose_probes`), so validation scales with the model rather than staying at whatever was
    enough for the narrower one.
    """
    for n in names:
        v = _feature_value(n, tile)
        yield n, v
        yield f"{n}@s", v * stages


def _piece_full(points, names: list[str], affine: bool = True) -> Piece | None:
    if affine:
        cols = [k for n in names for k in (n, f"{n}@s")]
        rows = [[v for _, v in _terms(t, st, names)] for t, st, _ in points]
    else:
        cols = list(names)
        rows = [[_feature_value(n, t) * (st - 1) for n in names] for t, st, _ in points]
    if len(points) < 2 * len(cols):
        return None
    coef = _solve(rows, [y for *_, y in points])
    return None if coef is None else Piece(coef=dict(zip(cols, coef, strict=True)))


def _piece(points: list[tuple[dict, float]], names: list[str]) -> Piece | None:
    if len(points) < 2:
        return None
    rows = [[_feature_value(n, t) for n in names] for t, _ in points]
    coef = _solve(rows, [y for _, y in points])
    return None if coef is None else Piece(coef=dict(zip(names, coef, strict=True)))


def dominated_by(c: dict, axes: list[str], bad: list[dict]) -> bool:
    """True when some config already MEASURED bad is <= this one on every axis.

    No model and no arithmetic: a larger tile with at least as many pipeline stages cannot need
    less shared memory. Measured over 12,377 comparable pairs of one kernel at fixed `num_warps`,
    zero violations -- and `num_warps` has to be fixed, because raising it lets the compiler stop
    staging an operand through smem, which made 18% of the steps up that axis go DOWN.

    This is what covers the warp counts where the linear fit does not hold. Shared with
    `compile_budget`, which anchors the same rule on configs the compile budget killed rather
    than on configs measured over the shared-memory limit -- the rule is about the ordering of
    configs, not about what was measured.
    """
    for b in bad:
        if b["num_warps"] != c["num_warps"]:
            continue
        if b["num_stages"] <= c["num_stages"] and all(b[a] <= c[a] for a in axes):
            return True
    return False


def comparison_holds(measured: dict[tuple, int], configs: list[dict]) -> bool:
    """Does "bigger on every axis needs at least as much shared memory" hold for THIS kernel?

    Checked, not assumed, and per kernel, because it is false for some. Counted over nine kernels
    at fixed `num_warps`: zero violations for the five attention and transpose kernels, and 399,
    147 and 6 for a reduction and two GEMM-shaped ones. A reduction's shared memory went DOWN as
    its tile grew -- 256 bytes at BLOCK=64 against 128 at BLOCK=256 -- because a wider tile leaves
    fewer partial sums per warp to reduce. The GEMM intuition is backwards there.

    Only the probe points are available to check against, so this can miss a violation that the
    sample does not contain. It is a filter for the kernels that are obviously wrong, not a proof.
    """
    axes = tile_axes(configs)
    pts = dict(measured)
    keys = list(pts)
    for a, b in itertools.combinations(keys, 2):
        if a[len(axes)] != b[len(axes)]:                      # num_warps must match
            continue
        if all(x <= y for x, y in zip(a, b, strict=True)) and pts[a] > pts[b]:
            return False
        if all(x >= y for x, y in zip(a, b, strict=True)) and pts[b] > pts[a]:
            return False
    return True


def classify(configs: list[dict], fits: dict[int, Fit | None], limit: int,
             measured_over: list[dict] | None = None,
             comparison_ok: bool = True) -> dict[str, list]:
    """Split the grid into what is worth compiling and what provably is not.

    `skip` is only ever populated for a warp count that produced a usable fit, and only for
    configs whose prediction clears `limit * MARGIN`. Everything else compiles, including every
    config of an unpredictable warp count -- the fallback is the old behaviour, not silence.
    """
    axes = tile_axes(configs)
    bad = (measured_over or []) if comparison_ok else []
    keep, skip = [], []
    for c in configs:
        f = fits.get(c["num_warps"])
        if f is None:
            # No usable fit for this warp count -- fall back to the comparison, which needs no
            # model at all, then to compiling it.
            (skip if dominated_by(c, axes, bad) else keep).append(c)
            continue
        pred = f.predict({a: c[a] for a in axes}, c["num_stages"])
        (skip if pred > limit else keep).append(c)
    return {"keep": keep, "skip": skip,
            "unpredictable_warps": sorted(w for w, f in fits.items() if f is None)}


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
    axes = tile_axes(configs)
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
