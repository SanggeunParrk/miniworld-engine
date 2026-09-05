"""One function per kernel that launches it once, with inputs the repo states outright.

The point is to stop inferring. ``registry.csv`` says which kernels exist and which driver runs
each one; this module holds the drivers; ``autotune.run_all`` calls them and records what
happened. Nothing scans for capability asserts or guesses shapes -- a kernel runs or it raises,
and the exception text is the reason it did not.

A kernel with no driver yet has an empty ``driver`` cell and is reported ``untested``. That is a
visible hole to close, never a pass.

Drivers take no arguments, run on the current device, and raise on failure. Keep them small: the
job is to reach the kernel, not to check numerics -- accuracy lives in the bench.

Tile alignment
--------------
Every extent these helpers hand out by default is a multiple of 128 (``pair`` 128, ``single``
384, ``rows2d`` 512x384, ``drivers.layernorm_linear._M`` 16384). The five config sets tile at 16/32/64/128, so
every extent divides every tile exactly and no kernel's boundary mask has ever been executed --
a missing or wrong ``mask=`` on a tail tile cannot be observed at these shapes.

``MINIWORLD_SHAPE_MODE=ragged`` subtracts a small amount from each extent that goes through
``ragged()``, putting a partial tile at the end of every axis. Set it before importing any
driver module; the constants are evaluated at import.

Activation dtype
----------------
``registry.csv`` declares 67 of the 103 kernels ``bf16|fp32``; every driver builds bf16.
``MINIWORLD_DRIVER_DTYPE=fp32`` builds them in fp32 instead, by the same import-time route as
``MINIWORLD_SHAPE_MODE`` and ``MINIWORLD_DRIVER_LENGTH``: it rebinds ``BF16``/``ACT_DTYPE``,
which every ``dtype=BF16`` default argument closes over. bf16 stays the default.

An extent a kernel genuinely cannot vary -- a fixed-shape CUDA GEMM, a power-of-two head dim,
a width baked into a weight layout -- goes through ``aligned_only(label, n, why)`` instead. That
keeps the extent aligned *and records why*, so the sweep reports it as an alignment requirement
rather than counting it as ragged coverage it never had.
"""

from __future__ import annotations

import os
from typing import TypedDict

import torch

#: "bf16" (default) or "fp32" -- see "Activation dtype" in the module docstring.
DTYPE_MODE = os.environ.get("MINIWORLD_DRIVER_DTYPE", "bf16").strip().lower()
if DTYPE_MODE not in ("bf16", "fp32"):
    raise ValueError(f"MINIWORLD_DRIVER_DTYPE must be 'bf16' or 'fp32', got {DTYPE_MODE!r}")

#: The activation dtype every driver builds its inputs at. Read at import for the same reason
#: ``SHAPE_MODE`` is: the drivers spell it as a default argument (``dtype=BF16``), which is
#: evaluated when the driver module is imported, so a per-call override would have to reach into
#: every driver module while a per-process one does not.
ACT_DTYPE = torch.bfloat16 if DTYPE_MODE == "bf16" else torch.float32

#: The name every driver and checker imports. It is the ACTIVATION dtype, bf16 unless
#: ``MINIWORLD_DRIVER_DTYPE=fp32`` says otherwise -- kept under the old name so the override
#: reaches all 90-odd ``dtype=BF16`` sites without editing them. An operand a kernel genuinely
#: requires in bf16 (a fixed-dtype CUDA extension, a weight layout baked bf16) spells
#: ``torch.bfloat16`` outright and is unaffected.
BF16 = ACT_DTYPE

#: "aligned" (default) or "ragged" -- see the module docstring.
SHAPE_MODE = os.environ.get("MINIWORLD_SHAPE_MODE", "aligned").strip().lower()
if SHAPE_MODE not in ("aligned", "ragged"):
    raise ValueError(f"MINIWORLD_SHAPE_MODE must be 'aligned' or 'ragged', got {SHAPE_MODE!r}")

#: label -> reason, for every extent a driver declared it cannot perturb.
ALIGNMENT_REQUIRED: dict[str, str] = {}

#: Sequence/atom length L to drive at, or None for each driver's own default.
_ENV_LEN = os.environ.get("MINIWORLD_DRIVER_LENGTH", "").strip()
DRIVER_LENGTH: int | None = int(_ENV_LEN) if _ENV_LEN else None

#: Base channel WIDTH to drive at, or None for each driver's own default. Same mechanism and the
#: same reason as DRIVER_LENGTH; see :func:`driver_width` for why it is one number and not one per
#: axis.
_ENV_WIDTH = os.environ.get("MINIWORLD_DRIVER_WIDTH", "").strip()
#: `or None` and not `if _ENV_WIDTH`: "0" is a truthy STRING, so the second form makes an
#: explicit MINIWORLD_DRIVER_WIDTH=0 mean width zero rather than "use the default", and
#: `ragged(0)` then builds a zero-width tensor. Only `if self.width` in OpUnit.env keeps
#: that unreachable today.
DRIVER_WIDTH: int | None = int(_ENV_WIDTH) or None if _ENV_WIDTH else None


class TensorKw(TypedDict, total=False):
    """Keyword args splatted into a torch factory (``torch.randn(..., **kw)``).

    A bare ``{"device": ..., "dtype": ..., "requires_grad": ...}`` infers as a dict with a joined
    value type, and no ``torch.randn`` overload accepts that -- 21 of this repo's type findings
    were one dict literal reused in three files. Naming the shape lets the splat match.
    """

    device: torch.device | str
    dtype: torch.dtype
    requires_grad: bool


def both_level_is_pair(length: int) -> bool:
    """For a ``level=both`` kernel, is this bucket the PAIR side or the linear side?

    Only when nothing says otherwise. The builder passes the side explicitly (see below), because
    a both-level kernel keys on ROWS and its two sides are separate buckets at the same length.
    This fallback is the rule for a driver run outside a build -- the checkers, a hand probe.

    ``BOTH_SHAPES`` is the UNION of the token set (128..512) and the atom set (256..8192), so a
    both-level kernel meets 512 and below as a pair activation (B, L, L, D) flattening to
    M = L*L, and 1024 and above as an atom activation (B, A, D) flattening to M = A. A driver that
    builds a pair at every bucket is therefore constructing shapes production never presents:
    at L=8192 it asks for M = 67,108,864 rows -- 16 GiB at D=128 -- where the model hands over
    8,192.

    That single mistake accounts for every skipped probe in the shape sweep: 20 CUDA OOMs at
    L=4096/8192 (on a 48 GB card, so no card in the fleet would survive them) and the one illegal
    memory access, which lands exactly at L=8192 because M*D = 8.6e9 is the first bucket to pass
    the int32 offset range. Promoting that kernel's offsets to int64 would have been the wrong fix:
    it spends registers to reach a shape the model never asks for.
    """
    from miniworld_engine.autotune.shape_key import TOKEN_SHAPES

    # An explicit side beats inferring one from the length. A `level=both` kernel keys on ROWS
    # (shape_key.BOTH_ROWS), so its two sides are separate buckets at the SAME length: an atom
    # A=256 launches 256 rows and a pair L=256 launches 65,536, and a work list that picks the
    # side from the length can only ever build one of them. `MINIWORLD_DRIVER_SIDE` is how the
    # builder asks for the other. Unset, the old rule stands.
    side = os.environ.get("MINIWORLD_DRIVER_SIDE", "").strip().lower()
    if side in ("pair", "atom", "token"):
        # "token" is a side too, and it is NOT a pair. The three DiT families are driven per side
        # for their LENGTHS and WIDTHS -- token counts 256..768 at d 384/512/768, atom counts
        # 1024..8192 at d 128 -- but the activation is (B, N, D) on both, never (B, L, L, D).
        # Falling through to the length rule instead would have called every token unit at 512 or
        # under a pair and built an L*L activation: 262,144 rows where the model hands over 512.
        return side == "pair"
    return length <= TOKEN_SHAPES[-1]


def driver_length(default: int) -> int:
    """The L this driver should build its activation at.

    Autotune results are per SHAPE BUCKET -- ``shape_key`` is in every kernel's ``key=[...]`` --
    so a driver frozen at one L can only ever tune one bucket. Driving the kernel directly at each
    L is what makes the work unit ``(op, bucket)`` instead of "a whole module at whatever shape
    that unit happened to be", which is the difference between one tuning per bucket and re-tuning
    the same op once per unit that reaches it.

    Read from the environment at import, exactly like ``SHAPE_MODE`` above, because that is the
    only ordering that works: these constants are module-level and evaluated on import, and the
    kernels read them through helpers that close over them. A per-call override would have to
    reach inside every driver module; a per-process one does not.

    L, not the row count. A pair kernel flattens (B, L, L, D) to M = L*L rows and a linear one to
    M = L, so the row count means different things in different files while L means the same
    thing everywhere -- and L is what ``token_key`` / ``atom_key`` / ``both_key`` bucket.
    """
    return DRIVER_LENGTH if DRIVER_LENGTH is not None else default


def driver_width(default: int) -> int:
    """The base channel WIDTH this driver should build its tensors at.

    The shape key carries the whole shape now -- the width axes are packed into it rather than
    standing beside it in ``key=[...]`` (plan.md G5) -- so a bucket is a (rows, widths) pair and a
    driver frozen at one width can only ever tune the widths its own harness happened to build.
    That is what left 363 lookups uncovered across 42 of 91 ops
    (docs/records/cache-coverage-replay-a6000.md) and what made a second, module-driven pass
    necessary to reach them.

    ONE number, not one per axis, because that is how the modules do it. A family derives every
    other width from a base: ``_DC = ragged(_D_BASE, by=5)``, ``_ND = 4 * _D_BASE``,
    ``_NH = _D // 32``. Overriding the base propagates exactly the way changing ``d_pair`` does in
    the model, and it keeps the sweep to one axis per family instead of a free cross product over
    all of them -- the cross product is 1,136 constexpr combinations, and almost none of them are
    reachable.

    Read from the environment at import, for the same ordering reason as :func:`driver_length`:
    these constants are module-level and the kernels read them through helpers that close over
    them, so a per-call override would have to reach inside every driver module.
    """
    return DRIVER_WIDTH if DRIVER_WIDTH is not None else default


def ragged(n: int, *, by: int = 3, floor: int = 16) -> int:
    """``n`` when aligned, ``n - by`` when ragged, never below ``floor``.

    ``by`` defaults to 3: odd, smaller than the smallest tile (16), and not a divisor of any tile
    width, so the tail tile is partial for every one of the five config sets at once.
    """
    if SHAPE_MODE != "ragged":
        return n
    return max(floor, n - by)


def aligned_only(label: str, n: int, why: str) -> int:
    """Return ``n`` unchanged and record that this extent must stay tile-aligned.

    Use this instead of leaving a bare constant, so the ragged sweep can distinguish "this axis
    was proven to mask correctly" from "this axis was never perturbed".
    """
    ALIGNMENT_REQUIRED[label] = why
    return n


def dev() -> torch.device:
    return torch.device("cuda")


def pair(b: int = 1, n: int = 128, d: int = 128, dtype: torch.dtype = BF16) -> torch.Tensor:
    """A pair activation [B, N, N, D] -- the shape most of these kernels are written for."""
    return torch.randn(b, n, n, d, device=dev(), dtype=dtype)


def single(b: int = 1, n: int = 384, d: int = 384, dtype: torch.dtype = BF16) -> torch.Tensor:
    return torch.randn(b, n, d, device=dev(), dtype=dtype)


def rows2d(m: int = 512, n: int = 384, dtype: torch.dtype = BF16) -> torch.Tensor:
    return torch.randn(m, n, device=dev(), dtype=dtype)


def vec(n: int = 384, dtype: torch.dtype = BF16) -> torch.Tensor:
    return torch.randn(n, device=dev(), dtype=dtype)


def norm_affine(n: int = 384) -> torch.Tensor:
    """A norm gamma/beta AS PRODUCTION HOLDS IT: fp32, whatever the activation dtype is.

    `modules/primitives.LayerNorm` mixes in `_Fp32ParamsMixin`, whose `_apply` pins the affine
    params to fp32 through the trunk's bulk `.to(torch.bfloat16)` -- at value 1.0 the bf16 ULP
    (2**-7) exceeds Adam's per-step update, so a bf16 gamma never trains. The affine reaches the
    fused kernels as a TENSOR OPERAND, and `cache.dtype_of_args` keys on the SET of float operand
    dtypes -- so a production launch keys `bfloat16+float32` while a driver building the affine
    with `vec()` (bf16) recorded plain `bfloat16`. Different bucket, permanent miss, on an axis
    no amount of shape or flag coverage can reach.

    Use this ONLY where the launcher really is handed a `primitives` norm's parameter. Most
    `vec()` call sites are biases or per-channel scales that production does keep in the
    activation dtype, and switching those would invent the mirror-image bug.
    """
    return torch.randn(n, device=dev(), dtype=torch.float32)


# ------------------------------------------------------------------------------------------------
# Promoted here from the per-family driver modules: each of these is used by more than one
# family, so it has exactly one home. A helper only one family uses lives in that family's
# module (see drivers/<family>.py).
# ------------------------------------------------------------------------------------------------

# adaln + conditioned_transition: the tail is fp32 io throughout.
FP32 = torch.float32

# adaln + conditioned_transition.
def _rand(*shape, dtype=BF16):
    return torch.randn(*shape, device=dev(), dtype=dtype)

# The three attention families. The "module docstring" it names is
# drivers/triangle_attention.py's, which is where the dense-grad reason is written out.
def _grad(out: torch.Tensor) -> None:
    """Reach the backward kernels with a dense grad (see the module docstring)."""
    out.backward(torch.randn_like(out))

# layernorm + layernorm_linear.
def _ln_stats(x: torch.Tensor, eps: float = 1e-5) -> tuple[torch.Tensor, torch.Tensor]:
    """(mean, rstd) fp32 [M] for a (M, N) x -- how bench_kernel_layernorm_bwd makes them."""
    xf = x.float()
    return xf.mean(-1), torch.rsqrt(xf.var(-1, unbiased=False) + eps)
