"""Which cards the engine holds Triton evidence for -- the one question ``modules.dispatch`` asks.

``implementation="miniworld"`` on a card without a native CUDA path routes to the engine's Triton kernel family only if
the engine has a RECORD for that exact card; otherwise it warns and runs the PyTorch reference. "Recorded" is not a
judgement made here, it is a lookup of two facts the repo already keeps:

  * ``autotune/manifests/<card>.csv`` -- written by ``python -m miniworld_engine.autotune.run_all`` on the card. A row with
    status ``ok`` means the kernel was driven on that card and checked against its reference (the ``#provenance`` row says
    when and at which commit); ``failed`` / ``untested`` rows and kernels with no row at all are NOT evidence.
  * ``kernels/registry/registry.csv`` -- what exists. The Triton rows of the kernel families an op launches
    (``OP_KERNELS`` below) are the denominator, minus the ``developed == no`` rows (A/B alternatives no module launches).

An op is recorded on a card FOR A MODE when every such kernel the mode launches has an ``ok`` row for that card at the dtype in use
(a kernel whose registry row does not declare the dtype does not count against it). The modes are the two calls a module makes:
``inference`` (no backward is recorded: forward kernels only) and ``training`` (everything). A kernel is a training kernel when its
name says it saves for or runs the backward -- a ``bwd``, ``saveact``, ``train`` or ``dw`` word (``kernel_mode``); a missing
backward kernel therefore blocks training calls only, and inference stays on Triton on a card whose forward kernels are recorded. The card is identified by its exact name and
capability, ``"NVIDIA RTX A6000 (sm86)"``: a card with the same capability and another name (A40, L40S) is not
recorded -- its shared memory, bandwidth and clocks differ, and nothing was run on it.

Cheap and import-safe: stdlib only (no torch, no GPU), the registry and each manifest are read once. Tests inject a
fake manifest directory with :func:`use_manifest_dir`.
"""

from __future__ import annotations

import contextlib
import csv
import fnmatch
import functools
from collections.abc import Iterator
from pathlib import Path

_PKG = Path(__file__).resolve().parent.parent
_REGISTRY = _PKG / "kernels" / "registry" / "registry.csv"
_MANIFESTS = _PKG / "autotune" / "manifests"
_PROVENANCE = "#provenance"
#: A manifest row without a ``dtype`` cell was written before the column existed: bf16 (see ``autotune.devices``).
_LEGACY_DTYPE = "bf16"

#: Op -> ``family/kernel`` patterns (``fnmatch``) over the registry's Triton rows: the kernels the op's ``miniworld`` route
#: launches on a card with no native CUDA path. The op names are the ``modules.dispatch`` ones. Family is the registry's
#: organisational column, not the launcher, so a family shared by two ops is split by kernel name (``gated_projection``:
#: the pair/token gate GEMMs belong to TriangleAttention, ``swa_gate_out_fwd`` to the SWA atom attention).
#: The token attention core's kernels (``augmented_attention/triton/main.py``), spelled out because the same family holds the
#: block-local window kernels (``*_window_triton``), which belong to the atom ops below and must not be held against the token ones.
_TOKEN_ATTENTION = (
    "augmented_attention/augmented_attention_fwd_triton",
    "augmented_attention/augmented_attention_bwd_pre_triton",
    "augmented_attention/augmented_attention_bwd_reduce_triton",
    "augmented_attention/augmented_attention_bwd_split_triton",
    "augmented_attention/augmented_attention_bwd_atomic_triton",
)
#: The block-local window attention (32 queries x 128 / 192 keys per window): the atom DiT block's core.
_WINDOW_ATTENTION = ("augmented_attention/augmented_attention_*_window_triton",)
#: The inference pair-bias producer of the portable DiT paths: LayerNorm of the pair and its projection to the heads in one pass.
_PAIR_BIAS = ("layernorm_linear/layernorm_linear_fwd_triton",)

OP_KERNELS: dict[str, tuple[str, ...]] = {
    # the residual path also launches the input LayerNorm's forward (layernorm family) and its backward with the residual gradient (trimul_inproj family)
    "transition": ("transition/*", "layernorm/layernorm_fwd_saveact_triton", "trimul_inproj/trimul_input_ln_residual_bwd_triton"),
    "triangle_multiplication": ("trimul_inproj/*",),
    "triangle_attention": (
        "triangle_attention/*",
        "layernorm_linear/*",
        "layernorm/*",
        "gated_projection/gated_projection_*",
        "bias_only_attention/*",
    ),
    "adaptive_layernorm": ("adaln/*",),
    "conditioned_transition": ("conditioned_transition/*", "adaln/*"),
    "augmented_attention": (*_TOKEN_ATTENTION, "adaln/*"),
    "outer_product_mean": ("outer_product_mean/*",),
    "msa_pair_weighted_averaging": ("pair_weighted_averaging/*",),
    "rmsnorm": ("rmsnorm/*",),
    # La-Proteina's pair builders and the AF3-style token-pair initialisation (kernels/_pair_route.py): the portable kernels of each; the decoder's CA-distance pair
    # shares the table-gradient and LayerNorm-backward kernels of the encoder's backbone pair
    "pair_embed_geom": ("pair_embed_geom/pair_embed_geom_*",),
    "pair_embed_ca": ("pair_embed_geom/pair_embed_ca_*", "pair_embed_geom/pair_embed_geom_fwd_*", "pair_embed_geom/pair_embed_geom_dw_*", "pair_embed_geom/pair_embed_geom_bwd_*"),
    "pair_embed_adaln": ("pair_embed_adaln/*",),
    "token_pair_init": ("token_pair_init/*",),
    "layernorm": ("layernorm/*",),
    # the smooth lDDT of the diffusion loss (kernels/smooth_lddt): prep + forward, and the backward in a training call
    "smooth_lddt": ("smooth_lddt/*",),
    # the ops.* facade entry points that launch a single Triton unit (kernels/*/whole_op.py)
    "gated_linear": ("gated_projection/gated_projection_*", "bias_only_attention/*"),
    "augmented_attention_pair_bias": _TOKEN_ATTENTION,
    "swiglu_ffn": ("transition/transition_expand_swiglu_triton",),
    "swa_atom_attention": ("rope/*", "rmsnorm/*", "gated_projection/swa_gate_out_fwd_triton"),
    "swa_dit": ("swa_dit/*", "rmsnorm/*"),
    # The attention cores of the portable DiT paths (the rest of those blocks is torch ops and the ops above, each with its own rule):
    # the AF3 / Protenix / OpenFold3 / ESMFold2 atom block (LocalDiTBlock), and the Boltz-2 / RoseTTAFold3 / Chai-1 token and atom stacks.
    "local_dit": (*_WINDOW_ATTENTION, *_PAIR_BIAS),
    "dit_variant": (*_TOKEN_ATTENTION, *_PAIR_BIAS),
    "dit_variant_atom": (*_WINDOW_ATTENTION, *_PAIR_BIAS),
}

#: The words of a kernel name that make it a training kernel: it saves activations for the backward (`saveact`), is the backward
#: (`bwd`, `dw`) or the fused training epilogue (`train`). Every other kernel of an op runs in the forward of an inference call too.
_TRAINING_WORDS = frozenset({"bwd", "saveact", "train", "dw"})

_manifest_dir: Path | None = None


def kernel_mode(kernel: str) -> str:
    """``"training"`` if ``kernel`` only runs where a backward is recorded, else ``"inference"`` (its forward serves both)."""
    return "training" if _TRAINING_WORDS & set(kernel.split("_")) else "inference"


def card_key(name: str, capability: tuple[int, int]) -> str:
    """``"NVIDIA RTX A6000 (sm86)"``: the manifest / cache key of a card (``autotune.cache.gpu_key`` from arguments)."""
    return f"{name} (sm{capability[0]}{capability[1]})".replace("/", "_")


def _dir() -> Path:
    return _manifest_dir if _manifest_dir is not None else _MANIFESTS


@contextlib.contextmanager
def use_manifest_dir(path: str | Path | None) -> Iterator[None]:
    """Read the manifests from ``path`` instead of the packaged ones (tests). ``None`` restores the packaged directory."""
    global _manifest_dir
    previous = _manifest_dir
    _manifest_dir = Path(path) if path is not None else None
    _load.cache_clear()
    try:
        yield
    finally:
        _manifest_dir = previous
        _load.cache_clear()


@functools.cache
def _registry() -> tuple[dict, ...]:
    with _REGISTRY.open(newline="") as handle:
        return tuple(csv.DictReader(handle))


@functools.cache
def _load(card: str) -> dict[tuple[str, str], str]:
    """``{(kernel, dtype): status}`` for ``card``; empty when the card has no manifest."""
    path = _dir() / f"{card}.csv"
    if not path.is_file():
        return {}
    with path.open(newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if r["kernel"] != _PROVENANCE]
    return {(r["kernel"], (r.get("dtype") or "").strip() or _LEGACY_DTYPE): r["status"] for r in rows}


def recorded_cards() -> tuple[str, ...]:
    """Every card with a manifest in the active directory."""
    return tuple(sorted(p.stem for p in _dir().glob("*.csv")))


def triton_kernels(op: str, dtype: str = "bf16", mode: str = "training") -> tuple[str, ...]:
    """The Triton kernels ``op``'s ``miniworld`` route launches in ``mode`` (developed rows that declare ``dtype``; ``"inference"``
    leaves out the training kernels, ``"training"`` is all of them)."""
    if mode not in ("inference", "training"):
        raise ValueError(f"mode must be 'inference' or 'training', not {mode!r}")
    patterns = OP_KERNELS.get(op)
    if patterns is None:
        return ()
    out = []
    for row in _registry():
        if row["backend"] != "triton" or row["developed"] != "yes":
            continue
        declared = {d.strip() for d in (row.get("dtypes") or "bf16").split("|") if d.strip()}
        if dtype not in declared:
            continue
        if mode == "inference" and kernel_mode(row["kernel"]) == "training":
            continue
        if any(fnmatch.fnmatchcase(f"{row['family']}/{row['kernel']}", p) for p in patterns):
            out.append(row["kernel"])
    return tuple(out)


def missing(op: str, name: str, capability: tuple[int, int], dtype: str = "bf16", mode: str = "training") -> tuple[str, ...]:
    """The kernels ``op`` launches in ``mode`` with no ``ok`` row for this exact card at ``dtype`` (empty means recorded). An op the
    table does not know has no kernels to vouch for: its only entry is a note naming the op."""
    kernels = triton_kernels(op, dtype, mode)
    if not kernels:
        return (f"<no portable kernel of {op!r} declares {dtype}>",)
    rows = _load(card_key(name, capability))
    return tuple(k for k in kernels if rows.get((k, dtype)) != "ok")


def recorded(op: str, name: str, capability: tuple[int, int], dtype: str = "bf16", mode: str = "training") -> bool:
    """Does the engine hold evidence that the Triton kernels ``op`` launches in ``mode`` run correctly on exactly this card at ``dtype``?"""
    return not missing(op, name, capability, dtype, mode)


def routing_table(dtype: str = "bf16", mode: str = "training") -> dict[str, dict[str, tuple[str, ...]]]:
    """``{card: {op: missing kernels}}`` over every recorded card -- the data behind ``docs/dispatch.md``."""
    return {
        c: {op: missing(op, c.rsplit(" (sm", 1)[0], _capability_of(c), dtype, mode) for op in OP_KERNELS}
        for c in recorded_cards()
    }


def _capability_of(card: str) -> tuple[int, int]:
    digits = card.rsplit("(sm", 1)[1].rstrip(")")
    return int(digits[:-1]), int(digits[-1])
