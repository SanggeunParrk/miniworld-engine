"""`docs/kernels/naming.md` declares a CLOSED vocabulary. Nothing enforced it.

The document fixes `<func>_<role>[_<detail>]_<backend>`, lists every legal token for each slot,
and even records the tokens it threw out and why. `test_registry_complete` validates `kind`,
`level` and `dtypes` against their vocabularies -- and the kernel NAME, which is what the whole
document is about, was checked by one assertion that the string "registry.csv" appears somewhere
in it. By this repository's own rule, that made the vocabulary decoration.

Two of 103 names were outside it when this test was written:

  * `transition_cast_cuda` -- `cast` was a real computation stage (a dtype conversion, and nothing
    else) with no word in the role list. The name was right and the list was short; `cast` is now
    declared.
  * `layernorm_bwd_privatized_triton` -- `privatized` is in the document's DISCARDED list, and
    carried by a kernel anyway. Its referent was `PRIVATIZE_DGDB`, a constexpr in the autotune
    key, which the same rule that excludes SAVE_GATE/SAVE_PREACT says earns no token. What that
    kernel actually contracts for is the prefolded `c1 = mean*rstd` it reads, so it is
    `layernorm_bwd_foldstats_triton` now -- the token its sibling
    `layernorm_fwd_recompute_foldstats_triton` already uses.

Drift direction this catches: a name using a token the document does not declare. The reverse --
a token added to the document and never used -- is not an error, so it is not checked.
"""
from __future__ import annotations

import csv

import pytest
from paths import REGISTRY as REG
from paths import ROOT

SPEC = ROOT / "docs/kernels/naming.md"

#: Declared here rather than parsed out of the prose, and pinned to the prose by
#: `test_every_token_is_in_the_document` below -- so a token deleted from the document fails, and
#: a name using a token missing from here fails. Longest first: the tokenizer is greedy.
FUNCS = ("layernorm_linear", "cond_transition", "trimul_outproj", "triangle_attention",
         "augmented_attention", "bias_only_attention", "gated_projection", "layernorm",
         # before "rmsnorm": the longer, more specific prefix, as layernorm_linear is
         "rmsnorm_adamod", "rmsnorm",
         "transition", "trimul", "adaln")
ROLES = ("bwd_reduce", "bwd_pre", "transpose", "epilogue", "layernorm", "sigmoid", "squeeze",
         "swiglu", "expand", "stats", "dbias", "dkdv", "dlnw", "gemm", "gate", "fold", "cast",
         "fwd", "bwd", "dx", "dw", "dq", "dk", "dv")
DETAILS = ("recompute", "foldstats", "noaffine", "rowscale", "dropres", "inplace", "ktiled",
           "strided", "mmajor", "extern", "packed", "atomic", "contig", "split", "flat", "fp32",
           "sm100", "sm90", "b2b", "saveact")
BACKENDS = ("triton", "cutlass", "cute", "cuda")

_PIECES = tuple(sorted(set(ROLES) | set(DETAILS), key=len, reverse=True))


def _names() -> list[str]:
    with REG.open() as fh:
        return [r["kernel"] for r in csv.DictReader(fh)]


def _fault(name: str) -> str | None:
    """Why `name` is outside the declared grammar, or None."""
    backend = next((b for b in BACKENDS if name.endswith("_" + b)), None)
    if backend is None:
        return f"ends in no declared <backend> (one of {', '.join(BACKENDS)})"
    middle = name[: -(len(backend) + 1)]
    func = next((f for f in FUNCS if middle == f or middle.startswith(f + "_")), None)
    if func is None:
        return "starts with no declared <func>"
    rest = middle[len(func):].lstrip("_")
    if not rest:
        return "has no <role>"
    pieces, tail = [], rest
    while tail:
        piece = next((t for t in _PIECES if tail == t or tail.startswith(t + "_")), None)
        if piece is None:
            return f"undeclared token at {tail!r}"
        pieces.append(piece)
        tail = tail[len(piece):].lstrip("_")
    if pieces[0] not in ROLES:
        return f"{pieces[0]!r} is a <detail>, and the slot after <func> is <role>"
    return None


def test_there_are_names_to_check() -> None:
    """Guard the guard. The floor is well under the registry's size (86 rows) on purpose: it is
    here to catch a parse that returned nothing, not to pin a count. It was 90, which turned the
    removal of five superseded kernels into a failure of the naming rules."""
    assert len(_names()) >= 50, f"only {len(_names())} rows in registry.csv"


def test_every_name_parses_into_declared_tokens() -> None:
    bad = {n: why for n in _names() if (why := _fault(n))}
    assert not bad, ("names outside docs/kernels/naming.md's vocabulary:\n  "
                     + "\n  ".join(f"{n}: {why}" for n, why in sorted(bad.items())))


@pytest.mark.parametrize("token", sorted(set(FUNCS) | set(ROLES) | set(DETAILS) | set(BACKENDS)))
def test_every_token_is_in_the_document(token: str) -> None:
    """Keeps this file and the prose from drifting apart in the direction that matters: a token
    enforced here but no longer documented is a rule nobody adding a kernel can find."""
    assert token in SPEC.read_text(), f"{token!r} is enforced here but absent from {SPEC.name}"


def test_the_grammar_rejects_something() -> None:
    """Not vacuous: the checker has to fail on names the document forbids."""
    for name, reason in (("transition_fwd", "no backend"),
                         ("nosuchfunc_fwd_triton", "no func"),
                         ("transition_triton", "no role"),
                         ("transition_wobble_triton", "undeclared token"),
                         ("transition_atomic_triton", "detail in the role slot")):
        assert _fault(name), f"{name} should have been rejected ({reason})"
