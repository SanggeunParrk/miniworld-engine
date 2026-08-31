"""A kernel's source has to call the shape-key function its registry `level` names.

`shape_key.py` states the contract in the docstrings: `token_key` is "for a token/pair-level
kernel (`level=token` in registry.csv)", `atom_key` for `level=atom`, and `both_key` is "for a
kernel used at both levels (`level=both`), from its ROW COUNT". Nothing enforced it, and
`fused_ln_mask` drifted: declared `level=token`, it keyed through `both_key(rows_of(x.shape))`.

The two functions bucket different quantities. A pair activation of side L has L*L rows, so the
cache recorded 16384/65536/147456/262144 where every other level=token kernel records
128/256/384/512. No launch was served the wrong config -- the wrapper is the only caller and it
keyed reads the same way it keyed writes -- but `dev audit` compares the cache against the
DECLARED bucket, which for level=token is L, so it reported all four buckets missing on a cache
that was complete. Four of the 795 declared pairs, and the only thing making `dev audit` exit 1
after a clean 1,889-unit build.

The assertion is one-directional on purpose: a file must USE its own level's key function, not
avoid the others. Three files legitimately call `both_key` as well, for a kernel that belongs to
another family and is declared `level=both` there -- `bias_only_attention/triton/gate_out.py` for
gated_projection's `_sigmul_fwd`, and `adaln`/`conditioned_transition` training for the borrowed
`layernorm_linear` helpers. Each says so at the call site. Forbidding the extra call would fail
all three and teach the next reader to delete a comment rather than fix a key.
"""
from __future__ import annotations

import ast
import collections
from pathlib import Path

from paths import ROOT, registry_rows

KEY_FNS = {"token_key", "atom_key", "both_key"}
#: registry `level` -> the shape_key function that level's docstring claims.
EXPECTED = {"token": "token_key", "atom": "atom_key", "both": "both_key"}


def _key_calls(path: Path) -> set[str]:
    """Shape-key functions actually CALLED in this file.

    ast, not a regex: the comment on the line this test was written for names `both_key`, and a
    regex reads that as a call.
    """
    tree = ast.parse(path.read_text())
    return {n.func.id for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in KEY_FNS}


def _single_level_files() -> dict[str, str]:
    """Source file -> its one `level`, for triton rows that declare a driver.

    Files whose rows disagree about `level` are skipped rather than guessed at; today there are
    none, and one appearing is a question for a person, not a failure.
    """
    by_file: dict[str, set] = collections.defaultdict(set)
    for r in registry_rows():
        if r["backend"] != "triton" or not (r.get("driver") or "").strip():
            continue
        if f := (r.get("file") or "").strip():
            by_file[f].add(r["level"])
    return {f: next(iter(lv)) for f, lv in by_file.items() if len(lv) == 1}


def test_a_kernel_keys_at_the_level_it_declares() -> None:
    bad = []
    for rel, level in sorted(_single_level_files().items()):
        src = ROOT / "src" / rel
        if not src.is_file():
            continue
        used = _key_calls(src)
        if not used:
            continue  # the key is computed elsewhere; this test has nothing to check here
        want = EXPECTED[level]
        if want not in used:
            bad.append(f"{rel}: level={level} so it must key through {want}(), "
                       f"but the file only calls {sorted(used)}")
    assert not bad, "\n  ".join(["a kernel keys at a level it does not declare:", *bad])
