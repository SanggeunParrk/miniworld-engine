"""Recover ``entry_grids`` for caches built before the field existed, from git history.

An incremental build subtracts the space a cache has ALREADY searched for a shape from the grid it
is about to sweep (``cache.configs_to_bench``). That subtraction is only as good as the record it
reads, and every cache committed before ``entry_grids`` existed carries no record at all -- 149 of
the 244 shipped caches do not even carry the file-level ``config_space``. For those, the honest
answer to "what was searched here?" is *unknown*, and unknown means measure it all: narrowing a
config ladder -- which this repository's own tests ask for -- then costs a full re-tune of kernels
whose source never changed. That is the 25-GPU-hour bill the incremental policy was written to
avoid, and it kept arriving because the policy had nothing to read.

The provenance is not lost. Every cache file records the ``config_space_hash`` of the grid its
build searched, and every grid this repository has ever shipped is a CSV in git. Hashing each
historical version of ``configs/grid/<op>.csv`` gives a hash -> grid index, and a cache's stored
hash then names its grid EXACTLY. Nothing is guessed: a file whose hash matches no version of its
CSV is skipped and keeps measuring everything, which is what it does today.

Two things this deliberately does not do:

* Stamp the CURRENT grid on a cache whose hash says otherwise. That would assert a build searched
  configs it never saw, and the failure mode is silent and permanent -- a config that is never
  measured because a file claims it already was.
* Assume one grid per FILE. ``store_ranked_configs`` no longer resets a file when the grid moves,
  so a file can hold entries from several builds and several grids. Each entry key is therefore
  resolved against the hash the file carried in the commit where THAT KEY first appeared, walking
  the file's own history oldest-first. A key that appeared under a smaller grid keeps the smaller
  space, and the next build measures the difference -- which is the whole point.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from miniworld_engine.autotune.cache import _sig, _sig_from_dict, config_space_hash

_REPO = Path(__file__).resolve().parents[3]
_DATA = Path(__file__).resolve().parent / "data"
_GRID_REL = "src/miniworld_engine/autotune/configs/grid"


@dataclass(frozen=True)
class Recovered:
    op: str
    gpu: str
    keys: int          # entries whose searched space was recovered
    unresolved: int    # entries whose grid hash matched no version of the CSV
    configs: int       # size of the recovered space, for the newest resolved key


def _git(*args: str) -> str | None:
    try:
        r = subprocess.run(("git", *args), cwd=_REPO, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout if r.returncode == 0 else None


def _parse_csv(text: str) -> list | None:
    """Historical CSV text -> config list, using TODAY's parser.

    Using today's parser is the right call and also self-checking: if the format has moved far
    enough that an old file parses differently, the grid it yields hashes to something the cache
    never recorded and the file is skipped rather than mis-stamped.
    """
    from miniworld_engine.autotune.configs import _read

    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as fh:
        fh.write(text)
        tmp = Path(fh.name)
    try:
        return _read(tmp)
    except Exception:
        return None
    finally:
        tmp.unlink(missing_ok=True)


def _grid_index(op: str) -> dict[str, list[str]]:
    """``config_space_hash`` -> the sorted signature reprs of that grid, over the CSV's history."""
    rel = f"{_GRID_REL}/{op}.csv"
    index: dict[str, list[str]] = {}
    revs = (_git("log", "--format=%H", "--", rel) or "").split()
    blobs = [_git("show", f"{rev}:{rel}") for rev in revs]
    live = _REPO / rel
    if live.is_file():
        blobs.append(live.read_text())
    for text in blobs:
        if not text:
            continue
        grid = _parse_csv(text)
        if not grid:
            continue
        index.setdefault(config_space_hash(grid),
                         sorted(repr(x) for x in {_sig(c) for c in grid}))
    return index


def _first_seen(rel_cache: str) -> dict[str, str]:
    """Entry key -> the ``config_space_hash`` the file carried when that key first appeared.

    Oldest commit first, so a key keeps the grid of the build that MEASURED it rather than the grid
    of whatever build happened to touch the file last.
    """
    seen: dict[str, str] = {}
    revs = (_git("log", "--format=%H", "--", rel_cache) or "").split()
    for rev in reversed(revs):                       # oldest -> newest
        blob = _git("show", f"{rev}:{rel_cache}")
        if not blob:
            continue
        try:
            data = json.loads(blob)
        except json.JSONDecodeError:
            continue
        h = data.get("config_space_hash")
        if not h:
            continue
        for key in (data.get("entries") or {}):
            seen.setdefault(key, h)
    return seen


#: Why caches were passed over on the last :func:`backfill`. Reported, never silent.
LAST_SKIPPED: Counter[str] = Counter()


def _contains_its_own_winners(space: list[str], ranked: list) -> bool:
    """Could this entry have been measured with this space? Its own winners answer.

    A cache entry stores the configs that WON for that shape. If one of them is not in the space
    the entry is stamped with, the entry cannot have been measured with that space -- the file is
    contradicting itself, and the stamp is wrong however plausible its provenance looked.

    This is the check that was missing. `_first_seen` pins a key to the `config_space_hash` of the
    commit the key first appeared in, and `store_ranked_configs` used to RESET a file when the grid
    moved and re-measure the same keys against the new one. A reset does not rename a key, so
    `setdefault` never saw the re-measurement and stamped the pre-reset grid: 225 entries across 11
    A100 caches, 140 of them not yet invalidated by a `build_rev` bump, together claiming 46,560
    configs as searched that nothing had ever measured.
    """
    if not ranked:
        return True
    have = set(space)
    return all(repr(_sig_from_dict(c)) in have for c in ranked)


def repair(*, apply: bool = False) -> list[tuple[str, str, int]]:
    """Drop stamps an entry's own winners contradict. Returns (op, gpu, entries dropped).

    Separate from :func:`backfill` because it fixes what backfill already wrote: `backfill` skips a
    file whose keys are all stamped, so a bad stamp is invisible to it forever. Dropping a stamp
    costs a re-measurement; keeping a wrong one costs a winner nobody timed.
    """
    out: list[tuple[str, str, int]] = []
    for jf in sorted(_DATA.glob("*/*.json")):
        try:
            data = json.loads(jf.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        refs = data.get("entry_grids")
        grids = data.get("grids") or {}
        if not isinstance(refs, dict) or not refs:
            continue
        entries = data.get("entries") or {}
        dropped = 0
        for key in list(refs):
            hashes = refs[key]
            hashes = [hashes] if isinstance(hashes, str) else list(hashes or [])
            space: list[str] = []
            for h in hashes:
                space.extend(grids.get(h) or [])
            if space and not _contains_its_own_winners(space, entries.get(key) or []):
                del refs[key]
                dropped += 1
        if not dropped:
            continue
        out.append((jf.parent.name, jf.stem, dropped))
        if apply:
            wanted = {h for hs in refs.values() for h in ([hs] if isinstance(hs, str) else hs)}
            data["grids"] = {h: v for h, v in grids.items() if h in wanted}
            if not refs:
                data.pop("entry_grids", None)
                data.pop("grids", None)
            jf.write_text(json.dumps(data, indent=2, sort_keys=True))
    return out


def backfill(*, apply: bool = False) -> list[Recovered]:
    """Give every cache an ``entry_space`` its own git history can prove. Reports unless ``apply``."""
    out: list[Recovered] = []
    skipped: Counter[str] = Counter()
    grids: dict[str, dict[str, list[str]]] = {}
    for jf in sorted(_DATA.glob("*/*.json")):
        op = jf.parent.name
        try:
            data = json.loads(jf.read_text())
        except (OSError, json.JSONDecodeError):
            skipped["unreadable JSON"] += 1
            continue
        entries = data.get("entries") or {}
        if not entries:
            skipped["no entries to describe"] += 1
            continue
        have = data.get("entry_grids")
        if isinstance(have, dict) and all(k in have for k in entries):
            skipped["already recorded"] += 1
            continue
        if op not in grids:
            grids[op] = _grid_index(op)
        index = grids[op]
        if not index:
            skipped["no config CSV in history (dispatch-only or renamed op)"] += 1
            continue
        rel_cache = str(jf.relative_to(_REPO))
        first = _first_seen(rel_cache)
        current_hash = data.get("config_space_hash")
        refs: dict[str, list[str]] = {k: list(v) for k, v in have.items()} if isinstance(have, dict) else {}
        grids_out: dict[str, list[str]] = dict(data.get("grids") or {})
        got = unresolved = 0
        size = 0
        for key in entries:
            if key in refs:
                continue
            # A key present in the working tree but in no commit was written since the last commit;
            # the file's CURRENT hash is the one that build recorded.
            h = first.get(key, current_hash)
            space = index.get(h) if h else None
            if space is None:
                unresolved += 1
                continue
            if not _contains_its_own_winners(space, entries.get(key) or []):
                # The recovered grid does not contain this entry's own winners, so the entry was
                # not measured with it -- most often because the file was RESET and the key
                # re-measured under a later grid, which `_first_seen` cannot see. Leave it
                # unresolved: a re-measurement is a cost, a wrong stamp is a wrong answer.
                unresolved += 1
                continue
            refs[key] = [h]
            grids_out[h] = space
            got += 1
            size = len(space)
        if not got:
            skipped["grid hash matches no version of the op's CSV"] += 1
            continue
        out.append(Recovered(op, jf.stem, got, unresolved, size))
        if apply:
            data["entry_grids"] = refs
            data["grids"] = grids_out
            # No trailing newline: match `_atomic.write_json`, the canonical writer, or every
            # rebuild re-diffs the file purely to strip one.
            jf.write_text(json.dumps(data, indent=2, sort_keys=True))
    LAST_SKIPPED.clear()
    LAST_SKIPPED.update(skipped)
    return out


def format_report(rows: list[Recovered], *, applied: bool) -> str:
    verb = "recorded" if applied else "would record"
    keys = sum(r.keys for r in rows)
    unres = sum(r.unresolved for r in rows)
    lines = [f"entry_grids backfill: {verb} the searched config space for {keys} cache entrie(s) "
             f"across {len(rows)} file(s), recovered from git history"]
    if unres:
        lines.append(f"  {unres} entrie(s) left unresolved -- their grid hash matches no version "
                     f"of the op's CSV, so they keep re-measuring the full grid")
    by_op: dict[str, int] = {}
    for r in rows:
        by_op[r.op] = by_op.get(r.op, 0) + r.keys
    if by_op:
        lines.append("")
        for op, n in sorted(by_op.items(), key=lambda kv: -kv[1])[:20]:
            lines.append(f"  {op:52} {n:4d} entrie(s)")
        if len(by_op) > 20:
            lines.append(f"  ... and {len(by_op) - 20} more op(s)")
    if LAST_SKIPPED:
        lines.append(f"\nSKIPPED ({sum(LAST_SKIPPED.values())}) -- these keep measuring everything:")
        for why, n in sorted(LAST_SKIPPED.items(), key=lambda kv: -kv[1]):
            lines.append(f"  {n:4d}  {why}")
    return "\n".join(lines)
