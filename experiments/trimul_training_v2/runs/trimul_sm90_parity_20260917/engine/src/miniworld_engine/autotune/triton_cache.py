"""Keep the build's triton cache from filling the filesystem, and clear it when the build is done.

A build compiles every config of every grid and triton keeps each result on disk. Measured on the
A6000 rebuild: 221,487 entries, 187 KB each, **40 GB** -- on a filesystem shared with the rest of
the lab. Sampled over 400 entries, that is:

    source   21.3%      cubin   15.7%
    llir     18.5%      json     1.2%
    ptx      17.0%
    ttgir    14.3%
    ttir     11.9%

The IR levels are 62% of it and nothing launches a kernel from them -- a cache HIT reads the
metadata json and the cubin. `TRITON_STORE_BINARY_ONLY` makes triton write only those two (plus
the source, which it puts outside that guard), which is 71 KB an entry instead of 187: **15 GB
instead of 40**.

Verified before making it the default: the knob is NOT one of triton's cache-invalidating
environment variables, so turning it on does not orphan a cache built without it -- the same
config compiles to the same hash either way, and the warm-hit path returns a kernel with its
metadata and launcher intact.

The cache is a BUILD ARTIFACT. What a build ships is the JSON under `autotune/data/`, which names
configs; nothing reads the triton cache afterwards. It has to survive the whole build, because
units share it -- a compiled binary is reused across input SIZES (measured: the first length adds
every entry, later lengths add zero, because triton's key carries the constexprs and the argument
specialisation, not the argument values). Once the shards are merged it is disposable in full.

Not done per op, deliberately: two ops can drive the SAME kernel -- `trimul_gemm_gate_triton` and
`trimul_bwd_gate_recompute_triton` both compile `fused_sigmoid_gate_fwd_kernel` -- so entries an
op looks finished with are still wanted by another op's units.
"""
from __future__ import annotations

import contextlib
import os
import shutil
from pathlib import Path

#: Triton names each entry with a base32 digest of a sha256 -- 52 characters in practice. The floor
#: is well under that so a future digest change does not make this refuse a real cache, and well
#: over any ordinary directory name so it does not accept something else.
_ENTRY_NAME_LEN = 40


def store_binary_only_env(env: dict[str, str], keep_ir: bool = False) -> None:
    """Set ``TRITON_STORE_BINARY_ONLY`` on a child's environment unless IR is wanted.

    Set on the CHILD rather than the parent because the parent does not compile, and because a
    build that wants the IR for one unit should not have to unset a global.
    """
    if keep_ir:
        env.pop("TRITON_STORE_BINARY_ONLY", None)
        return
    env.setdefault("TRITON_STORE_BINARY_ONLY", "1")


def looks_like_a_triton_cache(directory: Path) -> bool:
    """Does this directory hold triton cache entries, and NOTHING that says otherwise?

    Two conditions, both required, because the one mistake this module must not make is emptying a
    directory that is not a triton cache:

      * every child is either an entry directory -- a base32 digest, 52 characters in practice --
        or one of the loose files triton drops beside them (its launcher `.so`, a lock). A single
        foreign file is enough to refuse.
      * one of those entry directories really holds a metadata json. A directory of long-named
        empty directories is not a cache.

    Returns False for an empty directory: nothing to gain by emptying one, everything to lose by
    being wrong about which one it is.
    """
    if not directory.is_dir():
        return False
    entry = None
    try:
        with os.scandir(directory) as it:
            for item in it:
                # One pass, no recursion. A 221,487-entry directory on a shared filesystem is not
                # somewhere to go looking twice.
                if item.is_dir():
                    if len(item.name) < _ENTRY_NAME_LEN:
                        return False
                    entry = entry or item.path
                elif not (item.name.endswith(".so") or item.name.endswith(".lock")):
                    return False
    except OSError:
        return False
    if entry is None:
        return False
    try:
        with os.scandir(entry) as it:
            return any(f.name.endswith(".json") for f in it)
    except OSError:
        return False


def clear(directory: Path, dry_run: bool = False) -> tuple[int, int]:
    """Remove the CONTENTS of a triton cache directory. Returns (entries, bytes).

    The directory itself is kept, so a build that is still pointed at it keeps working. Refuses a
    directory that does not look like a triton cache -- see `looks_like_a_triton_cache`.

    Sizes are measured on the way through, not in a separate pass. `du` over 221,487 entries is
    the metadata storm this is supposed to prevent.
    """
    if not looks_like_a_triton_cache(directory):
        raise ValueError(f"{directory} does not look like a triton cache; refusing to empty it")
    entries = total = 0
    with os.scandir(directory) as it:
        for item in it:
            with contextlib.suppress(OSError):
                if not item.is_dir():
                    continue
                total += _bytes_under(item.path)
                if not dry_run:
                    shutil.rmtree(item.path, ignore_errors=True)
                entries += 1
    return entries, total


def _bytes_under(path: str) -> int:
    size = 0
    for root, _, files in os.walk(path):
        for name in files:
            with contextlib.suppress(OSError):
                size += os.path.getsize(os.path.join(root, name))
    return size
