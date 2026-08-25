# Reproducing a report from another machine

A consumer says "the build only found 527 units" or "it hung for ten hours". Attributing that
took a whole session once, because the failing environment was a different cluster and a fresh
clone and neither was reproducible here. Most of what makes a report hard to reproduce is not the
hardware — it is the state a working checkout accumulates. Strip that first; reach for a GPU last.

## What differs between a working checkout and a fresh one

| | working checkout | fresh clone |
|---|---|---|
| commit | whatever is checked out, plus uncommitted work | the pushed tip |
| autotune cache | shipped, plus whatever a local build merged | shipped only |
| JIT extensions | `~/.cache/torch_extensions`, possibly with a stale lock | empty |
| triton cache | `~/.triton`, warm | empty |
| config set | possibly pinned by `MINIWORLD_CONFIG_DIR` | the packaged default |
| environment | `.pixi/` as it has drifted | resolved from `pixi.lock` |

A report is about one of those far more often than about the card.

## The recipe

**1. Reproduce the commit, not the branch.** A detached worktree costs nothing and leaves your
checkout alone:

```bash
git worktree add --detach /tmp/repro <commit-or-tag>
```

**2. Isolate every cache.** These are the four that carry state between runs:

```bash
export TORCH_EXTENSIONS_DIR=/tmp/repro-cache/torch
export TRITON_CACHE_DIR=/tmp/repro-cache/triton
unset MINIWORLD_CONFIG_DIR
rm -rf /tmp/repro-cache
```

**3. Ask the cheap question first.** Most symptoms are visible without a GPU. The unit count, the
config set, the registry, and coverage are all CPU-only:

```bash
cd /tmp/repro && PYTHONPATH=src python -c "
from miniworld_engine.autotune.builder import op_units
print(len(op_units()), 'units')"
```

**4. Only then take a card.** With the caches isolated above, so what you measure is the fresh
path and not a warm one.

```bash
srun -p gpu --gres=gpu:1 --pty bash
python -m miniworld_engine.autotune.run_all
```

**5. Clean up.** A detached worktree left behind becomes the next person's confusion:

```bash
git worktree remove --force /tmp/repro
```

## Worked example: the 527-unit report

The report was that `build all` enumerated 527 units where it should have found 859. Steps 1 and 3
settle it in seconds, no GPU:

```
0854ac4^  (e1313dc)   527 units
main                  859 units
```

`0854ac4` is *fix(build): the sweep never drove fp32, and coverage could not see it missing*. The
consumer's clone predated it, so half of every fp32 kernel's work was never enumerated — and
coverage reported `missing 0`, because it counted against what the build enumerated rather than
against the registry. The answer was `git pull`, and no card was needed to find that.

`tests/builder/test_build_matrix.py` now pins the current count, so a regression to 527 fails in
CI rather than in someone else's ten-hour job.
