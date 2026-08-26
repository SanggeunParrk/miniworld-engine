"""Read the per-unit `.smem` logs a build leaves behind, and say what they mean for the next one.

A build writes one line per compile -- `<kernel>\\t<config sig>\\t<bytes>` -- from inside the
compile child (see `capture._compile_payload`). The point of keeping them is that a config which
exceeds the card's shared-memory limit is knowable BEFORE it is compiled, and compiling is 77% of
a unit's wall time while 40-74% of the configs of the larger grids are exactly that. What the logs
cannot say is why a config was slow or why ptxas took ten minutes on it; those are different
mechanisms and they need their own evidence.
"""
from __future__ import annotations

from pathlib import Path


def read(shard_dir: Path) -> dict[str, dict[str, int]]:
    """kernel -> {config sig: shared bytes}, merged over every unit in a shard directory."""
    out: dict[str, dict[str, int]] = {}
    for f in sorted(shard_dir.glob("*.smem")):
        for raw in f.read_text(errors="ignore").splitlines():
            parts = raw.split("\t")
            if len(parts) != 3:
                continue
            kernel, sig, val = parts
            try:
                out.setdefault(kernel, {})[sig] = int(val)
            except ValueError:
                continue
    return out


def over_limit(shard_dir: Path, limit: int) -> dict[str, tuple[int, int]]:
    """kernel -> (configs measured, configs that cannot launch on a card with `limit` bytes).

    The second number is what a predictor would have to reproduce to be worth having, and the
    first is how much evidence there is for that kernel. Reported per kernel because the answer
    is per kernel: one op's grid was 69% unusable while another's was 0%.
    """
    return {k: (len(v), sum(1 for b in v.values() if b > limit)) for k, v in read(shard_dir).items()}
