"""Read the per-unit `.smem` logs a build leaves behind, and say what they mean for the next one.

A build writes one line per compile -- `<kernel>\\t<config sig>\\t<bytes>` -- from inside the
compile child (see `capture._compile_payload`). The point of keeping them is that a config which
exceeds the card's shared-memory limit is knowable BEFORE it is compiled, and compiling is 72% of
a unit's wall time while 40-74% of the configs of the larger grids are exactly that.

A config the compile budget KILLED writes `!<kernel>\\t<config sig>\\t<budget>` instead. It is
a different mechanism -- ptxas grinding on register spill, not shared memory -- and it arrives at
the bench as the same undifferentiated +inf, so it needs its own evidence to be predictable at
all. It is worth having: measured on the A6000 rebuild, 13,875 of 869,844 configs were killed and
those 1.6% took 54% of the whole build's compile CPU. Concentrated, not spread: 205 of 461 rounds
kill nothing at all, and one op at three lengths accounts for 27.5 of the 231 CPU-hours.
"""
from __future__ import annotations

from pathlib import Path


def _rows(shard_dir: Path):
    """(killed, kernel, sig, value) for every well-formed line under `shard_dir`.

    `.smem.gz` is read too: a build's logs run to megabytes and the copies kept as test fixtures
    compress about eighteen-fold, which is the difference between a fixture and a liability.
    """
    for f in sorted([*shard_dir.glob("*.smem"), *shard_dir.glob("*.smem.gz")]):
        if f.suffix == ".gz":
            import gzip
            text = gzip.decompress(f.read_bytes()).decode(errors="ignore")
        else:
            text = f.read_text(errors="ignore")
        for raw in text.splitlines():
            parts = raw.split("\t")
            if len(parts) != 3:
                continue
            kernel, sig, val = parts
            try:
                number = int(val)
            except ValueError:
                continue
            yield kernel.startswith("!"), kernel.removeprefix("!"), sig, number


def read(shard_dir: Path) -> dict[str, dict[str, int]]:
    """kernel -> {config sig: shared bytes}, merged over every unit in a shard directory.

    Killed configs are NOT in here. They have no shared-memory reading -- they never got far
    enough to have one -- and folding their budget in as if it were a byte count would put a
    60 next to real values in the hundreds of thousands.
    """
    out: dict[str, dict[str, int]] = {}
    for killed, kernel, sig, number in _rows(shard_dir):
        if not killed:
            out.setdefault(kernel, {})[sig] = number
    return out


def killed(shard_dir: Path) -> dict[str, set[str]]:
    """kernel -> config sigs the compile budget killed."""
    out: dict[str, set[str]] = {}
    for is_killed, kernel, sig, _ in _rows(shard_dir):
        if is_killed:
            out.setdefault(kernel, set()).add(sig)
    return out


def over_limit(shard_dir: Path, limit: int) -> dict[str, tuple[int, int]]:
    """kernel -> (configs measured, configs that cannot launch on a card with `limit` bytes).

    The second number is what a predictor would have to reproduce to be worth having, and the
    first is how much evidence there is for that kernel. Reported per kernel because the answer
    is per kernel: one op's grid was 69% unusable while another's was 0%.
    """
    return {k: (len(v), sum(1 for b in v.values() if b > limit)) for k, v in read(shard_dir).items()}
