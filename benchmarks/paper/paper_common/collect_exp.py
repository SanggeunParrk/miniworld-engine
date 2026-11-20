"""Collect one experiment folder's raw records into <ID>_<card>.csv:  python paper_common/collect_exp.py KIND ID EXP_DIR CARD
KIND: speed (E1, E2, E5) | accuracy (E3) | memory (E6). E4 has its own collector (E4_fp8/collect.py)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paper_common import csvio


def main(kind: str, exp_id: str, exp_dir: str, card: str) -> None:
    d = Path(exp_dir)
    rows, extra = {
        "speed": (csvio.speed_rows, csvio.SPEED_EXTRA),
        "accuracy": (csvio.accuracy_rows, csvio.ACCURACY_EXTRA),
        "memory": (csvio.memory_rows, csvio.MEMORY_EXTRA),
    }[kind]
    csvio.write(csvio.csv_path(d, f"{exp_id}_{card}.csv"), rows(exp_id, d, card), extra)


if __name__ == "__main__":
    main(*sys.argv[1:5])
