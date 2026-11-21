#!/usr/bin/env bash
# CPU only (run via srun, never on the login node): regenerate paper_common/out/<card>/ and PAPER_DATA_SUMMARY.md
# for every card whose E1 CSV exists (h100, a100, rtx5090, b200).
set -euo pipefail
cd "$(dirname "$0")/.."
for c in h100 a100 rtx5090 b200; do
  if [ -f E1_operator_speed/E1_$c.csv ]; then python paper_common/tables.py --card $c --out paper_common/out/$c; fi
done
python paper_common/summarize.py
