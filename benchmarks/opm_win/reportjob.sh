#!/bin/bash
# merge the per-row fair-protocol records (v3) and write the table + graphs
D=/home/psk6950/miniworld-engine-scratch/wt-opm-win/benchmarks/opm_win
cat $D/raw/v3_kit_r*.jsonl > $D/raw/v3_kit.jsonl
cat $D/raw/v3_before_r*.jsonl > $D/raw/v3_before.jsonl
cat $D/raw/v3_after_r*.jsonl > $D/raw/v3_after.jsonl
python $D/report.py $D/raw/v3_kit.jsonl $D/raw/v3_before.jsonl $D/raw/v3_after.jsonl $D/results
