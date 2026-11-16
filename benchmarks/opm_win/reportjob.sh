#!/bin/bash
# merge the per-row fair-protocol records (v3) and write the table + graphs
D=/home/psk6950/miniworld-engine-scratch/wt-opm-win/benchmarks/opm_win
cat $D/raw/v3_kit_r*.jsonl > $D/raw/v3_kit.jsonl
cat $D/raw/v3_before_r*.jsonl > $D/raw/v3_before.jsonl
cat $D/raw/v3_after_r*.jsonl $D/raw/v3_after_lt_r*.jsonl $D/raw/v3_after_small_r*.jsonl > $D/raw/v3_after.jsonl   # later files win: the lt re-runs (N 256 / 384), the S <= 2048 re-runs (fused kernel restored)
python $D/report.py $D/raw/v3_kit.jsonl $D/raw/v3_before.jsonl $D/raw/v3_after.jsonl $D/results
