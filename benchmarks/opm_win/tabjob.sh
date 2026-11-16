#!/bin/bash
D=/home/psk6950/miniworld-engine-scratch/wt-opm-win/benchmarks/opm_win
python $D/tab.py $D/raw/base_kit.jsonl $D/raw/base_engine.jsonl:before $D/raw/after2_engine.jsonl:after > $D/raw/table.md
