#!/bin/bash
python bench.py --op opm --length 384 768 --out records/opm_t2.json 2>&1 | grep -E "^opm|^    |Error|error" | head -4
echo "=== 128-pair epilogue (same run)"; python bench.py --op opm --length 384 768 -D OPM_E_BIG=0 2>&1 | grep -E "^opm|^    |Error" | head -4
python bench_train.py --op opm --length 384 --no-time 2>&1 | grep -E "^opm|Error" | head -2
