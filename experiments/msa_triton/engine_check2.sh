#!/bin/bash
cd /home/psk6950/practice/miniworld-engine
export PYTHONPATH=$PWD/src
python -m miniworld_engine.autotune.run_all --only pair_weighted_averaging 2>&1 | grep -E "\[ok|\[FAIL"
MINIWORLD_SHAPE_MODE=ragged python -m miniworld_engine.autotune.run_all --only pair_weighted_averaging 2>&1 | grep -E "\[ok|\[FAIL"
python -m pytest tests/numerics/test_msa_triton_gpu.py -q -p no:cacheprovider -k dropout 2>&1 | grep -E "^E |passed|failed" | head -30
