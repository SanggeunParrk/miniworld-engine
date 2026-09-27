#!/bin/bash
cd /home/psk6950/practice/miniworld-engine
export PYTHONPATH=$PWD/src
echo "== run_all aligned"
python -m miniworld_engine.autotune.run_all --only outer_product_mean,pair_weighted_averaging --verbose
echo "== run_all ragged"
MINIWORLD_SHAPE_MODE=ragged python -m miniworld_engine.autotune.run_all --only outer_product_mean,pair_weighted_averaging --verbose
echo "== pytest"
python -m pytest tests/numerics/test_msa_triton_gpu.py tests/numerics/test_msa_triton_dispatch.py -q -p no:cacheprovider -x 2>&1 | tail -40
