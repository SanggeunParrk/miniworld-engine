#!/bin/bash
cd /home/psk6950/practice/miniworld-engine
export PYTHONPATH=$PWD/src
OUT=experiments/msa_triton/records/derive_check
mkdir -p $OUT
MINIWORLD_COMPILE_WRAP=disable python -m miniworld_engine.cli dev derive --out $OUT/registry_kernel.csv 2>&1 | grep -vE "launches$" | tail -30
grep -cE "^(outer_product_mean|pair_weighted_averaging)_" $OUT/registry_kernel.csv
grep -E "^(outer_product_mean|pair_weighted_averaging)_" $OUT/registry_kernel.csv | cut -d, -f1 | sort | uniq -c
echo "== existing GPU tests touching the MSA modules"
python -m pytest tests/numerics/test_op_contracts_gpu.py tests/integrations/test_anthropic_msa_gpu.py tests/integrations/test_pwa_train_gpu.py tests/integrations/test_opm_train_gpu.py tests/numerics/test_whole_op_reachable_gpu.py tests/numerics/test_stack_substitutability_gpu.py -q -p no:cacheprovider 2>&1 | grep -E "^FAILED|^ERROR|passed|failed" | tail -20
