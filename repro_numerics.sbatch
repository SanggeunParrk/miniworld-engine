#!/bin/bash
#SBATCH -J numerics-2
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH -o /home/psk6950/practice/miniworld-engine/repro_numerics.log
#SBATCH -t 30:00
cd /home/psk6950/practice/miniworld-engine
.pixi/envs/default/bin/python -m pytest \
  "tests/numerics/test_numerical.py::test_kernel_matches_its_reference[cond_transition_expand_swiglu_saveact_triton]" \
  "tests/numerics/test_numerical.py::test_kernel_matches_its_reference[cond_transition_squeeze_gate_saveact_triton]" \
  -q -p no:randomly --no-header -x 2>&1 | grep -vE "UserWarning|hit = _cached_subset" | tail -60
