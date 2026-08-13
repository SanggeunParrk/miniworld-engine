#!/bin/bash
# One shard of a PARALLEL autotune-cache build on the Ampere workstation cards (RTX A5000 /
# A6000, sm_86). Ampere twin of submits/build_shard.sbatch, which is pinned to the h100
# partition/qos/gres and to a checkout path that does not exist on this cluster.
#
# Each array task runs ONE bench.py capture command (line $SLURM_ARRAY_TASK_ID of $CMDFILE) and
# dumps its timings to its OWN shard json (`+autotune_shard=`), so N tasks never race on the
# committed src/miniworld_engine/autotune/data tree. A single-writer merge folds the shards in:
#
#   CARD=A6000 SHARDS=$HOME/.cache/mwk-shards/a6000 \
#     python submits/build_autotune_cache.py plan --targets ... --shards $SHARDS --shapes '...' > cmds
#   CMDFILE=cmds sbatch --array=1-32%8 --gres=gpu:A6000:1 submits/build_shard_ampere.sbatch
#   python submits/build_autotune_cache.py merge --shards $SHARDS --gpu "NVIDIA RTX A6000 (sm86)"
#
# The capture needs settings.run_autotune so the FULL grid is benched rather than the cached
# top-K; `+autotune_shard=` turns capture on by itself (bench.py: `or bool(conf.autotune_shard)`).
#SBATCH --job-name=cacheshard_ampere
#SBATCH --partition=gpu
#SBATCH --account=cssb
#SBATCH --qos=normal
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/psk6950/.cache/mwk-shardlog/%A_%a.out
set -uo pipefail
set -f
cd /home/psk6950/practice/miniworld-engine
export HYDRA_FULL_ERROR=1
: "${CMDFILE:?set CMDFILE}"
line=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$CMDFILE")
echo "host=$(hostname) task=$SLURM_ARRAY_TASK_ID :: $line"
pixi run --frozen python -c "
import torch
name = torch.cuda.get_device_name(0); cap = torch.cuda.get_device_capability(0)
print('shard on', name, 'sm', cap, torch.__version__)
assert cap == (8, 6), f'expected sm_86 (RTX A5000/A6000), got sm_{cap[0]}{cap[1]} ({name})'
"
pixi run --frozen bash -c 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-} ; PYTHONPATH=src python -u benchmarks/runners/bench.py '"$line"''
echo "[exit=$?]"
