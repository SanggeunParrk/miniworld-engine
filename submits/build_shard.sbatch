#!/bin/bash
# One shard of a parallel autotune-cache build: runs ONE bench.py capture command (line
# $SLURM_ARRAY_TASK_ID of $CMDFILE), which dumps its timings to its own autotune_shard file.
#SBATCH --job-name=cacheshard
#SBATCH --partition=h100
#SBATCH --account=cssb
#SBATCH --qos=normal_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/psk6950/pfcache/shardlog/%A_%a.out
set -uo pipefail
set -f
cd /home/psk6950/miniworld-engine
: "${CMDFILE:?set CMDFILE}"
line=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$CMDFILE")
echo "host=$(hostname) task=$SLURM_ARRAY_TASK_ID :: $line"
pixi run --frozen bash -c 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-} '"${CAP_EXPORT:-}"'; PYTHONPATH=src python -u benchmarks/runners/bench.py '"$line"''
echo "[exit=$?]"
