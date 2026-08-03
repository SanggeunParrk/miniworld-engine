#!/bin/bash
# Re-verify correctness (bmm->einsum, channel-last output rewrite of the contraction) AND
# re-measure speed for the B>1 triton trimul. einsum changes the math EXPRESSION, so this
# must confirm bidir+uni fwd/bwd still match the reference, then report new latencies.
#SBATCH --job-name=mwk_trimul_einsum_a100
#SBATCH --partition=A100
#SBATCH --qos=normal
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:50:00
#SBATCH --output=benchmarks/trimul_einsum_a100_%j.out

set -uo pipefail
set -f
cd /home/psk6950/miniworld-engine
echo "host=$(hostname) date=$(date)"; git status --short | head

R() { pixi run --frozen bash -c "export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\${LD_LIBRARY_PATH:-}; PYTHONPATH=src python $1"; }

echo "### correctness: bidir (B>1 + batch-independence) ###"; R submits/_bidir_bwise_verify.py; echo "[exit=$?]"
echo "### correctness: uni + front/back ###";                R submits/_trimul_uni_fb_verify.py; echo "[exit=$?]"
echo "### speed (batched vs loop vs pytorch) ###";           R submits/_trimul_speed.py; echo "[exit=$?]"
echo "TRIMUL_EINSUM DONE"
