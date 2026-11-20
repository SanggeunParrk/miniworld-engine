#!/bin/bash
#SBATCH -p A100 --gres=gpu:A100:1 --cpus-per-task=16 --mem=96G --time=00:10:00
#SBATCH --job-name=E5-a100 --output=slurm-%x-%j.log
# Resumable: submit through paper_common/chain.sh. Needs deepspeed / trifast in PAPER_EXT_PATH (README); a missing one is recorded unsupported.
HERE=${EXP_DIR:-$SLURM_SUBMIT_DIR}
. "$HERE/../paper_common/env_a100.sh"; . "$HERE/../paper_common/lib.sh"
EXT="ds4sci trifast"
stage E5_other_kernels ext cfg_engine.yaml
echo "end $(date +%T)"
