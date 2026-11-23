#!/bin/bash
#SBATCH -A cssb -p h100 --qos=cssb_h100 --gres=gpu:h100:1 --cpus-per-task=16 --mem=96G --time=00:10:00
#SBATCH --job-name=E4td-h100 --output=slurm-%x-%j.log
# Token DiT FP8 against the default (fp8_pair.py --config cfg_tokendit.yaml). Resumable. GATES=shipped|none, TAG=<label>, EXTRA=<selection overrides>
HERE=${EXP_DIR:-$SLURM_SUBMIT_DIR}
. "$HERE/../paper_common/env_h100.sh"
python "$HERE/fp8_pair.py" --config "$HERE/cfg_tokendit.yaml" --gates "${GATES:-shipped}" ${STATIC:+--static-weights} --rounds "${ROUNDS:-6}" \
  --out "${RECORDS:-$HERE/records_td}/h100/td_${TAG:-x}_${GATES:-shipped}${STATIC:+_static}.jsonl" --budget-seconds 540 $EXTRA
echo "end $(date +%T)"
