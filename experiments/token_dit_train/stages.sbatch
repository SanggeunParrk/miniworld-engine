#!/bin/bash
#SBATCH -p h100 -A cssb -q normal_h100
#SBATCH --gres=gpu:h100:1 -c 8 --mem=64G -t 120
#SBATCH --requeue --open-mode=append
#SBATCH -J tdit-train
# Runs "name|command" lines from $CFG as stages; a stage already recorded done in $OUT is skipped on a requeue.
OUT=${OUT:?}; CFG=${CFG:?}
cd /home/psk6950/miniworld-engine-dit2/experiments/token_dit_train
echo "## start $(date +%T) $(hostname) restart=${SLURM_RESTART_COUNT:-0}" >> $OUT
while IFS='|' read -r n cmd; do
  [ -z "$n" ] && continue
  grep -qF "#### $n done" $OUT && continue
  echo "#### $n" >> $OUT
  bash -c "$cmd" 2>&1 < /dev/null | grep -avE "^\s*$|ninja|detected during|^\s+\^|warning #|Compiling|Function properties|^Loading|Building|Emitting|^\[[0-9]+/[0-9]+\]|^ptxas info +: (Compiling|Function)" >> $OUT
  echo "#### $n done" >> $OUT
done < $CFG
echo "## all done $(date +%T)" >> $OUT
