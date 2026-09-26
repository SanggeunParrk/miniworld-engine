#!/bin/bash
#SBATCH -p h100 -A cssb -q normal_h100
#SBATCH --gres=gpu:h100:1 -c 8 --mem=64G -t 120
#SBATCH --requeue --open-mode=append
#SBATCH -J tdit-bwd
# Runs "name|command" lines from $CFG as stages; a stage already recorded done in $OUT is skipped on a requeue.
OUT=${OUT:?}; CFG=${CFG:?}
cd /home/psk6950/MiniWorld/runs/token_dit_bwd_20260926
echo "## start $(date +%T) restart=${SLURM_RESTART_COUNT:-0}" >> $OUT
while IFS='|' read -r n cmd; do
  [ -z "$n" ] && continue
  grep -qF "#### $n done" $OUT && continue
  echo "#### $n" >> $OUT
  bash -c "$cmd" 2>&1 < /dev/null | grep -avE "^\s*$|ninja|detected during|^\s+\^|const int|const uint8_t|warning #177|spill|bytes stack|ptxas|Compiling|Used [0-9]+ reg|Function properties|^Loading|Building|Emitting|^\[" >> $OUT
  echo "#### $n done" >> $OUT
done < $CFG
echo "## all done $(date +%T)" >> $OUT
