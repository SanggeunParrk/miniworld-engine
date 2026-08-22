#!/bin/bash
#SBATCH -J cachedir-ab
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -w gpu02
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH -t 03:00:00
#SBATCH -o /home/psk6950/practice/miniworld-engine/tools/kernel-audit/prof/ab-%j.log
set -u
R=/home/psk6950/practice/miniworld-engine
cd $R
export PYTHONPATH=src
export MINIWORLD_CONFIG_DIR=$R/configs/grid
OUT=$R/tools/kernel-audit/prof
PY=.pixi/envs/default/bin/python
OP=adaln_bwd_dx_dlnw_triton

echo "### host $(hostname)"
echo "### node-local candidates:"
for d in /tmp /dev/shm /scratch /local; do
  [ -d "$d" ] && printf '%-10s %s  %s\n' "$d" "$(findmnt -nT $d -o FSTYPE 2>/dev/null)" "$(df -h --output=avail $d 2>/dev/null | tail -1)"
done

run () {  # run <label> <cachedir>
  local label=$1 dir=$2
  export TRITON_CACHE_DIR=$dir
  mkdir -p "$dir"
  echo ""
  echo "############ $label  TRITON_CACHE_DIR=$dir"
  echo "### fs: $(findmnt -nT "$dir" -o FSTYPE 2>/dev/null)   entries: $(ls -U "$dir" 2>/dev/null | wc -l)"
  /usr/bin/time -f "### WALL %e s   maxrss %M KB" \
    $PY -u -m miniworld_engine.autotune.builder \
      --shard $OUT/ab-$label-$SLURM_JOB_ID \
      --compile-jobs 8 --bench-budget 3.0 \
      --op $OP --dtype bfloat16 --length 1024 2>&1 | grep -E '^\s*\[|WALL|unit ran'
}

# A: the current setup -- shared NFS cache, already holding 107k entries for this arch
run nfs-warm /public_data02/psk6950/tritoncache

# B: node-local, cold. Pays the full compile, then reads back off local disk.
LOCAL=/tmp/$USER/tritoncache-ab-$SLURM_JOB_ID
run local-cold $LOCAL

# C: node-local, warm. This is what every unit AFTER the first sees on a node.
run local-warm $LOCAL

rm -rf "$LOCAL"
