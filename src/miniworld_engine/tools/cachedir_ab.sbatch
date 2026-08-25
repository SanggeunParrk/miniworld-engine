#!/bin/bash
#SBATCH -J cachedir-ab
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -w gpu02
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH -t 03:00:00
#SBATCH -o src/miniworld_engine/tools/prof/ab-%j.log
set -u
R=${MINIWORLD_REPO:-$(git rev-parse --show-toplevel)}
cd "$R"
export PYTHONPATH=src
OUT=$R/src/miniworld_engine/tools/prof
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
run nfs-warm ${MINIWORLD_TRITON_CACHE:-$HOME/.cache/miniworld-triton}

# B: node-local, cold. Pays the full compile, then reads back off local disk.
LOCAL=/tmp/$USER/tritoncache-ab-$SLURM_JOB_ID
run local-cold $LOCAL

# C: node-local, warm. This is what every unit AFTER the first sees on a node.
run local-warm $LOCAL

rm -rf "$LOCAL"
