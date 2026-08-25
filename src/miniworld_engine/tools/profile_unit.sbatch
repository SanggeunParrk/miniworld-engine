#!/bin/bash
#SBATCH -J prof-unit
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -w gpu01
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH -t 01:00:00
#SBATCH -o src/miniworld_engine/tools/prof/prof-%j.log
set -u
R=${MINIWORLD_REPO:-$(git rev-parse --show-toplevel)}
cd "$R"
export PYTHONPATH=src
export TRITON_CACHE_DIR=${MINIWORLD_TRITON_CACHE:-$HOME/.cache/miniworld-triton}
OUT=$R/src/miniworld_engine/tools/prof
PY=.pixi/envs/default/bin/python

echo "### host $(hostname)"
echo "### mount for TRITON_CACHE_DIR:"
findmnt -T "$TRITON_CACHE_DIR" -o TARGET,SOURCE,FSTYPE 2>/dev/null
echo "### mount for /home:"
findmnt -T /home -o TARGET,SOURCE,FSTYPE 2>/dev/null

echo "### raw stat latency on 200 cache entries (cold-ish):"
$PY - <<'PY'
import os, time, itertools
d = os.environ["TRITON_CACHE_DIR"]
t0 = time.perf_counter()
names = list(itertools.islice(os.scandir(d), 200))
t1 = time.perf_counter()
n = 0; nb = 0
for e in names:
    try:
        for f in os.scandir(e.path):
            nb += os.stat(f.path).st_size; n += 1
    except OSError:
        pass
t2 = time.perf_counter()
print(f"scandir(200 dirs)      {t1-t0:8.3f}s")
print(f"stat {n} inner files   {t2-t1:8.3f}s  -> {(t2-t1)/max(n,1)*1000:.2f} ms/file, {nb/1e6:.1f} MB")
PY

echo "### profiling ONE unit ###"
$PY -u -m cProfile -o $OUT/unit-$SLURM_JOB_ID.pstats \
  -m miniworld_engine.autotune.builder \
  --shard $OUT/shard-$SLURM_JOB_ID \
  --compile-jobs 8 \
  --op adaln_bwd_dx_dlnw_triton --dtype bfloat16 --length 1024

echo "### TOP TOTTIME (self time -- where the wall clock actually goes) ###"
$PY - "$OUT/unit-$SLURM_JOB_ID.pstats" <<'PY'
import pstats, sys
st = pstats.Stats(sys.argv[1])
st.sort_stats("tottime").print_stats(30)
print("\n===== CUMULATIVE =====\n")
st.sort_stats("cumulative").print_stats(30)
PY
