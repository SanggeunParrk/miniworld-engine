#!/bin/bash
# One judge cycle on a Slurm cluster (docs/judge.md): plan the owed cases, prebuild what the cache lacks, measure the plan with four
# 1-GPU 10-minute work-queue jobs (resubmitted until every line is done), issue tokens, and report what is still owed.
#
#   tools/slurm/cycle.sh PLAN_DIR [judge owed options, e.g. --redo failed]
#
# Nothing here runs Python on the submitting (login) node: every Python step is a Slurm job. Site settings come from the environment:
#   SLURM_ACCOUNT, SLURM_PARTITION, SLURM_QOS   passed to sbatch when set
#   GPU_GRES   (default gpu:1)                   the GPU request of a measurement job
#   ENGINE_ENV (default <repo>/.pixi/envs/default)  the Python environment (its bin/ on PATH, lib/ on LD_LIBRARY_PATH)
#   CUTLASS_PATH, MATHDX_HOME                    forwarded to the prebuild job (see docs/kernels.md)
#   PREBUILD_CPUS, PREBUILD_MEM                  the prebuild job's CPUs / memory (default 48 / 240G; a cluster without such nodes: e.g. 16 / 64G)
#   CARD       (default h100_sm90)              the card the plan, the tokens and the records are for (a100_sm80 on an A100 cluster)
#   ARCH       (default sm90a)                  the prebuild target (sm80 for an A100)
#   PARALLEL   (default 4)                      measurement jobs (GPUs) at once
#   PLAN_GROUPS (default PARALLEL)              groups the plan is split into (more groups than GPUs keeps every free GPU busy)
#   NODELIST   (optional)                       --nodelist for every job (a cluster where only some nodes are free)
set -u
if [ "${1:-}" = "--job" ]; then                       # inside a measurement job: work through one group file
  shift; PLAN=$1; GROUP=$2
  start=$(date +%s); mkdir -p "$PLAN/out" "$PLAN/done"
  while read -r kind module cfg; do
    [ -z "$kind" ] && continue
    [ -e "$PLAN/done/$kind.$module" ] && continue
    left=$(( 570 - ($(date +%s) - start) )); [ $left -lt 60 ] && { echo "TIME UP before $kind $module"; exit 0; }
    echo "=== $kind $module (budget ${left}s) $(date +%T)"
    timeout $left python -m miniworld_engine.measure.$kind "$cfg" --out "$PLAN/out/$kind.$module.jsonl" \
      > "$PLAN/logs/$kind.$module.${SLURM_JOB_ID:-local}.log" 2>&1; rc=$?
    echo "=== $kind $module rc=$rc $(date +%T)"
    # done either way: a module cut by the time limit keeps the records it wrote and its other cases are owed again next cycle
    # (re-running it from the start every round would block its group for good)
    touch "$PLAN/done/$kind.$module"
  done < "$PLAN/$GROUP"
  echo GROUP DONE; exit 0
fi

PLAN=$(realpath -m "${1:?usage: cycle.sh PLAN_DIR [owed options]}"); shift
REPO=$(cd "$(dirname "$0")/../.." && pwd)
ENV=${ENGINE_ENV:-$REPO/.pixi/envs/default}
export PATH=$ENV/bin:$PATH LD_LIBRARY_PATH=$ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} PYTHONPATH=$REPO/src PYTHONUNBUFFERED=1
SB=(sbatch --parsable --time=00:10:00)
[ -n "${SLURM_ACCOUNT:-}" ] && SB+=(--account="$SLURM_ACCOUNT")
[ -n "${SLURM_PARTITION:-}" ] && SB+=(--partition="$SLURM_PARTITION")
[ -n "${SLURM_QOS:-}" ] && SB+=(--qos="$SLURM_QOS")
[ -n "${NODELIST:-}" ] && SB+=(--nodelist="$NODELIST")
mkdir -p "$PLAN/logs"; S=$PLAN/status.txt; say() { echo "$(date +%T) $*" | tee -a "$S"; }
wait_job() { while squeue -j "$1" -h 2>/dev/null | grep -q .; do sleep 15; done; }
cpu() {  # cpu NAME CPUS MEM COMMAND... : one CPU job, waited for, its log printed
  local j; j=$("${SB[@]}" --gres=none --cpus-per-task="$2" --mem="$3" -J "$1" -o "$PLAN/logs/$1-%j.log" --export=ALL --wrap="${*:4}")
  wait_job "$j"; grep -h "\[judge\]\|\[prebuild\] done\|Error\|Traceback" "$PLAN/logs/$1-$j.log"
}

say "plan"; cpu judge-owed 8 32G "python -m miniworld_engine.judge --gpu ${CARD:-h100_sm90} owed $* --groups ${PLAN_GROUPS:-${PARALLEL:-4}} --plan $PLAN"
say "prebuild"; for round in 1 2 3; do
  out=$(cpu prebuild ${PREBUILD_CPUS:-48} ${PREBUILD_MEM:-240G} "python -m miniworld_engine.kernels._prebuild --arch ${ARCH:-sm90a} --stop-after 480")
  echo "$out" | tee -a "$S" | grep -q " 0 of .* still absent" && break
done
run_group() {
  local g=$1 round j
  [ -s "$PLAN/$g" ] || return 0
  for round in $(seq 1 12); do
    j=$("${SB[@]}" --gres="${GPU_GRES:-gpu:1}" --cpus-per-task=8 --mem=64G -J "cycle-$g" -o "$PLAN/logs/%x-%j.log" --export=ALL \
        "$0" --job "$PLAN" "$g")
    say "$g round $round job $j"; wait_job "$j"
    grep -q "GROUP DONE" "$PLAN/logs/cycle-$g-$j.log" && { say "$g done"; return 0; }
  done
  say "$g NOT DONE after 12 jobs"
}
for g in $(ls "$PLAN" | grep -E '^G[0-9]+$'); do
  while [ "$(jobs -rp | wc -l)" -ge "${PARALLEL:-4}" ]; do sleep 15; done
  run_group "$g" &
done; wait
say "issue"; cpu judge-issue 8 32G "python -m miniworld_engine.judge --gpu ${CARD:-h100_sm90} issue $PLAN/out && python -m miniworld_engine.judge --gpu ${CARD:-h100_sm90} owed"
say "cycle finished (run again while owed > 0)"
