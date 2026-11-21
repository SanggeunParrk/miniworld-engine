#!/bin/bash
#SBATCH -A cssb -p h100 --qos=cssb_h100 --gres=gpu:h100:1 --cpus-per-task=16 --mem=96G --time=00:10:00
#SBATCH --job-name=paper-lane --output=slurm-%x-%j.log
# One lane of the H100 campaign: a 10-minute job that takes the first unfinished, unclaimed task of queue_h100.txt (EXP REP STAGE per line, priority
# order), runs it for the rest of the job's budget (the task is resumable: run_measure.py skips finished cases, writes a .done marker when complete),
# then the next task. Several lanes (a chain of such jobs each, paper_common/lane_chain.sh) share the queue: a task is claimed by a directory
# in $PAPER/.claims holding the claiming job id (a claim of a job that no longer runs is taken over). Every task keeps its own file and its own
# repeat number, so nothing is re-routed; the repeats are different Slurm jobs by construction.
PAPERDIR=${PAPERDIR:-$SLURM_SUBMIT_DIR}
CL=$PAPERDIR/.claims; mkdir -p "$CL"
BUDGET_JOB=${BUDGET_JOB:-450}
SECONDS=0
echo "lane job $SLURM_JOB_ID on $(hostname) $(date +%T)"
while read -r exp rep stage rest; do
  case "$exp" in ""|\#*) continue;; esac
  [ $((BUDGET_JOB - SECONDS)) -lt 60 ] && { echo "lane: budget used up"; break; }
  tag=$exp.r$rep.$stage
  # finished?  the .done marker of the stage's file
  case "$exp:$stage" in
    E4_fp8:*) f=$PAPERDIR/$exp/records/h100/fp8_$stage.r$rep.jsonl;;
    *) f=$PAPERDIR/$exp/records/h100/$stage.r$rep.jsonl;;
  esac
  [ -e "$f.done" ] && continue
  if ! mkdir "$CL/$tag" 2>/dev/null; then
    other=$(cat "$CL/$tag/job" 2>/dev/null)
    if [ -z "$other" ] || [ -n "$(squeue -h -j "$other" 2>/dev/null)" ]; then continue; fi
    echo "lane: taking over stale claim $tag (job $other)"; rm -rf "$CL/$tag"; mkdir "$CL/$tag" 2>/dev/null || continue
  fi
  echo "$SLURM_JOB_ID" > "$CL/$tag/job"
  echo "lane: task $tag $(date +%T), $((BUDGET_JOB - SECONDS)) s left"
  script=$PAPERDIR/$exp/h100.sbatch
  EXP_DIR=$PAPERDIR/$exp REP=$rep ONLY=$stage BUDGET=$((BUDGET_JOB - SECONDS)) bash "$script" < /dev/null; echo "lane: task $tag rc=$?"
  rm -rf "$CL/$tag"
done < "${QUEUE:-$PAPERDIR/paper_common/queue_h100.txt}"
echo "lane end $(date +%T)"
