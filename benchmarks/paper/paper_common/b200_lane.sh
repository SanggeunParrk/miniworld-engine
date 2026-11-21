#!/bin/bash
# b200_lane.sh GPU : one lane of the B200 campaign (the counterpart of h100_lane.sbatch without Slurm). Takes the first unfinished, unclaimed task of
# queue_b200.txt (EXP REP STAGE per line, priority order) and runs it on GPU $1 to completion, then the next. Several lanes (one per GPU 0-3) share
# the queue; a task is claimed by a directory in $PAPER/.claims_b200 holding the claiming lane's pid (a claim whose pid is gone is taken over).
# Stops when the queue has no open task or when $PAPER/.stop_b200 exists. Only GPUs 0-3 are allowed.
GPU=$1
case "$GPU" in 0|1|2|3) ;; *) echo "GPU must be 0-3 (4-7 belong to a colleague)"; exit 2;; esac
PAPER=$(cd "$(dirname "$0")/.." && pwd)
CL=$PAPER/.claims_b200; mkdir -p "$CL"
export CUDA_VISIBLE_DEVICES=$GPU
echo "lane gpu $GPU pid $$ on $(hostname) $(date +%T)"
while read -r exp rep stage rest; do
  case "$exp" in ""|\#*) continue;; esac
  [ -e "$PAPER/.stop_b200" ] && { echo "lane: stop file"; break; }
  tag=$exp.r$rep.$stage
  case "$exp:$stage" in
    E4_fp8:*) f=$PAPER/$exp/records/b200/fp8_$stage.r$rep.jsonl;;
    *) f=$PAPER/$exp/records/b200/$stage.r$rep.jsonl;;
  esac
  [ -e "$f.done" ] && continue
  if ! mkdir "$CL/$tag" 2>/dev/null; then
    other=$(cat "$CL/$tag/pid" 2>/dev/null)
    if [ -z "$other" ] || kill -0 "$other" 2>/dev/null; then continue; fi
    echo "lane: taking over stale claim $tag (pid $other)"; rm -rf "$CL/$tag"; mkdir "$CL/$tag" 2>/dev/null || continue
  fi
  echo $$ > "$CL/$tag/pid"
  # the GPU must be free of other processes (the colleague's work is on 4-7; never share a timing GPU with anything)
  used=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  if [ "${used:-0}" -gt 2000 ]; then echo "lane: GPU $GPU busy (${used} MiB used); lane ends"; rm -rf "$CL/$tag"; break; fi
  echo "lane: task $tag on GPU $GPU $(date +%T)"
  mkdir -p "$PAPER/logs_b200"
  bash "$PAPER/paper_common/b200_task.sh" "$exp" "$rep" "$stage" < /dev/null > "$PAPER/logs_b200/$tag.gpu$GPU.log" 2>&1; echo "lane: task $tag rc=$? $(date +%T)"
  rm -rf "$CL/$tag"
done < "${QUEUE:-$PAPER/paper_common/queue_b200.txt}"
echo "lane gpu $GPU end $(date +%T)"
