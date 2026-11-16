#!/bin/bash
# Submit LINKS dependent 1-GPU 10-minute jobs (afterany) of one resumable work list; finished cases are skipped on resume.
#   chain.sh NAME LINKS "STACK MODULE ROWS... " ...   (each group: STACK=engine|kit, module, rows; backends by stack)
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
NAME=$1; LINKS=$2; shift 2
CMDS=""
for g in "$@"; do
  set -- $g; STACK=$1; MOD=$2; shift 2; ROWS="$*"
  if [ "$STACK" = kit ]; then BE="pytorch anthropic"; else BE="pytorch cuda triton"; fi
  CMDS+="STACK=$STACK $HERE/run_job.sh --module $MOD --rows $ROWS --lengths 256 384 512 768 --depths 1024 2048 8192 16384 --backends $BE --iterations 30 --deadline 250 --out $HERE/raw/$NAME-$MOD-$STACK.jsonl; "
done
prev=""
for i in $(seq 1 "$LINKS"); do
  dep=""; [ -n "$prev" ] && dep="--dependency=afterany:$prev"
  prev=$(sbatch --parsable -A cssb -p h100 -q cssb_h100 --gres=gpu:h100:1 -c 8 --mem=96G -t 00:10:00 $dep -J msa-$NAME -o $HERE/logs/$NAME-%j.log --wrap="$CMDS")
  echo "$NAME link $i job $prev"
done
