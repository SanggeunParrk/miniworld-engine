#!/bin/bash
# sweep.sh OUT VARIANTS "L:S L:S ..."   kbench.py per case in one job (resumable: a case with records in OUT is skipped)
D=$(cd "$(dirname "$0")" && pwd); OUT=$1; VAR=$2; CASES=$3; T0=$(date +%s)
for c in $CASES; do
  L=${c%:*}; S=${c#*:}
  if [ -f "$OUT" ] && grep -q "\"L\": $L, \"S\": $S," "$OUT"; then continue; fi
  [ $(( $(date +%s) - T0 )) -gt 470 ] && { echo "DEADLINE"; exit 0; }
  echo "== L $L S $S"
  python $D/kbench.py $L $S $OUT --rounds 2 --idle 1.0 --variants $VAR
done
echo SWEEP DONE
