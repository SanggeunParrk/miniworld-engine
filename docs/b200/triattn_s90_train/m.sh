#!/bin/bash
# m.sh TAG : every training case of TriangleAttention (cuda), one exclusive measurer run per L. out in _s90/
W=/NHNHOME/WORKSPACE/26mohw002_A/psk6950; D=$W/b2511/dev.sh; A=s90-triattn-train; cd $W/b2511/agents/$A; mkdir -p _s90
for L in ${LS:-128 256 384 512 640 768}; do
  $D -x -g ${G:-3} -t 3600 $A python -m foldjet_engine.measure.speed _s90/ta$L.yaml --out _s90/spd_$1_L$L.jsonl 2>&1 | grep -v Warn | tail -2
  echo "L$L done $(date +%T)"
done
