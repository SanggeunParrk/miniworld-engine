#!/bin/bash
# ab188.sh "ENV=0,ENV=1" "L list": row 188 (starting + ending) training step graph A/B + accuracy of the last variant
W=/NHNHOME/WORKSPACE/26mohw002_A/psk6950; D=$W/b2511/dev.sh; A=s90-triattn-train; cd $W/b2511/agents/$A
IFS=, read -ra VS <<< "$1"
for rnd in 1 2; do for L in $2; do for st in true false; do
  for v in "${VS[@]}"; do
   r=$(env $v $D -x -g ${G:-3} -t 600 $A python _t/tl10.py TriangleAttention "{\"d_pair\":128,\"n_head\":4,\"p_drop\":0.25,\"starting\":$st,\"use_self_attention\":false}" $L 2>&1 | grep -m1 "graph")
   echo "$v st=$st $r" | sed 's/TriangleAttention//; s/mod\[\]://; s/.p_drop.: 0.25, //'
  done
done; done; done
