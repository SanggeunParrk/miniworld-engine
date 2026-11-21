#!/bin/bash
# ab.sh "ENVVAR=0,ENVVAR=1" "L list" : module training step graph time A/B (tl10.py), 2 rounds interleaved, all L128/L256 rows
W=/NHNHOME/WORKSPACE/26mohw002_A/psk6950; D=$W/b2511/dev.sh; A=s90-triattn-train; cd $W/b2511/agents/$A
IFS=, read -ra VS <<< "$1"
for rnd in 1 2; do for L in $2; do
 for kw in '{"d_pair":64,"n_head":2,"p_drop":0.25,"starting":true}' '{"d_pair":64,"n_head":4,"p_drop":0.25,"starting":true}' '{"d_pair":128,"n_head":4,"p_drop":0.25,"starting":false}' '{"d_pair":256,"n_head":8,"p_drop":0.25,"starting":true}' '{"d_pair":128,"n_head":4,"p_drop":0.25,"starting":true,"use_self_attention":false}'; do
  for v in "${VS[@]}"; do
   r=$(env $v $D -x -g ${G:-3} -t 600 $A python _t/tl10.py TriangleAttention "$kw" $L 2>&1 | grep -m1 "graph")
   echo "$v $r" | sed 's/TriangleAttention//; s/mod\[\]://; s/p_drop.: 0.25, //'
  done
 done
done; done
