#!/bin/bash
# tl.sh : kernel timelines of training steps (graph replay). usage: tl.sh "L list" ; cases below
W=/NHNHOME/WORKSPACE/26mohw002_A/psk6950; D=$W/b2511/dev.sh; A=s90-triattn-train; cd $W/b2511/agents/$A
run() { $D -x -g ${G:-3} -t 900 $A python _t/tl10.py TriangleAttention "$1" $2 2>&1 | grep -v "Warn\|USDT\|build\]"; }
for L in $LS; do
 run '{"d_pair":128,"n_head":4,"p_drop":0.25,"starting":true}' $L
 run '{"d_pair":384,"n_head":12,"p_drop":0.25,"starting":true}' $L
 run '{"d_pair":128,"n_head":4,"p_drop":0.25,"starting":true,"use_self_attention":false}' $L
 run '{"d_pair":64,"n_head":4,"p_drop":0.25,"starting":true}' $L
done
