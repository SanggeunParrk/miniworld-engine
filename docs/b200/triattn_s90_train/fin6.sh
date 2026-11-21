#!/bin/bash
W=/NHNHOME/WORKSPACE/26mohw002_A/psk6950; D=$W/b2511/dev.sh; A=s90-triattn-train; cd $W/b2511/agents/$A
export MINIWORLD_TRIATTN_B200_DBROWS=1
$D -g 3 -t 1500 $A python _s90/poison.py eager 2>&1 | grep -v Warn | tail -7
$D -g 3 -t 1500 $A python _s90/poison.py graph 2>&1 | grep -v Warn | tail -7
$D -g 3 -t 1500 $A python _t/detm.py 2>&1 | grep -E "TriangleAttention" | grep -vc "deterministic (5 runs)"
$D -g 3 -t 3000 $A python -m miniworld_engine.measure.accuracy _s90/acc_all.yaml --out _s90/acc_dbr.jsonl 2>&1 | grep -v Warn | tail -2
