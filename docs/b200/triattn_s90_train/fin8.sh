#!/bin/bash
W=/NHNHOME/WORKSPACE/26mohw002_A/psk6950; D=$W/b2511/dev.sh; A=s90-triattn-train; cd $W/b2511/agents/$A
MINIWORLD_TRIATTN_B200_DBROWS=1 $D -g 3 -t 3000 $A python -m foldjet_engine.measure.accuracy _s90/acc_all.yaml --out _s90/acc_dbr2.jsonl 2>&1 | grep -v Warn | tail -2
bash _s90/ab.sh "MINIWORLD_TRIATTN_B200_DBROWS=0,MINIWORLD_TRIATTN_B200_DBROWS=1" "128 256" 2>&1 | grep -v Warn | grep "graph" | head -40
bash _s90/ab_wide.sh "MINIWORLD_TRIATTN_B200_DBROWS=0,MINIWORLD_TRIATTN_B200_DBROWS=1" "384 512 768" 2>&1 | grep -v Warn | grep graph
