#!/bin/bash
W=/NHNHOME/WORKSPACE/26mohw002_A/psk6950; . $W/b2511/env.sh
cd $W/b2511/agents/s90-transition-inf
$W/b2511/dev.sh -g 2 -t 1500 s90-transition-inf python -m foldjet_engine.measure.accuracy b200cfg/accmerge.yaml --out out/accmerge.jsonl > out/accmerge.log 2>&1
echo "acc eval rc=$?"; grep -o '"light": "[A-Z]*"' out/accmerge.jsonl | sort | uniq -c
$W/b2511/dev.sh -g 2 -t 1500 s90-transition-inf python -m foldjet_engine.measure.accuracy b200cfg/acctrain.yaml --out out/acctrain.jsonl > out/acctrain.log 2>&1
echo "acc train rc=$?"; grep -o '"light": "[A-Z]*"' out/acctrain.jsonl | sort | uniq -c
