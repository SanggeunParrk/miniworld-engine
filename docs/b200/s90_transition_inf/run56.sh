#!/bin/bash
W=/NHNHOME/WORKSPACE/26mohw002_A/psk6950; . $W/b2511/env.sh
cd $W/b2511/agents/s90-transition-inf
$W/b2511/dev.sh -g 2 -t 900 s90-transition-inf python -m foldjet_engine.measure.accuracy b200cfg/acc33.yaml --out out/acc33_v56.jsonl > out/acc33_v56.log 2>&1
echo "acc rc=$?"; grep -o '"light": "[A-Z]*"' out/acc33_v56.jsonl | sort | uniq -c
bash b200cfg/s90m.sh v56 cuda eval 128 256 384 512 640 768 1024 1536 2048 2976 3968
