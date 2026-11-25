#!/bin/bash
W=/NHNHOME/WORKSPACE/26mohw002_A/psk6950; . $W/b2511/env.sh
cd $W/b2511/agents/s90-transition-inf
$W/b2511/dev.sh -g 2 -t 900 s90-transition-inf python -m foldjet_engine.measure.accuracy b200cfg/acc3334.yaml --out out/acc3334_v54.jsonl > out/acc3334_v54.log 2>&1
echo "acc rc=$?"; grep -o '"light": "[A-Z]*"' out/acc3334_v54.jsonl | sort | uniq -c
bash b200cfg/s90m3334.sh g54 cuda eval 128 256 384 512 640 768
bash b200cfg/s90m3334.sh g54b cuda eval 256 512 768
