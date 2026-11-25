#!/bin/bash
W=/NHNHOME/WORKSPACE/26mohw002_A/psk6950; . $W/b2511/env.sh
cd $W/b2511/agents/s90-transition-inf
$W/b2511/dev.sh -g 2 -t 1200 s90-transition-inf python -m foldjet_engine.measure.accuracy b200cfg/acc3132.yaml --out out/acc3132_k16.jsonl > out/acc3132_k16.log 2>&1
echo "acc rc=$?"; grep -o '"light": "[A-Z]*"' out/acc3132_k16.jsonl | sort | uniq -c
WATCHDOG=500 $W/b2511/dev.sh -g 2 -t 700 s90-transition-inf python b200cfg/tcheckt.py 384,2,16384,bf16 384,4,16384,bf16 2>&1 | grep -v "Warning\|warn\|miniworld-engine build"
bash b200cfg/s90m.sh v58 cuda eval 128 256 384 512 640 768 1024 1536 2048 2976 3968
