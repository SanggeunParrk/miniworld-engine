#!/bin/bash
W=/NHNHOME/WORKSPACE/26mohw002_A/psk6950; D=$W/b2511/dev.sh; A=s90-triattn-train; cd $W/b2511/agents/$A; mkdir -p _s90
$D -g 3 -t 3000 $A python -m miniworld_engine.measure.accuracy _s90/acc188.yaml --out _s90/acc_merged.jsonl 2>&1 | grep -v Warn | tail -3
LS="128 256 384 512 640 768" bash _s90/m.sh merged 2>&1 | grep "done"
