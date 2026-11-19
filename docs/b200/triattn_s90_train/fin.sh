#!/bin/bash
W=/NHNHOME/WORKSPACE/26mohw002_A/psk6950; D=$W/b2511/dev.sh; A=s90-triattn-train; cd $W/b2511/agents/$A
$D -t 3000 $A python -m pytest -x -q tests/gpu/test_triattn_b200.py tests/gpu/test_bidir_triattn_h100.py 2>&1 | grep -v Warn | tail -6
$D -t 3000 $A python -m miniworld_engine.measure.accuracy _s90/acc188.yaml --out _s90/acc_final.jsonl 2>&1 | grep -v Warn | tail -2
LS="128 256 384 512 640 768" bash _s90/m.sh final 2>&1 | grep done
