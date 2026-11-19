#!/bin/bash
W=/NHNHOME/WORKSPACE/26mohw002_A/psk6950; . $W/b2511/env.sh
cd $W/b2511/agents/s90-transition-inf
WATCHDOG=500 $W/b2511/dev.sh -x -g 2 -t 700 s90-transition-inf python b200cfg/tl.py 512,2,262144,bf16,eval 512,4,409600,bf16,eval 512,2,65536,bf16,eval 2>&1 | grep -v "Warning\|warn\|miniworld-engine build\|USDT"
