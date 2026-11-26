#!/bin/bash
W=/NHNHOME/WORKSPACE/26mohw002_A/psk6950; . $W/b2511/env.sh
cd $W/b2511/agents/s90-transition-inf
WATCHDOG=500 $W/b2511/dev.sh -x -g 2 -t 700 s90-transition-inf python b200cfg/tl.py 512,2,262144,bf16,eval 512,4,409600,bf16,eval 2>&1 | grep -v "Warning\|warn\|foldjet-engine build\|USDT" | cut -c1-120
WATCHDOG=300 $W/b2511/dev.sh -g 2 -t 400 s90-transition-inf python b200cfg/chk512.py 2,65536,0 4,65536,3 2>&1 | grep -v "Warning\|warn\|foldjet-engine build" | cut -c1-250
