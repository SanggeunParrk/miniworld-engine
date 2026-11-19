#!/bin/bash
W=/NHNHOME/WORKSPACE/26mohw002_A/psk6950; . $W/b2511/env.sh
cd $W/b2511/agents/s90-transition-inf
WATCHDOG=300 $W/b2511/dev.sh -x -g 2 -t 400 s90-transition-inf python b200cfg/tl.py 384,2,16384,bf16,eval 384,2,262144,bf16,eval 2>&1 | grep -v "Warning\|warn\|USDT" | cut -c1-200 | tail -25
