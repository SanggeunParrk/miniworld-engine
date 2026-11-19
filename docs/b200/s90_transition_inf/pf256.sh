#!/bin/bash
W=/NHNHOME/WORKSPACE/26mohw002_A/psk6950; . $W/b2511/env.sh
cd $W/b2511/agents/s90-transition-inf
WATCHDOG=300 $W/b2511/dev.sh -x -g 2 -t 400 s90-transition-inf python b200cfg/pf256.py 2,16384 4,16384 2,65536 2>&1 | grep -v "Warning\|warn\|exe_args\|USDT\|^\[miniworld"
