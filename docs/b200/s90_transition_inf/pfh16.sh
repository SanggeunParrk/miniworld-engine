#!/bin/bash
W=/NHNHOME/WORKSPACE/26mohw002_A/psk6950; . $W/b2511/env.sh
cd $W/b2511/agents/s90-transition-inf
WATCHDOG=500 $W/b2511/dev.sh -x -g 2 -t 600 s90-transition-inf python b200cfg/pfh16.py 128,16384 128,65536 128,147456 256,16384 256,65536 256,147456 2>&1 | grep -v "Warning\|warn\|exe_args\|USDT\|^\[miniworld"
