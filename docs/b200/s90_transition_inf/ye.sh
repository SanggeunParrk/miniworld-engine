#!/bin/bash
W=/NHNHOME/WORKSPACE/26mohw002_A/psk6950; . $W/b2511/env.sh
cd $W/b2511/agents/s90-transition-inf
WATCHDOG=500 $W/b2511/dev.sh -x -g 2 -t 700 s90-transition-inf python b200cfg/hvf.py 128,2,16384,YE=0 128,2,65536,YE=0 128,2,147456,YE=0 128,2,589824,YE=0 64,2,65536,YE=0 64,2,262144,YE=0 2>&1 | grep -v "Warning\|warn\|exe_args\|USDT\|^\[miniworld"
