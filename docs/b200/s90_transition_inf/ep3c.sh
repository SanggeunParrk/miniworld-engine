#!/bin/bash
W=/NHNHOME/WORKSPACE/26mohw002_A/psk6950; . $W/b2511/env.sh
cd $W/b2511/agents/s90-transition-inf
WATCHDOG=800 $W/b2511/dev.sh -x -g 2 -t 900 s90-transition-inf python b200cfg/hvf.py 128,2,16384,EP3=1 128,2,32768,EP3=1 128,2,65536,EP3=1 128,2,98304,EP3=1 128,2,147456,EP3=1 128,2,1048576,EP3=1 2>&1 | grep -v "Warning\|warn\|exe_args\|USDT\|^\[miniworld" | grep "fine\|rror"
