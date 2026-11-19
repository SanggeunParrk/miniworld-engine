#!/bin/bash
W=/NHNHOME/WORKSPACE/26mohw002_A/psk6950; . $W/b2511/env.sh
cd $W/b2511/agents/s90-transition-inf
WATCHDOG=800 $W/b2511/dev.sh -x -g 2 -t 900 s90-transition-inf python b200cfg/hvf.py 128,2,262144,EP3=1,EP3_A=1,EP3_B=1 128,2,262144,EP3=1,EP3_A=2,EP3_B=1 128,2,262144,EP3=1,EP3_A=1,EP3_B=2 128,2,262144,EP3=1,EP3_A=2,EP3_B=2 128,2,262144,EP3=1,EP3_A=1,EP3_B=0 128,2,262144,EP3=1,EP3_A=0,EP3_B=2 2>&1 | grep -v "Warning\|warn\|exe_args\|USDT\|^\[miniworld" | grep "var fine\|rror"
