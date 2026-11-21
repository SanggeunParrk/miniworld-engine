#!/bin/bash
W=/NHNHOME/WORKSPACE/26mohw002_A/psk6950; . $W/b2511/env.sh
cd $W/b2511/agents/s90-transition-inf
WATCHDOG=500 $W/b2511/dev.sh -x -g 2 -t 700 s90-transition-inf python b200cfg/hvf.py 128,2,16384,TBMMA=1 128,2,65536,TBMMA=1 128,2,262144,TBMMA=1 256,2,16384,TBMMA=1 256,2,65536,TBMMA=1 128,2,65536,TBEPI=1,TBMMA=1 128,2,65536,TBEPI=1,TBLN=2,TBMMA=1,HSIG=9 2>&1 | grep -v "Warning\|warn\|exe_args\|USDT\|^\[miniworld" | grep "fine\|diff" | grep -v "max |cur"
