#!/bin/bash
W=/NHNHOME/WORKSPACE/26mohw002_A/psk6950; . $W/b2511/env.sh
cd $W/b2511/agents/s90-transition-inf
M=147456
F=XPAR=1,WPAR=1,LN2=1,PF2=1
WATCHDOG=100 $W/b2511/dev.sh -x -g 2 -t 200 s90-transition-inf python b200cfg/s90t.py 256,2,$M,$F 256,2,$M,$F,X2=1 2>&1 | grep -v "Warning\|warn"
