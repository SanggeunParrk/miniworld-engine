#!/bin/bash
W=/NHNHOME/WORKSPACE/26mohw002_A/psk6950; . $W/b2511/env.sh
cd $W/b2511/agents/s90-transition-inf
F=XPAR=1,WPAR=1,LN2=1,PF2=1
args=""
for n in 2 4; do for M in 384 1152 16384 20480 65536 262144 589824; do args="$args 256,$n,$M,$F 256,$n,$M,$F,X2=1"; done; done
WATCHDOG=800 $W/b2511/dev.sh -x -g 2 -t 900 s90-transition-inf python b200cfg/s90t.py $args 2>&1 | grep -v "Warning\|warn\|REG\|err max"
