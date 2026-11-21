#!/bin/bash
W=/NHNHOME/WORKSPACE/26mohw002_A/psk6950; . $W/b2511/env.sh
cd $W/b2511/agents/s90-transition-inf
B=LNH=1
for M in 1048576 262144; do
WATCHDOG=250 $W/b2511/dev.sh -x -g 2 -t 400 s90-transition-inf python b200cfg/s90t.py 128,2,$M,$B,NXS_=3,NW_=2 128,2,$M,$B,RX=1,NXS_=3,NW_=2 128,2,$M,$B,RX=1,NXS_=2,NW_=4 128,2,$M,$B,RX=1,NXS_=2,NW_=3 2>&1 | grep -v "Warning\|warn\|REG\|err max"
done
