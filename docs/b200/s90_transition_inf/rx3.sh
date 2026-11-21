#!/bin/bash
W=/NHNHOME/WORKSPACE/26mohw002_A/psk6950; . $W/b2511/env.sh
cd $W/b2511/agents/s90-transition-inf
M=1048576
WATCHDOG=250 $W/b2511/dev.sh -x -g 2 -t 400 s90-transition-inf python b200cfg/s90t.py 64,2,$M,ROLES=1 64,2,$M,ROLES=1,RX=1 2>&1 | grep -v "Warning\|warn\|REG\|err max"
