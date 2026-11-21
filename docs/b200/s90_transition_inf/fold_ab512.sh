#!/bin/bash
W=/NHNHOME/WORKSPACE/26mohw002_A/psk6950; . $W/b2511/env.sh
cd $W/b2511/agents/s90-transition-inf
WATCHDOG=800 $W/b2511/dev.sh -x -g 2 -t 900 s90-transition-inf python b200cfg/fold_ab512.py 512,2,65536 512,2,147456 512,2,262144 512,2,409600 512,2,589824 512,4,65536 512,4,147456 512,4,262144 512,4,409600 512,4,589824 2>&1 | grep ratio
