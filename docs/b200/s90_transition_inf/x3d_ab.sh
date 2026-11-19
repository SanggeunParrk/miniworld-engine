#!/bin/bash
W=/NHNHOME/WORKSPACE/26mohw002_A/psk6950; . $W/b2511/env.sh
cd $W/b2511/agents/s90-transition-inf
WATCHDOG=800 $W/b2511/dev.sh -x -g 2 -t 900 s90-transition-inf python b200cfg/x3d_ab.py 384,2,65536 384,2,147456 384,2,262144 384,2,589824 384,4,65536 384,4,147456 384,4,262144 2>&1 | grep -v "Warning\|warn\|miniworld-engine build"
