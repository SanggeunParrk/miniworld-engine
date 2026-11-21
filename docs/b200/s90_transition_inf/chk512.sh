#!/bin/bash
W=/NHNHOME/WORKSPACE/26mohw002_A/psk6950; . $W/b2511/env.sh
cd $W/b2511/agents/s90-transition-inf
WATCHDOG=300 $W/b2511/dev.sh -g 2 -t 400 s90-transition-inf python b200cfg/chk512.py 2,65536,0 2,65536,3 4,65536,0 4,98304,3 2>&1 | grep -v "Warning\|warn\|miniworld-engine build"
