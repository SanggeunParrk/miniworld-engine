#!/bin/bash
W=/NHNHOME/WORKSPACE/26mohw002_A/psk6950; . $W/b2511/env.sh
cd $W/b2511/agents/s90-transition-inf
MW_FWD512=1 WATCHDOG=300 $W/b2511/dev.sh -g 2 -t 400 s90-transition-inf python b200cfg/chk512f.py 2,256,0 2,2048,0 2,65536,0 2>&1 | grep -v "Warning\|warn\|foldjet-engine build"
