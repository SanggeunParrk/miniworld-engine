#!/bin/bash
W=/NHNHOME/WORKSPACE/26mohw002_A/psk6950; . $W/b2511/env.sh
cd $W/b2511/agents/s90-transition-inf
MW_FWD512=1 WATCHDOG=300 $W/b2511/dev.sh -g 2 -t 400 s90-transition-inf python b200cfg/diag512.py 2>&1 | grep -v "Warning\|warn\|foldjet-engine build"
