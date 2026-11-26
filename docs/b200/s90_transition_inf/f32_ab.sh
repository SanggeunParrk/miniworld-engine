#!/bin/bash
W=/NHNHOME/WORKSPACE/26mohw002_A/psk6950; . $W/b2511/env.sh
cd $W/b2511/agents/s90-transition-inf
WATCHDOG=800 $W/b2511/dev.sh -x -g 2 -t 900 s90-transition-inf python b200cfg/f32_ab.py 128,2,16384 128,2,65536 128,2,147456 128,2,262144 256,2,16384 256,2,65536 256,2,147456 64,2,16384 64,2,65536 2>&1 | grep -v "Warning\|warn\|foldjet-engine build"
