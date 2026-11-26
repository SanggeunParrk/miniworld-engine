#!/bin/bash
W=/NHNHOME/WORKSPACE/26mohw002_A/psk6950; . $W/b2511/env.sh
cd $W/b2511/agents/s90-transition-inf
WATCHDOG=500 $W/b2511/dev.sh -g 2 -t 700 s90-transition-inf python b200cfg/tcheckt.py 256,2,65536,bf16 256,4,16384,bf16 256,2,16384,bf16 64,2,65536,bf16 2>&1 | grep -v "Warning\|warn\|foldjet-engine build"
