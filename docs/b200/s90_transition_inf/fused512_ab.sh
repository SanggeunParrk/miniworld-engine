#!/bin/bash
W=/NHNHOME/WORKSPACE/26mohw002_A/psk6950; . $W/b2511/env.sh
cd $W/b2511/agents/s90-transition-inf
MW_FWD512_FLAGS="-DNBS_=3 -DTBB" WATCHDOG=800 $W/b2511/dev.sh -x -g 2 -t 900 s90-transition-inf python b200cfg/fused512_ab.py 512,2,262144 2>&1 | grep ratio
