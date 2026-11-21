#!/bin/bash
W=/NHNHOME/WORKSPACE/26mohw002_A/psk6950; . $W/b2511/env.sh
cd $W/b2511/agents/s90-transition-inf
for f in "-DSQBASE=1" "-DTBW=1" "-DTBA=1" "-DTBE=1" "-DTBW=1 -DTBA=1" "-DTBW=1 -DTBA=1 -DTBE=1"; do
WATCHDOG=300 $W/b2511/dev.sh -x -g 2 -t 400 s90-transition-inf python b200cfg/sqprobe.py $f -- 2,262144 4,409600 2>&1 | grep "squeeze"
done
