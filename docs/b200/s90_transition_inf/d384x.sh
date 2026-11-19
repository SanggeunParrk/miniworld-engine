#!/bin/bash
W=/NHNHOME/WORKSPACE/26mohw002_A/psk6950; . $W/b2511/env.sh
cd $W/b2511/agents/s90-transition-inf
for f in "" "XFIRST=1" "XEF=1" "XFIRST=1 XEF=1"; do
echo "flags: $f"
WATCHDOG=300 $W/b2511/dev.sh -x -g 2 -t 400 s90-transition-inf python b200cfg/pf384.py 2,2359296 $f 2>&1 | grep -v "Warning\|warn\|exe_args\|USDT\|^\[miniworld" | grep "MMA expand\|LN x landed\|LN done\|rror"
done
