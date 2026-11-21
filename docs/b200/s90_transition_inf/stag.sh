#!/bin/bash
W=/NHNHOME/WORKSPACE/26mohw002_A/psk6950; . $W/b2511/env.sh
cd $W/b2511/agents/s90-transition-inf
for f in "" "-DSTAG=1" "-DSTAG=2" "-DSTAG=3" "-DSTAG=4"; do
  [ -z "$f" ] && fl="-DSTAGOFF=1" || fl="$f"
  WATCHDOG=300 $W/b2511/dev.sh -x -g 2 -t 600 s90-transition-inf python b200cfg/ab384.py $fl -- 384,2,262144 384,2,1048576 384,2,2359296 384,4,262144 384,4,1048576 2>&1 | grep median
done
