#!/bin/bash
W=/NHNHOME/WORKSPACE/26mohw002_A/psk6950; . $W/b2511/env.sh
cd $W/b2511/agents/s90-transition-inf
for fl in "-DTBX" "-DTBEPI" "-DTBX -DTBEPI -DTBSW -DTBB -DTBWS"; do
echo "FLAGS $fl"
MW_FWD512_FLAGS="$fl" WATCHDOG=300 $W/b2511/dev.sh -x -g 2 -t 400 s90-transition-inf python b200cfg/fused512_ab.py 512,2,262144 2>&1 | grep ratio
done
