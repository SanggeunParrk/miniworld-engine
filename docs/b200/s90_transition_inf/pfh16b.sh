#!/bin/bash
W=/NHNHOME/WORKSPACE/26mohw002_A/psk6950; . $W/b2511/env.sh
cd $W/b2511/agents/s90-transition-inf
for fl in "-DTBLN=2" "-DTBEPI=1" "-DTBLN=2 -DTBEPI=1 -DTBW=1"; do
echo "FLAGS $fl"
PF_FLAGS="$fl" PF_TAG="$(echo $fl | tr -d ' =-')" WATCHDOG=300 $W/b2511/dev.sh -x -g 2 -t 400 s90-transition-inf python b200cfg/pfh16.py 128,16384 256,16384 2>&1 | grep -E "^==|MMA sees|epilogue tile 0 end|last squeeze|SwiGLU c0"
done
