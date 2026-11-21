#!/bin/bash
W=/NHNHOME/WORKSPACE/26mohw002_A/psk6950; . $W/b2511/env.sh; export PATH=$ENV/bin:$PATH CUDA_HOME=$ENV
cd $W/b2511/agents/s90-transition-inf
for cfg in "16 384 1" "8 384 1"; do XFULL=1 $W/b2511/dev.sh -x -g 2 -t 300 s90-transition-inf python b200cfg/skel2.py $cfg; done
