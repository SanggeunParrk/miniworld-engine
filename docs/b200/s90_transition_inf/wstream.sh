#!/bin/bash
W=/NHNHOME/WORKSPACE/26mohw002_A/psk6950; . $W/b2511/env.sh; export PATH=$ENV/bin:$PATH CUDA_HOME=$ENV
cd $W/b2511/agents/s90-transition-inf
$W/b2511/dev.sh -x -g 2 -t 400 s90-transition-inf python b200cfg/wstream.py 2>&1 | grep -v "Warning\|warn"
