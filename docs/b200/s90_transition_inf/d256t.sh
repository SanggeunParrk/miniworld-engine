#!/bin/bash
W=/NHNHOME/WORKSPACE/26mohw002_A/psk6950; . $W/b2511/env.sh
cd $W/b2511/agents/s90-transition-inf
$W/b2511/dev.sh -x -g 2 -t 250 s90-transition-inf python b200cfg/pf3.py 256,2,589824,XPAR=1,WPAR=1,LN2=1 2>&1 | grep -v "Warning\|warn"
