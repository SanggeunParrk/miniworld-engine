#!/bin/bash
W=/NHNHOME/WORKSPACE/26mohw002_A/psk6950; . $W/b2511/env.sh
cd $W/b2511/agents/s90-transition-inf
bash b200cfg/s90m3132.sh pf384 cuda eval 512 768 1024 1536 2048
bash b200cfg/s90m3132.sh pf384b cuda eval 768 1536
