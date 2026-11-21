#!/bin/bash
W=/NHNHOME/WORKSPACE/26mohw002_A/psk6950; . $W/b2511/env.sh
cd $W/b2511/agents/s90-transition-inf
bash b200cfg/s90m3334.sh g512 cuda eval 256 384 512 640
bash b200cfg/s90m3334.sh g512b cuda eval 256 512 640
