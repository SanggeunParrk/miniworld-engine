#!/bin/bash
#SBATCH -J width-verify
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH -o /home/psk6950/practice/miniworld-engine/verify_width.log
#SBATCH -t 1:00:00
cd /home/psk6950/practice/miniworld-engine
echo "### GPU 테스트 스위트 (지금까지 116개가 skip 되던 것들)"
.pixi/envs/default/bin/python -m pytest tests -q -p no:randomly 2>&1 | tail -6
echo
echo "### 폭 검증"
.pixi/envs/default/bin/python verify_width.py
