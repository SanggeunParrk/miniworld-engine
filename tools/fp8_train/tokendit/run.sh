#!/bin/bash
# Token DiT FP8-training step benchmark: one CUDA graph per variant (fwd + bwd, weight pack + FP8 weight quantization inside),
# same process, interleaved with the 3.0.1 default captured twice (same-cubin control); cool = median of 5 interleaved replays
# after 2 s idle, sustained = mean over 3 x 5 back-to-back replays. Run on 1 H100 (sbatch), from the repo root:
#   PIN_TREE=$PWD OUT=speed.jsonl VARIANTS="default=;control=;bytes=bytes;rec=expand,nox,hsave;fast=expand,squeeze,nox,hsave" \
#   FOLDJET_ENGINE_TOKEN_DIT_FP8_TRAIN_MIN_ROWS=0 python tools/fp8_train/tokendit/speed.py 48:384 48:768 48:256 48:128 8:384
# (MIN_ROWS=0 measures the FP8 GEMMs at every M; leave it unset to measure the shipped gate.) Rows go to ../res/$OUT (create a res/ dir).
