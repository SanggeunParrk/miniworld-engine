#!/bin/bash
R=/home/psk6950/miniworld-engine-dit2/experiments/token_dit_overlap/vs_anthropic/run.sh
E=/home/psk6950/miniworld-engine-dit2/experiments
echo "== test (new)"; $R $E/token_dit_fused/tdit/cuda_core/test_core.py
echo "== det L768 (new)"; $R $E/token_dit_overlap/det_core2.py --length 768 --reps 300 | tail -4
echo "== det L384 (new)"; $R $E/token_dit_overlap/det_core2.py --length 384 --reps 300 | tail -4
echo "== cmp new"; $R core_vs_apb.py
echo "== cmp old"; ATTN_DEFS="QWID=64 DESCB=0" $R core_vs_apb.py
echo "== cmp 48 only"; ATTN_DEFS="DESCB=0" $R core_vs_apb.py
echo "== block bench (new)"; $R $E/token_dit_fused/bench.py
