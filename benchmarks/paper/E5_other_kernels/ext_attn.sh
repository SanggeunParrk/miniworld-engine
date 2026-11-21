#!/bin/bash
# The ext_attn stage of E5 on one card (after the card's environment is set up as for the other E5 stages: python / PYTHONPATH of the engine under test,
# PAPER_EXT_ATTN for the upstream trees; CARD = h100 | a100 | rtx5090): speed (REP = 1..3, a different job per repeat) or accuracy (KIND=accuracy).
#   REP=1 CARD=h100 ext_attn.sh            # records/<card>/ext_attn.r1.jsonl (resumable: a job stops after BUDGET seconds, chain it until ext_attn.r1.jsonl.done exists)
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CARD=${CARD:?CARD=h100|a100|rtx5090} REP=${REP:-1} BUDGET=${BUDGET:-480} KIND=${KIND:-speed}
EXT=${EXT:-"gnops of3tri megafold bionemo_ir turbopf ligo_pwa megafold_tr"}
if [ "$KIND" = accuracy ]; then CFG=cfg_ext_attn_acc.yaml OUT=$HERE/records/$CARD/ext_attn_acc.r$REP.jsonl; else CFG=cfg_ext_attn.yaml OUT=$HERE/records/$CARD/ext_attn.r$REP.jsonl; fi
# shellcheck disable=SC2086
python "$HERE/run_ext_attn.py" "$HERE/$CFG" --kind "$KIND" --out "$OUT" --budget-seconds "$BUDGET" --ext $EXT $EXTRA
