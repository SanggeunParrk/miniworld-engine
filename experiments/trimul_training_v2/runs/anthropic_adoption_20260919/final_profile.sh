set -euo pipefail
R=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919
python "$R/run_ncu.py" --lanes 1 --only adaln,swiglu,msa_ln,ln_linear,opm,gather,ln
cp "$R/profile-plan-0.json" "$R/profile-plan-extra.json"
python "$R/run_ncu.py" --lanes 1 --only triattn --modules
cp "$R/profile-plan-0.json" "$R/profile-plan-modules.json"
python "$R/bench.py" --family trimul --length 768 --width 128 --row tx_sm90a --output "$R/results/trimul-outgoing-L768-C128-tx_sm90a.json"
