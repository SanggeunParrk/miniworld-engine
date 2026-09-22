set -euo pipefail
R=/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919
python "$R/bench.py" --family opm_core --length 384 --width 256 --row carried --output "$R/results/opm_core-L384-carried.json" > "$R/logs/opm_core-L384-carried.log" 2>&1
python "$R/bench.py" --family opm_core --length 768 --width 256 --row carried --output "$R/results/opm_core-L768-carried.json" > "$R/logs/opm_core-L768-carried.log" 2>&1
python "$R/run_ncu.py" --lanes 1 --only opm_core,pwa,atom_window
cp "$R/profile-plan-0.json" "$R/profile-plan-msa.json"
