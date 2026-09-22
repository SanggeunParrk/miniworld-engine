set -e
bash runs/anthropic_adoption_20260919/env.sh python -u -B runs/anthropic_b7b12_fusion_20260920/compare_front.py --length 384 --sources front_rounding --output paired-qualified-L384.json
bash runs/anthropic_adoption_20260919/env.sh python -u -B runs/anthropic_b7b12_fusion_20260920/integrate_front.py --length 384 --output full-qualified-L384.json
