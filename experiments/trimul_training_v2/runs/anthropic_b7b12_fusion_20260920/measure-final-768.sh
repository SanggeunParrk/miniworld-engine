set -e
bash runs/anthropic_adoption_20260919/env.sh python -u -B runs/anthropic_b7b12_fusion_20260920/compare_front.py --length 768 --sources front_rounding --output paired-qualified-L768.json
bash runs/anthropic_adoption_20260919/env.sh python -u -B runs/anthropic_b7b12_fusion_20260920/integrate_front.py --length 768 --output full-qualified-L768.json
