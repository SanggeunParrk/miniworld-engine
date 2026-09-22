set -e
for tokens in 384 768; do
 bash runs/anthropic_adoption_20260919/env.sh ncu --set full --cache-control none --clock-control none --profile-from-start off --kernel-name 'regex:front_b7b12' --force-overwrite -o "runs/anthropic_b7b12_fusion_20260920/final-warm-L${tokens}" python -u -B runs/anthropic_b7b12_fusion_20260920/profile_front.py --length "$tokens" >"runs/anthropic_b7b12_fusion_20260920/final-warm-L${tokens}.log" 2>&1
 bash runs/anthropic_adoption_20260919/env.sh ncu --import "runs/anthropic_b7b12_fusion_20260920/final-warm-L${tokens}.ncu-rep" --page raw --csv >"runs/anthropic_b7b12_fusion_20260920/final-warm-L${tokens}.csv"
done
