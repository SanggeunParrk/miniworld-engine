set -e
for kernelcheck in memcheck racecheck synccheck; do
 timeout 240 bash runs/anthropic_adoption_20260919/env.sh compute-sanitizer --tool "$kernelcheck" --error-exitcode 42 --kernel-name 'kernel_substring=front_' python -u -B runs/anthropic_b7b12_fusion_20260920/sanitize_front.py --length 384 >"runs/anthropic_b7b12_fusion_20260920/final-${kernelcheck}-L384.log" 2>&1
done
timeout 240 bash runs/anthropic_adoption_20260919/env.sh compute-sanitizer --tool memcheck --error-exitcode 42 --kernel-name 'kernel_substring=front_' python -u -B runs/anthropic_b7b12_fusion_20260920/sanitize_front.py --length 384 --count 66 --part 1 >runs/anthropic_b7b12_fusion_20260920/final-memcheck-split-L384.log 2>&1
