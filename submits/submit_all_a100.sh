#!/bin/bash
# Submit the FULL A100 baseline bench sweep as independent SLURM jobs (detached: they run
# on compute nodes and survive logout / client shutdown). One job per module target + one
# job for the whole kernel-level sweep. They queue and drain overnight across the A100
# partition; check with `squeue -u $USER`. Logs land in benchmarks/{modules,kernels}/.
#
# This is a BASELINE pass on the current tree -> surfaces per-target A100 crashes / OOM /
# arch misroutes / perf-vs-baseline, the empirical half of the review. Login-node-safe:
# it only calls sbatch (no pixi/python/GPU here).
set -uo pipefail
cd /home/psk6950/miniworld-engine

MODULE_TARGETS=(
  transition
  triangle_multiplication
  bias_only_attention
  triangle_attention
  conditioned_transition
  adaptive_layernorm
  augmented_attention_token
  augmented_attention_atom
)

echo "=== submitting module bench jobs ==="
for t in "${MODULE_TARGETS[@]}"; do
  jid=$(sbatch --parsable -J "mwk_${t}" --export=ALL,BENCH_TARGET="$t" submits/run_bench_a100.sbatch)
  echo "  $t -> job $jid"
done

echo "=== submitting kernel-level bench sweep ==="
kjid=$(sbatch --parsable submits/run_kbench_a100.sbatch)
echo "  kernels -> job $kjid"

echo ""
echo "all submitted. queue:"
squeue -u "$USER" -o "%.10i %.22j %.8T %.10M %R"
