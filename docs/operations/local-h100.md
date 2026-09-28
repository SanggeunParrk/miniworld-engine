# Local H100 development (2026-09-27)

The user ended the Vast rental and requested local development with lower QoS.
Use Slurm from `/home/psk6950/miniworld-engine`; the login node has no GPU.

```sh
sinfo -p h100
squeue -u "$USER"
sbatch --partition=h100 --account=cssb --qos=normal_h100 \
  --gres=gpu:h100:1 --cpus-per-task=8 --mem=64G --time=00:30:00 \
  --output=.bench/transition-wide-local/job-%j.log \
  experiments/transition_wide_fusion/run.sh profile_stages
```

Create the output directory before submission. `normal_h100` has priority 100;
the higher `cssb_h100` QoS has priority 1000. Both are allowed for this account.
Use one GPU initially and let Slurm assign it. Check live availability rather
than assuming node02 remains idle. Do not override another job's allocation.

The experiment wrapper uses the existing MiniWorld Python 3.10 `cu128`
environment, CUDA toolkit 12.9, this checkout's `src`, and isolated caches under
`.bench/transition-wide-local`. It does not change the shared environment.
Record the actual device, loaded runtime, source hashes, and complete F+B
timing. Preserve existing forward paths and qualify full gradients,
changed-input CUDA Graph replay, and sanitizers before production dispatch.

Vast results remain at `.bench/vast-20260927/results`. Do not compare them to
local candidate timings as a paired speedup: measure the baseline locally too.
The Vast result watcher was already absent when switching to local Slurm;
its old PID file alone is not evidence of a live process.

The local wide Transition experiment is
[`experiments/transition_wide_fusion`](../../experiments/transition_wide_fusion/README.md).
Its selected D384/D512 tail fuses LayerNorm backward, residual addition and
compact affine partials. D256 retains the existing path. It is an explicit
experiment, not a global dispatch change. Use its qualification records before
adopting it; failed broad-fusion candidates and their logs are retained.
