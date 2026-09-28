# Session handoff: local H100 development

## Current execution target (2026-09-27)

The user terminated the Vast instance to save money. Do not connect to Vast,
restart synchronization, or rent/restart an instance. Its instructions below
are historical. GPU development now runs on this cluster through Slurm with
`--partition=h100 --account=cssb --qos=normal_h100` (priority 100, below
`cssb_h100` priority 1000). Start with one `gpu:h100:1`; do not run GPU work or
heavy CUDA compilation on the login node. See
[docs/operations/local-h100.md](docs/operations/local-h100.md).

## Historical Vast handoff

Before remote/GPU work, read [docs/operations/vast-h100.md](docs/operations/vast-h100.md).
It contains SSH, activation, synchronization, locks, result locations and measured limits.

- This local checkout is the authoritative source; do not automatically merge remote edits.
- Use `scripts/vast-sync.sh push [NEW_FILE...]` between jobs. New untracked files must be named explicitly. Never copy credentials, local environments, or use `--delete`.
- Run GPU jobs via `scripts/vast-sync.sh run 0|1 COMMAND...`. It holds a shared source lock and an exclusive GPU lock. A busy lock returns failure; inspect the other session's work, do not bypass it or kill its jobs.
- Do not write shared remote source while jobs run. Keep experiments isolated and record source identity/configuration and full F+B timing.
- Pull results with `scripts/vast-sync.sh pull`; automatic result pulls may already be running. The watcher ends at Seoul midnight, not indefinitely.
- NCU is unavailable (`ERR_NVGPUCTRPERM`); the user explicitly chose to proceed without it. Use CUDA events/graphs and activity profiling; do not claim hardware-counter or SOL results.
- Preserve existing work, validated forward paths, and strict gradient/graph/sanitizer gates. Experimental wins are not automatically production dispatch changes.
- Do not restart, stop, destroy, or extend the rented instance as part of this workflow.
