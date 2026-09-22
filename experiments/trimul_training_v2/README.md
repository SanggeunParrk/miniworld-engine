# Latest TriMul training and development archive

Start with the [2.0.0 release map](../../docs/releases/2.0.0.md) and the
[current results](runs/trimul_cuda_widths_opt_20260923/README.md).

**Selected entry:** `runs/trimul_training_current.py`; **standalone validation:**
`bash experiments/trimul_training_v2/env.sh python -B experiments/trimul_training_v2/verify_release.py --index 2`.
Run from the repository root with a CUDA 12.9 compiler and PyTorch cu128 environment.
The environment script takes the Python executable from PATH. No compiled binary
is committed. D128 is tuned; further optimization of the other widths is deferred.

`runs/` preserves the dependency chain and research iterations from Sept 19–23.
The dated directories and original JSON paths are historical evidence. Start from
the selected entry, not an older `selected.py` or a historical report generator.
Some old job/deployment scripts record machine-specific paths; they are retained
as records, not advertised as portable commands. `verify_release.py` explicitly
rejects reads from the original workspaces while validating the selected entry.

`env.sh` deliberately isolates the pinned benchmark engine under
`runs/trimul_sm90_parity_20260917/engine/src`. It does not modify the installed
engine or its automatic dispatch. Keep separate processes for the capsule and
production engine. Training expects one visible full H100, B1, BF16, L384/768,
width64/128/256/384/512 and explicit mask/dropout scales. Plans are stream-bound;
the autograd wrapper allocates independent state per forward.

`original-sources.json` records imported source hashes before packaging changes.
`MANIFEST.json` records the actual packaged files. Large profiler captures,
compiler caches, binaries and transient job logs are omitted. Compact timings,
configurations, NCU summaries, selected sanitizer logs and diagrams are retained.
[Directory index](INDEX.md) lists the archived iterations.

We previously developed inference kernels, but Anthropic's implementation was
better on the measured workloads. This work inherits it and adds training.
See the root third-party notices and preserved upstream licenses. Research
status and unsuccessful candidates are kept explicit; no universal speedup or
SoL90 claim is made.
