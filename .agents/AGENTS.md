# AGENTS

## CRITICAL: LOGIN NODE SAFETY

This repository is often opened from a cluster login node.

Do not run heavy or wide-scope commands on the login node.

Forbidden on the login node:
- Recursive scans outside this repository, including commands like `find $HOME ...`.
- Large filesystem walks under `$HOME`, parent directories, shared storage, or other repos.
- Benchmark runs, compilation, installs, profiling, or any GPU-dependent command.
- Long-running shell probes whose scope is broader than this repository.

Allowed on the login node:
- Small repo-local reads such as `rg`, `sed`, `nl`, `git status`, and file inspection inside this repo only.
- Narrow searches rooted at `<repo root>` only.

If GPU work, benchmarking, or heavy inspection is needed:
- Use `srun` or an allocated compute node.
- Keep login-node activity limited to lightweight repo-local inspection.

If there is any doubt, choose the safer option and do less on the login node.
