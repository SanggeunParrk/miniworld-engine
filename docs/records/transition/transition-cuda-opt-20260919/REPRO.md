# Reproduction

The scripts in `repro/` are exact archived copies. They expect the original run
layout and installed engine, CUDA 12.9, CUTLASS/MathDx, PyTorch and Triton environment.
Use the original workspace run directory:

`/home/psk6950/MiniWorld/runs/transition_cuda_opt_20260919`

- `validate_final.py --d D`: independent values/gradients, identity and graph checks.
- `compare_grad_baseline.py`: archived old binary versus integrated new operation.
- `compare_modules.py --d D --length L`: the existing module benchmark fixture.
- `sanitize.py --file final-selections.json --production`: direct kernel launches;
  run under Compute Sanitizer memcheck or racecheck.
- `audit_native.py --file production-resolved.json`: disassemble actual binaries.

Run on node02 in a Slurm GPU allocation. The old baseline binaries in the recorded
paths correspond to the included old source snapshot; they must be rebuilt from
that snapshot if the local extension cache is unavailable. SASS/PTX/NCU binary
artifacts remain in the original run directory; JSON and text summaries are
archived here. This is a workspace reproduction record, not a standalone package.
