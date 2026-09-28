# Local H100 D128 compact bias experiment

User ended the Vast rental and requested development on local H100s at lower
QoS. All GPU work uses `h100`, account `cssb`, QoS `normal_h100` (priority 100,
versus `cssb_h100` priority 1000), one GPU, bounded wall time. No Vast access.

Candidate copies current `bias_fusion.cu` into this isolated directory. Eight-row
bias sums remain FP32, are rounded once to BF16 for HBM storage, and are reduced
in FP32. dK/dV math, barriers, row group, Q pipeline, forward and the rest of
backward are unchanged. L768 partial buffer is 452,984,832 versus 905,969,664
bytes; combined write/read savings are 905,969,664 bytes per backward. L384
saves 113,246,208 write/read bytes. BF16 storage retains FP32-like exponent range.

This changes bias-gradient rounding, so timing alone is insufficient. Qualification
includes an FP64 core oracle, exact dK/dV parity, high-magnitude dO, masks/zero
fixtures, full parameter/input gradients, changed-state graph replay, live
dropout, actual native dispatch, and sanitizer checks. Production files and
dispatch are untouched; the harness substitutes the extension only in-process.

`run.sbatch` submitted as job 19731. It builds the isolated extension and runs
paired L768 starting/ending full F+B comparison, 90 samples each per arm. JSON
results preserve source hashes, traces, correctness and all samples. Environment
uses the existing local Python 3.10/cu128 environment and CUDA 12.9 toolchain.

Run commands:

```
sbatch experiments/triattn_local_d128/run.sbatch
bash experiments/triattn_local_d128/env.sh python experiments/triattn_local_d128/stress.py
```

The second command must itself be executed within a Slurm allocation.

Completed result: full F+B improves 1.79–1.95% on both lengths/directions.
Qualification jobs 19732 and 19737 passed. Job 19734 passed whole-module
memcheck and eight matched-dropout comparisons; its later broad racecheck was
cancelled and replaced by targeted native checks in 19737. Full details:
`docs/reports/triattn-local-d128-compact-20260927.md`. Production dispatch unchanged.
