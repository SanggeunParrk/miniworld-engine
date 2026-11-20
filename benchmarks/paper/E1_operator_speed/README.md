# E1: speed of every operator class

All 39 operator classes, every row of `src/miniworld_engine/requirements/requirements_module.csv`, inference (`eval`) and training (`train`),
dtypes of the row (`master`, `float32`, ...), all lengths of a row (`lengths: all`), minimum MSA depth / samples (the judge's canonical speed
run, `measure/configs/speed.yaml`; iterations 100 instead of 50). Backends: `pytorch` (the reference, measured in the same process), `cuda`, `triton`,
`cueq` (stage `engine`) and `anthropic` (stage `kit`, through the card's kit stack: H100 `kit_stack.sh`, A100 `kit_stack_compat.sh` with
`MINIWORLD_ANTHROPIC_ROOT` = a cu129 / py312 release root, cssb3 `~/w45-jobs/upstream`; the RTX 5090 runs it in the engine env).
The paper numbers are ratios against PyTorch, geometric mean per class over the same cases (`paper_common/tables.py`).

Files: `cfg_engine.yaml`, `cfg_kit.yaml`, `h100.sbatch`, `a100.sbatch`, `rtx5090.sh`, `collect.py`.

    cd benchmarks/paper
    REP=1 paper_common/chain.sh 12 E1_operator_speed/h100.sbatch      # REP=1,2,3: three repeats (different Slurm jobs); resumable
    python E1_operator_speed/collect.py --card h100                    # in a CPU job -> E1_h100.csv

Output `E1_<card>.csv`: the common columns of `../README.md` + `stack,row,variant,samples,depth,repeats,timed_as`
(`timed_as: eager` = a case whose step cannot be captured in a graph, timed eagerly by the measurer: an upper bound, flagged).
Smoke: add `EXTRA="--module Transition --lengths 128 --max-cases 2"`.
