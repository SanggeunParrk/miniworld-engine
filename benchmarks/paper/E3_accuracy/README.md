# E3: accuracy against float64

`miniworld_engine.measure.accuracy` on the E1 cases (every requirements row, all lengths, minimum depth / samples): the module is built once with
seeded random weights; gold = the `pytorch` backend in float64; per tensor (outputs, and in `train` the gradients of every floating input and
parameter) `err = max|x - gold| / max|gold|`; `tol = max(err_pytorch, fixed[dtype])` with `fixed` = bf16 / master 8e-3, float32 1e-5 (the
judge's standard, `cfg_*.yaml`); `err <= tol` GREEN, `<= 5 tol` YELLOW, beyond RED; a call's light is its worst tensor. External
backends (cueq, anthropic) are measured the same way, their light is informational. The Anthropic stage runs on the kit stack (as E1).
Deterministic: one run per card (no repeats).

    cd benchmarks/paper && paper_common/chain.sh 30 E3_accuracy/h100.sbatch ; python E3_accuracy/collect.py --card h100

Output `E3_<card>.csv`: common columns (ms / peak columns empty, `status` = `measured | unsupported | skipped | error`) + `stack,row,variant,samples,depth,light,err_backend,err_pytorch,tol,tensor,external`.
`paper_common/tables.py` makes `accuracy_lights.csv` (GREEN / YELLOW / RED per backend and card).
