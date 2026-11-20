# E2: TriMul / TriAttn sweeps

`TriangleMultiplication` (outgoing) and `TriangleAttention` (starting node, 32 channels per head: `n_head = d/32`), dtype `master`:
- L sweep at d=128: L in 128, 256, 384, 512, 640, 768, 1024, 1536, 2048;
- D sweep at L=768: d in 64, 128, 256, 384, 512 (TriAttn heads 2, 4, 8, 12, 16; d=128 is the L sweep's L=768 point).
Inference and training. Backends as E1 (`pytorch`, `cuda`, `triton`, `cueq`; `anthropic` on the kit stack).

`requirements_e2.csv` is the extra requirements CSV (the format of `requirements_module.csv`, precedent: the TriAttn d=512 sweep row): it
holds the whole grid because TriAttn d=512 is used by no predictor (no requirements row; the judge never measures it) and the requirements
rows train only up to L=768. Cases beyond the card's memory are recorded `skipped` (OOM), not removed.

    cd benchmarks/paper
    REP=1 paper_common/chain.sh 6 E2_sweeps/h100.sbatch ; python E2_sweeps/collect.py --card h100
    python paper_common/tables.py --card h100 --out OUT       # sweeps.csv for the paper scripts

Output `E2_<card>.csv` (columns as E1). Files: `cfg_engine.yaml`, `cfg_kit.yaml`, `requirements_e2.csv`, `h100.sbatch`, `a100.sbatch`, `rtx5090.sh`, `collect.py`.
