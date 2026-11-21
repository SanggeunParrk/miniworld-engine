# E5: external triangle-multiplication kernels

Backends of the measurer for `TriangleMultiplication` (outgoing / incoming, LN + gating, bf16 master layout), installed at run time exactly as
`externals.py` does for the attention libraries: nothing under `src/` changes. Each adapter is its own file (`ext_<name>_trimul.py`); `externals.py` (`TRIMUL_REGISTRY`) lists them and installs them: `run_measure.py ... --ext fast_trimul gpumode geonn`.

| backend | library (third_party/) | license | function served | notes |
|---|---|---|---|---|
| `fast_trimul` | fast_trimul 3.0.6 (CuTe DSL) | Apache-2.0 | whole update (LN, gated projections, contraction, LN_out, output gate) | fp16 kernels (bf16 cast), L % 8 == 0, no mask, training = kernel forward + torch-recompute backward |
| `gpumode` | arseniivanov/trimul, GPU MODE competition winners (Triton; a100 / h100 / b200 file by card, h100 file on sm120) | MIT | outgoing forward only | output-gate placement differs from the engine's: remapped (zero gate weight, 2 x W_o), engine gate + residual by the adapter (extra LN + GEMM in the time) |
| `geonn` | AMD geometric-nn-ops 1.0.0 (Triton) | Apache-2.0 | whole update, cuEquivariance signature | forward only (backward = its PyTorch path: training recorded unsupported); validated by AMD on MI300 only |

Checked and not connected: Hugging Face `kernels-community/trimul_gpumode` `kernel_global` (derived from a repository without a license: not used; its per-GPU files are the
`gpumode` ones above), the TTT-Discover H100 kernel (printed in the paper's appendix only), TurboPairFormer (no public code found), cuEquivariance / Anthropic (E1 baselines).

Install on a card: `third_party/<entry>/fetch.sh [TARGET]`; the adapters look in `$PAPER_EXT_PATH` and `third_party/<entry>/upstream`.
Run (history; now one unified config): `python paper_common/run_measure.py E5_other_kernels/cfg_e5_all.yaml --kind speed --module TriangleMultiplication --out E5_other_kernels/records/<card>/<stage>.r<N>.jsonl --ext fast_trimul gpumode geonn` (the records of this file family were taken with the former `cfg_trimul.yaml` / `run_ext_trimul.py`, merged into `cfg_e5_all.yaml` / `externals.py`)
(`--kind accuracy` for the accuracy against the fp64 reference).

## Records and scope

`records/<card>/ext_trimul.r{1,2,3}.jsonl` (three repeats, a different job each), `records/ext_trimul_table.md` (table of the former `collect_trimul.py`; the unified collector is `collect.py`), `records/acc/ext_trimul_acc.<card>.jsonl`
(`acc_trimul.py --grads`: error against the module's own fp64 PyTorch result). Reported scope: TriMul outgoing AF3 setup (d_pair=128, p_drop 0.25), L 512 / 1024 / 2048,
eval and train where the backend serves them, 20 timed iterations (now the TriMul row of `requirements_e5_all.csv`, `cfg_e5_all.yaml`). The H100 and A100 records were taken with that config;
the 5090 records were taken before the scope reduction on the wider E2 grid with 100 iterations (same cells included; `ext_trimul.r*.jsonl.meta.json` has the settings): the
cells of the reported scope are reused. Not connected as separate operator classes: LayerNorm / gate kernels of fast_trimul are internal helpers of its fused module (no
module-level API to wrap), the other libraries expose only the whole update.

5090 train L=2048 (OOM in the three repeats on the shared GPU): one extra run, records/rtx5090/fill_l2048_train.r1.jsonl (20 iterations): pytorch 468.3 ms, cuda 66.9 ms, fast_trimul 516.5 ms (x7.72 of cuda).
