# E5: external triangle-multiplication kernels

Backends of the measurer for `TriangleMultiplication` (outgoing / incoming, LN + gating, bf16 master layout), installed at run time exactly as
`externals.py` does for the attention libraries: nothing under `src/` changes. Each adapter is its own file; `ext_trimul_registry.py` lists them
(one line each; the coordinator merges it into `externals.py`); `run_ext_trimul.py` is the launcher that routes `--ext` names to the registry.

| backend | library (third_party/) | license | function served | notes |
|---|---|---|---|---|
| `fast_trimul` | fast_trimul 3.0.6 (CuTe DSL) | Apache-2.0 | whole update (LN, gated projections, contraction, LN_out, output gate) | fp16 kernels (bf16 cast), L % 8 == 0, no mask, training = kernel forward + torch-recompute backward |
| `gpumode` | arseniivanov/trimul, GPU MODE competition winners (Triton; a100 / h100 / b200 file by card, h100 file on sm120) | MIT | outgoing forward only | output-gate placement differs from the engine's: remapped (zero gate weight, 2 x W_o), engine gate + residual by the adapter (extra LN + GEMM in the time) |
| `geonn` | AMD geometric-nn-ops 1.0.0 (Triton) | Apache-2.0 | whole update, cuEquivariance signature | forward only (backward = its PyTorch path: training recorded unsupported); validated by AMD on MI300 only |

Checked and not connected: Hugging Face `kernels-community/trimul_gpumode` `kernel_global` (derived from a repository without a license: not used; its per-GPU files are the
`gpumode` ones above), the TTT-Discover H100 kernel (printed in the paper's appendix only), TurboPairFormer (no public code found), cuEquivariance / Anthropic (E1 baselines).

Install on a card: `third_party/<entry>/fetch.sh [TARGET]`; the adapters look in `$PAPER_EXT_PATH` and `third_party/<entry>/upstream`.
Run: `python E5_other_kernels/run_ext_trimul.py E5_other_kernels/cfg_trimul.yaml --kind speed --out E5_other_kernels/records/<card>/ext_trimul.r<N>.jsonl --ext fast_trimul gpumode geonn`
(`--kind accuracy` for the accuracy against the fp64 reference).

## Records and scope

`records/<card>/ext_trimul.r{1,2,3}.jsonl` (three repeats, a different job each), `records/ext_trimul_table.md` (`collect_trimul.py --e5`), `records/acc/ext_trimul_acc.<card>.jsonl`
(`acc_trimul.py --grads`: error against the module's own fp64 PyTorch result). Reported scope: TriMul outgoing AF3 setup (d_pair=128, p_drop 0.25), L 512 / 1024 / 2048,
eval and train where the backend serves them, 20 timed iterations (`requirements_trimul_e5.csv`, `cfg_trimul.yaml`). The H100 and A100 records were taken with that config;
the 5090 records were taken before the scope reduction on the wider E2 grid with 100 iterations (same cells included; `ext_trimul.r*.jsonl.meta.json` has the settings): the
cells of the reported scope are reused. Not connected as separate operator classes: LayerNorm / gate kernels of fast_trimul are internal helpers of its fused module (no
module-level API to wrap), the other libraries expose only the whole update.

5090 train L=2048 (OOM in the three repeats on the shared GPU): one extra run, records/rtx5090/fill_l2048_train.r1.jsonl (20 iterations): pytorch 468.3 ms, cuda 66.9 ms, fast_trimul 516.5 ms (x7.72 of cuda).
