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
