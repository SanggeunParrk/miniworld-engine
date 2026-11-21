# GPU MODE TriMul competition kernels (E5 baseline `gpumode`)

- Upstream: https://github.com/arseniivanov/trimul @ 558b9f26 (MIT, Copyright (c) 2025 Arseni Ivanov): the Triton submissions `triton_a100.py`
  (A100: 1st), `triton_b200.py` (B200: 2nd), `triton_h100.py` (H100: 3rd) of the GPU MODE "TriMul outgoing forward" competition
  (https://www.gpumode.com/v2/leaderboard/496). Also redistributed at https://huggingface.co/kernels-community/trimul_gpumode (no license declared there).
- Task (reference.py): AF3 outgoing TriMul, LN_in, left / right projections + gates, mask `[B, L, L]`, contraction, LN_out, output gate in the
  HIDDEN width, `to_out`; fp32 input, forward only, no residual, `dim` 128..768, `hidden_dim` 128. Differences from the engine's update and how
  the adapter maps them: see `ext_gpumode_trimul.py` (output-gate placement is re-expressed with a zero gate weight and a doubled W_o; the engine's gate
  and residual are applied by the adapter, so the measured time includes one extra LN + d x d GEMM).
- Not fetched: `kernel_global` of the Hugging Face repo (adapted from https://github.com/davidberard98/gpumode-trimul, which carries no license file: not used,
  nothing copied). Other winning code: the TTT-Discover kernel (H100, 1161 us) is printed in the paper's appendix only (the repo
  test-time-training/discover, MIT, has the environment but not the kernel); the msuiche blog post is a personal write-up without a license.
- Install: `third_party/gpumode_trimul/fetch.sh [TARGET]` (raw files, sha256 pinned). Patches: none.
- Adapter: `benchmarks/paper/E5_other_kernels/ext_gpumode_trimul.py`.
