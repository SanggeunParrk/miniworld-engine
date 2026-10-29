# Protenix-v2 Anthropic kit (fast) at 128-2048 tokens, timed with FoldForge's forward definition (2026-10-29)

The comparison number for FoldForge is `model_seconds_warm` (`foldforge/models/execution.py::measured_forward`): `torch.cuda.synchronize()`,
one complete model call (trunk passes + diffusion sampler + confidence head), `torch.cuda.synchronize()`. The kit's matching number is
its `PHASE ... fwd_s` line: trunk, sampler and confidence, each phase CUDA-synchronized, summed. The kit's `Model forward time` log line is
NOT comparable: it also covers the output handling after the forward (0.13-0.27 s more at 128 tokens, 2.6-3.0 s at 2048; table below), and it
was the number earlier comparisons used for the v1 kit.

Setup: one H100 80GB HBM3 (node02), job 39129. Kit `protenix_v2` as installed in `~/foldforge-scratch/anthropic-kits/protenix_v2` (pinned
weights, `run.sh pred --config h100 --mode fast`), run as its reports ran it: Ubuntu 22.04 container with the CUDA 13.0 forward-compat libcuda
on driver 575 and the conda GCC runtime (`cu130run.sh`, `CU130_EXTRA_LD`); `[protenix-opt] ACTIVE mode=fast` in every log. Without the
container the kit fails its driver / CXXABI gates and runs stock (`NOT ACTIVE`, exit 3) -- that attempt is not reported.
Input: 5I28 azurin (128 residues) x N chains, N = 1, 2, 4, 8, 12, 16 (128-2048 tokens); no MSA, no templates; 4 recycles (`-c 4`), 200 steps,
5 samples, bf16; seeds 0-3 in one process, seed 0 cold, seeds 1-3 warm. Logs: `~/miniworld-engine-scratch/kitv2sweep/logs/kit_fast_x<N>.log`.

## Result (seconds, warm = median of seeds 1-3)

| N | tokens | trunk | sampler | confidence | **fwd_s (= FoldForge definition)** | `Model forward time` (not comparable) | cold fwd_s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 128 | 0.126 | 0.414 | 0.052 | **0.600** | 0.78 | 3.650 |
| 2 | 256 | 0.320 | 0.579 | 0.054 | **0.966** | 1.16 | 4.507 |
| 4 | 512 | 1.020 | 1.030 | 0.062 | **2.149** | 2.59 | 7.505 |
| 8 | 1024 | 4.735 | 2.130 | 0.207 | **7.197** | 8.07 | 9.866 |
| 12 | 1536 | 12.655 | 3.463 | 0.509 | **16.901** | 18.30 | 19.329 |
| 16 | 2048 | 26.050 | 5.563 | 7.312 | **41.0 (40.586 / 40.898 / 41.996)** | 43.49 | 47.577 |

The trunk dominates from 1024 tokens on (66 % at N = 8, 64 % at N = 16); at N = 16 the confidence head jumps to 7.3 s. At 128 tokens the
sampler is 69 % of the forward.

Note the kit runs `--use_template false`; FoldForge's Protenix-v2 spec with `template: {}` still runs its template stack (four empty
templates, batched) -- 2.7 ms of a 128-token eager forward, so the definitions differ by at most that.
