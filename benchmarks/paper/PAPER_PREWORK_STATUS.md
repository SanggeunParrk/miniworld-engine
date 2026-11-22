# Paper pre-work status (engine + FoldForge, four cards), 2026-11-22

Everything below is on `main`: engine `miniworld-engine` (this repo) and FoldForge `origin/main`. Numbers come from the CSVs; see `PAPER_DATA_SUMMARY.md` (engine, computed from `E*/E*_<card>.csv`) and the FoldForge `benchmarks/paper/F*/README.md` DATA_STATUS sections.

## Engine releases used by the paper

| release | content | PyPI / tag |
|---|---|---|
| 3.0.6 | 5090 AF3 NaN fix (Transition fp16 residual accumulator), long-L PWA / TokenDiT, harness | tag `v3.0.6` |
| 3.0.7 | H100 fp32 Transition with TF32 off on CUDA, bf16 opt-in for the bias-only pair-bias logits (fp32 default), B200 kernel set (source / JIT only) | tag `v3.0.7` |
| 3.0.8 | A100 / RTX 5090: OpenDDE `TokenDiTProtenix` `extra_pair_bias` on CUDA (was Triton) | tag `v3.0.8` |
| 3.0.9 | B200: TriAttn `ex2_poly2` upper clamp (non-finite on real AF3 / Chai-1), fp32 Transition `ieee_f32_sm100`, OpenDDE `extra_pair_bias` | tag `v3.0.9` |

There is no B200 kernels wheel: B200 rows are measured on source trees (labels in the CSVs, `engine_label`).

## Engine experiments (E1-E6)

| card | E1 speed | E2 sweeps | E3 accuracy | E4 FP8 | E5 external kernels | E6 memory |
|---|---|---|---|---|---|---|
| H100 | done | done | done | done | done (+ ds4sci eager correction) | done |
| A100 | done | done | done | n/a | done | done |
| RTX 5090 | done | done | done | n/a (no sm120 FP8 route) | done | done |
| B200 | done | done | done | done | done (+ fp64 accuracy) | done |

Timing: CUDA graph, cold L2, medians; the iteration count (100 or 20) does not change the quantity measured and is not a caveat. E5: AF3 setup (d_pair 128, 4 heads), L 512 / 1024 / 2048, inference and training, all op classes, 13 external candidates; cells without a candidate (OPM, LayerNorm) or without support are marked, not missing.

## FoldForge experiments (F-series)

| experiment | H100 | A100 | RTX 5090 | B200 |
|---|---|---|---|---|
| F1 / F2 FoldBench (AF3) | done (3.0.1; 32 targets > 2048 tokens re-run on 3.0.6); launch mode stated (multi, GPU-h = 2 x wall) | n/a | n/a | n/a |
| F3 5I28 time | 3.0.1 / 3.0.2 values (labelled) | 3.0.6, 33/33 | 3.0.6 (ESMFold2 family not measured: 16 GB host RAM) | done (boltz2 / chai1 / rosettafold3 per-size processes) |
| F4 27 targets | 3.0.1 (labelled; largest target 2008 tokens, outside the PWA hole) | 3.0.6 + OpenDDE 3.0.8, 22/22 | 3.0.6 + OpenDDE 3.0.8 | done, engine 291 ok / PyTorch 268 ok |
| F5 sample comparison | 3.0.1 | done | done | done (1,340 pairs) |
| F6 training speed | engine arm re-measured, Transition on CUDA (0.8121 / 0.7870 s per example) | 3.0.6 | 3.0.6 | done (8/8 configurations) |
| F7 finetune loss | PyTorch run 433 steps vs engine 1424: compared on the 433 common steps | n/a | n/a | n/a |
| F8 finetuned FoldBench | done | n/a | n/a | n/a |
| F9 inference limit | 3.0.6 | 22/22 | 9 predictors (ESMFold2 family not measured) | done (22 limits; chai1 n=40 CUDA IMA at 98 % reserved classified OOM) |
| F10 training limit | dropped from the paper | | | |
| F11 FP8 accuracy | done | n/a | n/a | n/a |

## Defects found by the end-to-end runs and fixed (all on `main`)

- 5090 AF3 NaN (3.0.6); H100 fp32 Transition on Triton (3.0.7); A100 / 5090 OpenDDE `extra_pair_bias` on Triton (3.0.8); B200 TriAttn non-finite, fp32 Transition and OpenDDE on Triton (3.0.9).
- Open, documented, not in the paper's experiments: B200 fp32 inference S=1, L=100 / 104 NaN (composed runner; bf16 unaffected); B200 multi-size CUDA-graph replay non-finite inside one process (workaround: one process per size); B200 judge tokens cannot be issued (`on_kit_stack`), B200 is not a supported release target.
- Memory: engine 3.0.8 keeps 18.6 MiB per OpenDDE TokenDiT block resident (0.44 GiB at 24 blocks); one 5090 cell (7wvs) therefore runs out of memory on 3.0.8 (documented in the F4 README).

## What the data supports (scope)

- Across the measured operator classes the engine CUDA backend is faster than PyTorch eager on every card (geometric means in `PAPER_DATA_SUMMARY.md`), and no surveyed public implementation in E5 is faster than the engine in any measured cell. Baselines faster than the engine are LayerNorm (declared exception) and cells within +-3 % noise; they are listed in `PAPER_DATA_SUMMARY.md` section 1.3 / 1.3a.
- The comparison set is the public implementations found by the survey (listed in `E5_other_kernels/E5_results.md` and `third_party/README.md`); claims should name that set. Non-public or unsurveyed implementations are outside the data.
- The wording of claims in the paper is the authors' decision; this file only records what was measured.
