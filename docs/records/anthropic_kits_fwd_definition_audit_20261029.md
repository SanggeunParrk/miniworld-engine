# Anthropic kits: which forward each recorded time covers, against FoldForge's definition (2026-10-29)

FoldForge's number is `model_seconds_warm` (`foldforge/models/execution.py::measured_forward`): `torch.cuda.synchronize()`, one complete
model call -- trunk passes, diffusion sampler, confidence head -- `torch.cuda.synchronize()`. Load, featurisation and output writing are
outside it. ESMFold2's ESM-C language-model forward is also outside it (FoldForge runs it before the model call; `ff_stage.py` reports it
as `LMTIME cold warm`).

The kit times compared so far are `warm_median` of the `fast` rows in `~/foldforge-scratch/anthropic-kits/results.jsonl` (timer rules:
`PROTOCOL.md` there; each kit's `REPORT.md`). Audited from those files and the run logs; nothing re-run except Protenix-v2
(`protenix_v2_kit_fwd_aligned_20261029.md`).

## Verdict per kit

| kit | timer behind the recorded `warm_median` | same span as FoldForge? |
|---|---|---|
| AF3 (Torch) | kit `ITEM ... forward_s`: synchronize, `A.inference()` = trunk + 5-sample sampler + confidence + distogram, synchronize | yes |
| Boltz-2 | kit `PHASE fwd_s` = `Boltz2.forward`, CUDA-synchronized | yes |
| OpenFold3 (ob0) | kit line named `Model forward time`, but it wraps `OpenFold3.forward` with synchronization | yes |
| RoseTTAFold3 | `RF3WithConfidence.forward`, synchronized | yes |
| AF3 (JAX) | `Running model inference with seed N took X seconds` = the jitted model call + the device-to-host copy of its outputs | yes, plus the result copy |
| Protenix-v1 | CLI `Model forward time` | **no**: it also covers post-forward handling; the logs' `PHASE ... total_s` (trunk + sampler + confidence, synchronized) is the matching number |
| Protenix-v2 | CLI `Model forward time` (earlier comparisons) | **no**: as v1; re-measured with `PHASE fwd_s` in `protenix_v2_kit_fwd_aligned_20261029.md` |
| OpenDDE | CLI `Model forward time` | **no**: as v1; `PHASE total_s` matches |
| Chai-1 | kit `FORWARD` line = `run_folding_on_context`, synchronized | **no**: the function also ranks the samples and writes the CIF files (`output_dir`); the log has no finer split, so a re-run with a timer on trunk + diffusion + confidence is needed to align it |
| ESMFold2 | `ESMFold2InputBuilder.fold()`, synchronized | **no**: it includes the ESM-C LM forward; compare it with FoldForge's model forward + `LMTIME` warm |

## Re-read numbers (seconds; warm forwards of the `pin_*_fast_default_n<N>` runs, from their logs)

Protenix-v1 (kit `fast`): `PHASE total_s` against the recorded `Model forward time`.

| N | tokens | PHASE total_s (FoldForge span) | recorded (`Model forward time`) | recorded / PHASE |
|---:|---:|---:|---:|---:|
| 1 | 128 | 1.096 | 1.20 | 1.09 |
| 2 | 256 | 1.234 | 1.405 | 1.14 |
| 4 | 512 | 1.99 | 2.295 | 1.15 |
| 6 | 768 | 3.537 | 3.985 | 1.13 |
| 8 | 1024 | 5.98 | 6.565 | 1.10 |
| 12 | 1536 | 13.743 | 14.61 | 1.06 |
| 16 | 2048 | 30.156 | 31.31 | 1.04 |

OpenDDE (kit `fast`):

| N | tokens | PHASE total_s (FoldForge span) | recorded (`Model forward time`) | recorded / PHASE |
|---:|---:|---:|---:|---:|
| 1 | 128 | 3.905 | 4.10 | 1.05 |
| 2 | 256 | 3.545 | 4.05 | 1.14 |
| 4 | 512 | 6.80 | 8.32 | 1.22 |
| 6 | 768 | 13.20 | 16.32 | 1.24 |
| 8 | 1024 | 23.19 | 28.49 | 1.23 |
| 12 | 1536 | 58.70 | 70.17 | 1.20 |

ESMFold2: kit `fold()` 0.303 s (N = 1) / 1.286 s (N = 4) includes the LM. FoldForge (engine 2.3.12, `fwd_esmfold2*_u1029_miniworld`
in `~/foldforge-scratch/ffbench`): model forward 0.236 / 0.808 s + LM warm 0.043 / 0.045 s = 0.279 / 0.853 s on the same span. The LM
time barely grows from 128 to 512 tokens on the FoldForge side (four identical chains); whether the kit's `fold()` runs the LM once per
chain was not checked.

Chai-1: no aligned number; the recorded `fast` warm (4.263 s at N = 1, 9.941 s at N = 4) over-states the forward by the ranking and
CIF writing time.
