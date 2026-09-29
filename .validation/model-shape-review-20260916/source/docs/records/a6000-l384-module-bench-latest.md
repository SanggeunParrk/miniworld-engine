# Latest L384 A6000 module benchmark table

Updated 2026-09-10. BF16 inputs/trunk and FP32 normalization affine, actual torch.compile, depth1, mask probability .125, TF32 OFF. L384; atom length3072. Values are median milliseconds of three native CLI processes.
Inference: A5, CUDA Graph ON, dropout 0. Training: A48, CUDA Graph OFF, dropout .25 wherever the module has dropout, forward+backward without optimizer. Pair modules have no augmentation axis.

This is a consolidation: triangle training uses the completed dropout .25 run; triangle inference was remeasured with current code; SWA rows use the corrected independent PyTorch baseline; unchanged modules retain the verified L384 results. Different rows may use different physical A6000 cards, while all implementations within a row share a card. Do not sum these rows into a whole-model latency claim.

## inference

| Module | PyTorch | cuEquivariance | MiniWorld |
|---|---:|---:|---:|
| AdaLN | 0.067 | — | 0.052 |
| Trimul outgoing | 4.901 | 1.140 | 0.865 |
| Trimul incoming | 4.907 | 1.124 | 0.843 |
| Bidirectional trimul | 9.681 | 2.761 | 1.678 |
| Triangle attention (starting) | 5.029 | 2.059 | 1.529 |
| Transition | 1.794 | — | 0.954 |
| ConditionedTransition | 0.321 | — | 0.214 |
| Token attention | 0.776 | — | 0.561 |
| Atom attention | 5.388 | — | 2.627 |
| SWA attention | 0.365 | — | 0.404 |
| DiT | 1.038 | — | 0.748 |
| SWA DiT | 0.620 | — | 0.570 |

## training

| Module | PyTorch | cuEquivariance | MiniWorld |
|---|---:|---:|---:|
| AdaLN | 1.233 | — | 1.610 |
| Trimul outgoing | 14.111 | 4.740 | 4.381 |
| Trimul incoming | 14.104 | 4.729 | 4.402 |
| Bidirectional trimul | 26.985 | 8.810 | 7.657 |
| Triangle attention (starting) | 13.808 | 8.385 | 6.467 |
| Transition | 4.836 | — | 3.762 |
| ConditionedTransition | 6.023 | — | 6.201 |
| Token attention | 11.994 | — | 11.598 |
| Atom attention | 92.719 | — | 72.843 |
| SWA attention | 14.361 | — | 10.112 |
| DiT | 18.075 | — | 17.912 |
| SWA DiT | 21.756 | — | 16.551 |

`—` means no equivalent implementation was included. Bidirectional cuEquivariance is the equivalent vendor-primitive composition with shared output normalization. SWA PyTorch uses torch operations except for the allowed FlashAttention backend.
The RMSNorm fix changed validation and its BF16 forward tolerance, not the production kernel. It therefore does not require replacing these steady-state timings.

Checked 168 selected native rows across 56 groups. [Raw rows, CSV hashes, source and GPU provenance](a6000-l384-module-bench-latest.json).

## Other GPU build status

`miniworld-engine build all` is not yet certified to cover the requested workload on a new GPU.
- The module build case still hardcodes augmentation 2 in augmented attention; other cases invoke inputs with batch1. The registry does not declare inference A5 / training A48. A successful declared-plan coverage check cannot certify inputs absent from that plan.
- The stored derivation is sm86 and its source identity is stale after the recent edits. A read-only plan.load check rejected it for sm86, sm90 and sm100. cmd_build checks current per-architecture derivation after merging, so a bare build can spend work and still fail certification.
- Other architectures take different kernels, resource limits and dispatch branches. A6000 runtime and numerical checks do not establish their correctness or complete cache coverage.
The remaining prerequisite is to align the build workload with A5/A48, derive it for the target architecture, and validate coverage plus runtime/numerics on that GPU. This table update did not change the builder or launch a build on another card.
