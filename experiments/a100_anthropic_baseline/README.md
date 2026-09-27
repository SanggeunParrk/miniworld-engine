# A100 Anthropic 커널 baseline — 2026-09-24

A100용 커널 개발을 시작하기 전에 Anthropic 공개 커널(`anthropics/uplifting-biomolecular-modeling` @ `f4f62fa6`, `common/opt_core`)을
A100에서 측정했다. H100 비교표([version-compare-20260923](../../verdicts/version-compare-20260923/README.md))와 같은 모듈·shape·측정법을 쓴다.

## 조건

- **GPU**: NVIDIA **A100 80GB PCIe** (gpu01, gpu05), driver 570.211 (CUDA 12.8). SXM이 아니므로 clock과 대역폭이 upstream 표(A100-SXM4)보다 낮다.
- **SW**: torch 2.10.0+cu128, triton 3.6.0, cuequivariance 0.11.1, Python 3.12 (FoldForge `.venv`), nvcc 12.9.
- **shape**: B=1, BF16 activation, FP32 LN affine. TriMul D128(hidden 128, 양방향은 방향별 128), Transition expansion 4,
  OPM/PWA는 MSA depth 1024 / MSA64 / pair128, PWA head 8×32. mask는 약 10% false. dropout은 TriMul 25%, PWA 15%(학습에만 적용).
- **측정**: `(module, L, arm)`마다 새 프로세스를 띄운다. CUDA graph를 1회 캡처한 뒤 약 300ms warm replay, 이어서 7×50 replay의 회당 중앙값(ms)을 쓴다.
  - PyTorch / cuEq / Engine: 정적 `torch.compile(fullgraph)`.
  - Anthropic: eager 호출을 그대로 graph에 캡처했다.
  - 학습은 fwd+bwd이며, 입력과 전체 parameter의 gradient, dropout RNG를 포함한다.
- **정확도**: 모든 추론 출력을 같은 weight의 FP32 PyTorch 모듈과 비교했다(rel-RMS). 모든 행이 2e-3~4e-3 범위다.
- **Anthropic 최선**: upstream이 제공하는 모든 row/config 중 A100에서 가장 빠른 값이다. 셀 테이블이 고르는 `fast` 단어만 쓴 값이 아니다.
  PWA는 upstream에 A100 튜닝 행이 없어서 이 방식이 공정한 기준이다.
- **Engine(main, A100)**: 현재 main의 `implementation="miniworld"`이다. H100 CUDA 경로는 A100에서 작동하지 않는다.
  TriMul은 Triton 경로로, OPM/PWA는 PyTorch와 같은 경로로 떨어진다.

## Anthropic 측이 무엇인가 (모듈별)

| 모듈 | Anthropic 구현 (A100) | 비고 |
|---|---|---|
| 단방향 TriMul | `kernels.trimul` row: `fast`, `v4`, `tmk3_*`, `esm_v5_fwd`, `esm_k1ptr`, `native_rebuilt` | `native`는 아래 참고 |
| 양방향 TriMul | sm_80 native K1(h256) → outgoing 절반 NT bmm + incoming 절반 TN bmm → K3(h256) | upstream에 양방향 entry가 없다. H100 `integrations/anthropic_trimul.py`와 같은 조합이며, 호출 glue만 우리 코드다 |
| Transition | `kernels.transition` row: `fast`, `v2`, `v1`, `pf`, `lnl`, `af3_fused` | 전부 Triton이다. CUDA `flash_sm90a`/`esm_t16`은 sm_90 전용이다 |
| MiniPairformer 1블록 | 위 native 양방향 + Transition row | |
| OPM | `ops.msa_opm.forward_mask_norm`, A100 행 `a2`(default)와 `CFG_VARIANTS` | 기본 `f1`/`p8`은 TMA를 쓰고 198KB smem이 필요해 sm_80에서 불가능하다. `a4`는 180KB가 필요해 실패했다 |
| PWA | `ops.msa_pwa.forward_masked`, `_CFG` default와 `CFG_VARIANTS` | A100 측정 기록이 없는 행이다 |
| Token DiT | 측정하지 않음 | `dit_exact`는 sm_90 전용이다. A100 셀은 우리 shape에서 `sdpa`를 고른다 |

**native TriMul cubin 재빌드.** release의 sm_80 cubin은 CUDA 13.0 image여서, 이 드라이버(CUDA 12.8)에서 `CUDA_ERROR_INVALID_IMAGE`가 난다.
그래서 `pkg/v5` 트리를 `~/.cache/miniworld-a100/native_rebuilt/v5`로 복사한 뒤, upstream `build.py`로 `build/sm_80`만 nvcc 12.9로 다시 빌드했다.
- 쓰는 tile 4개의 register 수와 spill이 원본 ptxas 기록과 같다.
- upstream sm80 witness vector 206/206이 bitwise로 일치한다. 10건은 설계상 skip이고, cubin digest pin만 `ignore_build`로 우회했다.
- upstream 트리(`refs/`)는 수정하지 않았다.

## 추론 forward (ms)

| 모듈 | L | PyTorch | cuEquivariance | Engine(main, A100) | **Anthropic 최선** | 최선 row | Anthropic rel-RMS |
|---|---:|---:|---:|---:|---:|---|---:|
| 양방향 TriMul | 384 | 2.560 | 1.314 | 0.864 | **0.790** | native_bidir | 2.67e-03 |
| 양방향 TriMul | 768 | 20.359 | 5.464 | 3.827 | **3.417** | native_bidir | 2.64e-03 |
| 단방향 TriMul (outgoing) | 384 | 1.371 | 0.620 | 0.516 | **0.407** | esm_v5_fwd | 2.87e-03 |
| 단방향 TriMul (outgoing) | 768 | 10.759 | 2.642 | 2.183 | **1.776** | esm_k1ptr | 2.47e-03 |
| Transition | 384 | 0.976 | — | 0.551 | **0.395** | pf | 3.27e-03 |
| Transition | 768 | 3.779 | — | 2.181 | **1.542** | pf | 3.26e-03 |
| MiniPairformer 1블록 | 384 | 3.532 | 2.282 | 1.445 | **1.226** | fast | 4.15e-03 |
| MiniPairformer 1블록 | 768 | 24.173 | 9.261 | 5.985 | **5.228** | fast | 4.14e-03 |
| OPM | 384 | 2.343 | — | 2.362 | **5.540** | a3 | 1.68e-03 |
| OPM | 768 | 8.892 | — | 8.991 | **19.042** | a3 | 1.68e-03 |
| PWA | 384 | 2.062 | — | 2.135 | **1.919** | g_fo2p | 1.69e-03 |
| PWA | 768 | 5.316 | — | 5.174 | **5.618** | g_fo2p | 1.68e-03 |

## 학습 forward + backward (ms) — Anthropic 공개 커널에는 backward가 없음

| 모듈 | L | PyTorch | cuEquivariance | Engine(main, A100) |
|---|---:|---:|---:|---:|
| 양방향 TriMul | 384 | 7.377 | 4.435 | 3.198 |
| 양방향 TriMul | 768 | 59.464 | 18.270 | 14.154 |
| 단방향 TriMul (outgoing) | 384 | 4.247 | 2.614 | 1.976 |
| 단방향 TriMul (outgoing) | 768 | 31.925 | 10.609 | 8.170 |
| Transition | 384 | 2.487 | — | 2.109 |
| Transition | 768 | 9.363 | — | 8.358 |
| MiniPairformer 1블록 | 384 | 10.058 | 7.108 | 5.374 |
| MiniPairformer 1블록 | 768 | 69.603 | 28.156 | 22.228 |
| OPM | 384 | 6.616 | — | 6.947 |
| OPM | 768 | 25.030 | — | 25.999 |
| PWA | 384 | 5.628 | — | 6.239 |
| PWA | 768 | 14.290 | — | 15.223 |

## Anthropic row/config 전체 (추론 ms, rel-RMS vs fp32)

**양방향 TriMul**

| row | L384 | L768 |
|---|---|---|
| native_bidir | 0.790 (2.67e-03) | 3.417 (2.64e-03) |

**단방향 TriMul (outgoing)**

| row | L384 | L768 |
|---|---|---|
| esm_k1ptr | 0.446 (2.49e-03) | 1.776 (2.47e-03) |
| esm_v5_fwd | 0.407 (2.87e-03) | 1.779 (2.84e-03) |
| fast | 0.462 (2.65e-03) | 1.777 (2.84e-03) |
| native_rebuilt | 0.425 (2.65e-03) | 1.811 (2.62e-03) |
| tmk3_exact | 0.590 (2.65e-03) | 2.486 (2.62e-03) |
| tmk3_fast | 0.526 (2.65e-03) | 2.269 (2.62e-03) |
| v4 | 0.459 (2.65e-03) | 1.929 (2.62e-03) |

**Transition**

| row | L384 | L768 |
|---|---|---|
| af3_fused | 0.429 (2.99e-03) | 1.672 (2.99e-03) |
| fast | 0.458 (3.13e-03) | 1.786 (3.13e-03) |
| lnl | 0.435 (2.99e-03) | 1.669 (2.99e-03) |
| pf | 0.395 (3.27e-03) | 1.542 (3.26e-03) |
| v1 | 0.895 (2.99e-03) | 3.443 (2.99e-03) |
| v2 | 0.453 (3.13e-03) | 1.777 (3.13e-03) |

**MiniPairformer 1블록**

| row | L384 | L768 |
|---|---|---|
| fast | 1.226 (4.15e-03) | 5.228 (4.14e-03) |
| v2 | 1.236 (4.15e-03) | 5.244 (4.14e-03) |

**OPM**

| row | L384 | L768 |
|---|---|---|
| a1 | 6.352 (1.68e-03) | 22.303 (1.68e-03) |
| a3 | 5.540 (1.68e-03) | 19.042 (1.68e-03) |
| a4 | 실패: triton.runtime.errors.OutOfResources: out of resource: shared memory, Required: 180224, Ha | 실패: triton.runtime.errors.OutOfResources: out of resource: shared memory, Required: 180224, Ha |
| a5 | 6.433 (1.68e-03) | 22.697 (1.68e-03) |
| a6 | 6.829 (1.68e-03) | 23.927 (1.68e-03) |
| default | 6.335 (1.68e-03) | 22.256 (1.68e-03) |
| v1 | 15.359 (1.68e-03) | 58.510 (1.68e-03) |
| v2 | 14.720 (1.68e-03) | 55.965 (1.68e-03) |
| v3 | 14.983 (1.68e-03) | 57.194 (1.68e-03) |
| v4 | 14.338 (1.68e-03) | 54.102 (1.68e-03) |

**PWA**

| row | L384 | L768 |
|---|---|---|
| a4 | 3.117 (1.69e-03) | 7.915 (1.68e-03) |
| default | 1.946 (1.69e-03) | 5.755 (1.68e-03) |
| fp_fo4 | 3.045 (1.69e-03) | 7.633 (1.68e-03) |
| g_fo1 | 1.934 (1.69e-03) | 5.750 (1.68e-03) |
| g_fo2p | 1.919 (1.69e-03) | 5.618 (1.68e-03) |
| g_fo3 | 2.031 (1.69e-03) | 5.965 (1.68e-03) |
| g_fo4 | 1.992 (1.69e-03) | 5.819 (1.68e-03) |
| g_fo7 | 3.021 (1.69e-03) | 7.496 (1.68e-03) |
| g_fohp1 | 2.336 (1.69e-03) | 6.453 (1.68e-03) |
| h_fg1 | 1.985 (1.69e-03) | 6.089 (1.68e-03) |
| h_fg3 | 1.945 (1.69e-03) | 5.698 (1.68e-03) |
| v0 | 3.704 (1.69e-03) | 10.249 (1.68e-03) |


## 요약 — A100 개발 목표선

| 모듈 | 추론 목표 (현재 최고) | 보유자 | 학습 목표 (현재 최고) | 보유자 |
|---|---|---|---|---|
| 양방향 TriMul L384 / L768 | 0.790 / 3.417 | Anthropic native | 3.198 / 14.154 | Engine(main) |
| 단방향 TriMul L384 / L768 | 0.407 / 1.776 | Anthropic esm_v5_fwd / esm_k1ptr | 1.976 / 8.170 | Engine(main) |
| Transition L384 / L768 | 0.395 / 1.542 | Anthropic pf | 2.109 / 8.358 | Engine(main) |
| MiniPairformer L384 / L768 | 1.226 / 5.228 | Anthropic | 5.374 / 22.228 | Engine(main) |
| OPM L384 / L768 | 2.343 / 8.892 | PyTorch compile | 6.616 / 25.030 | PyTorch compile |
| PWA L384 / L768 | 1.919 / 5.174 | Anthropic g_fo2p / Engine(main) | 5.628 / 14.290 | PyTorch compile |

- TriMul, Transition, 블록 추론에서는 Anthropic이 현재 main 엔진보다 A100에서 9–30% 빠르다. H100 때처럼 이 기준을 넘는 것이 첫 목표다.
- **OPM은 Anthropic A100 커널이 compile된 PyTorch보다 2.1–2.4배 느리다.** `_opm_kernel`이 L768 기준 15.7ms로 82%를 차지한다.
  upstream의 "x1.21"은 non-compile stock 대비 수치다. 따라서 OPM의 기준선은 PyTorch compile이다.
- PWA는 L384에서 Anthropic이 7% 빠르고, L768에서는 PyTorch/main보다 느리다. `_pwa_fo_kernel`이 2.6ms, `softmax`가 1.2ms, `_ln_vg`가 0.8ms다.
- Anthropic 공개 커널에는 backward가 전혀 없다. 학습의 비교 대상은 cuEq, PyTorch, 현재 main이다.
- 한계: 모두 B=1, pair D128 shape다. Token DiT와 MSA module 1블록은 아직 측정하지 않았다. PCIe 카드 한 종류만 측정했다.

## 재현

```bash
cd experiments/a100_anthropic_baseline
# 1) native sm_80 재빌드 (한 번만): pkg/v5 -> ~/.cache/miniworld-a100/native_rebuilt/v5, python -m trimul_native.build --archs sm_80 --nvcc /usr/local/cuda-12.9/bin/nvcc
sbatch run.sbatch           # 4 GPU array, 전체 spec
python report.py            # results.csv, results-tables.md
```

원본: `results/*.json`. 각 파일에 replay 7회 값, 커널별 시간, Anthropic selection, native gate 결과를 담았다. slurm job 51083.
