# B1–B4: LN parameter 합산 / TMA 저장 중첩 / CTA 내부 Phase 전환

2026-09-21, node01 H100. 기준은 직전 v49 개발 경로 `trimul_b1_sol90_selected_20260921`. Anthropic 추론이나 cuEquivariance 대비 수치가 아니다. 저장은 입력 affine x_n BF16, 원본 tri BF16, 출력 mean/rstd FP32. 출력 LayerNorm activation은 저장하지 않는다.

## 검증된 전체 학습 측정

양방향 C128/H256 BF16, L384/L768, mask/dropout25%/residual. 구간별600회 교대 CUDA graph 중앙값. Packing/Wp 전치/cuBLAS/B7/전체11 gradients 포함. Optimizer/RNG 생성/컴파일/CPU dispatch 제외. FWD는 같은 구현이며 작은 차이는 변동이다.

| L | 경로 | FWD ms | B1–B4 ms | BWD ms | 전체 ms |
|---|---|---|---|---|---|
| 384 | 직전 v49 B1 | 0.2936 | 0.2063 | 0.9118 | 1.2062 |
| 384 | 현재 LN 합산/TMA/동기화 개선 | 0.2940 | 0.1919 | 0.8979 | 1.1892 |
| 768 | 직전 v49 B1 | 1.1818 | 0.6959 | 3.6797 | 4.9155 |
| 768 | 현재 LN 합산/TMA/동기화 개선 | 1.1806 | 0.6444 | 3.6237 | 4.8572 |

- L384: B1 7.00%, BWD 1.52%, 전체 1.40% 시간 감소.
- L768: B1 7.39%, BWD 1.52%, 전체 1.19% 시간 감소.

## 바뀐 구현

1. 출력 LN dgamma/dbeta의 warp shuffle 합산을32KiB shared scratch를 통한 균형 이진 합산으로 바꿨다. 죽은 dy/current x_n 버퍼를 재사용한다. 기존 덧셈 순서를 유지하며 별도 HBM tensor를 추가하지 않는다.
2. dNorm WGMMA 직후 다음 tri/x_n/stats TMA를 발행한다. **두 warp-group의 완료를 CTA barrier로 확인한 후에만** dProj가 있던 반대 슬롯을 덮어쓴다.
3. dTri TMA 저장을 먼저 발행하고, LN parameter 합산 중에 저장을 진행한다. 타일을 재사용하기 전 완료를 기다린다.
4. Phase B는 같은 CTA가 Phase A에서 쓴 dGate 행들만 읽으므로 첫 전체 grid barrier를 제거했다. CTA fence와 TMA 완료 보장은 유지한다. **모든 CTA의 parameter partial을 합치는 마지막 grid barrier는 유지한다.**

BF16 반올림/합산 순서/보존 tensor는 그대로다. B7와 cuBLAS도 바꾸지 않았다.

## 발견하고 고친 경쟁 조건

초기 조기 TMA 후보는 일반 벤치와 gradient 비교를 통과했지만 memcheck 환경에서 dTri/LN gradient가 달라졌다. WGMMA wait는 warp-group 단위라서 group0이 다음 입력으로 dProj를 덮을 때 group1이 아직 읽을 수 있었다. CTA barrier를 추가한 이 수정본으로 전체 측정과 sanitizer를 다시 실행했다. 초기 후보 `trimul_b1_epilogue_20260921`는 선택하거나 게시하지 않았다. 아래 결과는 수정 후 결과다.

## 제외한 실험

- dWgate 단일 루프 및 accumulator shared parking: 레지스터 spill/추가 이동 비용으로 느려 제외.
- dWgate TMA 버퍼2~7개 및 소유 타일2~3개 묶기: 추가 이득이 작아 제외.
- LN scratch 전치 +128-bit shared load: 더 느려 제외. 현재는 행 배치와 scalar shared load를 사용한다.
- dy 선행 로드: 추가 이득이 없고 LN scratch와 겹치므로 사용하지 않는다.

## NCU와 SoL

| L | 경로 | 시간 | DRAM read | DRAM write | DRAM peak | Tensor peak | long-scoreboard/issue |
|---|---|---|---|---|---|---|---|
| 384 | baseline | 206.272000 us | 245.561344 Mbyte | 141.553920 Mbyte | 55.990997 % | 19.249851 % | 1.140388 inst |
| 384 | optimized | 191.456000 us | 245.661696 Mbyte | 141.593600 Mbyte | 60.352207 % | 21.140525 % | 1.135382 inst |
| 768 | baseline | 693.024000 us | 928.684800 Mbyte | 481.425408 Mbyte | 60.699193 % | 23.113598 % | 0.782934 inst |
| 768 | optimized | 639.008000 us | 928.753152 Mbyte | 481.152512 Mbyte | 65.820569 % | 25.184043 % | 0.721779 inst |

| L | 실측 트래픽 roofline 효율 | 낙관적 고유 payload 모델 효율 |
|---|---|---|
| 384 | 60.4% | 41.4% |
| 768 | 65.9% | 49.6% |

**SoL90 미달.** NCU DRAM 처리율은 전체 SoL과 같지 않다. 이전과 같은 단순 모델 `max(1800*L²/3.35TBps, 262144*L²/989.5TFps)`를 사용했다. 고유 큰 tensor의 단일 입출력과 GEMM을 계산하며 scalar math/shared-memory/instruction/dependency/작은 scratch 비용을 생략한 낙관적 모델이다. 실측 트래픽 모델은 별도 dWgate의 x_n/dGate 재읽기도 포함하므로 알고리즘 효율과 구분한다. NCU sampling의 중첩 inline source 귀속을 합산해 시간 비율처럼 표시하지 않았다.

[NVIDIA H100 사양](https://www.nvidia.com/en-us/data-center/h100/)의 SXM3.35TB/s와 dense BF16 989.5TF/s(표의 sparsity1979TF/s 절반)를 사용했다. 별도 node01 streaming 교정은3.105TB/s였다. 정확한 전체 알고리즘 최단시간이 증명된 것은 아니다.

## 검증

- 일반 및 변경 입력/가중치/mask/dropout/dy, gamma_out=0: 출력과11 gradients가 기준과 bit-exact.
- Graph replay/eager bit-exact. BF16 원본tri + FP32mean/rstd 정책 검사 통과.
- memcheck L384/L7680 errors, racecheck L3840 hazards.
- CTA 일부를 Phase B 직전에 고의로 지연: 두 길이×3seed×20graph replay에서 B1 여섯 출력 bit-exact.
- 기존 B7 독립 기준 L768 dWL 상대L2 0.055569%(한도0.05%) 문제는 남아 있다. 이번 검증은 개발 경로 개선이며 production 승격을 뜻하지 않는다.
