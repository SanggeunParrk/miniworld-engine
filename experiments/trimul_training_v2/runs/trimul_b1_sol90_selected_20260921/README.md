# B1–B4: node01 검증 완료, SoL90 최적화 진행 중

2026-09-21. Anthropic 파생 CUDA/TMA/WGMMA 학습 개발 경로. 기준은 직전 `trimul_b1_tri_opt_20260921`을 node01에서 함께 측정한 결과다. Anthropic 추론이나 cuEquivariance 대비 수치가 아니다.

## 결과

양방향 C128/H256 BF16, L384/L768, mask/dropout25%/residual. 구간별600회 교대 CUDA graph 중앙값. Live packing, Wp 전치, cuBLAS, B7, 전체11 gradients 포함. Optimizer/RNG 생성/컴파일/CPU dispatch 제외. FWD는 같은 구현이므로 작은 차이는 변동이다.

| L | 경로 | FWD ms | B1–B4 ms | BWD ms | 전체 ms |
|---|---|---|---|---|---|
| 384 | 직전 선택 B1 | 0.2943 | 0.2217 | 0.9276 | 1.2218 |
| 384 | 이번 최적화 B1 | 0.2949 | 0.2050 | 0.9111 | 1.2044 |
| 768 | 직전 선택 B1 | 1.1834 | 0.7484 | 3.7133 | 4.9580 |
| 768 | 이번 최적화 B1 | 1.1832 | 0.6955 | 3.6676 | 4.9076 |

- L384: B1 7.52%, BWD 1.77%, 전체 1.42% 지연 감소.
- L768: B1 7.07%, BWD 1.23%, 전체 1.02% 지연 감소.

## 구현

- 다음 tri/x_n/mean/rstd의 TMA를 출력 LN 미분 중으로 앞당겼다. FP32 통계를 double buffer하여 현재 값이 덮이지 않게 했다.
- 출력 LN affine 재구성을 두 warp-group에 분산했다.
- 두 dNorm WGMMA 타일을 함께 발행했다.
- BF16 dNorm 조각을 레지스터에 유지하여 shared-memory 왕복을 줄였다. 정규화 tri 값은 계속 원본 tri와 mean/rstd에서 재구성한다.
- dTri TMA 저장 폭을 튜닝했다. CTA128/120/96은132보다 느려 채택하지 않았다.

저장 정책: 입력 affine x_n BF16, 원본 tri BF16, 출력 mean/rstd FP32. 출력 LN activation 저장 없음. HBM activation/gradient와 수식, 반올림, CTA별 reduction 순서는 그대로다. B7/cuBLAS도 같다. 상세 실험은 `../trimul_b1_sol90_20260921/README.md` 및 단계별 JSON에 있다.

## NCU

| L | 경로 | 시간 | DRAM read | DRAM write | DRAM peak | Tensor peak | long-scoreboard/issue |
|---|---|---|---|---|---|---|---|
| 384 | baseline | 223.584000 us | 245.524480 Mbyte | 141.729024 Mbyte | 51.682763 % | 17.891872 % | 1.314547 inst |
| 384 | optimized | 206.208000 us | 245.549824 Mbyte | 141.467904 Mbyte | 55.992812 % | 19.524991 % | 1.158319 inst |
| 768 | baseline | 752.192000 us | 928.702208 Mbyte | 481.252352 Mbyte | 55.917529 % | 21.137160 % | 0.958259 inst |
| 768 | optimized | 694.368000 us | 928.662016 Mbyte | 481.421568 Mbyte | 60.582899 % | 23.000683 % | 0.779515 inst |

NCU 시간은 위 CUDA event 벤치와 별개다. 처리율은 전체 커널의 SoL과 같지 않다.

## SoL90 미달

낙관적 streaming roofline은 `max(1800*L²/3.35TBps, 262144*L²/989.5TFps)`로 계산한다. 이는 큰 입력/출력을 한 번씩 읽고 쓰는 payload 기준이며 작은 parameter/scratch와 pointwise/shared-memory/instruction/dependency 비용을 생략한 단순 모델이다. 따라서 달성 가능한 정확한 최단시간이라고 주장하지 않는다. 현재 구현이 실제 이동한 DRAM 바이트를 사용하는 고정 스케줄 roofline도 별도로 표시한다. 추가 재읽기를 필수 연산량으로 둔갑시켜 SoL을 높게 표시하지 않는다.

| L | 고유 payload 메모리 하한 μs | GEMM 하한 μs | 단순 이상 모델 효율 | 실측 트래픽 roofline 효율 |
|---|---|---|---|---|
| 384 | 79.23 | 39.06 | 38.4% | 56.0% |
| 768 | 316.92 | 156.26 | 45.6% | 60.6% |

[NVIDIA H100 공식 사양](https://www.nvidia.com/en-us/data-center/h100/): SXM 3.35TB/s, BF16 1979TF/s는 sparsity 포함이므로 dense989.5TF/s를 사용했다. node01 독립 대역폭 교정은3.105TB/s였다. 현 경로에는 dWgate를 위한 x_n/dGate 재읽기가 남아 있어, 스케줄을 더 바꿀 여지가 있다. 출력 LN 미분/저장 단계와 재계산 단계의 대기가 주요 후속 대상이다.

## 검증 및 남은 제한

일반 입력, 입력/가중치/mask/dropout/dy 변경, gamma_out=0, graph/eager 모두 출력과11 gradients가 직전 기준과 bit-exact다. memcheck L384/L768 0 errors, racecheck L384 0 hazards. 새 B1의 검증 완료이며, 기존 B7 L768 독립 기준 dWL 상대L2 0.055569%(한도0.05%) 문제는 남아 있다. Production 승격이나 SoL90 달성을 뜻하지 않는다.
