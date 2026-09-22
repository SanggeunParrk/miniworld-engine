# B1–B4: shared 복사 제거와 다음 tri 타일의 조기 로드

2026-09-21 · Anthropic 파생 CUDA/TMA/WGMMA 학습 개발 경로. 저장 정책은 BF16 tri + FP32 mean/rstd, 입력 affine x_n 저장을 그대로 유지한다. 출력 LayerNorm activation은 저장하지 않는다.

## 변경

1. **dProj 16 KiB shared 복사 제거.** B4의 dNorm GEMM이 dProj의 원래 shared 위치를 직접 읽는다. dNorm의32 KiB와 dProj의16 KiB는 겹치지 않는다.
2. **중복 CTA 동기화 제거.** 통계 저장 분기에서 gate 재계산 직후 연속 실행되던 두 fence/barrier 중 하나를 제거했다.
3. **다음 입력 TMA 조기 발행.** 현재 B4가 dNorm/dProj를 모두 소비한 후, 현재 dTri의 TMA store가 완료되기를 기다리는 동안 다음 tri/x_n/mean/rstd를 반대 슬롯으로 가져온다. 저장 통계는 현재 LN 미분에서 이미 소비한 뒤에 덮어쓴다.

수식·BF16 반올림·CTA별 reduction 순서·HBM 보존 텐서는 그대로다. 동일 CUDA launch의 Phase A/B 구조도 유지한다.

## 설정 탐색

직전의132 CTA, affine unroll L384=4/L768=8을 유지하고 복사 제거/중복 동기화 제거/dWproj 두 GEMM 묶기/조기 tri 로드/dy 프리패치의24개 유효 조합을 길이별로 비교했다. 모든 후보의 B1 여섯 출력이 bit-exact였다.

두 길이 모두 DIRECT_DP=1, NO_REDUNDANT_SYNC=1, EARLY_RAW=1, PAIR_WP=0, PREFETCH_DY=0을 선택했다. dWproj 묶기와 dy 프리패치는 추가 이득이 작거나 없었다. 이번 탐색은 구현 옵션24개이며 전체 타일/CTA 공간을 다시 탐색했다는 뜻은 아니다.

## 전체 학습 재측정

node02 H100 GPU2개에서 길이별 독립 실행. 양방향 C128/H256 BF16 · mask/dropout25%/residual · 구간별600회 교대 CUDA graph 중앙값. Live packing, Wp 전치, cuBLAS, B7, 전체11개 gradient 포함. Optimizer/RNG 생성/compile/CPU dispatch 제외. FWD는 같은 구현이며 작은 차이는 측정 변동이다.

| L | 경로 | FWD ms | B1–B4 ms | BWD ms | 전체 ms |
|---|---|---|---|---|---|
| 384 | 직전 tri+통계 B1 | 0.2930 | 0.2444 | 0.9521 | 1.2417 |
| 384 | 현재 복사 제거+조기 로드 | 0.2928 | 0.2221 | 0.9300 | 1.2179 |
| 768 | 직전 tri+통계 B1 | 1.1807 | 0.8168 | 3.7424 | 4.9664 |
| 768 | 현재 복사 제거+조기 로드 | 1.1789 | 0.7337 | 3.6565 | 4.8710 |

- L384: B1 지연 9.09% 감소, BWD 2.32% 감소, 전체 1.91% 감소.
- L768: B1 지연 10.17% 감소, BWD 2.30% 감소, 전체 1.92% 감소.

## 변경별 영향

| L | dProj 복사 제거 | 중복 sync 제거 | 조기 raw 로드 | B1 μs | 해당 교대 기준 대비 감소 |
|---|---|---|---|---|---|
| 384 | 0 | 0 | 0 | 243.456 | 0.01% |
| 384 | 0 | 0 | 1 | 229.440 | 5.76% |
| 384 | 0 | 1 | 0 | 242.624 | 0.32% |
| 384 | 0 | 1 | 1 | 228.832 | 5.82% |
| 384 | 1 | 0 | 0 | 238.032 | 2.90% |
| 384 | 1 | 0 | 1 | 221.024 | 8.84% |
| 384 | 1 | 1 | 0 | 237.120 | 3.11% |
| 384 | 1 | 1 | 1 | 220.400 | 9.23% |
| 768 | 0 | 0 | 0 | 817.360 | -0.05% |
| 768 | 0 | 0 | 1 | 762.768 | 6.68% |
| 768 | 0 | 1 | 0 | 812.288 | 0.38% |
| 768 | 0 | 1 | 1 | 761.584 | 6.75% |
| 768 | 1 | 0 | 0 | 789.216 | 3.57% |
| 768 | 1 | 0 | 1 | 734.528 | 10.23% |
| 768 | 1 | 1 | 0 | 787.040 | 3.85% |
| 768 | 1 | 1 | 1 | 732.704 | 10.39% |

각 행은 해당 후보와 기존 커널의 같은 시점 교대 측정이다. 작은 차이는 최적 조합의 전체 벤치로 다시 확인했다.

## NCU

| L | 경로 | 시간 | DRAM read | DRAM write | DRAM peak % | Tensor peak % | long-scoreboard/issue |
|---|---|---|---|---|---|---|---|
| 384 | baseline | 245.440000 us | 245.551360 Mbyte | 141.711616 Mbyte | 47.072782 % | 15.971965 % | 1.709715 inst |
| 384 | optimized | 224.928000 us | 245.521152 Mbyte | 141.751296 Mbyte | 51.374682 % | 17.994494 % | 1.293431 inst |
| 768 | baseline | 834.272000 us | 928.806400 Mbyte | 481.250560 Mbyte | 50.419787 % | 19.267056 % | 1.302970 inst |
| 768 | optimized | 745.824000 us | 928.662272 Mbyte | 481.308416 Mbyte | 56.395553 % | 21.294549 % | 0.939957 inst |

DRAM 바이트는 사실상 동일하다. shared 복사와 대기를 줄여 동일한 작업을 더 빨리 처리한다. long-scoreboard/issue는 활성 발행당 메모리 의존 대기 지표이며 전체 시간의 백분율이 아니다. NCU profiling 시간은 위 CUDA event 벤치와 별도다. DRAM/Tensor peak 활용률만으로 알고리즘 SoL90을 주장하지 않는다.

## 검증

- 일반 입력 및 입력/가중치/mask/dropout/dy 변경·gamma_out=0에서 출력과11개 gradient가 직전 tri+통계 기준과 bit-exact.
- CUDA graph replay와 eager bit-exact. 원본 BF16 tri와 저장 FP32 통계만 남는 정책 검사 통과.
- 상세 sanitizer/NCU 실행 상태는 verification-L*.json.
- 기존 B7 L768 dWL 독립 기준 상대L2 0.055569%(한도0.05%) 문제는 남아 있다. 기존 구현과 일치한다는 검증이므로 production 승격을 뜻하지 않는다.

새 B1: L384·768 memcheck 0 errors, L384 racecheck 0 hazards. 두 길이 NCU 완료.
