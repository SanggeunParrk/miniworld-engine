# B1–B4: dW 저장 순서와 TMA 캐시 재사용 개선

2026-09-22 · node01 H100 · 양방향 BF16 C128/H256 · dropout 25%, mask, residual.

## 선택된 변경

1. dWproj FP32 누산값을 dWgate 단계가 끝날 때까지 레지스터에 유지한다.
   기존 global partial 버퍼에 쓰는 시점만 늦춘다. 새 activation·scratch를 추가하지 않는다.
2. dGate TMA 출력에는 L2 evict_last 힌트를 준다.
3. dWgate가 소비한 dGate는 evict_first로 읽는다. L384에서는 x_n 읽기도 함께 낮춘다.
   캐시 힌트가 지켜져야만 맞는 코드가 아니며, 기존 TMA 완료 대기와 CTA/grid barrier를 유지한다.

계산식, FP32 누산 순서, BF16 반올림, 입력 x_n·원본 tri·출력 LN mean/rstd 저장 정책은 그대로다.
단일 CUDA 호출, 두 계산 warpgroup, dynamic shared 228352B와 132 CTA를 유지한다.
L384 B1_LATE_WEIGHT=3 / B1_TMA_PRIORITY=2 / B1_GATE_DEMOTE=3,
L768은 B1_GATE_DEMOTE=1을 사용한다. 설정은 policy.py에서 고정했다.

## 동일 실행 3자 비교

단위 µs. 각 구간 5라운드 × 200회 교차 CUDA graph 측정의 pooled median.
B1 구간에서 5개 라운드 모두 새 후보가 v51보다 빨랐다.
전체 구간은 기존 v51 회귀 벤치의 B7를 그대로 둔 측정이며 최신 B7 후보 비교는 아니다.
각 구간은 별도로 측정하므로 시간을 더해 전체를 만들지 않는다.

| L | 구간 | v51 | 저장만 지연 | 저장 지연 + 캐시 정책 | 시간 단축 |
|---|---|---:|---:|---:|---:|
| 384 | b1 | 190.528 | 187.520 | 182.176 | 4.38% |
| 384 | backward | 900.752 | 900.336 | 895.344 | 0.60% |
| 384 | forward_backward | 1193.984 | 1193.168 | 1188.704 | 0.44% |
| 768 | b1 | 640.960 | 640.160 | 636.128 | 0.75% |
| 768 | backward | 3590.544 | 3585.552 | 3585.408 | 0.14% |
| 768 | forward_backward | 4810.256 | 4803.920 | 4800.736 | 0.20% |

전체 학습 시간 변화는 작다. 위 B1 개선율을 전체 fwd+bwd 개선율로 사용하지 않는다.
forward는 동일한 코드다. live weight packing과 전체 11 gradients를 포함하며,
optimizer·RNG 생성·컴파일 시간·CPU dispatch는 제외한다.

## NCU: HBM 읽기 감소

별도 profiler 실행. cache-control none, clock-control none, 5회 워밍업 후 1회 수집.
출력 쓰기의 일부가 캐시에 남으면 kernel 범위 DRAM write counter에 포함되지 않을 수 있다.
아래 실제 바이트 감소를 논리적 출력 버퍼 삭제로 해석하지 않는다.

| L | 구현 | NCU µs | HBM 읽기 MB | HBM 쓰기 MB | HBM peak 비율 |
|---|---|---:|---:|---:|---:|
| 384 | baseline | 188.480 | 245.571 | 140.247 | 61.08% |
| 384 | optimized | 180.768 | 218.436 | 131.685 | 57.79% |
| 768 | baseline | 632.384 | 928.789 | 477.982 | 66.36% |
| 768 | optimized | 630.752 | 905.412 | 465.518 | 64.84% |

캐시 재사용으로 HBM 읽기를 줄이면서 속도가 개선됐다. 전체 알고리즘의 최소 시간을
입증한 것은 아니다. HBM peak 비율이 낮아졌어도 해야 할 연산이 늘거나 느려졌다는 뜻은 아니다.
**SoL90 미달 / 전체 알고리즘 SoL90 달성 근거 없음.**

## 검증

- 일반·변경 입력, weights, dy, mask/dropout, gamma_out[0]=0: forward 및 11 gradients bit-exact.
- graph/eager bit-exact, 원본 tri·통계·x_n 저장 pointer/dtype 확인.
- 별도 B1 probe: 3개 입력 변형, 6개 출력 bit-exact, NaN scratch poison, 반복 replay, counter 0.
- L384/L768 각각 동일 cubin의 memcheck·racecheck 통과. SHA-256은 summary.json에 포함.
- 현재 개발 runs/trimul_training_current.py에도 새 B1을 연결했다. 기존 B7 class를 유지했고,
  일반·변경 입력에서 전체 출력/11 gradients 및 graph/eager bit-exact를 별도로 확인했다.
- 기존 B7의 독립 수학 참조 정확도 문제를 해결한 실험은 아니다. 생산용 dispatch로 승격하지 않았다.

## 함께 구현하고 제외한 구조

- 2/4/8/16 타일씩 주 계산과 dWgate 교대: FP32 partial 왕복으로 느려짐.
- dWproj를 레지스터에 유지하는 묶음 교대: 손해는 줄었지만 기본보다 느림.
- dWgate 역순 타일 방문: FP32 누산 순서가 바뀌어 일부 입력에서 상대 L2 0.01% 한도를 초과.
  성능을 채택하지 않고 제외했다. 이번 선택은 역순 방문을 사용하지 않는다.
- x_n·tri·dTri까지 캐시 우선순위를 바꾼 조합은 추가 이득이 없어 제외했다.

상세 모든 후보·원본 결과·소스/cubin 식별자는 summary.json 및 각 실험 폴더에 보관했다.
역순 구현의 최초 job15403은 전처리 줄바꿈 오류로 실패했고 수정 후 job15408을 검사했다.
개발 연결 검사 job15435는 Python 모듈 이름 충돌로 시작 단계에서 실패했고,
절대 파일 경로 import로 수정한 job15438을 사용했다.

PTX cache-policy 의미와 구문은 NVIDIA 공식 문서를 확인했다:
https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-tensor
성능 수치는 이 문서가 아니라 위 H100 실험의 결과다.
