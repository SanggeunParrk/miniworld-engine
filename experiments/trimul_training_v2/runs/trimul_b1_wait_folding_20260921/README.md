# B1–B4 대기 축소: 검증된 개발 경로

2026-09-21 · node01 H100. 비교 기준은 직전 v50 자체 학습 경로이며 Anthropic 추론이나 cuEquivariance가 아니다.

양방향 C128/H256 BF16, dropout25%/mask/residual, L384/L768. 구간별600회 교대 CUDA graph 중앙값. 전체 수치는 live packing/Wp 전치/cuBLAS/B7/11 gradients를 포함하며 optimizer/RNG 생성/compile/CPU dispatch를 제외한다. FWD 구현은 동일하므로 그 차이는 측정 변동이다.

| L | 경로 | FWD ms | B1–B4 ms | BWD ms | 전체 ms |
|---|---|---|---|---|---|
| 384 | 직전 v50 | 0.2924 | 0.1904 | 0.9012 | 1.1883 |
| 384 | 대기 축소 | 0.2919 | 0.1885 | 0.8992 | 1.1857 |
| 768 | 직전 v50 | 1.1665 | 0.6467 | 3.5883 | 4.7890 |
| 768 | 대기 축소 | 1.1699 | 0.6359 | 3.5766 | 4.7740 |

## 실제 변경

1. dGate TMA 저장 완료를 dNorm WGMMA 뒤로 옮겨 기존 CTA barrier와 합쳤다. dGate가 있던 shared scratch를 LN parameter 합산이 재사용하기 전에 저장 완료를 반드시 기다린다.
2. dTri TMA 저장 뒤의 warp-group barrier는 바로 다음 호출자의 CTA barrier가 포괄하므로 제거했다. 저장 완료 대기와 호출자의 CTA barrier는 유지한다.

**다음 raw TMA 전에 두 warp-group의 WGMMA 완료를 확인하는 CTA barrier를 유지한다.** 이전 경쟁 조건을 재도입하지 않았다. FP32 덧셈 순서, BF16 반올림, 입력 affine x_n 저장, 원본 tri BF16와 출력 mean/rstd FP32 저장 정책은 그대로다. 출력 LN activation은 저장하지 않는다. B7/cuBLAS는 변경하지 않았다.

## SoL90 진행 상황

| L | NCU B1 μs | DRAM peak | 실측 트래픽 roofline | 고유 payload 모델 |
|---|---|---|---|---|
| 384 | 188.768 | 61.19% | 61.22% | 41.97% |
| 768 | 624.320 | 67.38% | 67.42% | 50.76% |

**SoL90 미달이다.** DRAM 처리율을 전체 알고리즘 SoL로 표시하지 않는다. 이전과 같은 낙관적 모델 `max(1800*L²/3.35TBps, 262144*L²/989.5TFps)`은 scalar/shared/instruction/의존성 및 작은 scratch 비용을 생략한다. 실측 트래픽 모델에는 dWgate의 x_n/dGate 재읽기가 포함된다. 정확한 도달 가능한 최소시간을 증명한 수치는 아니다.

[H100 공식 사양](https://www.nvidia.com/en-us/data-center/h100/)의 SXM3.35TB/s, dense BF16 989.5TF/s를 사용한다. 공식1979TF/s는 sparsity 수치다. 별도 node01 streaming 교정은3.105TB/s였다.

### dWgate만 분리한 진단

| L | 단독 CUDA event μs | 단독 NCU μs | 단독 DRAM peak |
|---|---|---|---|
| 384 | 37.024 | 35.136 | 73.71% |
| 768 | 109.472 | 109.984 | 85.00% |

원래 CTA 소유 행/입력/부분합을 그대로 쓰고 shared reservation도 동일하게 유지한 진단이다. 최종 CTA 간 합산은 제외한다. 단독 부분합은 원래 B1과 bit-exact. 단독 측정은 전체 실행 중 해당 구간의 직접 타이밍과 다르며, 단독85%를 전체 SoL85%로 해석하면 안 된다.

## 검증

- 두 길이 출력·전체11 gradients: 기준과 bit-exact.
- 입력/가중치/dy/dropout/mask 변경 및 gamma_out=0: bit-exact.
- CUDA graph 재실행/eager, 원본tri·stats 저장 정책 검사 통과.
- memcheck 두 길이0 errors, racecheck L3840 hazards.
- CTA 일부 및 WG1을 의도적으로 지연: 두 길이×3seed×20replay의 B1 여섯 출력 bit-exact.

개발 경로이며 production 승격이 아니다. 기존 B7 독립 기준 L768 dWL 상대L2 0.055569%가 한도0.05%를 넘는 문제는 그대로 남아 있다.

## 추가 시도

- dWproj와 dNorm WGMMA를 연속 발행: 유효하지만 추가 이득이 작아 제외.
- 출력 LN affine과 gate 재계산, sigmoid와 projection 겹치기: 추가 이득이 작아 제외.
- LN parameter 합산 동기화를 warp-group으로 축소: 추가 이득이 없어 CTA 동기화 유지.
- 마지막 partial의 L1 우회/일반 읽기: 추가 이득 없어 기존 volatile 읽기 유지.
- 마지막 CTA partial 합산 unroll1~132: 기존 컴파일러 설정보다 유의한 개선 없음. 낮은 unroll은 오히려 느리다.
