# Backward 분리 실험 · 2026-09-21

## 판단

B7–B12를 dW와 dX 역할의 두 CUDA 호출로 분리했다. **분리만 하면 전체 학습은 2–3% 느려진다. 입력 LN activation을 저장·재사용하면서 분리하면 직전 재계산형 대비 L384 4.51%, L768 5.70% 시간이 줄었다.** 저장만 하고 B7을 통합 유지하면 약2.6–2.8% 감소다. 분리 효과와 LN 저장 효과를 구분해야 한다.

PC1과 PC2 분리형 차이는 전체 시간0.1–0.2%로 작다. PC1은 dW의 실제 두 CTA/SM 배치를 확인한 개발 후보다. **L768 변경 입력의 독립 gradient 검증 미통과가 남아 있어 production 기본값으로 채택하지 않았다.** 목표1.7배 또는 알고리즘 SoL90 달성 결과가 아니다.

## 같은 실행에서 측정한 전체 모듈

node02 H100 두 장, BF16, batch1, C128, L384/768, 공유 출력 LN256의 양방향 TriMul, mask/dropout25%/residual. CUDA graph 경로별600회 교대. 매 호출 weight packing·forward 저장·전체11개 gradient·추가 커널 호출·최종 reduction 포함. RNG 생성, optimizer, CPU dispatch, 컴파일 제외. 합계 열은 fwd+bwd 한 호출 직접 측정이며 단독 중앙값의 합이 아니다.

| L | 배선 | fwd ms | bwd ms | fwd+bwd ms | 전체 시간 변화 |
|---|---|---|---|---|---|
| 384 | 직전 재계산형 | 0.284 | 1.097 | 1.373 | +0.00% |
| 384 | 분리만 · LN 저장 없음 | 0.284 | 1.140 | 1.419 | +3.31% |
| 384 | 입력 LN 저장 · 통합 B7 | 0.294 | 1.045 | 1.335 | -2.83% |
| 384 | 입력 LN 저장 · 분리 PC2 | 0.294 | 1.024 | 1.313 | -4.41% |
| 384 | 입력 LN 저장 · 분리 PC1 | 0.296 | 1.020 | 1.312 | -4.51% |
| 768 | 직전 재계산형 | 1.166 | 4.439 | 5.653 | +0.00% |
| 768 | 분리만 · LN 저장 없음 | 1.166 | 4.578 | 5.791 | +2.44% |
| 768 | 입력 LN 저장 · 통합 B7 | 1.195 | 4.269 | 5.504 | -2.64% |
| 768 | 입력 LN 저장 · 분리 PC2 | 1.195 | 4.118 | 5.342 | -5.51% |
| 768 | 입력 LN 저장 · 분리 PC1 | 1.194 | 4.110 | 5.331 | -5.70% |


기준은 runs/trimul_recompute_next_20260921의 직전 on-chip 재계산 구현이다. Anthropic 원본 추론이나 cuEquivariance 대비 수치가 아니다. 기존 activation 전체 저장형보다 이 재계산형이 빠르다는 뜻도 아니다. 과거 저장형의 다른 실행 수치를 이번 표에 섞지 않았다.

## 무엇을 분리했나

- 기존 단일 B7 커널도 dW CTA와 dX CTA가 별개였다. 한 thread가 두 역할의 최대 레지스터를 동시에 유지하던 구조는 아니다.
- 기존: dW64 CTA + dX68 CTA가 하나의132-CTA cooperative launch. 384 threads/CTA, compiled168 registers/thread, dynamic shared221184B를 모든 CTA에 예약.
- 새 dW:256 CTA,256 threads/CTA. TMA producer1 WG + WGMMA consumer1 WG. producer32/consumer224 register budget, compiled128 registers/thread. Dynamic shared65536B. NCU에서 register/shared 상한 모두2 CTA/SM, 실측 occupancy 약24%를 확인했다.
- 새 dX:132 CTA,256 threads/CTA. 두 active WG만 남김. Compiled180 registers/thread, dynamic shared221184B,1 CTA/SM. dX의 thread당 레지스터 수가 감소한 것은 아니다. 불필요한 세 번째 WG를 제거해 CTA 전체 자원과 작업 분배를 바꿨다.
- 두 호출은 같은 stream에서 순서대로 실행한다. 두 stream 동시 실행 벤치가 아니다. 각 호출 내부에서 해당 파라미터 미분 최종 reduction까지 끝낸다. 새 B7 dW/dX와 수정 B1 모두 ptxas stack/spill0.

## Forward 저장과 backward 재사용

K1 → cuBLAS2개 → K3를 유지. K3가 이미 계산하는 BF16 affine 입력 LN 출력 x_n(C128)만 TMA store한다. B1은 출력 gate 재계산에서, B7의 dW/dX는 입력 projection/gate 재계산에서 이를 읽는다. B1 출력 LN 재계산은 그대로다.

Left/right/tri는 기존처럼 유지한다. Mean/rstd, 출력 xn_out, projection, gate, pL/pR/gL/gR preactivation을 새로 저장하지 않는다. B7 입력 LN 미분은 raw x에서 통계를 다시 계산한다. 추가 forward activation은 L38437.75MB/L768150.99MB. dW partial scratch도 기존8.39MB에서16.78MB로 증가하므로 peak memory가 이 activation 증가분과 같다는 뜻은 아니다.

## Backward trace 보조 수치

| L | 경로 | B1 μs | B7 두 역할 합 μs |
|---|---|---|---|
| 384 | 직전 재계산형 | 324.80 | 590.53 |
| 384 | 입력 LN 저장 · 통합 B7 | 312.18 | 550.59 |
| 384 | 입력 LN 저장 · 분리 PC1 | 312.46 | 524.69 |
| 768 | 직전 재계산형 | 1270.69 | 2325.67 |
| 768 | 입력 LN 저장 · 통합 B7 | 1229.10 | 2168.86 |
| 768 | 입력 LN 저장 · 분리 PC1 | 1226.60 | 2020.26 |


Profiler trace5회 평균이며 위 event 중앙값과 계측 범위가 다르다.

## NCU

| L | 커널 | 실측 occupancy | NCU SM | NCU memory | Tensor pipe |
|---|---|---|---|---|---|
| 384 | dW | 24.2% | 36.6% | 59.6% | 36.6% |
| 384 | dX | 12.5% | 33.3% | 34.4% | 26.6% |
| 768 | dW | 24.2% | 39.2% | 62.1% | 39.2% |
| 768 | dX | 12.4% | 35.1% | 36.4% | 28.1% |


NCU memory는 L1/L2/DRAM 등 memory throughput 중 bottleneck 지표다. HBM만의 대역폭도, 알고리즘 최소 시간 대비 효율도 아니다. 두 커널 모두 SoL90에 근접했다고 볼 근거가 없다. dX는 여전히1 CTA/SM이고 WG 동기화·WGMMA 대기가 남아 있다. DRAM 왕복만으로 병목을 설명할 수 없다.

## 정확도 · 모든 실패 포함

초기 입력: 모든 경로의 y는 bit-exact,11개 gradient 상대L2≤5e-4. 저장 LN과 no-save forward는 같은 출력이다. x/WL/Wg/dy/dropout scale/mask를 변경한 graph replay는 각 경로의 새 eager 호출과 모든 출력 bit-exact.

변경 입력의 독립 reference 및 직전 재계산형과의 비교:

| L | 경로 | 독립 기준 최대 상대L2 | 독립 기준≤0.05% | 직전 경로≤0.05% |
|---|---|---|---|---|
| 384 | 직전 재계산형 | 0.02464% dWproj | 통과 | 통과 |
| 384 | 분리만 · LN 저장 없음 | 0.02464% dWproj | 통과 | 통과 |
| 384 | 입력 LN 저장 · 통합 B7 | 0.02464% dWproj | 통과 | 통과 |
| 384 | 입력 LN 저장 · 분리 PC2 | 0.02464% dWproj | 통과 | 통과 |
| 384 | 입력 LN 저장 · 분리 PC1 | 0.02464% dWproj | 통과 | 통과 |
| 768 | 직전 재계산형 | 0.05485% dWL | 실패 | 통과 |
| 768 | 분리만 · LN 저장 없음 | 0.05557% dWL | 실패 | 통과 |
| 768 | 입력 LN 저장 · 통합 B7 | 0.03633% dWproj | 통과 | 실패 |
| 768 | 입력 LN 저장 · 분리 PC2 | 0.05557% dWL | 실패 | 통과 |
| 768 | 입력 LN 저장 · 분리 PC1 | 0.05557% dWL | 실패 | 통과 |


L768 새 분리형 dWL0.05557%가0.05% 한도를 넘는다. 기존 경로도0.05485%로 실패한다. 새 분리형은 기존 경로와는0.05% 이내지만 이것만으로 독립 검증 통과라고 하지 않는다. 저장+통합 경로는 독립 기준을 통과하고 기존 경로와의 비교를 실패했다. 두 판단을 분리해 기록했다. 허용오차를 늘리지 않았다.

추가 누적 분할/CTA 후보를 검사했다.4개 누적 segment는 초기 입력부터 최대0.05770%로 미통과. PC1 DW splits6은 독립 검증을 통과하나 B7 약2.865ms로 느리다. splits10/12/14/15는 변경 입력 검증 미통과. splits17은 cooperative residency 한도를 넘어 launch가 거부되어 후보에서 제외했고 guard를 추가했다. 더 많은 후보를 검증했다고 주장하지 않는다.

L384 B7 racecheck0 hazards/0 errors/0 warnings. L768 전체 호출 unfiltered memcheck0 errors. 이는 수치 검증 미통과를 대체하지 않는다. GPU 실험 할당13363은 종료했고 기존 학습13228은 유지했다.

## 구현과 재현

- b7_roles.cu / b7_pipe_dw.inc: split-role 및 register/shared specialization.
- role_plan.py: config별 SHA-256 cubin, cooperative launch, separate reduction scratch.
- b1_fused.cu / saved_plans.py: saved x_n 재사용.
- bench.py: 원형/분리만/저장만/둘 다 전체 연결 및600회 paired timing.
- validate.py: 초기·변경 입력·graph/eager 독립 비교. 실패도 명시 저장.
- capture_ncu.py: 선택된 두 B7 launch만 profiler capture.

node02 할당 내 MINIWORLD_TRIMUL_TRAIN_BUILD_DIR=/home/psk6950/MiniWorld/runs/anthropic_b7b12_fusion_20260920/k3-audit-build 설정. bash runs/anthropic_adoption_20260919/env.sh python -B runs/trimul_split_bwd_20260921/bench.py --length 384를 실행한다. L768도 동일. --only split_xn_pc1 --check-only는 초기 입력 검사만 수행하므로 변경 입력 검증을 대신하지 않는다.

Anthropic Apache-2.0 native v5 TMA/WGMMA·LN 구현을 계승한 학습 실험이다. 엔진 production dispatch는 변경하지 않았다. Source/cubin hashes는 results JSON에 고정했다.
