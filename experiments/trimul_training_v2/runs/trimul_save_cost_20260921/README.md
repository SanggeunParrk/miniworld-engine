# TriMul forward: 통계 및 projection/gate 저장 비용 · 2026-09-21

## 결론

평균·역표준편차 저장 비용은 작다. 입력 LN 통계만 추가하면 L384 +0.64%, L768 +1.33%. 입력·출력 LN 통계를 모두 추가하면 +0.14%, +1.02%다. 출력 통계만 저장한 경우는 소폭 빨라졌으며 컴파일러 스케줄·spill 변화까지 포함된 실제 지연이다. 저장되는 바이트가 실행시간을 단조롭게 증가시킨다는 뜻은 아니다.

Projection/gate는 큰 비용이다. 입력 x_n + 양쪽 LN 통계를 저장하는 동일 기준에서 출력 쪽만 +10.57%/+9.75%, 입력 쪽만 +31.58%/+27.96%, 양쪽 모두 +41.43%/+38.16%다.

## 조건

node02 H100 두 장, BF16, B1/C128, 공유 출력 LN256 양방향 TriMul, L384/768, mask/dropout25%/residual. **Forward만** 비교. 각 경로600회 교대 CUDA graph, 매 호출 weight packing과 모든 저장 포함. RNG 생성, optimizer, backward, CPU dispatch, 컴파일 제외. 기존 학습13228과 독립 에이전트13364는 건드리지 않았고 실험 할당13365는 종료했다.

## 1. 평균·역표준편차

기존 input x_n BF16 저장에 FP32 mean/rstd만 추가한다. LN 내부에서 이미 계산한 값을 기록하며 재계산·별도 LN 커널을 추가하지 않는다. Input/output statistics 모두 K3에서 저장한다.

| L | 저장 조건 | 전체 fwd ms | x_n만 대비 μs | 변화 |
|---|---|---|---|---|
| 384 | 입력 x_n만 | 0.294 | +0.000 | +0.00% |
| 384 | 입력 LN 통계 추가 | 0.296 | +1.872 | +0.64% |
| 384 | 출력 LN 통계 추가 | 0.293 | -1.056 | -0.36% |
| 384 | 입력·출력 LN 통계 추가 | 0.294 | +0.416 | +0.14% |
| 768 | 입력 x_n만 | 1.186 | +0.000 | +0.00% |
| 768 | 입력 LN 통계 추가 | 1.202 | +15.760 | +1.33% |
| 768 | 출력 LN 통계 추가 | 1.179 | -7.248 | -0.61% |
| 768 | 입력·출력 LN 통계 추가 | 1.198 | +12.112 | +1.02% |


추가 크기: 입력 또는 출력 한 LN의 통계는 L3841.18MB/L7684.72MB, 두 LN 모두2.36MB/9.44MB. 세 측정 block에서 양쪽 통계 추가의 변화는 L384 +0.08~0.21%, L768 +0.85~1.09%. 정확히0 비용이라고 하지는 않는다.

## 2. Projection/gate 저장

이 표의 기준은 input x_n + 입력/출력 mean/rstd다. Output xn_out은 추가하지 않았다.

| L | 저장 조건 | 전체 fwd ms | 통계 기준 대비 μs | 변화 | 추가 MB |
|---|---|---|---|---|---|
| 384 | 입력·출력 LN 통계 추가 | 0.294 | +0.000 | +0.00% | 0.00 |
| 384 | 출력 projection·gate 추가 | 0.325 | +31.120 | +10.57% | 75.50 |
| 384 | 입력 projection·gate 추가 | 0.387 | +92.944 | +31.58% | 301.99 |
| 384 | 입력·출력 projection·gate 추가 | 0.416 | +121.952 | +41.43% | 377.49 |
| 768 | 입력·출력 LN 통계 추가 | 1.198 | +0.000 | +0.00% | 0.00 |
| 768 | 출력 projection·gate 추가 | 1.315 | +116.768 | +9.75% | 301.99 |
| 768 | 입력 projection·gate 추가 | 1.533 | +334.960 | +27.96% | 1207.96 |
| 768 | 입력·출력 projection·gate 추가 | 1.655 | +457.168 | +38.15% | 1509.95 |


- 입력: pL, pR와 gL/gR의 sigmoid 이전 gate logit. 모두 BF16이며 기존 backward와 같은 [g0,p0,g1,p1,...] × M interleaved layout. 네 값 전체1024채널을 저장한다.
- 출력: BF16 projection128채널 및 BF16 sigmoid(BF16 gate logit)128채널. 둘 다 저장하며 residual/dropout 반영 전 값이다.
- 입력 gate logit과 출력 sigmoid gate를 구분한다. 입력의 sigmoid 이후 값을 FP32로 저장하는 실험은 아니다.
- Left/right/tri는 원래 유지하던 값이다. 추가 MB는 이 기존 값과 allocator/peak memory를 제외한 새 tensor의 논리적 크기다.

## 전체 저장량

| L | 조건 | fwd ms | 기존 left/right/tri 외 추가 MB |
|---|---|---|---|
| 384 | 입력 x_n만 | 0.294 | 37.75 |
| 384 | 입력 LN 통계 추가 | 0.296 | 38.93 |
| 384 | 출력 LN 통계 추가 | 0.293 | 38.93 |
| 384 | 입력·출력 LN 통계 추가 | 0.294 | 40.11 |
| 384 | 출력 projection·gate 추가 | 0.325 | 115.61 |
| 384 | 입력 projection·gate 추가 | 0.387 | 342.10 |
| 384 | 입력·출력 projection·gate 추가 | 0.416 | 417.60 |
| 384 | 출력 xn_out까지 전부 추가 | 0.443 | 493.09 |
| 768 | 입력 x_n만 | 1.186 | 150.99 |
| 768 | 입력 LN 통계 추가 | 1.202 | 155.71 |
| 768 | 출력 LN 통계 추가 | 1.179 | 155.71 |
| 768 | 입력·출력 LN 통계 추가 | 1.198 | 160.43 |
| 768 | 출력 projection·gate 추가 | 1.315 | 462.42 |
| 768 | 입력 projection·gate 추가 | 1.533 | 1368.39 |
| 768 | 입력·출력 projection·gate 추가 | 1.655 | 1670.38 |
| 768 | 출력 xn_out까지 전부 추가 | 1.753 | 1972.37 |


## 실제 CUDA 구현

K1은 Anthropic native v5 원본 fused LN + projection/gate + mask와 타일(2,64,8,2)을 유지했다. 추가 gate/projection을 이미 존재하는 FP32 accumulator에서 BF16으로 기록한다. 원래 a/b 계산과 반올림은 유지해 최종 출력이 bit-exact다.

K3도 기존 fused LN + 출력 projection/gate + dropout/residual 구조와 config(2,64,4,1,regs24/240,serialLN)를 유지했다. LN 통계는 ln_fragment의 반환값을 저장하고, projection/gate는 같은 epilogue에서 추가 저장한다. Full tile/config 재튜닝이 아니라 저장 스케줄 두 종류 비교다.

| L | 커널 | 방법0 μs | 방법1 μs | 선택 |
|---|---|---|---|---|
| 384 | k1 | 201.632 | 208.160 | 0 |
| 384 | k3 | 125.328 | 161.296 | 0 |
| 768 | k1 | 800.656 | 801.504 | 0 |
| 768 | k3 | 451.424 | 616.160 | 0 |


K1 방법0은 gate/projection 전용 staging을32KiB 추가해 a/b와 함께 TMA store, 방법1은 기존 stage를 재사용하며 TMA read 완료를 기다린다. K3 방법0은 별도32KiB staging과 TMA, 방법1은 register에서 vector STG. 둘 다 방법0을 선택했다. L768 K1의0.1% 차이는 유의미한 우위로 단정하지 않는다. 단독 시간과 전체 fwd 시간은 캐시 상태가 달라 차이를 그대로 더하지 않는다.

## Compiler 영향과 이전 설명 정정

원래 input x_n만 TMA 저장하는 K3에도 ptxas52B spill stores/68B spill loads가 있었다. 과거 현황판의 '저장ON 후보는 모두 spill0' 설명은 잘못되어 이번에 수정했다. 당시 compiler JSON에도 이 값이 남아 있었다. Input 통계만 추가한 후보는40B/56B, 양쪽 통계 및 선택한 projection/gate 저장 후보는0이다. 따라서 작은 ±1% 차이를 순수 HBM store 시간이라고 해석하지 않는다.

수정된 저장OFF 대조군과 실제 이전 커널도 함께 측정했다. L384 none285.152 vs original285.312μs, x_n294.624 vs original293.920μs. L768 none1157.072 vs original1158.320μs, x_n1187.280 vs original1186.080μs. 본 표의 통계 기준은 실제 이전 original_xn을 사용한다.

## 검증

- 모든 후보의 최종 y가 이전 forward와 bit-exact. K1 a/b도 bit-exact.
- 저장 projection/logit/gate는 FP32 PyTorch GEMM·sigmoid 기준과 비교해 상대L2≤3.82e-5(0.00382%). 한도5e-4를 그대로 유지했다.
- FP32 mean/rstd는 독립 mean/variance/rsqrt 수식과 상대L2≤2e-6. LN activation은 검증된 fused LN 기준과 비교.
- x, 입력/출력 gamma, WL/Wg, dropout scale, pair mask를 바꾼 동일 graph replay에서도 y와 저장값 검증 통과. 저장값이 초기 capture 값으로 고정되지 않는다.
- L384 새 save_K1/K3 필터 racecheck0 hazards/0 errors/0 warnings. L768 전체 check-only unfiltered memcheck0 errors. 두 저장 방법을 모두 검사했다.
- 이 결과는 forward 저장 비용이며 backward 연결·절감량을 검증한 결과가 아니다. 지난 B7 분리 후보의 L768 gradient 제한을 해결했다는 뜻도 아니다. Production dispatch는 그대로다.

## 재현

node02 H100 할당에서 MINIWORLD_TRIMUL_TRAIN_BUILD_DIR=/home/psk6950/MiniWorld/runs/anthropic_b7b12_fusion_20260920/k3-audit-build 설정 후:

    bash runs/anthropic_adoption_20260919/env.sh python -B runs/trimul_save_cost_20260921/bench.py --length 384

L768도 동일. 저장 값 생성은 save_cost_core.py와 save_k1.cu/save_k3.cu, 원본에서의 추출·변경은 derive.py 및 derivation.json으로 기록했다. Source/cubin SHA-256은 결과 JSON에 있다. Anthropic Apache-2.0 native v5 개발을 계승한 저장 비용 분석이다.
