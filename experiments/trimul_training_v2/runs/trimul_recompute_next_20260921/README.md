# TriMul 재계산 배선: dWproj 불필요한 gate 계산 제거 · 2026-09-21

## 결과

L384: 직전 대비 5.5% 시간 감소 (1.058배), 최초 재계산 대비 1.289배. 저장형 대비 지연은 아직 19.6% 더 큼.

L768: 직전 대비 4.8% 시간 감소 (1.051배), 최초 재계산 대비 1.296배. 저장형 대비 지연은 아직 21.3% 더 큼.

**1.7배 미달 / SoL90 미입증. Production 기본값은 변경하지 않았다.** 같은 H100 node02에서 네 경로를 동시에 비교한 결과다. 공개 Anthropic inference나 cuEquivariance 대비 새 성능 주장이 아니다.

- saved: Anthropic 파생 저장형 학습 forward + 기존 최적화 CUDA backward.
- initial_recompute: `trimul_fused_recompute_20260920` 최초 내부 재계산.
- previous_recompute: `trimul_recompute_optimized_20260920` 직전 선택.
- optimized_recompute: 이 디렉터리의 `selected-L384.json`, `selected-L768.json`.

BF16 / batch1 / C128 / outgoing·incoming 각 hidden128 / L384·768 / dropout25% / mask·residual / 모든 11개 gradient. 경로별·범위별 CUDA graph 600회 교대 측정. Live weight packing 포함. Optimizer, RNG 생성, CPU dispatch, 컴파일 제외. 전체 시간은 fwd+bwd를 별도 graph에서 직접 측정한 값이다.

| L | 경로 | Forward ms | Backward ms | 전체 실측 ms |
|---|---|---|---|---|
| 384 | Anthropic 파생 저장형 | 0.437 | 0.714 | 1.150 |
| 384 | 최초 내부 재계산 | 0.284 | 1.538 | 1.773 |
| 384 | 직전 재계산 최적화 | 0.283 | 1.174 | 1.455 |
| 384 | 이번 선택 | 0.284 | 1.095 | 1.375 |
| 768 | Anthropic 파생 저장형 | 1.751 | 2.774 | 4.537 |
| 768 | 최초 내부 재계산 | 1.144 | 5.895 | 7.133 |
| 768 | 직전 재계산 최적화 | 1.145 | 4.585 | 5.782 |
| 768 | 이번 선택 | 1.143 | 4.318 | 5.502 |

## 변경한 배선

Forward는 동일한 Anthropic 파생 no-save K1 → cuBLAS 두 번 → no-save K3다. 유지하는 forward activation은 left/right/tri. x_n, projection/gate, LN 통계는 backward CUDA 안에서 다시 계산한다. Global activation 복원 커널은 0개다. Gradient 전달 텐서와 dW reduction scratch의 global 메모리는 계속 필요하다.

**B1–B4의 dWproj 역할에서만 재계산을 줄였다.** 기존에는 각 projection 출력 절반을 맡은 CTA에서도 output gate 128채널을 전부 계산했다. 이제 해당 CTA가 실제 쓰는 64채널만 계산한다. WG0은 input LN → 필요한 gate64 → dy·dropout·gate 곱을 처리한다. WG1은 그동안 output LN256을 처리한다. CTA 동기화 후 두 WG가 dWproj WGMMA를 실행한다. 재계산 activation의 새로운 HBM 저장은 없다.

CTA 배분은 dWgate36 / dWproj52 / dtri44에서 **36 / 48 / 48**로 바꿨다. Row64, 256 threads, 총132 CTA, shared227840 B. `DW_SPLITS=24`, `GATE_SPLITS=36`, `TRAIN_L=384/768`, `B1_DWPROJ_PIPE=1`. 새 B1 ptxas는255 registers, stack/spills0. Gate/projection 반올림과 미분 계산식을 유지한다. dW reduction partition 변경으로 weight gradient의 FP32 합산 묶음은 달라지며 수치 오차 검증으로 확인했다.

**B7–B12는 직전 선택을 유지한다.** 64 dW CTA +68 dX CTA, TMA producer1+consumer2, dX 방향별 resident packed weights128 KiB. 후보가 안정적인 추가 이득을 주지 못해 active B7 source와 plan 코드를 이전 스냅샷으로 되돌렸다. 실험 소스는 `rejected-b7/`에 별도 보관했다.

B5–B6 cuBLAS 네 번 유지. Forward 중간값의 논리적 유지량은 L384 226.492 MB, L768 905.970 MB로 직전 재계산과 같다. 저장형은719.585 /2878.341 MB. Input/weights/mask/output 제외 tensor-size 합계로, peak GPU memory 실측은 아니다.

## 구간별 trace

아래는 별도 profiler trace이며 graph 중앙값과 차이가 있을 수 있다.

| L | 구간 | 직전 μs | 이번 μs | 시간 감소 |
|---|---|---|---|---|
| 384 | B1–B4 | 402.9 | 322.7 | 19.9% |
| 384 | B7–B12 | 588.7 | 589.2 | -0.1% |
| 768 | B1–B4 | 1515.4 | 1246.6 | 17.7% |
| 768 | B7–B12 | 2289.5 | 2287.3 | 0.1% |

## 동일 버퍼 800회 교대 비교

| L | 구간 | 후보1 | 후보2 | 후보3 |
|---|---|---|---|---|
| 384 | b1 | previous: 404.26 μs | partition_only: 396.42 μs | projection_pipe: 327.23 μs |
| 384 | b7 | previous: 590.72 μs | stats: 585.86 μs | stats_early_mask: 583.39 μs |
| 768 | b1 | previous: 1579.92 μs | partition_only: 1507.78 μs | projection_pipe: 1248.24 μs |
| 768 | b7 | previous: 2254.54 μs | stats: 2242.40 μs | stats_early_mask: 2269.42 μs |


- B1 partition_only는 CTA 배분만 변경. projection_pipe가 실제 선택이다.
- B7 stats는 LN 통계 shared 재사용, stats_early_mask는 추가로 weight TMA 조기 발행 + mask hoist. Stats만의 차이는 양쪽 L 모두1% 미만. Combined 후보는 L768에서 역효과. 미채택.
- B7 consumer3은 ptxas spill 때문에 실행 전에 제외. Shared-input WGMMA로 register lifetime을 줄이는 시도도 consumer3 spill을 없애지 못했고 consumer2에서는 더 느렸다.
- B1 input-gradient / gate-weight 전용 중첩과12주기 dropout mask register cache도 더 느려 미채택. Config 공간을 모두 탐색했다는 주장은 하지 않는다.

## NCU: B1–B4 최종 커널

| L | DRAM | SM | Tensor pipe | Occupancy | No eligible |
|---|---|---|---|---|---|
| 384 | 31.4% | 44.5% | 15.1% | 12.5% | 55.0% |
| 768 | 47.4% | 46.5% | 15.9% | 12.5% | 53.3% |


`--set full`, 추가 tensor-pipe metric, cache-control none, clock-control none. NCU instrumented 시간 대신 위 CUDA graph 시간을 성능 판단에 쓴다. 낮은 eligible warp 비율과 tensor 활용률이 남는다. SM/DRAM 사용률을 알고리즘 SoL 비율과 동일시할 수 없으며 **SoL90 달성을 입증하지 못했다**. B7은 소스가 동일해 이번에는 다시 NCU를 수집하지 않았다. 직전 프로파일은 별도 이전 보고서에 유지한다.

## 재현

node02의 할당받은 H100에서 `runs/anthropic_adoption_20260919/env.sh`로 cu128 환경을 실행한다. `MINIWORLD_TRIMUL_TRAIN_BUILD_DIR`은 `/home/psk6950/MiniWorld/runs/anthropic_b7b12_fusion_20260920/k3-audit-build`로 지정한다. `bench.py --length 384` 또는 `--length 768`이 최종 선택 설정과 네 비교 경로를 로드한다. `paired_b1.py --length N`은 같은 버퍼의 B1 후보 비교다. B7 후보 소스는 active code에서 제외했으므로 당시 실험 재현에는 `rejected-b7/` 스냅샷이 필요하다. 검증 명령은 `verify.py`에 기록했다.

## 검증과 남은 제한

- 일반 입력: forward bit-exact. 11개 gradient 모두 독립 기준 상대L2≤0.05%.
- 구간 검사: B1 dg/dtri bit-exact. dW≤0.05%, LN parameter≤0.0005%.
- x/W/dy/dropout/mask를 바꾼 CUDA graph replay는 새 eager와 모든 출력 bit-exact.
- L384 B1 memcheck/racecheck, L768 전체 모듈 unfiltered memcheck: 오류0, race hazard0.
- 선택된 cubin: stack/spills0. 소스·설정·cubin SHA-256를 결과JSON에 고정했다.
- **L768 변경 입력의 dWL 독립 기준 검사는 실패가 남아 있다.** 이번0.0548456%, 저장형0.0541300%, 허용0.05%. 이전 재계산에서도 같은 문제다. 허용치를 올리지 않았고, 새 경로와 저장형 비교는 기준 안이다. 전체 검증 완료로 표현하지 않는다.

Anthropic Apache-2.0 native v5의 TMA/WGMMA, LN, layout 구현을 계승한 학습 확장이다. 다음 병목은 B7–B12 재계산과 WGMMA 의존성, 그리고 저장형보다 커진 backward 비용이다.
