# 출력 LN 저장 정책 최적화: 통계만 저장하고 TMA로 prefetch

2026-09-21 · node02 H100 · 양방향 BF16 C128/H256 · dropout25%, mask, residual.
선택한 **개발 기본값**: 입력 affine x_n 저장 유지 + 출력 LN 평균/역표준편차 FP32 저장.
원본 tri를 유지하고 B1의 큰 출력 측 입력으로 한 번 읽는다. 출력 x̂/affine LN activation은 저장하지 않는다.
학습 어댑터: `training.py:Training`, 구현 `save_k3.cu`, `b1_fused.cu`, `lowreg_stats.inc`, `replace_plan.py`.
기존 B7 independent-reference 이슈가 남아 있으므로 production 승격은 별개다.

## 최종 직접 측정

기준은 직전 shared B1 + split_xn_pc1 B7이며 Anthropic 원본 inference와의 비교가 아니다.

| L | 경로 | FWD ms | B1–B4 ms | BWD ms | 전체 ms | 전체 변화 |
|---|---|---|---|---|---|---|
| 384 | 직전 shared B1 | 0.2952 | 0.2576 | 0.9689 | 1.2615 | +0.00% |
| 384 | 출력 통계 TMA 저장 (선택) | 0.2942 | 0.2407 | 0.9496 | 1.2421 | -1.53% |
| 384 | FP32 x̂ 대체 + TMA 저장 | 0.3483 | 0.2990 | 1.0118 | 1.3538 | +7.32% |
| 768 | 직전 shared B1 | 1.1775 | 0.8825 | 3.8012 | 5.0148 | +0.00% |
| 768 | 출력 통계 TMA 저장 (선택) | 1.1703 | 0.8195 | 3.7288 | 4.9346 | -1.60% |
| 768 | FP32 x̂ 대체 + TMA 저장 | 1.3920 | 1.0118 | 3.9382 | 5.3724 | +7.13% |

Slurm 13474_0/1. 경로별600회(200×3) 교대 CUDA graph 중앙값. Live weight packing과 모든11개 gradient 포함, optimizer/RNG 생성/CPU dispatch/compile 제외.
구간 중앙값을 합산하지 않고 전체를 직접 측정했다. 검증·NCU 프로파일링 시간은 벤치 시간에 포함하지 않았다.

## 무엇을 바꿨나

- K3가 이미 계산한 출력 LN mean/rstd를 FP32 [L²]×2 버퍼에 저장. 추가 저장은 L384 1.18MB /L768 4.72MB.
- B1이 x_n/tri 입력 타일과 함께 두 통계를 TMA로 미리 로드한다. 기존 input barrier의 transaction에512B를 추가해 별도 kernel/전역 scalar load 대기를 피했다.
- B1은 저장 통계로 affine와 LN 미분을 계산한다. 출력 LN mean/variance reduction은 재계산하지 않는다. 출력 projection/gate 및 B7 수식과 저장 정책은 그대로다.
- 큰 출력 측 global 읽기는 tri 하나. tri와 출력 LN activation을 함께 읽는 경로가 아니다.
- CTA 96/112/120/128/132 × affine unroll 1/2/4/8의20개 조합을 각 길이에서 검증/튜닝. 기존 경로도 같은 공간을 점검했다.
- 최종 통계 경로: 132 CTA ×256threads, L384 unroll4 /L768 unroll8. 기존 기준은132 CTA/unroll4; 기존 L768 unroll8의 sweep 이득은 약0.15%로 작았다.
- K1/contraction/output GEMM/B7는 동일. 이 실험은 출력 LN 저장 정책과 B1 구현에 한정한다.

## NCU

원본 raw CSV 단위(Mbyte/Gbyte, us/ms)를 변환한 값은 `ncu-summary.json`에 있다. `--cache-control none --clock-control none`, 워밍업 후 B1 한 호출, full metrics. 프로파일링 latency는 CUDA graph 벤치 latency와 구분한다.
L768 B1 실제 DRAM 읽기: 기준924.06MB → 통계 저장928.69MB → FP32 x̂ 대체1228.34MB.
통계 경로는 DRAM bytes 절감이 아니라 재계산 제거와 prefetch로 빨라졌다. Register/thread는255→243, occupancy는12.5%로 같다. Tensor active는17.5→19.0% 정도다.
FP32 x̂ 대체의 추가 읽기도 실측으로 확인했으며, 이는 두 원본을 읽어서가 아니라 FP32로 원소 바이트가 늘어났기 때문이다.
이 지표들은 SoL90이나 이 구조의 성능 상한을 의미하지 않는다.

## 시도 후 제외한 후보

1. FP16 x̂ 직접 저장: gradient 상대L2 최대 약0.24%, 0.05% 기준 실패. BF16보다 정밀해도 affine를 BF16으로 재구성하는 경계에서 차이가 생긴다.
2. FP16 x̂ + mean/rstd로 원래 BF16 값을 복원: 오차는 줄었지만 L384 최대 약0.0743%로 실패하고, 나눗셈/복원 비용 때문에 느렸다. L768만 통과한 결과로 일반 채택하지 않았다.
3. 통계만 저장하되 일반 load: 정확하지만 초기에는 gate 계산과의 overlap 손실/통계 load 대기로 이득이 없었다. 병렬 배치를 복원한 뒤 TMA prefetch에서 개선됐다.
4. FP32 x̂ scalar store → TMA 타일 store: 정확도를 유지하고 forward 비용을 줄였지만, 최종 전체는 기준보다 약7.1~7.3% 느렸다. 64B swizzle 주소식이 잘못된 중간 V2는 정확도 실패로 제외하고 V3에서 수정했다.
5. FP32 B1 gate weight 재로드 제거: shared 공간을 다시 배치하고 gradient global store를 바꿔 정확도는 유지했으나 더 느렸다. 채택하지 않았다.

실험 원본은 sibling 폴더 `trimul_ln_policy_tune_20260921`(FP16/복원), `trimul_ln_policy_v2_20260921`(중간), `trimul_ln_policy_v3_20260921`(TMA 저장/기존 baseline tuning), `trimul_ln_policy_v5_20260921`(gate weight 유지)에 보관했다. V4가 최종 선택이다.

## 검증

- 일반 입력과 x/weight/gamma/beta/mask/dropout mask/dy 변경 후 전체 y 및11개 gradient가 기준과 bit-exact. gamma_out[0]=0 포함. CUDA graph replay도 eager와 bit-exact.
- 대체 FP32 경로는 tri NaN poison 검증 통과. 선택된 통계 경로는 의도적으로 tri를 유지하므로 이 검증의 대상이 아니다.
- 최종 memcheck L384/768 모두0 errors. L384 B1+K3 racecheck0 hazards (0 errors,0 warnings).
- 기존 B7의 L768 변형 입력 독립 reference dWL 오차0.055569%가0.05%를 넘는 문제는 해결한 것이 아니다. 기존과 동일한 gradient이므로 production 승격은 하지 않았다.
- 재현: node02 Slurm `final.sbatch`, `verify.sbatch`. 결과 JSON에 소스 SHA-256/cubin 경로/600개 timing sample 기록.
- 현재 SVG는 mean/rstd의 K3 WRITE → B1 TMA READ와 PyTorch 수식을 반영한다.
