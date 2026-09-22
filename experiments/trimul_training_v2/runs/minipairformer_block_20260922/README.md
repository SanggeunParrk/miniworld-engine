# MiniPairformer 한 블록: 학습·추론 비교 · 2026-09-22

> **후속 해결:** 이 문서는 수정 전 측정 이력이다. 이후 gate 미분 곱셈 순서를 복원하여 L384 엄격한 LN gradient 검증을 통과했다. [수정·검증 결과](../trimul_ln_gradient_20260922/README.md). 원래 실패 기록과 당시 시간은 보존한다.

**블록 구성 = 양방향 TriMul → Transition. 최신 조합 추론 0.447ms, 학습 fwd+bwd 1.947ms.**

## 동일 실행 비교

| 모드 | 이전 TriMul Triton | 이전 TriMul H100 | 최신 TriMul 후보 | Triton 대비 | H100 대비 |
|---|---:|---:|---:|---:|---:|
| 추론 | 0.598ms | 0.575ms | **0.447ms** | 1.34배 | 1.29배 |
| 학습 forward | 0.674ms | 0.624ms | **0.465ms** | 1.45배 | 1.34배 |
| 학습 fwd+bwd | 2.502ms | 2.457ms | **1.947ms** | 1.28배 | 1.26배 |

## 측정 범위

job15573, node01 H10080GB HBM3, B1/L384/C128, TriMul hidden128 per direction (packed256), Transition expansion4.
pair-only MiniPairformerBlock과 같은 두 연산을 직접 연결한 커널 조합 벤치다. 16블록 스택·single track·triangle attention은 포함하지 않는다.
최신 수동 backward는 Transition의 실제 입력 gradient를 TriMul로 전달한다. 두 모듈의 따로 잰 시간을 더한 값이 아니다.
학습은 고정 row dropout25%, 추론은 dropout0 및 backward용 저장 없음. 두 residual과 합성 pair mask, live weight packing, 입력+15개 파라미터 gradient를 포함한다.
optimizer, dropout RNG 생성, compilation, CPU dispatch는 제외한다. static compile+manual CUDA graph, 모드마다5×250회 교차 측정.
Transition projection은 BF16, LN 파라미터는 FP32. squeeze는 nonzero 초기화하여 실제 branch와 gradients를 검사했다.

## 비교 대상의 정확한 의미

세 조합 모두 동일한 D128 Triton full-K/residual Transition과 동일 가중치를 쓴다. 바뀌는 것은 TriMul이다.
이전 TriMul은 보존된 Anthropic 도입 전 Triton/H100 경로와 기존 선택 config를 사용한다. 과거 전체 설치 환경 또는 옛 Transition 구현까지 재현한 비교는 아니다.
실제 profiler에서 이전 H100 FrontSingleWarpgroup/ParityF567/DualBackwardSm90, 최신 b1_fused/b7_joint를 확인했다.
추론 최신 경로는 infer_k1/infer_k3이며 학습 save_k3와 B7이 없다. 두 경로 모두 Transition kernel을 이어 실행한다.
최신 조합은 기본 MiniWorld 자동 dispatch로 승격한 상태가 아니다. 이전에 확인한 LN gradient 제한이 남아 있다.

## 검증과 한계

경로 간 비교는 기존 BF16 cross-algorithm 기준 forward0.5%, gradient1%를 사용했다. 이는 엄격한 동일 알고리즘 검증과 별개다.
최신 경로의 이전 Triton 대비 최대 상대L2: 학습 전체 0.0012525, 학습 forward 0.00018623, 추론 0.00345068.
새 입력/가중치 변형 후 graph/eager 비교에서는 LN gradient만5e-6 한도를 사용하고 나머지는 bit-exact를 요구했다. 실제 최대는1.19233e-06다.
기존 LN 구현의 FP32 atomic 누산은 반복 실행에서 마지막 비트가 달라진다. job15566/15569는 이를 모두 bit-exact로 요구해 중단했고, 원래 LN 수치 한도를 사용한 최종 job에서 검증을 통과했다.
앞선 최신 TriMul 전체 검사에서 입력 LN gradient 최대9.535e-6 >5e-6였던 문제를 이 비교가 해결한 것은 아니다. **최신 학습 시간은 정확도 검증이 남은 후보의 진단값**이다.
Transition 및 일부 LN autotune cache miss는 타이밍 전에 heuristic24개 후보로 처리했다. 완전한 config-space 최적 튜닝을 했다는 주장은 아니다.
Transition은 모든 arm에서 같은 구현을 썼다. 런타임 autotuner가 best_config/cache를 노출하지 않아 해당 객체의 config 결과는 null/empty로 기록되었다.
반복 측정에 사용한 cubin/source·cache 환경을 이전 실행과 동일하다고 가정하지 않는다. 비율은 이번 실행 내부에서만 계산했다.

## 재현·자료

- [원본 결과 / 모든 시간표본 / kernel trace 목록](results.json)
- [요약과 telemetry](summary.json)
- [벤치 코드](bench.py)
- [이전 경로의 고정 config](pin_previous.py)
- [최신 TriMul 한계](../trimul_full_latest_20260922/README.md)

저장소 루트에서 `sbatch --job-name=minipair-block runs/minipairformer_block_20260922/bench.sbatch`. 새 production 코드·기본 배선은 변경하지 않았다.
