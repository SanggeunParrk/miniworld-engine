# MiniPairformer 1블록: 최신 TriMul + 새 CUDA Transition

> **후속 해결:** 이 문서는 수정 전 측정 이력이다. 이후 gate 미분 곱셈 순서를 복원하여 L384 엄격한 LN gradient 검증을 통과했다. [수정·검증 결과](../trimul_ln_gradient_20260922/README.md). 원래 실패 기록과 당시 시간은 보존한다.

job15581 · 2026-09-22 · 단위 ms

| 모드 | 이전 Triton TriMul + Triton Transition | 이전 H100 TriMul + Triton Transition | 최신 TriMul + Triton Transition | 최신 TriMul + 새 CUDA Transition |
|---|---|---|---|---|
| 추론 | 0.600 | 0.578 | 0.450 | 0.406 |
| 학습 forward | 0.676 | 0.626 | 0.467 | 0.431 |
| 학습 fwd+bwd | 2.522 | 2.470 | 1.965 | 1.582 |

- 추론: 이전 Triton 조합 대비 1.48배, 이전 H100 TriMul 조합 대비 1.42배; 새 Transition 연결 자체는 1.11배.
- 학습 forward: 이전 Triton 조합 대비 1.57배, 이전 H100 TriMul 조합 대비 1.45배; 새 Transition 연결 자체는 1.08배.
- 학습 fwd+bwd: 이전 Triton 조합 대비 1.59배, 이전 H100 TriMul 조합 대비 1.56배; 새 Transition 연결 자체는 1.24배.

## 배선과 비교 범위

- 한 블록은 양방향 TriMul → Transition이다. backward도 Transition의 실제 입력 gradient를 TriMul로 전달한다. 개별 모듈 시간의 합계가 아니다.
- 최신 Transition은 `/home/psk6950/miniworld-engine-tbwd`의 D128 hand-CUDA 구현을 그대로 복사한 `transition_snapshot/`이다. 원본 경로와 SHA-256은 `transition_sources.json`에 보존했다.
- Forward는 LN → expand/SwiGLU → squeeze/residual을 `transition_fwd_fused`에 융합한다. 학습에서는 x_n과 LN 통계를 저장하고 추론에서는 저장하지 않는다.
- Backward는 `transition_bwd_fused`와 작은 `reduce_partials`로 입력 및 모든 파라미터 gradient를 계산한다. 기존 Triton backward로 돌아가지 않았는지 profiler trace로 확인한다.
- 최신 TriMul은 Anthropic 유래 forward + 선택한 cache-policy B1 + K128 단일 B7 후보다. 추론은 infer_k1/infer_k3를 사용한다.
- 이전 TriMul은 보존한 Anthropic 도입 전 Triton/H100 경로다. 두 이전 조합의 Transition은 앞선 벤치와 같은 Triton full-K/residual 경로다. 과거 전체 설치 환경을 재현한 비교는 아니다.
- 네 조합 모두 동일 입력과 가중치, 동일 정밀도를 쓴다. 최신 TriMul + 기존 Transition 행을 추가하여 새 Transition만의 기여를 분리했다.

## 측정 조건

H100 node01, B=1/L=384/C=128, BF16, TriMul hidden128 per direction, Transition expansion4.
학습은 고정 row dropout25%, 추론은 dropout0. pair mask, 두 residual, live weight packing, 입력과 15개 파라미터 gradient를 포함한다.
Transition projection은 BF16, LN 파라미터는 FP32, squeeze 가중치는 nonzero다.
static compile + 수동 CUDA graph로 모드마다 5×250회 교차 측정한다.
optimizer, dropout RNG 생성, 컴파일, CPU dispatch는 제외한다. 일부 기존 Triton cache miss는 측정 전 최대24개 후보로 처리한다.

## 정확도와 적용 상태

경로 간 BF16 비교는 forward 상대L2 0.5%, gradient 1% 이내를 검사한다.
입력과 squeeze 가중치를 바꿔 graph/eager도 비교한다. 기존 LN atomic 누산에는 5e-6 한도를 쓰고 나머지는 bit-exact를 요구한다.
**최신 TriMul의 앞선 엄격 검사에서 입력 LN gradient가 최대9.535e-6로 5e-6 한도를 넘은 문제는 남아 있다.**
이 비교의 통과가 그 문제를 해결한 것은 아니다. 최신 학습 시간은 개발 후보의 진단 결과이며, production 기본 배선으로 승격하지 않았다.

## 재현 및 근거

- [벤치 코드](bench.py) · [Slurm 제출 파일](bench.sbatch)
- [원본 결과와 kernel trace 목록](results.json) · [요약](summary.json)
- [Transition 소스 SHA-256](transition_sources.json)
- [앞선 Transition 고정 비교](../minipairformer_block_20260922/index.html)

저장소 루트에서 `sbatch runs/minipairformer_block_cuda_20260922/bench.sbatch`.
